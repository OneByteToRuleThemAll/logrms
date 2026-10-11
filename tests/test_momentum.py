from __future__ import annotations

import copy
import shutil

import pytest
import torch

from logrms import LogAdam, LogAdamCompiled
from logrms.momentum import _RATIO_MAX, _decode_ratio, _encode_ratio


def make_pair(*, stochastic_rounding: bool = True, weight_decay: float = 0.01, **compiled):
    torch.manual_seed(123)
    initial = [torch.randn(37), torch.randn(5, 11), torch.randn(3, 4, 5)]
    reference_params = [torch.nn.Parameter(value.clone()) for value in initial]
    compiled_params = [torch.nn.Parameter(value.clone()) for value in initial]
    common = dict(
        lr=3e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        max_norm=None,
        stochastic_rounding=stochastic_rounding,
        rounding_seed=991,
    )
    compiled.setdefault("compile_backend", "eager")
    compiled.setdefault("compile_mode", "")
    reference = LogAdam(reference_params, **common)
    fast = LogAdamCompiled(
        compiled_params, **common, fallback_on_compile_error=False, **compiled
    )
    return reference_params, compiled_params, reference, fast


def set_grads(reference_params, compiled_params, step: int) -> None:
    generator = torch.Generator().manual_seed(10_000 + step)
    for index, (reference, compiled) in enumerate(zip(reference_params, compiled_params)):
        grad = torch.randn(reference.shape, generator=generator)
        if step % 3 == 0:
            grad.reshape(-1)[:: 5 + index] = 0.0
        reference.grad = grad.clone()
        compiled.grad = grad.clone()


def test_two_persistent_int8_bytes_per_element() -> None:
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 257))
    optimizer = LogAdam([parameter], max_norm=None)
    parameter.grad = torch.linspace(1.0, -1.0, parameter.numel())
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["v"].dtype == torch.int8
    assert state["m"].dtype == torch.int8
    assert optimizer.state_bytes() == 2 * parameter.numel()
    assert optimizer.projected_state_bytes() == 2 * parameter.numel()


def test_ratio_codec_roundtrip_and_range() -> None:
    codes = torch.arange(-127, 128, dtype=torch.int8)
    decoded = _decode_ratio(codes)
    assert float(decoded.max()) == pytest.approx(_RATIO_MAX)
    assert float(decoded.min()) == pytest.approx(-_RATIO_MAX)
    encoded = _encode_ratio(decoded, stochastic_rounding=False, step=1, seed=0)
    assert torch.equal(encoded, codes)
    clipped = _encode_ratio(
        torch.tensor([1e6, -1e6]), stochastic_rounding=True, step=1, seed=0
    )
    assert clipped.tolist() == [127, -127]


def test_stochastic_ratio_rounding_is_unbiased() -> None:
    target = torch.full((200_000,), 0.2371)
    total = torch.zeros(())
    for step in range(1, 9):
        codes = _encode_ratio(target, stochastic_rounding=True, step=step, seed=5)
        total = total + _decode_ratio(codes).mean()
    assert float(total / 8) == pytest.approx(0.2371, rel=2e-3)


def test_first_step_is_sign_sized_like_adam() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    optimizer = LogAdam([parameter], lr=1e-2, max_norm=None)
    parameter.grad = torch.tensor([0.25, -3.0, 0.0])
    optimizer.step()
    torch.testing.assert_close(
        parameter, torch.tensor([0.99, -1.99, 0.5]), rtol=0, atol=1e-6
    )


def test_tracks_fp32_adamw_on_a_noisy_quadratic() -> None:
    torch.manual_seed(0)
    start = torch.randn(4096)
    ours = torch.nn.Parameter(start.clone())
    theirs = torch.nn.Parameter(start.clone())
    kwargs = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    logadam = LogAdam([ours], max_norm=None, **kwargs)
    adamw = torch.optim.AdamW([theirs], **kwargs)
    generator = torch.Generator().manual_seed(1)
    for _ in range(300):
        noise = torch.randn(start.shape, generator=generator)
        ours.grad = ours.detach() + noise
        theirs.grad = theirs.detach() + noise
        logadam.step()
        adamw.step()
    ours_loss = float(ours.detach().square().mean())
    theirs_loss = float(theirs.detach().square().mean())
    assert ours_loss < 0.1 * float(start.square().mean())
    assert ours_loss == pytest.approx(theirs_loss, rel=0.25)


def test_nonfinite_detection_precedes_state_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = LogAdam([parameter], max_norm=1.0, error_if_nonfinite=True)
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.tensor([1.0]))


def test_checkpoint_roundtrip_preserves_int8_state_and_trajectory() -> None:
    torch.manual_seed(3)
    source_parameter = torch.nn.Parameter(torch.randn(64))
    source = LogAdam([source_parameter], max_norm=None, rounding_seed=17)
    for _ in range(5):
        source_parameter.grad = torch.randn_like(source_parameter)
        source.step()
    restored_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    restored = LogAdam([restored_parameter], max_norm=None)
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    state = restored.state[restored_parameter]
    assert state["v"].dtype == torch.int8 and state["m"].dtype == torch.int8
    assert torch.equal(state["v"], source.state[source_parameter]["v"])
    assert torch.equal(state["m"], source.state[source_parameter]["m"])
    grad = torch.randn_like(source_parameter)
    source_parameter.grad = grad.clone()
    restored_parameter.grad = grad.clone()
    source.step()
    restored.step()
    assert torch.equal(source_parameter, restored_parameter)


@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_compiled_group_matches_reference(stochastic_rounding: bool) -> None:
    reference_params, compiled_params, reference, compiled = make_pair(
        stochastic_rounding=stochastic_rounding
    )
    for step in range(12):
        set_grads(reference_params, compiled_params, step)
        reference.step()
        compiled.step()
        for ref_param, fast_param in zip(reference_params, compiled_params):
            torch.testing.assert_close(ref_param, fast_param, rtol=2e-6, atol=2e-7)
            ref_state, fast_state = reference.state[ref_param], compiled.state[fast_param]
            assert ref_state["step"] == fast_state["step"]
            assert torch.equal(ref_state["v"], fast_state["v"])
            assert torch.equal(ref_state["m"], fast_state["m"])


def test_reference_checkpoint_loads_into_compiled() -> None:
    reference_params, compiled_params, reference, compiled = make_pair()
    for step in range(4):
        set_grads(reference_params, compiled_params, step)
        reference.step()
    compiled.load_state_dict(copy.deepcopy(reference.state_dict()))
    for compiled_param, reference_param in zip(compiled_params, reference_params):
        compiled_param.data.copy_(reference_param)
    set_grads(reference_params, compiled_params, 99)
    reference.step()
    compiled.step()
    for ref_param, fast_param in zip(reference_params, compiled_params):
        torch.testing.assert_close(ref_param, fast_param, rtol=2e-6, atol=2e-7)
        assert torch.equal(reference.state[ref_param]["m"], compiled.state[fast_param]["m"])


def test_compile_failure_runs_the_step_exactly_once() -> None:
    reference_params, compiled_params, reference, compiled = make_pair()

    def fail(*args, **kwargs):
        raise RuntimeError("intentional compile failure")

    compiled.fallback_on_compile_error = True
    compiled._compiled_group_step = fail
    for step in range(3):
        set_grads(reference_params, compiled_params, step)
        reference.step()
        if step == 0:
            with pytest.warns(RuntimeWarning, match="torch.compile failed"):
                compiled.step()
        else:
            compiled.step()
    assert compiled.compile_failures == 1
    assert not compiled.compilation_active
    for ref_param, fast_param in zip(reference_params, compiled_params):
        torch.testing.assert_close(ref_param, fast_param, rtol=2e-6, atol=2e-7)
        assert reference.state[ref_param]["step"] == compiled.state[fast_param]["step"] == 3


@pytest.mark.skipif(shutil.which("g++") is None and shutil.which("c++") is None,
                    reason="Inductor needs a C++ compiler on CPU")
def test_real_inductor_graph_stays_close_to_reference() -> None:
    """Exercise the real Inductor backend, not only ``compile_backend='eager'``.

    Generated code may round log2/exp2 differently from eager PyTorch, so a
    stored code can occasionally land on the other side of a rounding
    boundary. The check is therefore statistical rather than bit-exact.
    """
    reference_params, compiled_params, reference, compiled = make_pair(
        compile_backend="inductor", compile_mode="default"
    )
    for step in range(8):
        set_grads(reference_params, compiled_params, step)
        reference.step()
        compiled.step()
    assert compiled.compilation_active and compiled.compile_failures == 0
    total = mismatched = 0
    for ref_param, fast_param in zip(reference_params, compiled_params):
        torch.testing.assert_close(ref_param, fast_param, rtol=0, atol=3e-3)
        for key in ("v", "m"):
            difference = reference.state[ref_param][key].int() - compiled.state[fast_param][key].int()
            assert int(difference.abs().max()) <= 2
            mismatched += int(difference.ne(0).sum())
            total += difference.numel()
    assert mismatched <= 0.02 * total
