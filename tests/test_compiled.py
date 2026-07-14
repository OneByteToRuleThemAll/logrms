from __future__ import annotations

import copy

import pytest
import torch

from logrms import CONFIDENCE_DEFAULT, LogRMS, LogRMSCompiled


def make_pair(*, stochastic_rounding: bool = True, weight_decay: float = 0.01):
    torch.manual_seed(123)
    initial = [torch.randn(37), torch.randn(5, 11), torch.randn(3, 4, 5)]
    reference_params = [torch.nn.Parameter(value.clone()) for value in initial]
    compiled_params = [torch.nn.Parameter(value.clone()) for value in initial]
    common = dict(
        lr=3e-3,
        beta2=0.999,
        eps=1e-8,
        weight_decay=weight_decay,
        backend="8bit",
        confidence_table=CONFIDENCE_DEFAULT,
        max_norm=None,
        error_if_nonfinite=True,
        stochastic_rounding=stochastic_rounding,
        rounding_seed=991,
    )
    reference = LogRMS(reference_params, **common)
    compiled = LogRMSCompiled(
        compiled_params,
        **common,
        compile_backend="eager",
        compile_mode="",
        fullgraph=True,
        fallback_on_compile_error=False,
    )
    return reference_params, compiled_params, reference, compiled


def set_grads(reference_params, compiled_params, step: int) -> None:
    generator = torch.Generator().manual_seed(10_000 + step)
    for index, (reference, compiled) in enumerate(
        zip(reference_params, compiled_params)
    ):
        grad = torch.randn(reference.shape, generator=generator)
        if step % 3 == 0:
            grad.reshape(-1)[:: 5 + index] = 0.0
        reference.grad = grad.clone()
        compiled.grad = grad.clone()


def assert_parity(reference_params, compiled_params, reference, compiled) -> None:
    for reference_param, compiled_param in zip(reference_params, compiled_params):
        torch.testing.assert_close(
            reference_param, compiled_param, rtol=2e-6, atol=2e-7
        )
        reference_state = reference.state[reference_param]
        compiled_state = compiled.state[compiled_param]
        assert reference_state["step"] == compiled_state["step"]
        assert reference_state["rounding_seed"] == compiled_state["rounding_seed"]
        assert torch.equal(reference_state["v"], compiled_state["v"])
        assert torch.equal(reference_state["history"], compiled_state["history"])


@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_compiled_group_matches_reference(stochastic_rounding: bool) -> None:
    reference_params, compiled_params, reference, compiled = make_pair(
        stochastic_rounding=stochastic_rounding
    )
    for step in range(12):
        set_grads(reference_params, compiled_params, step)
        reference.step()
        compiled.step()
        assert_parity(reference_params, compiled_params, reference, compiled)
    assert reference.state_bytes() == compiled.state_bytes()
    assert compiled.projected_state_bytes() == 2 * sum(
        parameter.numel() for parameter in compiled_params
    )


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
    assert_parity(reference_params, compiled_params, reference, compiled)


def test_compiled_checkpoint_loads_into_reference() -> None:
    reference_params, compiled_params, reference, compiled = make_pair()
    for step in range(4):
        set_grads(reference_params, compiled_params, step)
        compiled.step()
    reference.load_state_dict(copy.deepcopy(compiled.state_dict()))
    for reference_param, compiled_param in zip(reference_params, compiled_params):
        reference_param.data.copy_(compiled_param)
    set_grads(reference_params, compiled_params, 101)
    reference.step()
    compiled.step()
    assert_parity(reference_params, compiled_params, reference, compiled)


def test_compile_failure_uses_same_tensor_update() -> None:
    reference_params, compiled_params, reference, compiled = make_pair()

    def fail(*args, **kwargs):
        raise RuntimeError("intentional compile failure")

    compiled.fallback_on_compile_error = True
    compiled._compiled_group_step = fail
    set_grads(reference_params, compiled_params, 1)
    reference.step()
    with pytest.warns(RuntimeWarning, match="torch.compile failed"):
        compiled.step()
    assert compiled.compile_failures == 1
    assert not compiled.compilation_active
    assert_parity(reference_params, compiled_params, reference, compiled)


def test_zero_gradients_do_not_advance_history() -> None:
    reference_params, compiled_params, reference, compiled = make_pair(
        stochastic_rounding=False, weight_decay=0.0
    )
    for reference_param, compiled_param in zip(reference_params, compiled_params):
        reference_param.grad = torch.zeros_like(reference_param)
        compiled_param.grad = torch.zeros_like(compiled_param)
    reference.step()
    compiled.step()
    assert_parity(reference_params, compiled_params, reference, compiled)
    for parameter in compiled_params:
        assert torch.count_nonzero(compiled.state[parameter]["history"]) == 0
