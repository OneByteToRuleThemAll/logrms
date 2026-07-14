from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from logrms import CONFIDENCE_DEFAULT, LogRMS
from logrms.reference import _history_confidence_and_update_


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src" / "logrms" / "reference.py"
EXPECTED_SHA256 = "1a59b6b948d565c646a435ad9c97f84b95579c347b981a32ebd81b119858e4f4"


def test_reference_source_hash() -> None:
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_log8_uses_two_persistent_tensor_bytes_per_element() -> None:
    parameter = torch.nn.Parameter(torch.linspace(-1.0, 1.0, 257))
    optimizer = LogRMS([parameter], backend="8bit", max_norm=None)
    parameter.grad = torch.linspace(1.0, -1.0, parameter.numel())
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["v"].dtype == torch.int8
    assert state["history"].dtype == torch.uint8
    assert optimizer.state_bytes() == 2 * parameter.numel()
    assert optimizer.projected_state_bytes() == 2 * parameter.numel()


def test_float32_parameters_keep_log8_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = LogRMS([parameter], lr=1e-2, backend="8bit", max_norm=None)
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    assert parameter.dtype == torch.float32
    assert optimizer.state[parameter]["v"].dtype == torch.int8


def test_confidence_is_neutral_until_eight_nonzero_signs() -> None:
    history = torch.zeros(1, dtype=torch.uint8)
    observed = []
    for step in range(8):
        sign = 1.0 if step % 2 == 0 else -1.0
        confidence = _history_confidence_and_update_(
            history, torch.tensor([sign]), CONFIDENCE_DEFAULT
        )
        observed.append(float(confidence.item()))
    assert observed[:7] == [1.0] * 7
    assert observed[7] == pytest.approx(0.2)


def test_zero_gradient_does_not_change_history() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = LogRMS([parameter], backend="8bit", max_norm=None, weight_decay=0.0)
    parameter.grad = torch.tensor([1.0, -1.0])
    optimizer.step()
    before = optimizer.state[parameter]["history"].clone()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.equal(optimizer.state[parameter]["history"], before)


def test_decoupled_weight_decay_applies_with_zero_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = LogRMS(
        [parameter], lr=0.1, weight_decay=0.2, backend="8bit", max_norm=None
    )
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([1.96]))


def test_nonfinite_detection_precedes_state_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = LogRMS(
        [parameter], backend="8bit", max_norm=1.0, error_if_nonfinite=True
    )
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(parameter, torch.tensor([1.0]))


def test_stochastic_rounding_is_reproducible_for_fixed_seed() -> None:
    left = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 129))
    right = torch.nn.Parameter(left.detach().clone())
    kwargs = dict(
        lr=3e-3,
        backend="8bit",
        max_norm=None,
        stochastic_rounding=True,
        rounding_seed=991,
    )
    first = LogRMS([left], **kwargs)
    second = LogRMS([right], **kwargs)
    for step in range(12):
        generator = torch.Generator().manual_seed(10_000 + step)
        gradient = torch.randn(left.shape, generator=generator)
        left.grad = gradient.clone()
        right.grad = gradient.clone()
        first.step()
        second.step()
    assert torch.equal(left, right)
    assert torch.equal(first.state[left]["v"], second.state[right]["v"])
    assert torch.equal(first.state[left]["history"], second.state[right]["history"])


def test_checkpoint_roundtrip_preserves_integer_state() -> None:
    source_parameter = torch.nn.Parameter(torch.randn(17))
    source = LogRMS([source_parameter], backend="8bit", max_norm=None)
    source_parameter.grad = torch.randn_like(source_parameter)
    source.step()
    restored_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    restored = LogRMS([restored_parameter], backend="8bit", max_norm=None)
    restored.load_state_dict(copy.deepcopy(source.state_dict()))
    state = restored.state[restored_parameter]
    assert state["v"].dtype == torch.int8
    assert state["history"].dtype == torch.uint8
    assert torch.equal(state["v"], source.state[source_parameter]["v"])
    assert torch.equal(state["history"], source.state[source_parameter]["history"])


def test_closure_and_noncontiguous_parameter() -> None:
    parameter = torch.nn.Parameter(torch.randn(4, 3).t())
    assert not parameter.is_contiguous()
    optimizer = LogRMS([parameter], backend="8bit", max_norm=None)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = parameter.square().mean()
        loss.backward()
        return loss

    returned = optimizer.step(closure)
    assert returned is not None and torch.isfinite(returned)
    assert optimizer.state[parameter]["v"].shape == parameter.shape
