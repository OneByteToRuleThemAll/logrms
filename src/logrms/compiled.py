"""torch.compile implementation of the LogRMS log8 update.

Validation, global clipping, state initialization, checkpoint metadata, and
closures remain eager. Dense tensors in each parameter group are passed to one
compiled graph. Step numbers and hyperparameters are tensor inputs, which
avoids specializing a new graph for every training step.

The log8 per-coordinate state remains exactly one int8 second moment plus one
uint8 history byte.  Runtime staging tensors are optimizer-level caches, are
not serialized, and are not counted as per-coordinate optimizer state.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from .reference import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_RMS,
    CONFIDENCE_V106,
    CONFIDENCE_V107_FIXED,
    VERSION as REFERENCE_VERSION,
    LogRMS,
    _get_luts,
)

__all__ = [
    "LogRMSCompiled",
    "COMPILED_VERSION",
    "REFERENCE_VERSION",
    "CONFIDENCE_DEFAULT",
    "CONFIDENCE_RMS",
    "CONFIDENCE_V106",
    "CONFIDENCE_V107_FIXED",
]

COMPILED_VERSION = "current"
VERSION = COMPILED_VERSION
_LOG8_SCALE = 2.0
_UINT32_MASK = 0xFFFFFFFF
_UNIFORM_MANTISSA_MASK = 0x00FFFFFF
_UNIFORM_DENOMINATOR = float(1 << 24)


def _compile(function: Any, *, backend: str, mode: str, fullgraph: bool, dynamic: bool):
    options: dict[str, Any] = {
        "backend": backend,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    if backend == "inductor" and mode:
        options["mode"] = mode
    return torch.compile(function, **options)


def _compiled_log8_group_step_(
    params: list[Tensor],
    grads: list[Tensor],
    v_states: list[Tensor],
    histories: list[Tensor],
    steps: Tensor,
    seeds: Tensor,
    bias_corrections2: Tensor,
    hyperparameters: Tensor,
    lengths_lut: Tensor,
    bits_lut: Tensor,
    offsets_lut: Tensor,
    confidence_lut: Tensor,
    maximize: bool,
    stochastic_rounding: bool,
) -> None:
    """Mutate one fixed log8 parameter group; intended as one Inductor graph."""
    lr = hyperparameters[0]
    beta2 = hyperparameters[1]
    eps = hyperparameters[2]
    weight_decay = hyperparameters[3]

    for index in range(len(params)):
        param = params[index]
        grad_fp32 = grads[index].detach().to(dtype=torch.float32)
        if maximize:
            grad_fp32 = -grad_fp32

        v_state = v_states[index]
        decoded = torch.exp2(v_state.to(dtype=torch.float32) / _LOG8_SCALE)
        decoded = torch.where(v_state.eq(-128), torch.zeros_like(decoded), decoded)
        v_fp32 = decoded.mul(beta2).add(
            grad_fp32.mul(grad_fp32).mul(1.0 - beta2)
        )

        positive = v_fp32.gt(0)
        safe_v = torch.clamp_min(v_fp32, torch.finfo(torch.float32).tiny)
        scaled_log = torch.log2(safe_v) * _LOG8_SCALE
        if stochastic_rounding:
            lower = torch.floor(scaled_log).clamp(-127, 127)
            upper = torch.ceil(scaled_log).clamp(-127, 127)
            lower_value = torch.exp2(lower / _LOG8_SCALE)
            upper_value = torch.exp2(upper / _LOG8_SCALE)
            width = upper_value - lower_value
            probability = (v_fp32 - lower_value).div(
                torch.clamp_min(width, torch.finfo(torch.float32).tiny)
            ).clamp_(0.0, 1.0)
            probability = torch.where(
                width.gt(0), probability, torch.zeros_like(probability)
            )

            flat = torch.arange(
                v_fp32.numel(), dtype=torch.int64, device=v_fp32.device
            )
            offset = torch.bitwise_and(
                steps[index] * 0x9E3779B9 + seeds[index] * 0x85EBCA6B,
                _UINT32_MASK,
            )
            random_bits = torch.bitwise_and(flat + offset, _UINT32_MASK)
            random_bits = torch.bitwise_xor(
                random_bits,
                torch.bitwise_and(
                    torch.bitwise_left_shift(random_bits, 13), _UINT32_MASK
                ),
            )
            random_bits = torch.bitwise_xor(
                random_bits, torch.bitwise_right_shift(random_bits, 17)
            )
            random_bits = torch.bitwise_xor(
                random_bits,
                torch.bitwise_and(
                    torch.bitwise_left_shift(random_bits, 5), _UINT32_MASK
                ),
            )
            uniform = torch.bitwise_and(
                random_bits, _UNIFORM_MANTISSA_MASK
            ).to(torch.float32)
            uniform = uniform.mul(1.0 / _UNIFORM_DENOMINATOR).reshape(v_fp32.shape)
            quantized = torch.where(uniform.lt(probability), upper, lower)
        else:
            quantized = torch.round(scaled_log).clamp(-127, 127)

        quantized = quantized.to(dtype=torch.int8)
        v_state.copy_(
            torch.where(positive, quantized, torch.full_like(quantized, -128))
        )

        history = histories[index]
        history_index = history.to(dtype=torch.long)
        length = lengths_lut[history_index]
        packed = bits_lut[history_index]
        nonzero = grad_fp32.ne(0)
        sign = grad_fp32.gt(0).to(dtype=torch.uint8)
        sequence = torch.bitwise_or(torch.bitwise_left_shift(packed, 1), sign)
        ready = torch.logical_and(nonzero, length.eq(7))
        sequence_confidence = confidence_lut[sequence.to(dtype=torch.long)]
        confidence = torch.where(
            ready, sequence_confidence, torch.ones_like(grad_fp32)
        )
        next_length = torch.clamp(
            length.to(dtype=torch.int16) + 1, max=7
        ).to(torch.uint8)
        next_bits = torch.bitwise_and(sequence, 0x7F)
        next_code = (
            offsets_lut[next_length.to(dtype=torch.long)].to(dtype=torch.int16)
            + next_bits.to(dtype=torch.int16)
        ).to(dtype=torch.uint8)
        history.copy_(torch.where(nonzero, next_code, history))

        bias_correction2 = bias_corrections2[index]
        denom = v_fp32.div(bias_correction2).sqrt().add(eps)
        update = torch.where(
            nonzero,
            grad_fp32.div(denom),
            torch.zeros_like(grad_fp32),
        ).mul(confidence)

        param.mul_(1.0 - lr * weight_decay)
        param.add_(update.to(dtype=param.dtype) * (-lr))


@dataclass
class _RuntimeBatch:
    signature: tuple[int, ...]
    steps_cpu: Tensor
    steps_device: Tensor
    seeds_device: Tensor
    bias_corrections2_cpu: Tensor
    bias_corrections2_device: Tensor
    hyperparameters_cpu: Tensor
    hyperparameters_device: Tensor


class LogRMSCompiled(LogRMS):
    """Whole-parameter-group compiled LogRMS implementation.

    The optimizer accepts the same public arguments and checkpoints as
    :class:`logrms.reference.LogRMS`. Non-log8 or non-strided configurations
    use the frozen eager step.  Dense log8 groups are captured as fixed-list
    Inductor graphs; changing the group structure can cause a new compilation.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        backend: Optional[str] = "8bit",
        confidence_table: Optional[Sequence[float]] = None,
        max_norm: Optional[float] = 1.0,
        error_if_nonfinite: bool = True,
        maximize: bool = False,
        stochastic_rounding: bool = True,
        rounding_seed: int = 0,
        *,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = True,
        dynamic: bool = False,
        fallback_on_compile_error: bool = True,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            backend=backend,
            confidence_table=confidence_table,
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
            maximize=maximize,
            stochastic_rounding=stochastic_rounding,
            rounding_seed=rounding_seed,
        )
        if not hasattr(torch, "compile"):
            raise RuntimeError("LogRMSCompiled requires torch.compile")
        self.compile_backend = str(compile_backend)
        self.compile_mode = str(compile_mode)
        self.compile_fullgraph = bool(fullgraph)
        self.compile_dynamic = bool(dynamic)
        self.fallback_on_compile_error = bool(fallback_on_compile_error)
        self.compile_failures = 0
        self._compiled_enabled = True
        self._runtime_batches: dict[tuple[int, str, Optional[int]], _RuntimeBatch] = {}
        self._compiled_group_step = _compile(
            _compiled_log8_group_step_,
            backend=self.compile_backend,
            mode=self.compile_mode,
            fullgraph=self.compile_fullgraph,
            dynamic=self.compile_dynamic,
        )

    @property
    def compilation_active(self) -> bool:
        return self._compiled_enabled and all(
            str(group["backend"]) == "8bit" for group in self.param_groups
        )

    def _runtime_batch(
        self,
        group_index: int,
        params: list[Tensor],
        steps: list[int],
        seeds: list[int],
        group: dict[str, Any],
    ) -> _RuntimeBatch:
        device = params[0].device
        key = (group_index, device.type, device.index)
        signature = tuple(id(param) for param in params)
        runtime = self._runtime_batches.get(key)
        if runtime is None or runtime.signature != signature:
            pin = device.type == "cuda"
            steps_cpu = torch.empty(len(params), dtype=torch.int64, pin_memory=pin)
            hparams_cpu = torch.empty(4, dtype=torch.float32, pin_memory=pin)
            runtime = _RuntimeBatch(
                signature=signature,
                steps_cpu=steps_cpu,
                steps_device=torch.empty(len(params), dtype=torch.int64, device=device),
                seeds_device=torch.tensor(seeds, dtype=torch.int64, device=device),
                bias_corrections2_cpu=torch.empty(
                    len(params), dtype=torch.float32, pin_memory=pin
                ),
                bias_corrections2_device=torch.empty(
                    len(params), dtype=torch.float32, device=device
                ),
                hyperparameters_cpu=hparams_cpu,
                hyperparameters_device=torch.empty(4, dtype=torch.float32, device=device),
            )
            self._runtime_batches[key] = runtime

        for index, step in enumerate(steps):
            runtime.steps_cpu[index] = step
            runtime.bias_corrections2_cpu[index] = (
                1.0 - float(group["beta2"]) ** step
            )
        runtime.hyperparameters_cpu[0] = float(group["lr"])
        runtime.hyperparameters_cpu[1] = float(group["beta2"])
        runtime.hyperparameters_cpu[2] = float(group["eps"])
        runtime.hyperparameters_cpu[3] = float(group["weight_decay"])
        runtime.steps_device.copy_(runtime.steps_cpu, non_blocking=device.type == "cuda")
        runtime.bias_corrections2_device.copy_(
            runtime.bias_corrections2_cpu, non_blocking=device.type == "cuda"
        )
        runtime.hyperparameters_device.copy_(
            runtime.hyperparameters_cpu, non_blocking=device.type == "cuda"
        )
        return runtime

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        if not self.compilation_active:
            return LogRMS.step(self, closure)

        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grouped_work: list[
            tuple[int, dict[str, Any], list[tuple[Tensor, Tensor, int]]]
        ] = []
        params_with_grad: list[Tensor] = []
        parameter_ordinal = 0
        for group_index, group in enumerate(self.param_groups):
            self._normalize_and_validate_group(group)
            work: list[tuple[Tensor, Tensor, int]] = []
            for param in group["params"]:
                grad = param.grad
                if grad is not None:
                    if grad.is_sparse:
                        raise RuntimeError("LogRMS does not support sparse gradients")
                    if grad.layout != torch.strided:
                        raise RuntimeError(
                            f"LogRMS requires strided gradients, got {grad.layout}"
                        )
                    work.append((param, grad, parameter_ordinal))
                    params_with_grad.append(param)
                parameter_ordinal += 1
            if work:
                grouped_work.append((group_index, group, work))

        self.last_step_skipped = False
        self.last_grad_norm = None
        if not grouped_work:
            return loss

        if self.max_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                params_with_grad,
                max_norm=self.max_norm,
                error_if_nonfinite=self.error_if_nonfinite,
            )
            self.last_grad_norm = total_norm.detach()
            if not bool(torch.isfinite(total_norm)):
                self.last_step_skipped = True
                return loss
        elif self.error_if_nonfinite:
            total_norm = torch.nn.utils.clip_grad_norm_(
                params_with_grad,
                max_norm=float("inf"),
                error_if_nonfinite=True,
            )
            self.last_grad_norm = total_norm.detach()

        for group_index, group, work in grouped_work:
            by_device: dict[torch.device, list[tuple[Tensor, Tensor, int]]] = {}
            for item in work:
                by_device.setdefault(item[0].device, []).append(item)

            for device_work in by_device.values():
                params, grads, v_states, histories = [], [], [], []
                steps, seeds = [], []
                for param, grad, ordinal in device_work:
                    state = self._init_or_validate_state(
                        param, "8bit", parameter_ordinal=ordinal
                    )
                    state["step"] += 1
                    params.append(param)
                    grads.append(grad)
                    v_states.append(state["v"])
                    histories.append(state["history"])
                    steps.append(int(state["step"]))
                    seeds.append(int(state["rounding_seed"]))

                runtime = self._runtime_batch(
                    group_index, params, steps, seeds, group
                )
                luts = _get_luts(params[0].device, self.confidence_table)
                arguments = (
                    params,
                    grads,
                    v_states,
                    histories,
                    runtime.steps_device,
                    runtime.seeds_device,
                    runtime.bias_corrections2_device,
                    runtime.hyperparameters_device,
                    *luts,
                    bool(group["maximize"]),
                    self.stochastic_rounding,
                )
                try:
                    if self._compiled_enabled:
                        self._compiled_group_step(*arguments)
                    else:
                        _compiled_log8_group_step_(*arguments)
                except Exception as error:
                    if not self.fallback_on_compile_error:
                        raise
                    self.compile_failures += 1
                    self._compiled_enabled = False
                    warnings.warn(
                        "torch.compile failed for LogRMS's log8 group; the "
                        "current group will run through the same tensor implementation "
                        f"eagerly; future steps use the frozen reference path: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _compiled_log8_group_step_(*arguments)

        return loss

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["_logrms_compiled"] = {
            "compiled_version": COMPILED_VERSION,
            "reference_version": REFERENCE_VERSION,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "fullgraph": self.compile_fullgraph,
            "dynamic": self.compile_dynamic,
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copied = dict(state_dict)
        copied.pop("_logrms_compiled", None)
        super().load_state_dict(copied)
        self._runtime_batches.clear()
