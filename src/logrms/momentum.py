"""LogAdam: a two-byte Adam-style optimizer built on the LogRMS log8 codec.

LogAdam is a separate mathematical variant, not a change to :class:`LogRMS`.
It keeps the int8 log-encoded second moment unchanged and uses the second byte
for a quantized first moment instead of sign history.

Persistent tensor state per parameter element (exactly two bytes):

* ``v`` (``int8``): log-encoded second moment, identical to the LogRMS log8
  codec. Code ``-128`` represents exact zero.
* ``m`` (``int8``): first moment stored *relative to* the square root of the
  decoded second moment, ``r = m / sqrt(decode(v))``, with square-root
  companding over ``[-8, 8]``.

Because the ratio ``m / sqrt(v)`` is bounded by the optimizer's own
recurrences, one fixed range covers every tensor and no per-tensor or
per-block scale metadata is stored.

Update
------

    v_t   = beta2 * v_(t-1) + (1 - beta2) * g_t^2
    m_t   = beta1 * m_(t-1) + (1 - beta1) * g_t
    p     = p * (1 - lr * weight_decay)
    p     = p - lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)

Both recurrences and the current update are computed in FP32 from the decoded
state. Quantization affects only what the next step reads back.

``LogAdam`` is the per-tensor eager reference. ``LogAdamCompiled`` sends each
parameter group through one ``torch.compile`` graph, following the same
boundary as :class:`logrms.compiled.LogRMSCompiled`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from .reference import (
    _decode_v,
    _encode_v_,
    _new_v_state,
    _stateless_uniform_like,
)

__all__ = ["LogAdam", "LogAdamCompiled", "LOGADAM_VERSION"]

LOGADAM_VERSION = "0.1"

_LOG8_SCALE = 2.0
_UINT32_MASK = 0xFFFFFFFF
_UNIFORM_MANTISSA_MASK = 0x00FFFFFF
_UNIFORM_DENOMINATOR = float(1 << 24)
_GOLDEN = 0x9E3779B9
_SEED_MIX = 0x85EBCA6B
# Decorrelates the first-moment dither stream from the second-moment stream.
_M_SEED_SALT = 0xA5A5A5A5
# |m| / sqrt(v) peaks near 6.6 for a run of same-sign gradients early in
# training at beta1=0.9, beta2=0.999, and tends to at most 1 in steady state.
_RATIO_MAX = 8.0


def _decode_ratio(code: Tensor) -> Tensor:
    """Decode an int8 companded ratio code to FP32 ``m / sqrt(v)``."""
    scaled = code.to(dtype=torch.float32) / 127.0
    return torch.sign(scaled) * scaled * scaled * _RATIO_MAX


def _encode_ratio(
    ratio: Tensor,
    *,
    stochastic_rounding: bool,
    step: int,
    seed: int,
) -> Tensor:
    """Quantize ``m / sqrt(v)`` to int8 with square-root companding."""
    ratio = ratio.clamp(-_RATIO_MAX, _RATIO_MAX)
    position = torch.sign(ratio) * torch.sqrt(ratio.abs() / _RATIO_MAX) * 127.0
    if not stochastic_rounding:
        return torch.round(position).clamp(-127, 127).to(dtype=torch.int8)
    lower = torch.floor(position).clamp(-127, 127)
    upper = (lower + 1.0).clamp(-127, 127)
    lower_scaled = lower / 127.0
    upper_scaled = upper / 127.0
    lower_value = torch.sign(lower_scaled) * lower_scaled * lower_scaled * _RATIO_MAX
    upper_value = torch.sign(upper_scaled) * upper_scaled * upper_scaled * _RATIO_MAX
    width = upper_value - lower_value
    probability = (ratio - lower_value).div(
        torch.clamp_min(width, torch.finfo(torch.float32).tiny)
    ).clamp(0.0, 1.0)
    probability = torch.where(width.gt(0), probability, torch.zeros_like(probability))
    uniform = _stateless_uniform_like(ratio, step=step, seed=seed)
    return torch.where(uniform.lt(probability), upper, lower).to(dtype=torch.int8)


class LogAdam(Optimizer):
    """Two-byte Adam-style optimizer: int8 log second moment + int8 first moment.

    Args:
        params: Iterable of parameters or parameter-group dictionaries.
        lr: Learning rate.
        betas: ``(beta1, beta2)`` exponential decay rates, each in ``[0, 1)``.
        eps: Non-negative denominator stabilizer.
        weight_decay: Non-negative decoupled weight decay.
        max_norm: Optional global gradient-norm clipping threshold applied
            inside the optimizer, as in :class:`LogRMS`. ``None`` disables it.
        error_if_nonfinite: Raise instead of updating when the global gradient
            norm is NaN or infinite.
        stochastic_rounding: Use stateless unbiased rounding for both stored
            moments.
        rounding_seed: Integer seed for the stateless rounding dither. It does
            not consume or alter PyTorch's random-number-generator state.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_norm: Optional[float] = 1.0,
        error_if_nonfinite: bool = True,
        stochastic_rounding: bool = True,
        rounding_seed: int = 0,
    ) -> None:
        if not math.isfinite(float(lr)) or float(lr) < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        beta1, beta2 = (float(betas[0]), float(betas[1]))
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if not math.isfinite(float(eps)) or float(eps) < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if max_norm is not None and (
            not math.isfinite(float(max_norm)) or float(max_norm) <= 0.0
        ):
            raise ValueError("max_norm must be a positive finite number or None")

        self.max_norm = None if max_norm is None else float(max_norm)
        self.error_if_nonfinite = bool(error_if_nonfinite)
        self.stochastic_rounding = bool(stochastic_rounding)
        self.rounding_seed = int(rounding_seed) & _UINT32_MASK
        self.last_grad_norm: Optional[Tensor] = None
        self.last_step_skipped = False

        defaults = dict(
            lr=float(lr),
            betas=(beta1, beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if not param.is_floating_point():
                    raise TypeError(
                        f"LogAdam only supports real floating-point parameters, got {param.dtype}"
                    )

    # ------------------------------------------------------------------ state

    def _init_state(self, param: Tensor, parameter_ordinal: int) -> dict[str, Any]:
        state = self.state[param]
        if not state:
            state["step"] = 0
            state["rounding_seed"] = (
                self.rounding_seed + int(parameter_ordinal) * _GOLDEN
            ) & _UINT32_MASK
            state["v"] = _new_v_state(param, "8bit")
            state["m"] = torch.zeros(
                tuple(param.shape), dtype=torch.int8, device=param.device
            )
        return state

    def _collect(
        self,
    ) -> tuple[list[tuple[dict[str, Any], list[tuple[Tensor, Tensor, int]]]], list[Tensor]]:
        grouped: list[tuple[dict[str, Any], list[tuple[Tensor, Tensor, int]]]] = []
        params_with_grad: list[Tensor] = []
        parameter_ordinal = 0
        for group in self.param_groups:
            work: list[tuple[Tensor, Tensor, int]] = []
            for param in group["params"]:
                grad = param.grad
                if grad is not None:
                    if grad.is_sparse or grad.layout != torch.strided:
                        raise RuntimeError("LogAdam requires dense strided gradients")
                    work.append((param, grad, parameter_ordinal))
                    params_with_grad.append(param)
                parameter_ordinal += 1
            if work:
                grouped.append((group, work))
        return grouped, params_with_grad

    def _clip(self, params_with_grad: list[Tensor]) -> bool:
        """Apply global clipping. Returns False when the step must be skipped."""
        self.last_step_skipped = False
        self.last_grad_norm = None
        if self.max_norm is None and not self.error_if_nonfinite:
            return True
        total_norm = torch.nn.utils.clip_grad_norm_(
            params_with_grad,
            max_norm=float("inf") if self.max_norm is None else self.max_norm,
            error_if_nonfinite=self.error_if_nonfinite,
        )
        self.last_grad_norm = total_norm.detach()
        if not bool(torch.isfinite(total_norm)):
            self.last_step_skipped = True
            return False
        return True

    # ------------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grouped, params_with_grad = self._collect()
        if not grouped or not self._clip(params_with_grad):
            return loss

        for group, work in grouped:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for param, raw_grad, ordinal in work:
                state = self._init_state(param, ordinal)
                state["step"] += 1
                step = int(state["step"])
                seed = int(state["rounding_seed"])

                grad = raw_grad.detach().to(dtype=torch.float32)
                v_prev = _decode_v(state["v"], "8bit")
                m_prev = _decode_ratio(state["m"]) * v_prev.sqrt()

                v = v_prev.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)
                m = m_prev.mul(beta1).add(grad, alpha=1.0 - beta1)

                stored_v = _encode_v_(
                    state["v"],
                    v,
                    "8bit",
                    stochastic_rounding=self.stochastic_rounding,
                    step=step,
                    seed=seed,
                )
                scale = stored_v.sqrt()
                ratio = torch.where(
                    scale.gt(0),
                    m.div(torch.clamp_min(scale, torch.finfo(torch.float32).tiny)),
                    torch.zeros_like(m),
                )
                state["m"].copy_(
                    _encode_ratio(
                        ratio,
                        stochastic_rounding=self.stochastic_rounding,
                        step=step,
                        seed=seed ^ _M_SEED_SALT,
                    )
                )

                denom = v.div(1.0 - beta2**step).sqrt_().add_(eps)
                update = torch.where(
                    m.ne(0),
                    m.div(1.0 - beta1**step).div(denom),
                    torch.zeros_like(m),
                )
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-lr)

        return loss

    # ------------------------------------------------------------- accounting

    def state_bytes(self) -> int:
        """Allocated persistent tensor payload bytes, excluding Python metadata."""
        total = 0
        for state in self.state.values():
            for key in ("v", "m"):
                value = state.get(key)
                if isinstance(value, Tensor):
                    total += value.numel() * value.element_size()
        return total

    def projected_state_bytes(self) -> int:
        """Projected tensor payload after all parameters have received gradients."""
        return 2 * sum(
            param.numel() for group in self.param_groups for param in group["params"]
        )

    # ------------------------------------------------------------ checkpoints

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["_logadam"] = {
            "version": LOGADAM_VERSION,
            "max_norm": self.max_norm,
            "error_if_nonfinite": self.error_if_nonfinite,
            "stochastic_rounding": self.stochastic_rounding,
            "rounding_seed": self.rounding_seed,
            "ratio_max": _RATIO_MAX,
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a checkpoint and restore the int8 state dtypes.

        PyTorch's generic loader casts state tensors to the parameter dtype.
        Every int8 code is exactly representable in FP16, BF16, and FP32, so
        the values survive that cast and are converted back here.
        """
        copied = dict(state_dict)
        metadata = copied.pop("_logadam", {})
        copied.pop("_logadam_compiled", None)
        ratio_max = float(metadata.get("ratio_max", _RATIO_MAX))
        if ratio_max != _RATIO_MAX:
            raise ValueError(
                f"LogAdam checkpoint uses ratio_max={ratio_max}, expected {_RATIO_MAX}"
            )
        super().load_state_dict(copied)
        for param, state in self.state.items():
            for key in ("v", "m"):
                value = state.get(key)
                if isinstance(value, Tensor):
                    if tuple(value.shape) != tuple(param.shape):
                        raise RuntimeError(
                            f"invalid LogAdam checkpoint: {key} state shape mismatch"
                        )
                    state[key] = value.round().to(dtype=torch.int8)
            if "step" in state:
                state["step"] = int(state["step"])
        if metadata:
            loaded_max_norm = metadata.get("max_norm", self.max_norm)
            self.max_norm = None if loaded_max_norm is None else float(loaded_max_norm)
            self.error_if_nonfinite = bool(
                metadata.get("error_if_nonfinite", self.error_if_nonfinite)
            )
            self.stochastic_rounding = bool(
                metadata.get("stochastic_rounding", self.stochastic_rounding)
            )
            self.rounding_seed = int(
                metadata.get("rounding_seed", self.rounding_seed)
            ) & _UINT32_MASK


# --------------------------------------------------------------------------
# Compiled execution
# --------------------------------------------------------------------------


def _hash_uniform(count: int, shape: Any, device: torch.device, offset: Tensor) -> Tensor:
    """Tensor form of ``reference._stateless_uniform_like``."""
    flat = torch.arange(count, dtype=torch.int64, device=device)
    value = torch.bitwise_and(flat + offset, _UINT32_MASK)
    value = torch.bitwise_xor(
        value, torch.bitwise_and(torch.bitwise_left_shift(value, 13), _UINT32_MASK)
    )
    value = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 17))
    value = torch.bitwise_xor(
        value, torch.bitwise_and(torch.bitwise_left_shift(value, 5), _UINT32_MASK)
    )
    uniform = torch.bitwise_and(value, _UNIFORM_MANTISSA_MASK).to(torch.float32)
    return uniform.mul(1.0 / _UNIFORM_DENOMINATOR).reshape(shape)


def _logadam_group_step_(
    params: list[Tensor],
    grads: list[Tensor],
    v_states: list[Tensor],
    m_states: list[Tensor],
    v_offsets: Tensor,
    m_offsets: Tensor,
    bias_corrections: Tensor,
    hyperparameters: Tensor,
    stochastic_rounding: bool,
) -> None:
    """Mutate one parameter group; intended as one Inductor graph.

    ``hyperparameters`` holds ``[beta1, 1-beta1, beta2, 1-beta2, eps, decay,
    -lr]`` as FP32. ``bias_corrections`` has one ``[1-beta1^t, 1-beta2^t]``
    row per tensor. All values are computed in Python and passed as tensors so
    that a new step does not specialize a new graph.
    """
    beta1 = hyperparameters[0]
    one_minus_beta1 = hyperparameters[1]
    beta2 = hyperparameters[2]
    one_minus_beta2 = hyperparameters[3]
    eps = hyperparameters[4]
    decay = hyperparameters[5]
    negative_lr = hyperparameters[6]
    tiny = torch.finfo(torch.float32).tiny

    for index in range(len(params)):
        param = params[index]
        grad = grads[index].detach().to(dtype=torch.float32)
        v_state = v_states[index]
        m_state = m_states[index]

        v_prev = torch.exp2(v_state.to(dtype=torch.float32) / _LOG8_SCALE)
        v_prev = torch.where(v_state.eq(-128), torch.zeros_like(v_prev), v_prev)
        r_prev = m_state.to(dtype=torch.float32) / 127.0
        m_prev = torch.sign(r_prev) * r_prev * r_prev * _RATIO_MAX * v_prev.sqrt()

        v = v_prev.mul(beta2).add(grad.mul(grad).mul(one_minus_beta2))
        m = m_prev.mul(beta1).add(grad.mul(one_minus_beta1))

        # ---- second moment: LogRMS log8 codec
        positive = v.gt(0)
        scaled_log = torch.log2(torch.clamp_min(v, tiny)) * _LOG8_SCALE
        if stochastic_rounding:
            lower = torch.floor(scaled_log).clamp(-127, 127)
            upper = torch.ceil(scaled_log).clamp(-127, 127)
            lower_value = torch.exp2(lower / _LOG8_SCALE)
            upper_value = torch.exp2(upper / _LOG8_SCALE)
            width = upper_value - lower_value
            probability = (v - lower_value).div(torch.clamp_min(width, tiny)).clamp(0.0, 1.0)
            probability = torch.where(
                width.gt(0), probability, torch.zeros_like(probability)
            )
            uniform = _hash_uniform(v.numel(), v.shape, v.device, v_offsets[index])
            quantized = torch.where(uniform.lt(probability), upper, lower)
        else:
            quantized = torch.round(scaled_log).clamp(-127, 127)
        quantized = torch.where(positive, quantized, torch.full_like(quantized, -128.0))
        v_state.copy_(quantized.to(dtype=torch.int8))

        # ---- first moment: companded ratio to the *stored* second moment
        stored_v = torch.exp2(quantized / _LOG8_SCALE)
        stored_v = torch.where(positive, stored_v, torch.zeros_like(stored_v))
        scale = stored_v.sqrt()
        ratio = torch.where(
            scale.gt(0), m.div(torch.clamp_min(scale, tiny)), torch.zeros_like(m)
        ).clamp(-_RATIO_MAX, _RATIO_MAX)
        position = torch.sign(ratio) * torch.sqrt(ratio.abs() / _RATIO_MAX) * 127.0
        if stochastic_rounding:
            r_lower = torch.floor(position).clamp(-127, 127)
            r_upper = (r_lower + 1.0).clamp(-127, 127)
            lower_scaled = r_lower / 127.0
            upper_scaled = r_upper / 127.0
            r_lower_value = torch.sign(lower_scaled) * lower_scaled * lower_scaled * _RATIO_MAX
            r_upper_value = torch.sign(upper_scaled) * upper_scaled * upper_scaled * _RATIO_MAX
            r_width = r_upper_value - r_lower_value
            r_probability = (ratio - r_lower_value).div(
                torch.clamp_min(r_width, tiny)
            ).clamp(0.0, 1.0)
            r_probability = torch.where(
                r_width.gt(0), r_probability, torch.zeros_like(r_probability)
            )
            r_uniform = _hash_uniform(m.numel(), m.shape, m.device, m_offsets[index])
            r_code = torch.where(r_uniform.lt(r_probability), r_upper, r_lower)
        else:
            r_code = torch.round(position).clamp(-127, 127)
        m_state.copy_(r_code.to(dtype=torch.int8))

        # ---- parameter update from the fresh FP32 moments
        denom = v.div(bias_corrections[index, 1]).sqrt().add(eps)
        update = torch.where(
            m.ne(0), m.div(bias_corrections[index, 0]).div(denom), torch.zeros_like(m)
        )
        param.mul_(decay)
        param.add_(update.to(dtype=param.dtype) * negative_lr)


class LogAdamCompiled(LogAdam):
    """Whole-parameter-group ``torch.compile`` implementation of LogAdam.

    Accepts the same arguments and checkpoints as :class:`LogAdam`. The first
    step incurs compilation time; changing which parameters receive gradients
    can cause a new compilation.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_norm: Optional[float] = 1.0,
        error_if_nonfinite: bool = True,
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
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_norm=max_norm,
            error_if_nonfinite=error_if_nonfinite,
            stochastic_rounding=stochastic_rounding,
            rounding_seed=rounding_seed,
        )
        if not hasattr(torch, "compile"):
            raise RuntimeError("LogAdamCompiled requires torch.compile")
        self.compile_backend = str(compile_backend)
        self.compile_mode = str(compile_mode)
        self.fallback_on_compile_error = bool(fallback_on_compile_error)
        self.compile_failures = 0
        self._compiled_enabled = True
        options: dict[str, Any] = {
            "backend": self.compile_backend,
            "fullgraph": bool(fullgraph),
            "dynamic": bool(dynamic),
        }
        if self.compile_backend == "inductor" and self.compile_mode:
            options["mode"] = self.compile_mode
        self._compiled_group_step = torch.compile(_logadam_group_step_, **options)

    @property
    def compilation_active(self) -> bool:
        return self._compiled_enabled

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        if not self._compiled_enabled:
            return LogAdam.step(self, closure)

        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grouped, params_with_grad = self._collect()
        if not grouped or not self._clip(params_with_grad):
            return loss

        for group, work in grouped:
            by_device: dict[torch.device, list[tuple[Tensor, Tensor, int]]] = {}
            for item in work:
                by_device.setdefault(item[0].device, []).append(item)
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])

            for device, device_work in by_device.items():
                params, grads, v_states, m_states = [], [], [], []
                v_offsets, m_offsets, corrections = [], [], []
                for param, grad, ordinal in device_work:
                    state = self._init_state(param, ordinal)
                    params.append(param)
                    grads.append(grad)
                    v_states.append(state["v"])
                    m_states.append(state["m"])
                    step = int(state["step"]) + 1
                    seed = int(state["rounding_seed"])
                    v_offsets.append((step * _GOLDEN + seed * _SEED_MIX) & _UINT32_MASK)
                    m_offsets.append(
                        (step * _GOLDEN + (seed ^ _M_SEED_SALT) * _SEED_MIX) & _UINT32_MASK
                    )
                    corrections.append([1.0 - beta1**step, 1.0 - beta2**step])

                arguments = (
                    params,
                    grads,
                    v_states,
                    m_states,
                    torch.tensor(v_offsets, dtype=torch.int64, device=device),
                    torch.tensor(m_offsets, dtype=torch.int64, device=device),
                    torch.tensor(corrections, dtype=torch.float32, device=device),
                    torch.tensor(
                        [
                            beta1,
                            1.0 - beta1,
                            beta2,
                            1.0 - beta2,
                            float(group["eps"]),
                            1.0 - lr * float(group["weight_decay"]),
                            -lr,
                        ],
                        dtype=torch.float32,
                        device=device,
                    ),
                    self.stochastic_rounding,
                )
                try:
                    self._compiled_group_step(*arguments)
                except Exception as error:
                    if not self.fallback_on_compile_error:
                        raise
                    # A compile failure happens before any tensor is mutated.
                    # State was not advanced yet, so the eager kernel can run
                    # the same step once. Later steps use the reference path.
                    self.compile_failures += 1
                    self._compiled_enabled = False
                    warnings.warn(
                        "torch.compile failed for LogAdam; this group runs the same "
                        f"tensor kernel eagerly and later steps use the reference path: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _logadam_group_step_(*arguments)
                for param, _, _ in device_work:
                    self.state[param]["step"] += 1

        return loss

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["_logadam_compiled"] = {
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
        }
        return result
