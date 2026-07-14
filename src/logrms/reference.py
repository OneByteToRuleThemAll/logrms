"""
LogRMS v11.1 — compact sign-consistency optimizer
==================================================

LogRMS is an RMS-normalized optimizer with a per-coordinate gain determined by
recent gradient-sign consistency. It intentionally has no first-moment buffer.

Persistent tensor state per parameter element:

* ``float16`` / ``log16`` backend: 2-byte float16 log2(second moment) bit pattern
  + 1-byte sign history = 3 bytes.
* ``8bit`` / ``log8`` backend: 1-byte fixed-point log2(second moment)
  + 1-byte sign history = 2 bytes.
* ``float32`` / ``fp32`` reference backend: 4-byte second moment bit pattern
  + 1-byte sign history = 5 bytes.

A small Python integer step counter is also stored per parameter tensor. As with
other PyTorch optimizers, Python/Tensor object metadata is not included in the
per-element payload figures above.

Update
------

    v_t        = beta2 * v_(t-1) + (1 - beta2) * g_t^2
    v_hat      = v_t / (1 - beta2^t)
    confidence = table[sign transitions over the latest 8 nonzero signs]
    p          = p * (1 - lr * weight_decay)
    p          = p - lr * confidence * g_t / (sqrt(v_hat) + eps)

Confidence is neutral until eight nonzero signs have been observed for a
coordinate. A zero gradient neither changes history nor counts as a negative
sign. The one-byte history uses a compact variable-length prefix encoding: the
state stores up to seven prior signs, and the current sign supplies the eighth
observation used for the confidence calculation. The evidence-backed default
is the damp-only v10.6 table; the corrected boost-and-dampen v10.7 table remains
available as an explicit experimental option.

Low-precision EMA recurrences need more than exponent range: deterministic
rounding can erase small increments or decays indefinitely. Compact backends
therefore use stateless stochastic rounding by default. It is unbiased in the
represented second-moment domain across coordinates and steps, consumes no
model RNG state, and needs no per-coordinate residual. Set
``stochastic_rounding=False`` to reproduce the deterministic notebook codec.

The old direct-bfloat16 EMA backend is intentionally not used. Passing
``backend='bfloat16'`` maps to the log16 backend with a warning.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from itertools import chain
from typing import Any, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "LogRMS",
    "VERSION",
    "CONFIDENCE_DEFAULT",
    "CONFIDENCE_RMS",
    "CONFIDENCE_V106",
    "CONFIDENCE_V107_FIXED",
    "memory_audit",
]

VERSION = "11.1"

# Corrected experimental v10.7 table: neutral at 3-4 transitions.
CONFIDENCE_V107_FIXED: tuple[float, ...] = (
    1.50,
    1.30,
    1.10,
    1.00,
    1.00,
    0.67,
    0.33,
    0.00,
)
CONFIDENCE_V106: tuple[float, ...] = tuple(1.0 - t * (0.8 / 7.0) for t in range(8))
CONFIDENCE_RMS: tuple[float, ...] = (1.0,) * 8
# The notebook supports v10.6 directly; the small held-out benchmark also found
# it more stable than the unvalidated boost table.
CONFIDENCE_DEFAULT = CONFIDENCE_V106

# Half-octave int8 quantization covers v from roughly 2**-63.5 to 2**63.5.
# Quarter-octave quantization used by v11.0 clipped ordinary small-gradient
# moments (for example g**2 around 1e-12) before rounding quality even mattered.
_LOG8_SCALE = 2.0
_UINT32_MASK = 0xFFFFFFFF
_UNIFORM_MANTISSA_MASK = 0x00FFFFFF
_UNIFORM_DENOMINATOR = float(1 << 24)
_LOG16_ZERO_BITS = int(
    torch.tensor(float("-inf"), dtype=torch.float16).view(torch.int16).item()
)

# LUTs are immutable and tiny. Device copies are cached lazily.
_LUT_CACHE: dict[
    tuple[str, Optional[int], tuple[float, ...]],
    tuple[Tensor, Tensor, Tensor, Tensor],
] = {}


def _validate_confidence_table(table: Sequence[float]) -> tuple[float, ...]:
    if len(table) != 8:
        raise ValueError("confidence_table must contain exactly 8 values")
    normalized = tuple(float(x) for x in table)
    if any(not math.isfinite(x) or x < 0.0 for x in normalized):
        raise ValueError("confidence_table values must be finite and non-negative")
    return normalized


def _canonical_backend(backend: Optional[str]) -> str:
    if backend is None:
        return "float16"
    name = str(backend).lower().replace("_", "").replace("-", "")
    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "log16": "float16",
        "8bit": "8bit",
        "int8": "8bit",
        "log8": "8bit",
        "float32": "float32",
        "fp32": "float32",
        "reference": "float32",
    }
    if name == "bfloat16" or name == "bf16":
        warnings.warn(
            "The direct bfloat16 EMA backend was removed because the EMA can stall. "
            "Using the float16 log-encoded backend instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "float16"
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'float16', '8bit', or 'float32'"
        ) from exc


def _build_cpu_luts(table: tuple[float, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build history decode/encode and confidence lookup tables on CPU."""
    lengths = torch.zeros(256, dtype=torch.uint8)
    bits = torch.zeros(256, dtype=torch.uint8)

    # Code ranges for a sequence of length L are:
    # [2**L - 1, 2**(L+1) - 2], for L=0..7. Code 255 is invalid/unused.
    for length in range(8):
        offset = (1 << length) - 1
        width = 1 << length
        for packed in range(width):
            code = offset + packed
            lengths[code] = length
            bits[code] = packed

    offsets = torch.tensor([(1 << length) - 1 for length in range(8)], dtype=torch.uint8)

    confidence = torch.empty(256, dtype=torch.float32)
    for sequence in range(256):
        xor = (sequence ^ (sequence >> 1)) & 0x7F
        transitions = xor.bit_count()
        confidence[sequence] = table[transitions]

    return lengths, bits, offsets, confidence


def _get_luts(device: torch.device, table: tuple[float, ...]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key = (device.type, device.index, table)
    cached = _LUT_CACHE.get(key)
    if cached is None:
        cpu_luts = _build_cpu_luts(table)
        cached = tuple(t.to(device=device) for t in cpu_luts)  # type: ignore[assignment]
        _LUT_CACHE[key] = cached
    return cached


def _history_confidence_and_update_(
    history: Tensor,
    grad_fp32: Tensor,
    table: tuple[float, ...],
) -> Tensor:
    """Return confidence and append each nonzero gradient sign in-place.

    The uint8 code stores a variable-length sequence of at most seven prior
    nonzero signs. Once seven prior signs exist, appending the current sign gives
    the eight-sign window used to count seven adjacent transitions.
    """
    lengths_lut, bits_lut, offsets_lut, confidence_lut = _get_luts(history.device, table)

    index = history.to(dtype=torch.long)
    length = lengths_lut[index]
    packed = bits_lut[index]

    nonzero = grad_fp32.ne(0)
    sign = grad_fp32.gt(0).to(dtype=torch.uint8)
    sequence = torch.bitwise_or(torch.bitwise_left_shift(packed, 1), sign)

    ready = torch.logical_and(nonzero, length.eq(7))
    sequence_confidence = confidence_lut[sequence.to(dtype=torch.long)]
    confidence = torch.where(ready, sequence_confidence, torch.ones_like(grad_fp32))

    next_length = torch.clamp(length.to(dtype=torch.int16) + 1, max=7).to(torch.uint8)
    next_bits = torch.bitwise_and(sequence, 0x7F)
    next_code = (
        offsets_lut[next_length.to(dtype=torch.long)].to(dtype=torch.int16)
        + next_bits.to(dtype=torch.int16)
    ).to(dtype=torch.uint8)
    history.copy_(torch.where(nonzero, next_code, history))
    return confidence


def _new_v_state(param: Tensor, backend: str) -> Tensor:
    shape = tuple(param.shape)
    if backend == "float16":
        return torch.full(shape, _LOG16_ZERO_BITS, dtype=torch.int16, device=param.device)
    if backend == "8bit":
        return torch.full(shape, -128, dtype=torch.int8, device=param.device)
    if backend == "float32":
        return torch.zeros(shape, dtype=torch.int32, device=param.device)
    raise AssertionError(f"unhandled backend: {backend}")


def _decode_v(v_state: Tensor, backend: str) -> Tensor:
    if backend == "float16":
        return torch.exp2(v_state.view(torch.float16).to(dtype=torch.float32))
    if backend == "8bit":
        decoded = torch.exp2(v_state.to(dtype=torch.float32) / _LOG8_SCALE)
        return torch.where(v_state.eq(-128), torch.zeros_like(decoded), decoded)
    if backend == "float32":
        return v_state.view(torch.float32)
    raise AssertionError(f"unhandled backend: {backend}")


def _stateless_uniform_like(reference: Tensor, *, step: int, seed: int) -> Tensor:
    """Return deterministic U[0, 1) dither without touching PyTorch RNG state."""
    flat = torch.arange(reference.numel(), dtype=torch.int64, device=reference.device)
    offset = (int(step) * 0x9E3779B9 + int(seed) * 0x85EBCA6B) & _UINT32_MASK
    value = torch.bitwise_and(flat + offset, _UINT32_MASK)
    value = torch.bitwise_xor(
        value,
        torch.bitwise_and(torch.bitwise_left_shift(value, 13), _UINT32_MASK),
    )
    value = torch.bitwise_xor(value, torch.bitwise_right_shift(value, 17))
    value = torch.bitwise_xor(
        value,
        torch.bitwise_and(torch.bitwise_left_shift(value, 5), _UINT32_MASK),
    )
    uniform = torch.bitwise_and(value, _UNIFORM_MANTISSA_MASK).to(torch.float32)
    return uniform.mul_(1.0 / _UNIFORM_DENOMINATOR).reshape(reference.shape)


def _choose_unbiased_upper(
    target: Tensor,
    lower_value: Tensor,
    upper_value: Tensor,
    *,
    step: int,
    seed: int,
) -> Tensor:
    """Choose between adjacent representations with E[choice] == target."""
    width = upper_value - lower_value
    probability = (target - lower_value).div(
        torch.clamp_min(width, torch.finfo(torch.float32).tiny)
    ).clamp_(0.0, 1.0)
    probability = torch.where(width.gt(0), probability, torch.zeros_like(probability))
    return _stateless_uniform_like(target, step=step, seed=seed).lt(probability)


def _encode_v_(
    v_state: Tensor,
    v_fp32: Tensor,
    backend: str,
    *,
    stochastic_rounding: bool = False,
    step: int = 0,
    seed: int = 0,
) -> Tensor:
    """Quantize v into v_state and return the represented FP32 value."""
    if backend == "float16":
        positive = v_fp32.gt(0)
        safe_v = torch.clamp_min(v_fp32, torch.finfo(torch.float32).tiny)
        log_v = torch.log2(safe_v)
        encoded = log_v.to(dtype=torch.float16)

        if stochastic_rounding:
            encoded_fp32 = encoded.to(dtype=torch.float32)
            lower = torch.where(
                encoded_fp32.gt(log_v),
                torch.nextafter(encoded, torch.full_like(encoded, float("-inf"))),
                encoded,
            )
            upper = torch.where(
                encoded_fp32.lt(log_v),
                torch.nextafter(encoded, torch.full_like(encoded, float("inf"))),
                encoded,
            )
            lower_value = torch.exp2(lower.to(dtype=torch.float32))
            upper_value = torch.exp2(upper.to(dtype=torch.float32))
            choose_upper = _choose_unbiased_upper(
                v_fp32,
                lower_value,
                upper_value,
                step=step,
                seed=seed,
            )
            encoded = torch.where(choose_upper, upper, lower)

        zero = torch.full_like(encoded, float("-inf"))
        encoded = torch.where(positive, encoded, zero)
        v_state.copy_(encoded.view(torch.int16))
        return torch.exp2(v_state.view(torch.float16).to(dtype=torch.float32))

    if backend == "8bit":
        positive = v_fp32.gt(0)
        safe_v = torch.clamp_min(v_fp32, torch.finfo(torch.float32).tiny)
        scaled_log = torch.log2(safe_v) * _LOG8_SCALE
        if stochastic_rounding:
            lower = torch.floor(scaled_log).clamp(-127, 127)
            upper = torch.ceil(scaled_log).clamp(-127, 127)
            lower_value = torch.exp2(lower / _LOG8_SCALE)
            upper_value = torch.exp2(upper / _LOG8_SCALE)
            choose_upper = _choose_unbiased_upper(
                v_fp32,
                lower_value,
                upper_value,
                step=step,
                seed=seed,
            )
            quantized = torch.where(choose_upper, upper, lower)
        else:
            quantized = torch.round(scaled_log).clamp(-127, 127)
        quantized = quantized.to(dtype=torch.int8)
        v_state.copy_(torch.where(positive, quantized, torch.full_like(quantized, -128)))
        decoded = torch.exp2(v_state.to(dtype=torch.float32) / _LOG8_SCALE)
        return torch.where(v_state.eq(-128), torch.zeros_like(decoded), decoded)

    if backend == "float32":
        represented = v_state.view(torch.float32)
        represented.copy_(v_fp32)
        return represented

    raise AssertionError(f"unhandled backend: {backend}")


class LogRMS(Optimizer):
    """RMS-normalized optimizer with compact sign-consistency gain state.

    Args:
        params: Iterable of parameters or parameter-group dictionaries.
        lr: Learning rate.
        beta2: Second-moment exponential decay in ``[0, 1)``.
        eps: Non-negative denominator stabilizer.
        weight_decay: Non-negative decoupled weight decay.
        backend: ``'float16'`` (3 B/element), ``'8bit'`` (2 B/element),
            or ``'float32'`` (5 B/element reference implementation).
        confidence_table: Eight values indexed by the number of transitions in
            the latest eight nonzero signs. The default is the damp-only v10.6
            table; pass ``CONFIDENCE_V107_FIXED`` to test boost-and-dampen behavior.
        max_norm: Optional global gradient-norm clipping threshold. ``None``
            disables clipping.
        error_if_nonfinite: Raise instead of updating when the global gradient
            norm is NaN or infinite.
        maximize: Maximize the objective instead of minimizing it.
        stochastic_rounding: Use stateless unbiased rounding for compact
            second-moment storage. Disable only for legacy/notebook parity.
        rounding_seed: Integer seed for stateless rounding dither. This does not
            consume or alter PyTorch's random-number-generator state.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        backend: Optional[str] = None,
        confidence_table: Optional[Sequence[float]] = None,
        max_norm: Optional[float] = 1.0,
        error_if_nonfinite: bool = True,
        maximize: bool = False,
        stochastic_rounding: bool = True,
        rounding_seed: int = 0,
    ) -> None:
        self.confidence_table = _validate_confidence_table(
            CONFIDENCE_DEFAULT if confidence_table is None else confidence_table
        )
        self.max_norm = None if max_norm is None else float(max_norm)
        self.error_if_nonfinite = bool(error_if_nonfinite)
        self.stochastic_rounding = bool(stochastic_rounding)
        self.rounding_seed = int(rounding_seed) & _UINT32_MASK
        self.last_grad_norm: Optional[Tensor] = None
        self.last_step_skipped = False

        canonical_backend = _canonical_backend(backend)
        self._validate_hyperparameters(
            lr=lr,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            max_norm=self.max_norm,
        )

        defaults = dict(
            lr=float(lr),
            beta2=float(beta2),
            eps=float(eps),
            weight_decay=float(weight_decay),
            backend=canonical_backend,
            maximize=bool(maximize),
        )
        super().__init__(params, defaults)

        if not any(group["params"] for group in self.param_groups):
            raise ValueError("optimizer got no parameters across all parameter groups")

        seen: set[Tensor] = set()
        for group in self.param_groups:
            self._normalize_and_validate_group(group)
            for param in group["params"]:
                if param in seen:
                    raise ValueError("a parameter appears more than once in the optimizer")
                seen.add(param)
                self._validate_parameter(param)

    @staticmethod
    def _validate_hyperparameters(
        *,
        lr: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        max_norm: Optional[float],
    ) -> None:
        if not math.isfinite(float(lr)) or float(lr) < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not math.isfinite(float(beta2)) or not 0.0 <= float(beta2) < 1.0:
            raise ValueError(f"invalid beta2: {beta2}")
        if not math.isfinite(float(eps)) or float(eps) < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if max_norm is not None and (not math.isfinite(max_norm) or max_norm <= 0.0):
            raise ValueError("max_norm must be a positive finite number or None")

    @staticmethod
    def _validate_parameter(param: Tensor) -> None:
        if not param.is_floating_point():
            raise TypeError(f"LogRMS only supports real floating-point parameters, got {param.dtype}")
        if param.layout != torch.strided:
            raise TypeError(f"LogRMS only supports strided parameters, got layout {param.layout}")

    def _normalize_and_validate_group(self, group: dict[str, Any]) -> None:
        group["backend"] = _canonical_backend(group.get("backend", "float16"))
        group["lr"] = float(group["lr"])
        group["beta2"] = float(group["beta2"])
        group["eps"] = float(group["eps"])
        group["weight_decay"] = float(group["weight_decay"])
        group["maximize"] = bool(group.get("maximize", False))
        self._validate_hyperparameters(
            lr=group["lr"],
            beta2=group["beta2"],
            eps=group["eps"],
            weight_decay=group["weight_decay"],
            max_norm=self.max_norm,
        )

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Optimizer.__init__ calls this override. Suppress its duplicate warning;
        # duplicate parameters are rejected deterministically below.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="optimizer contains a parameter group with duplicate parameters.*",
            )
            super().add_param_group(param_group)

        group = self.param_groups[-1]
        try:
            self._normalize_and_validate_group(group)
            if len(group["params"]) != len(set(group["params"])):
                raise ValueError("a parameter appears more than once in a parameter group")
            for param in group["params"]:
                self._validate_parameter(param)
        except Exception:
            self.param_groups.pop()
            raise

    @property
    def backend(self) -> str:
        backends = {str(group["backend"]) for group in self.param_groups}
        return next(iter(backends)) if len(backends) == 1 else "mixed"

    @property
    def lr(self) -> float:
        return float(self.param_groups[0]["lr"])

    @lr.setter
    def lr(self, new_lr: float) -> None:
        new_lr = float(new_lr)
        self._validate_hyperparameters(
            lr=new_lr,
            beta2=0.0,
            eps=0.0,
            weight_decay=0.0,
            max_norm=self.max_norm,
        )
        for group in self.param_groups:
            group["lr"] = new_lr

    def _init_or_validate_state(
        self,
        param: Tensor,
        backend: str,
        *,
        parameter_ordinal: int,
    ) -> dict[str, Any]:
        state = self.state[param]
        if not state:
            state["step"] = 0
            state["rounding_seed"] = (
                self.rounding_seed + int(parameter_ordinal) * 0x9E3779B9
            ) & _UINT32_MASK
            state["v"] = _new_v_state(param, backend)
            state["history"] = torch.zeros(tuple(param.shape), dtype=torch.uint8, device=param.device)
            return state

        expected_v_dtype = {
            "float16": torch.int16,
            "8bit": torch.int8,
            "float32": torch.int32,
        }[backend]
        v_state = state.get("v")
        history = state.get("history")
        if not isinstance(v_state, Tensor) or tuple(v_state.shape) != tuple(param.shape):
            raise RuntimeError("invalid LogRMS checkpoint: second-moment state shape mismatch")
        if v_state.dtype != expected_v_dtype or v_state.device != param.device:
            raise RuntimeError(
                "invalid LogRMS checkpoint: second-moment state dtype/device does not match backend"
            )
        if not isinstance(history, Tensor) or tuple(history.shape) != tuple(param.shape):
            raise RuntimeError("invalid LogRMS checkpoint: history state shape mismatch")
        if history.dtype != torch.uint8 or history.device != param.device:
            raise RuntimeError("invalid LogRMS checkpoint: history must be uint8 on the parameter device")
        state["step"] = int(state.get("step", 0))
        state["rounding_seed"] = int(
            state.get(
                "rounding_seed",
                (self.rounding_seed + int(parameter_ordinal) * 0x9E3779B9)
                & _UINT32_MASK,
            )
        ) & _UINT32_MASK
        return state

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        loss: Optional[Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        work: list[tuple[Tensor, Tensor, dict[str, Any], int]] = []
        params_with_grad: list[Tensor] = []
        parameter_ordinal = 0
        for group in self.param_groups:
            self._normalize_and_validate_group(group)
            for param in group["params"]:
                grad = param.grad
                if grad is not None:
                    if grad.is_sparse:
                        raise RuntimeError("LogRMS does not support sparse gradients")
                    if grad.layout != torch.strided:
                        raise RuntimeError(f"LogRMS requires strided gradients, got {grad.layout}")
                    work.append((param, grad, group, parameter_ordinal))
                    params_with_grad.append(param)
                parameter_ordinal += 1

        self.last_step_skipped = False
        self.last_grad_norm = None
        if not work:
            return loss

        # GradScaler.step() unscales ordinary Optimizer subclasses before this
        # method runs. Direct AMP users should likewise call scaler.unscale_()
        # before any separate clipping operation.
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

        for param, raw_grad, group, parameter_ordinal in work:
            backend = str(group["backend"])
            state = self._init_or_validate_state(
                param,
                backend,
                parameter_ordinal=parameter_ordinal,
            )
            state["step"] += 1
            step = int(state["step"])

            grad_fp32 = raw_grad.detach().to(dtype=torch.float32)
            if group["maximize"]:
                grad_fp32 = -grad_fp32

            beta2 = float(group["beta2"])
            v_state = state["v"]
            v_fp32 = _decode_v(v_state, backend)
            v_fp32 = v_fp32.mul(beta2).addcmul(grad_fp32, grad_fp32, value=1.0 - beta2)
            _encode_v_(
                v_state,
                v_fp32,
                backend,
                stochastic_rounding=self.stochastic_rounding,
                step=step,
                seed=int(state["rounding_seed"]),
            )

            confidence = _history_confidence_and_update_(
                state["history"], grad_fp32, self.confidence_table
            )
            bias_correction2 = 1.0 - beta2**step
            # The current update uses the freshly computed FP32 moment. Storage
            # quantization affects the next recurrence, matching the notebook
            # semantics and preventing avoidable first-step quantization error.
            denom = v_fp32.div(bias_correction2).sqrt_().add_(float(group["eps"]))
            # Preserve an exact zero update when eps=0 and both g and v are zero;
            # a raw 0/0 division would otherwise manufacture NaNs.
            update = torch.where(
                grad_fp32.ne(0),
                grad_fp32.div(denom),
                torch.zeros_like(grad_fp32),
            ).mul_(confidence)

            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update.to(dtype=param.dtype), alpha=-lr)

        return loss

    def state_bytes(self) -> int:
        """Allocated persistent tensor payload bytes, excluding Python metadata."""
        total = 0
        for state in self.state.values():
            for key in ("v", "history"):
                value = state.get(key)
                if isinstance(value, Tensor):
                    total += value.numel() * value.element_size()
        return total

    def projected_state_bytes(self) -> int:
        """Projected tensor payload after all parameters have received gradients."""
        total = 0
        bytes_per_element = {"float16": 3, "8bit": 2, "float32": 5}
        for group in self.param_groups:
            total += sum(param.numel() for param in group["params"]) * bytes_per_element[
                str(group["backend"])
            ]
        return total

    def state_dict(self) -> dict[str, Any]:
        result = super().state_dict()
        result["_logrms"] = {
            "version": VERSION,
            "confidence_table": list(self.confidence_table),
            "max_norm": self.max_norm,
            "error_if_nonfinite": self.error_if_nonfinite,
            "stochastic_rounding": self.stochastic_rounding,
            "rounding_seed": self.rounding_seed,
            "log8_scale": _LOG8_SCALE,
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a checkpoint while preserving LogRMS's encoded tensor dtypes.

        PyTorch's generic Optimizer loader casts non-step tensor state to the
        parameter dtype.  LogRMS stores FP16/FP32 bit patterns in integer
        tensors and its history in uint8, so that generic cast would corrupt the
        representation.  This mirrors the standard loader's parameter mapping
        and hook behavior but moves state tensors without changing dtype.
        """
        copied = deepcopy(state_dict)
        metadata = copied.pop("_logrms", {})
        loaded_version = str(metadata.get("version", ""))
        old_log8_scale = float(
            metadata.get(
                "log8_scale",
                4.0 if loaded_version == "11.0" else _LOG8_SCALE,
            )
        )
        if not math.isfinite(old_log8_scale) or old_log8_scale <= 0.0:
            raise ValueError("invalid LogRMS checkpoint: log8_scale must be positive")

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, copied)
            if hook_result is not None:
                copied = hook_result

        groups = self.param_groups
        saved_groups = deepcopy(copied["param_groups"])
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")

        current_lengths = [len(group["params"]) for group in groups]
        saved_lengths = [len(group["params"]) for group in saved_groups]
        if current_lengths != saved_lengths:
            raise ValueError(
                "loaded state dict contains a parameter group that does not "
                "match the size of the optimizer's group"
            )

        id_map = dict(
            zip(
                chain.from_iterable(group["params"] for group in saved_groups),
                chain.from_iterable(group["params"] for group in groups),
                strict=True,
            )
        )

        def move_preserving_dtype(value: Any, param: Tensor) -> Any:
            if isinstance(value, Tensor):
                return value.detach().clone().to(device=param.device)
            if isinstance(value, dict):
                return {key: move_preserving_dtype(item, param) for key, item in value.items()}
            if isinstance(value, list):
                return [move_preserving_dtype(item, param) for item in value]
            if isinstance(value, tuple):
                return tuple(move_preserving_dtype(item, param) for item in value)
            return deepcopy(value)

        restored_state: defaultdict[Any, dict[Any, Any]] = defaultdict(dict)
        for saved_id, saved_state in copied["state"].items():
            if saved_id in id_map:
                param = id_map[saved_id]
                moved_state = move_preserving_dtype(saved_state, param)
                v_state = moved_state.get("v") if isinstance(moved_state, dict) else None
                if (
                    isinstance(v_state, Tensor)
                    and v_state.dtype == torch.int8
                    and old_log8_scale != _LOG8_SCALE
                ):
                    zero = v_state.eq(-128)
                    migrated = torch.round(
                        v_state.to(torch.float32) * (_LOG8_SCALE / old_log8_scale)
                    ).clamp(-127, 127).to(torch.int8)
                    moved_state["v"] = torch.where(
                        zero,
                        torch.full_like(migrated, -128),
                        migrated,
                    )
                restored_state[param] = moved_state
            else:
                restored_state[saved_id] = deepcopy(saved_state)

        def update_group(
            current_group: dict[str, Any],
            saved_group: dict[str, Any],
        ) -> dict[str, Any]:
            saved_group["params"] = current_group["params"]
            if "param_names" in current_group and "param_names" not in saved_group:
                saved_group["param_names"] = current_group["param_names"]
            return saved_group

        restored_groups = [
            update_group(current, saved)
            for current, saved in zip(groups, saved_groups, strict=True)
        ]
        self.__setstate__({"state": restored_state, "param_groups": restored_groups})

        if metadata:
            self.confidence_table = _validate_confidence_table(
                metadata.get("confidence_table", self.confidence_table)
            )
            loaded_max_norm = metadata.get("max_norm", self.max_norm)
            self.max_norm = None if loaded_max_norm is None else float(loaded_max_norm)
            self.error_if_nonfinite = bool(
                metadata.get("error_if_nonfinite", self.error_if_nonfinite)
            )
            # v11.0 used deterministic nearest rounding. Preserve its behavior
            # when resuming an old checkpoint unless newer metadata says otherwise.
            self.stochastic_rounding = bool(
                metadata.get(
                    "stochastic_rounding",
                    False if loaded_version == "11.0" else self.stochastic_rounding,
                )
            )
            self.rounding_seed = int(
                metadata.get("rounding_seed", self.rounding_seed)
            ) & _UINT32_MASK

        parameter_ordinal = 0
        for group in self.param_groups:
            self._normalize_and_validate_group(group)
            for param in group["params"]:
                if param in self.state and self.state[param]:
                    self._init_or_validate_state(
                        param,
                        str(group["backend"]),
                        parameter_ordinal=parameter_ordinal,
                    )
                parameter_ordinal += 1

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)


def _fmt_bytes(number: int) -> str:
    value = float(number)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def memory_audit(model: torch.nn.Module, optimizer: LogRMS, *, print_report: bool = True) -> dict[str, Any]:
    """Return and optionally print an accurate optimizer-memory summary."""
    parameters = list(model.parameters())
    parameter_count = sum(param.numel() for param in parameters)
    weight_bytes = sum(param.numel() * param.element_size() for param in parameters)
    allocated_state = optimizer.state_bytes()
    projected_state = optimizer.projected_state_bytes()

    result = {
        "version": VERSION,
        "backend": optimizer.backend,
        "parameter_count": parameter_count,
        "weight_bytes": weight_bytes,
        "allocated_state_bytes": allocated_state,
        "projected_state_bytes": projected_state,
        "allocated_total_bytes": weight_bytes + allocated_state,
        "projected_total_bytes": weight_bytes + projected_state,
    }

    if print_report:
        print("=" * 68)
        print(f"MEMORY AUDIT — LogRMS v{VERSION} ({parameter_count:,} parameters) [{optimizer.backend}]")
        print("=" * 68)
        print(f"Model weights:                 {_fmt_bytes(weight_bytes)}")
        print(f"Allocated optimizer payload:   {_fmt_bytes(allocated_state)}")
        print(f"Projected optimizer payload:   {_fmt_bytes(projected_state)}")
        if weight_bytes:
            print(f"Projected state / weights:     {100.0 * projected_state / weight_bytes:.1f}%")
        print(f"Projected weights + state:     {_fmt_bytes(weight_bytes + projected_state)}")
        print("Note: Python objects and per-tensor step/rounding-seed integers are excluded.")
        print("=" * 68)

    return result
