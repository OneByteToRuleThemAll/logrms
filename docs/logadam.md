# LogAdam: an 8-bit first moment in the second byte

LogAdam is an experimental variant, separate from LogRMS. It keeps the LogRMS
`int8` log second moment unchanged and replaces the `uint8` sign history with
an `int8` first moment. Persistent tensor state is still exactly two bytes per
parameter element.

| Persistent state | Dtype | Bytes/parameter |
|---|---:|---:|
| Log second-moment code (`v`) | `int8` | 1 |
| First moment relative to `sqrt(v)` (`m`) | `int8` | 1 |
| **Total** |  | **2** |

## Why

A small ablation (see below) found that the sign-history byte behaved like a
constant 0.6x learning-rate multiplier: default LogRMS and LogRMS with a flat
0.6 confidence table were indistinguishable, and stored sign-flip rates sat
near 0.47, where the confidence formula gives about 0.6 with only random
jitter. The remaining gap to AdamW matched the gap between AdamW with and
without a first moment. LogAdam spends the second byte on that first moment.

## First-moment codec

The first moment is stored relative to the square root of the decoded second
moment:

```text
r = m / sqrt(decode(q))
```

This ratio is bounded by the two recurrences themselves. With `beta1 = 0.9`
and `beta2 = 0.999`, one isolated gradient gives
`(1 - beta1) / sqrt(1 - beta2) ~= 3.16`, a run of same-sign gradients early in
training peaks near 6.6, and the steady-state value is at most about 1. One
fixed range therefore covers every tensor and no per-tensor or per-block scale
is stored.

Codes `-127..127` decode with square-root companding over `[-8, 8]`:

```text
decode(c) = sign(c) * (c / 127)^2 * 8
```

Resolution is about `0.0005` near zero and coarser toward the range limit.
Encoding uses the same stateless stochastic rounding as the second moment,
with a separately salted seed, so `E[decode(code)]` equals the ratio away from
range clipping. The ratio is taken against the *stored* second-moment code, so
the stochastic jump in that code does not bias the decoded first moment.

Different `beta` values change the bound. `beta1 = 0.9, beta2 = 0.99` gives a
smaller ratio; a `beta2` much closer to 1 than `0.999` with a small `beta1`
could exceed 8 early in training and would be clipped.

## Update

```text
v_t   = beta2 * v_(t-1) + (1 - beta2) * g_t^2
m_t   = beta1 * m_(t-1) + (1 - beta1) * g_t
theta <- theta * (1 - lr * weight_decay)
theta <- theta - lr * (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
```

Both recurrences and the update use FP32 values decoded from state.
Quantization affects only the next step.

## Implementations

- `LogAdam` is the per-tensor eager reference.
- `LogAdamCompiled` sends each parameter group through one `torch.compile`
  graph, at the same boundary as `LogRMSCompiled`.

Tests check exact state parity between the two with `compile_backend="eager"`,
and a statistical check under real Inductor on CPU. Bit-exact parity is not
expected under Inductor: generated code can round `log2`/`exp2` differently,
which occasionally moves a stored code across a rounding boundary.

## Evidence so far

`benchmarks/char_lm_ablation.py` trains a ~0.36M-parameter character-level
transformer on CPU for 1,200 steps (batch 32, context 64, warmup then cosine
decay, weight decay 0.01, global clipping 1.0). Mean validation loss over
three seeds, from `results/char_lm_ablation.csv`:

| Arm | State B/param | lr 3e-3 | lr 1e-3 |
|---|---:|---:|---:|
| LogAdam (int8 v + int8 m) | 2 | 1.888 | 2.228 |
| AdamW, FP32 | 8 | 1.889 | 2.240 |
| LogRMS, all-ones table (log v only) | 2 (1 used) | 1.977 | 2.291 |
| AdamW, beta1 = 0, FP32 | 8 | 2.010 | 2.299 |
| LogRMS, flat 0.6 table | 2 (1 used) | 2.113 | 2.365 |
| LogRMS, default | 2 | 2.120 | 2.370 |
| LogRMS, FP32 second moment | 5 | 2.131 | 2.374 |

Limits of this evidence:

- One very small model, short runs, three seeds, two learning rates.
- The corpus is the Python standard library source of the machine that ran
  it, so absolute losses are not portable.
- No GPU timing or peak-memory measurement of `LogAdamCompiled` yet.
- No comparison with torchao or bitsandbytes 8-bit AdamW yet, which is the
  direct competitor at this state size.

The next step is the 49.5M TinyStories benchmark with fused AdamW, torchao
8-bit AdamW, LogRMS, and `LogAdamCompiled`, over at least three seeds.
