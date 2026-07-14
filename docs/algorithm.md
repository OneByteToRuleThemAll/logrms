# LogRMS algorithm

## Persistent state

For each parameter element, LogRMS stores:

- `q` (`int8`), the log-encoded second moment. Code `-128` represents exact
  zero.
- `h` (`uint8`), a variable-length encoding of up to seven previous nonzero
  gradient signs.

One integer step counter and one rounding seed are stored per parameter tensor.
These fixed-size values are not part of the per-element payload.

## Log codec

For codes from `-127` through `127`:

```text
decode(q) = 2 ** (q / 2)
```

Adjacent values differ by `sqrt(2)`. Encoding uses stateless stochastic
rounding between adjacent representable second-moment values. The random value
is derived from the element index, parameter ordinal, step, and rounding seed;
it does not consume the model's random-number stream.

## Update

For every optimizer step:

1. Collect dense gradients and reject sparse gradients.
2. Check for nonfinite gradients and apply global norm clipping, if enabled,
   before changing optimizer state.
3. Decode the stored moment and compute in FP32:

   ```text
   v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
   ```

4. Encode `v_t` into the `int8` state for the next recurrence.
5. Append the current sign where `g_t != 0`. Zero gradients do not change sign
   history.
6. Once eight nonzero signs are available, count adjacent sign transitions
   `T_t` and calculate:

   ```text
   c_t = 1                         before the window is ready
   c_t = 1 - 0.8 * T_t / 7         after the window is ready
   ```

7. Apply decoupled weight decay and the normalized update:

   ```text
   theta <- theta * (1 - lr * weight_decay)
   theta <- theta - lr * c_t * g_t /
                    (sqrt(v_t / (1 - beta2^step)) + eps)
   ```

The current update uses the newly computed FP32 `v_t`. Quantization affects the
next recurrence, not the current denominator. Model parameters may use FP32,
BF16, or FP16; the optimizer arithmetic is FP32 before the final update is cast
to the parameter dtype.

## Compiled execution

The compiled implementation keeps optimizer bookkeeping and validation in
ordinary Python, then sends the dense tensor update for each parameter group to
`torch.compile`. Parameters and state tensors are mutated in place. PyTorch may
warn that CUDA graphs are skipped because inputs are mutated; this does not mean
the compiled Inductor graph failed.
