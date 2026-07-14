# LogRMS

LogRMS is an experimental PyTorch optimizer that stores exactly **two
persistent tensor bytes per parameter**:

- one `int8` log-encoded second moment; and
- one `uint8` record of recent nonzero gradient signs.

The optimizer computes its second-moment recurrence and current update in FP32.
It then stochastically quantizes the moment for the next step. Sign history
conservatively reduces updates on coordinates whose recent gradients alternate
frequently. It does not maintain Adam's first-moment tensor.

## Current evidence

In a controlled 49,515,008-parameter TinyStories run on an NVIDIA L4, compiled
LogRMS used 2.0000 persistent state bytes per parameter and had a median
optimizer-step time of 8.77 ms. Fused AdamW used 8.0000 bytes per parameter and
7.61 ms. Median full-step time was 107.37 ms for LogRMS and 107.26 ms for fused
AdamW. LogRMS test perplexity was 3.7698, compared with 3.6911 for fused AdamW
under the same untuned schedule.

These are encouraging single-run results, not a claim of universal parity.
The repository includes the paper and CSV evidence so the measurements and
limitations can be inspected directly.

| Optimizer | State bytes/parameter | Median optimizer step | Test loss |
|---|---:|---:|---:|
| Fused AdamW | 8.0000 | 7.61 ms | 1.3059 |
| LogRMS, compiled | **2.0000** | 8.77 ms | 1.3270 |
| AdamW, foreach | 8.0000 | 18.44 ms | 1.3061 |
| torchao AdamW, 8-bit | 2.0460 | 113.98 ms | 1.3125 |

## Installation

```bash
git clone https://github.com/OneByteToRuleThemAll/logrms.git
cd logrms
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[test]'
pytest
```

## Usage

```python
import torch
from logrms import LogRMSCompiled

model = torch.nn.Linear(32, 4).cuda()
optimizer = LogRMSCompiled(
    model.parameters(),
    lr=3e-3,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
    max_norm=1.0,
)

optimizer.zero_grad(set_to_none=True)
loss = model(torch.randn(16, 32, device="cuda")).square().mean()
loss.backward()
optimizer.step()
```

The first compiled step incurs compilation time. The default
`compile_mode="reduce-overhead"` is intended for repeated CUDA training steps.
The optimizer performs global gradient clipping internally when `max_norm` is
set, so do not clip the same gradients a second time outside the optimizer.

`LogRMS` exposes the slower reference implementation used for numerical parity
and checkpoint tests. `LogRMSCompiled` is the intended training interface.

## Memory accounting

| Persistent state | Dtype | Bytes/parameter |
|---|---:|---:|
| Log second-moment code | `int8` | 1 |
| Sign-history code | `uint8` | 1 |
| **Total** |  | **2** |

This is the persistent tensor payload after state initialization. It does not
include parameters, gradients, activations, allocator reservations, compiler
caches, temporary tensors, or fixed Python metadata. Peak PyTorch allocation is
a whole-training-process measurement and should not be interpreted as optimizer
state alone.

## Repository contents

- `src/logrms/reference.py` — canonical correctness reference.
- `src/logrms/compiled.py` — current `torch.compile` implementation.
- `tests/` — state, semantics, checkpoint, and compiled-parity tests.
- `notebooks/` — self-contained Colab notebooks for the 5M compiled-parity run
  and 50M systems benchmark.
- `docs/algorithm.md` — state encoding and update rule.
- `docs/reproducibility.md` — experiment boundaries and verification steps.
- `results/` — current paper's CSV evidence.
- `paper/` — current PDF, LaTeX source, bibliography, and figures.

## Status

LogRMS is research software. The current evidence establishes that the
two-byte design can train the tested transformers and can run near fused AdamW
step speed after compilation. It does not yet establish multi-seed convergence
equivalence, large-scale distributed behavior, or superiority across tasks.

## Author and citation

LogRMS was developed by Matthew Belcher. Citation metadata is available in
[`CITATION.cff`](CITATION.cff).

Code is licensed under Apache License 2.0. The paper, figures, documentation,
and experimental tables are available under CC BY 4.0 as described in
[`NOTICE`](NOTICE).
