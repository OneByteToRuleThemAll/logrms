# Reproducibility and claim boundaries

## Registered configuration

```text
backend:              8bit log-encoded second moment
confidence:           damp-only sign consistency
beta2:                0.999
eps:                  1e-8
global clip norm:     1.0
stochastic rounding:  enabled
```

Learning rate and weight decay remain task-specific hyperparameters.

## Included evidence

The repository contains two current studies:

- A 5.06M-parameter TinyStories study comparing FP32 AdamW, torchao 8-bit
  AdamW, and LogRMS over two dataset passes.
- A 49.5M-parameter TinyStories systems study comparing AdamW foreach, fused
  AdamW, torchao 8-bit AdamW, and compiled LogRMS.

The 49.5M study reports optimizer and full-step wall times separately, exact
persistent tensor state, complete validation trajectories, final test loss, and
whole-process peak PyTorch allocation. It uses one final-run seed and a shared
schedule that was not independently tuned for every optimizer.

## Interpretation limits

- Single-run transformer comparisons do not establish statistical equivalence.
- Peak PyTorch allocation covers the entire training process, not only
  optimizer state.
- Persistent state bytes do not include temporary tensors, compiler caches,
  allocator reservations, or fixed Python metadata.
- The first compiled step includes a substantial one-time compilation cost.
- Results on TinyStories do not establish behavior on larger transformers,
  sparse gradients, distributed training, or other task families.

## Verification

```bash
python -m pip install -e '.[test]'
pytest
python examples/minimal_training.py
```

To rebuild the paper:

```bash
cd paper
pdflatex logrms_paper.tex
bibtex logrms_paper
pdflatex logrms_paper.tex
pdflatex logrms_paper.tex
```
