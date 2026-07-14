# Colab notebooks

## 5M compiled parity

[`LogRMS_TinyStories_5M_Compiled_Parity.ipynb`](LogRMS_TinyStories_5M_Compiled_Parity.ipynb)
trains compiled LogRMS for two complete passes over the official TinyStories
training split. The notebook embeds the current reference and compiled source,
verifies their hashes, and compares the run against recorded reference
telemetry at matching steps and token counts. It does not require an earlier
Google Drive run.

## 50M systems benchmark

[`LogRMS_TinyStories_50M_Systems_Benchmark.ipynb`](LogRMS_TinyStories_50M_Systems_Benchmark.ipynb)
compares AdamW foreach, fused AdamW, torchao 8-bit AdamW, and compiled LogRMS on
the same approximately 50M-parameter TinyStories model. It records optimizer
step time, full-step time, total wall time, throughput, persistent optimizer
state, validation trajectory, test loss, and whole-process peak PyTorch
allocation to CSV.

Both notebooks are self-contained apart from downloading their public Python
dependencies and the official TinyStories dataset. Their output cells are
cleared so repository diffs remain small and reproducible.
