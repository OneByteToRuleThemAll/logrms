"""Small CPU ablation: AdamW vs LogRMS variants vs LogAdam on a char-level LM.

One run per invocation; prints one JSON line. Example:

    python benchmarks/char_lm_ablation.py logadam 3e-3 0 1200

The corpus is the first 120 top-level ``.py`` files of the running Python's
standard library, so absolute losses depend on the Python version. Compare
arms run on the same machine. This is a quick directional check on a
~0.36M-parameter model, not a substitute for the TinyStories benchmarks.
"""

from __future__ import annotations

import glob
import json
import math
import sys
import sysconfig
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from logrms import CONFIDENCE_DEFAULT, CONFIDENCE_RMS, LogAdam, LogRMS

torch.set_num_threads(1)
CTX, BATCH, D, LAYERS, HEADS = 64, 32, 96, 3, 4

LOGRMS_ARMS = {
    "logrms8": ("8bit", CONFIDENCE_DEFAULT),      # current default, 2 B/param
    "logrms_fp32": ("float32", CONFIDENCE_DEFAULT),
    "rms8": ("8bit", CONFIDENCE_RMS),             # log second moment only
    "rms8_x0.6": ("8bit", (0.6,) * 8),            # sign history -> constant
    "rms_fp32": ("float32", CONFIDENCE_RMS),
}
ARMS = ["adamw", "adamw_b1_0", "logadam", *LOGRMS_ARMS]


def corpus() -> tuple[torch.Tensor, torch.Tensor]:
    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(f"{stdlib}/*.py"))[:120]
    data = b"".join(open(f, "rb").read() for f in files)
    tokens = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    split = int(len(tokens) * 0.95)
    return tokens[:split], tokens[split:]


class LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(256, D)
        self.pos = nn.Parameter(torch.zeros(CTX, D))
        layer = nn.TransformerEncoderLayer(
            D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        self.mask = nn.Transformer.generate_square_subsequent_mask(CTX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tr(self.emb(x) + self.pos, mask=self.mask, is_causal=True)
        return F.linear(self.ln(h), self.emb.weight)


def batch(data: torch.Tensor, generator: torch.Generator):
    ix = torch.randint(0, len(data) - CTX - 1, (BATCH,), generator=generator)
    x = torch.stack([data[i : i + CTX] for i in ix])
    y = torch.stack([data[i + 1 : i + CTX + 1] for i in ix])
    return x, y


def run(arm: str, lr: float, seed: int, steps: int) -> dict:
    train, val = corpus()
    torch.manual_seed(seed)
    model = LM()
    generator = torch.Generator().manual_seed(1000 + seed)
    common = dict(lr=lr, weight_decay=0.01)
    external_clip = arm.startswith("adamw")
    if arm == "adamw":
        opt = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), **common)
    elif arm == "adamw_b1_0":
        opt = torch.optim.AdamW(model.parameters(), betas=(0.0, 0.999), **common)
    elif arm == "logadam":
        opt = LogAdam(model.parameters(), max_norm=1.0, rounding_seed=seed, **common)
    else:
        backend, table = LOGRMS_ARMS[arm]
        opt = LogRMS(
            model.parameters(), backend=backend, confidence_table=table,
            max_norm=1.0, rounding_seed=seed, **common,
        )
    warm = int(0.05 * steps)

    def schedule(s: int) -> float:
        if s < warm:
            return (s + 1) / warm
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm)))

    started = time.time()
    for s in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * schedule(s)
        x, y = batch(train, generator)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if external_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    eval_generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        val_loss = sum(
            F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
            for x, y in (batch(val, eval_generator) for _ in range(60))
        ) / 60
    result = dict(arm=arm, lr=lr, seed=seed, steps=steps, val_loss=val_loss,
                  train_loss_last=loss.item(), seconds=time.time() - started)
    if hasattr(opt, "state_bytes"):
        result["state_bytes_per_param"] = opt.state_bytes() / sum(
            p.numel() for p in model.parameters()
        )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: char_lm_ablation.py {{{'|'.join(ARMS)}}} LR SEED STEPS")
    print(json.dumps(run(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))), flush=True)
