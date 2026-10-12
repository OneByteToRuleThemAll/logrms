"""49.5M-parameter TinyStories benchmark for LogRMS, LogAdam and AdamW baselines.

Follows the model, tokenizer, data split and optimizer settings of
``notebooks/LogRMS_TinyStories_50M_Systems_Benchmark.ipynb`` as a plain script
that runs on any single CUDA GPU, Windows included.

    python benchmarks/tinystories_50m.py prepare --data DATA
    python benchmarks/tinystories_50m.py systems --data DATA --out OUT
    python benchmarks/tinystories_50m.py train --data DATA --out OUT \
        --optimizer logadam_compiled --seed 0 --updates 6000

``systems`` measures step time, optimizer time, peak allocation and exact
optimizer state for every available optimizer on identical batches.
``train`` runs one optimizer for a fixed number of updates with the notebook's
warmup + cosine schedule compressed to that length, then evaluates on the
full validation and test halves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, CTX, D_MODEL, HEADS, LAYERS, FF = 4096, 256, 512, 8, 15, 2048
MICROBATCH, ACCUMULATION, EVAL_BATCH = 16, 2, 32
PEAK_LR, WEIGHT_DECAY, BETA2, EPS, MAX_NORM = 1e-3, 0.01, 0.999, 1e-8, 1.0
WARMUP_FRACTION, MIN_LR_RATIO = 0.05, 0.01
BASE_MODEL_SEED, BASE_DATA_SEED = 2701, 3701
TOKENIZER_CHARS = 50_000_000


# ----------------------------------------------------------------------------- data


def prepare(data: Path) -> None:
    import sentencepiece as spm
    from datasets import load_dataset

    data.mkdir(parents=True, exist_ok=True)
    train = load_dataset("roneneldan/TinyStories", split="train")
    official_validation = load_dataset("roneneldan/TinyStories", split="validation")
    midpoint = len(official_validation) // 2
    splits = {
        "train": train,
        "validation": official_validation.select(range(midpoint)),
        "test": official_validation.select(range(midpoint, len(official_validation))),
    }

    prefix = data / "tinystories_bpe4096"
    if not prefix.with_suffix(".model").exists():
        text_path = data / "tokenizer_training_text.txt"
        written = 0
        with text_path.open("w", encoding="utf-8") as handle:
            for row in train:
                remaining = TOKENIZER_CHARS - written
                if remaining <= 0:
                    break
                text = row["text"][:remaining].replace("\x00", " ")
                handle.write(text + "\n")
                written += len(text)
        spm.SentencePieceTrainer.train(
            input=str(text_path), model_prefix=str(prefix), vocab_size=VOCAB,
            model_type="bpe", character_coverage=1.0, pad_id=0, unk_id=1, eos_id=2,
            bos_id=-1, byte_fallback=True, num_threads=max(1, os.cpu_count() or 1),
            shuffle_input_sentence=False, input_sentence_size=0, minloglevel=2,
        )
    processor = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))
    assert processor.vocab_size() == VOCAB

    for name, dataset in splits.items():
        binary = data / f"{name}.uint16.bin"
        if binary.exists():
            print(f"{name}: exists, {binary.stat().st_size // 2:,} tokens")
            continue
        temporary = binary.with_suffix(".tmp")
        count = 0
        started = time.time()
        with temporary.open("wb") as handle:
            for start in range(0, len(dataset), 4096):
                texts = dataset[start : start + 4096]["text"]
                encoded = processor.encode(texts, out_type=int, num_threads=os.cpu_count() or 1)
                flat = np.fromiter(
                    (t for ids in encoded for t in (*ids, processor.eos_id())), dtype=np.uint16
                )
                flat.tofile(handle)
                count += flat.size
                if start % (4096 * 50) == 0:
                    print(f"{name}: {start:,}/{len(dataset):,} stories, {count:,} tokens, "
                          f"{time.time() - started:.0f}s", flush=True)
        os.replace(temporary, binary)
        print(f"{name}: {count:,} tokens", flush=True)


class Tokens:
    """Packed token stream cut into fixed blocks, as in the notebook."""

    def __init__(self, path: Path):
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)
        self.blocks = math.ceil(max(0, len(self.tokens) - 1) / CTX)

    def batch(self, indices: list[int], device: torch.device):
        inputs = np.zeros((len(indices), CTX), dtype=np.int64)
        labels = np.full((len(indices), CTX), -100, dtype=np.int64)
        for row, index in enumerate(indices):
            start = index * CTX
            raw = np.asarray(self.tokens[start : min(start + CTX + 1, len(self.tokens))], dtype=np.int64)
            inputs[row, : raw.size - 1] = raw[:-1]
            labels[row, : raw.size - 1] = raw[1:]
        return (torch.from_numpy(inputs).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(labels).pin_memory().to(device, non_blocking=True))


# ---------------------------------------------------------------------------- model


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x):
        b, n, w = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, HEADS, w // HEADS).unbind(dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.proj(out.transpose(1, 2).contiguous().view(b, n, w))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(D_MODEL), Attention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, FF), nn.GELU(), nn.Linear(FF, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyStoriesLM50M(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB, D_MODEL)
        self.position_embedding = nn.Embedding(CTX, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.final_norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.apply(self._init)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        hidden = self.token_embedding(tokens) + self.position_embedding(
            torch.arange(tokens.shape[1], device=tokens.device)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.final_norm(hidden))


# ----------------------------------------------------------------------- optimizers


def parameter_groups(model):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(name: str, model: nn.Module, seed: int, device: torch.device):
    groups = parameter_groups(model)
    if name == "adamw_fused":
        return torch.optim.AdamW(groups, lr=PEAK_LR, betas=(0.9, BETA2), eps=EPS, fused=True)
    if name == "adamw_foreach":
        return torch.optim.AdamW(groups, lr=PEAK_LR, betas=(0.9, BETA2), eps=EPS, foreach=True)
    if name == "logrms_compiled":
        from logrms import CONFIDENCE_DEFAULT, LogRMSCompiled
        return LogRMSCompiled(
            groups, lr=PEAK_LR, beta2=BETA2, eps=EPS, backend="8bit",
            confidence_table=CONFIDENCE_DEFAULT, max_norm=None, error_if_nonfinite=False,
            stochastic_rounding=True, rounding_seed=BASE_MODEL_SEED + seed,
            compile_mode="reduce-overhead", fallback_on_compile_error=False,
        )
    if name == "logadam_compiled":
        from logrms import LogAdamCompiled
        return LogAdamCompiled(
            groups, lr=PEAK_LR, betas=(0.9, BETA2), eps=EPS, max_norm=None,
            error_if_nonfinite=False, stochastic_rounding=True,
            rounding_seed=BASE_MODEL_SEED + seed, compile_mode="reduce-overhead",
            fallback_on_compile_error=False,
        )
    if name == "adamw8bit_torchao":
        from torchao.optim import AdamW8bit
        return AdamW8bit(groups, lr=torch.tensor(PEAK_LR, device=device),
                         betas=(0.9, BETA2), eps=EPS, block_size=256)
    if name == "adamw8bit_bnb":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=PEAK_LR, betas=(0.9, BETA2), eps=EPS)
    raise KeyError(name)


def set_lr(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(value)
        else:
            group["lr"] = value


def scheduled_lr(step: int, total: int) -> float:
    warmup = max(1, round(total * WARMUP_FRACTION))
    if step < warmup:
        return PEAK_LR * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return PEAK_LR * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine)


def state_bytes(optimizer) -> int:
    """Unique persistent optimizer-state storage, including tensor subclasses."""
    seen: set[tuple[int, int]] = set()
    total = 0

    def add(t: torch.Tensor) -> None:
        nonlocal total
        if type(t) is not torch.Tensor and hasattr(t, "__tensor_flatten__"):
            names, _ = t.__tensor_flatten__()
            for name in names:
                add(getattr(t, name))
            return
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if key not in seen and t.numel() > 1:
            seen.add(key)
            total += t.untyped_storage().nbytes()

    for state in optimizer.state.values():
        for key, value in state.items():
            if key == "step" or not torch.is_tensor(value):
                continue
            add(value)
    return total


def available(name: str) -> bool:
    try:
        if name == "adamw8bit_torchao":
            import torchao.optim  # noqa: F401
        if name == "adamw8bit_bnb":
            import bitsandbytes  # noqa: F401
        return True
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"skipping {name}: {type(error).__name__}: {error}")
        return False


# --------------------------------------------------------------------- train loop


def train_step(model, optimizer, batches, *, timing: bool):
    if timing:
        torch.cuda.synchronize()
    started = time.perf_counter()
    total_loss = 0.0
    for inputs, labels in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=-100)
        (loss / len(batches)).backward()
        total_loss += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
    if timing:
        torch.cuda.synchronize()
    optimizer_started = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if timing:
        torch.cuda.synchronize()
    finished = time.perf_counter()
    return total_loss / len(batches), norm, (finished - started) * 1e3, (finished - optimizer_started) * 1e3


@torch.inference_mode()
def evaluate(model, data: Tokens, device, max_batches: int | None = None) -> float:
    model.eval()
    loss_sum, count = 0.0, 0
    for start in range(0, data.blocks, EVAL_BATCH):
        if max_batches is not None and start // EVAL_BATCH >= max_batches:
            break
        inputs, labels = data.batch(list(range(start, min(start + EVAL_BATCH, data.blocks))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                                          ignore_index=-100, reduction="sum"))
        count += int((labels != -100).sum())
    model.train()
    return loss_sum / count


def systems(args) -> None:
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.data) / "train.uint16.bin"
    data = Tokens(path) if path.exists() else None
    generator = torch.Generator().manual_seed(BASE_DATA_SEED)
    order = torch.randperm(data.blocks if data else 1_000_000, generator=generator).tolist()
    rows = []
    for name in args.optimizers:
        if not available(name):
            continue
        torch.manual_seed(BASE_MODEL_SEED)
        model = TinyStoriesLM50M().to(device)
        optimizer = make_optimizer(name, model, 0, device)
        parameters = sum(p.numel() for p in model.parameters())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        full, opt = [], []
        first_step_ms = None
        for step in range(args.warmup + args.steps):
            batches = []
            for micro in range(ACCUMULATION):
                index = (step * ACCUMULATION + micro) * MICROBATCH
                ids = order[index : index + MICROBATCH]
                if data is not None:
                    batches.append(data.batch(ids, device))
                else:
                    tokens = torch.randint(0, VOCAB, (MICROBATCH, CTX + 1), device=device)
                    batches.append((tokens[:, :-1], tokens[:, 1:]))
            set_lr(optimizer, PEAK_LR)
            _, _, full_ms, opt_ms = train_step(model, optimizer, batches, timing=True)
            if step == 0:
                first_step_ms = full_ms
            if step >= args.warmup:
                full.append(full_ms)
                opt.append(opt_ms)
        row = {
            "optimizer": name,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "parameters": parameters,
            "state_bytes": state_bytes(optimizer),
            "state_bytes_per_parameter": state_bytes(optimizer) / parameters,
            "first_step_ms": first_step_ms,
            "median_full_step_ms": statistics.median(full),
            "p90_full_step_ms": float(np.percentile(full, 90)),
            "median_optimizer_step_ms": statistics.median(opt),
            "p90_optimizer_step_ms": float(np.percentile(opt, 90)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "timed_steps": len(full),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        del model, optimizer
        torch.cuda.empty_cache()
    with (out / "systems.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args) -> None:
    device = torch.device("cuda")
    out = Path(args.out) / f"{args.optimizer}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print("already done", out)
        return
    data_root = Path(args.data)
    train_data = Tokens(data_root / "train.uint16.bin")
    valid_data = Tokens(data_root / "validation.uint16.bin")
    test_data = Tokens(data_root / "test.uint16.bin")

    torch.manual_seed(BASE_MODEL_SEED + args.seed)
    model = TinyStoriesLM50M().to(device)
    optimizer = make_optimizer(args.optimizer, model, args.seed, device)
    order = torch.randperm(
        train_data.blocks, generator=torch.Generator().manual_seed(BASE_DATA_SEED + args.seed)
    ).tolist()

    evaluations = []
    started = time.perf_counter()
    loss_ema = None
    full_times, opt_times = [], []
    skipped = 0
    for step in range(args.updates):
        set_lr(optimizer, scheduled_lr(step, args.updates))
        batches = []
        for micro in range(ACCUMULATION):
            index = (step * ACCUMULATION + micro) * MICROBATCH
            batches.append(train_data.batch(order[index : index + MICROBATCH], device))
        timing = step % 50 == 0
        loss, norm, full_ms, opt_ms = train_step(model, optimizer, batches, timing=timing)
        if timing and step >= 20:
            full_times.append(full_ms)
            opt_times.append(opt_ms)
        if (step + 1) % 50 == 0:
            value = float(loss)
            if not math.isfinite(value):
                skipped += 1
            loss_ema = value if loss_ema is None else 0.9 * loss_ema + 0.1 * value
        if (step + 1) % args.eval_every == 0 or step + 1 == args.updates:
            validation = evaluate(model, valid_data, device, max_batches=args.eval_batches)
            evaluations.append({
                "optimizer": args.optimizer, "seed": args.seed, "step": step + 1,
                "tokens": (step + 1) * MICROBATCH * ACCUMULATION * CTX,
                "train_loss_ema": loss_ema, "validation_subset_loss": validation,
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(json.dumps(evaluations[-1]), flush=True)
    train_seconds = time.perf_counter() - started
    summary = {
        "optimizer": args.optimizer, "seed": args.seed, "updates": args.updates,
        "tokens": args.updates * MICROBATCH * ACCUMULATION * CTX,
        "final_validation_loss": evaluate(model, valid_data, device),
        "test_loss": evaluate(model, test_data, device),
        "train_seconds": train_seconds,
        "median_full_step_ms_sampled": statistics.median(full_times) if full_times else None,
        "median_optimizer_step_ms_sampled": statistics.median(opt_times) if opt_times else None,
        "state_bytes_per_parameter": state_bytes(optimizer) / sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "nonfinite_loss_samples": skipped,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    summary["test_perplexity"] = math.exp(summary["test_loss"])
    with (out / "evaluations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluations[0]))
        writer.writeheader()
        writer.writerows(evaluations)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    s = sub.add_parser("systems")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--optimizers", nargs="+", default=[
        "adamw_fused", "adamw_foreach", "logrms_compiled", "logadam_compiled",
        "adamw8bit_bnb", "adamw8bit_torchao"])
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--optimizer", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--updates", type=int, required=True)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-batches", type=int, default=128)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(Path(args.data))
    elif args.command == "systems":
        systems(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
