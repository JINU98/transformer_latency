from __future__ import annotations

import argparse
import contextlib
import csv
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
DECODE_TOKENS = 10

PHI_CONFIG = {
    "shape_name": "phi_old_baseline_like",
    "vocab_size": 30257,
    "context_length": 16348,
    "d_model": 3072,
    "num_heads": 32,
    "num_layers": 32,
    "d_ff": 12288,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

NAME_MAP = {
    "block.norm1": "LayerNorm (Pre-Attention)",
    "attn.qkv_projection": "QKV Projection",
    "attn.matmul_qk": "QK^T MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.softmax": "Softmax",
    "attn.weighted_sum": "Attention Weighted Sum",
    "attn.out_projection": "Output Projection",
    "block.norm2": "LayerNorm (Pre-FFN)",
    "ff.linear1": "FFN Linear 1",
    "ff.gelu": "GELU Activation",
    "ff.linear2": "FFN Linear 2",
    "model.output_head": "Output Head Projection",
    "attn.cache_concat": "KV Cache Concat",
}

ORDER = [
    "block.norm1",
    "attn.qkv_projection",
    "attn.matmul_qk",
    "attn.apply_causal_mask",
    "attn.softmax",
    "attn.weighted_sum",
    "attn.out_projection",
    "block.norm2",
    "ff.linear1",
    "ff.gelu",
    "ff.linear2",
    "model.output_head",
    "attn.cache_concat",
]


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return SEQ_LENS
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def pretty_name(name: str) -> str:
    return NAME_MAP.get(name, name.replace(".", " ").replace("_", " ").title())


def component_group(name: str) -> str:
    if name.startswith("attn."):
        return "Self-Attention"
    if name.startswith("ff."):
        return "Feed Forward Network"
    if name.startswith("block."):
        return "Layer Normalization"
    if name.startswith("model."):
        return "Model Output"
    return "Other"


class LatencyRecorder:
    def __init__(self, use_cuda: bool, torch_module) -> None:
        self.use_cuda = use_cuda
        self.torch = torch_module
        self.samples_ms: dict[str, list[float]] = defaultdict(list)

    def sync(self) -> None:
        if self.use_cuda and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    @contextlib.contextmanager
    def record(self, name: str):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.samples_ms[name].append((time.perf_counter() - start) * 1000.0)


def make_model_classes(torch, nn, F, recorder: LatencyRecorder):
    PastKV = tuple[object, object]

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in: int, d_out: int, num_heads: int, qkv_bias: bool = False) -> None:
            super().__init__()
            if d_out % num_heads != 0:
                raise ValueError("d_out must be divisible by num_heads")
            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)

        def forward(self, x, past_kv: Optional[PastKV] = None, use_cache: bool = False):
            batch, seq_len, _ = x.shape
            with recorder.record("attn.qkv_projection"):
                qkv = self.qkv(x)
            qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            past_len = 0
            if past_kv is not None:
                past_k, past_v = past_kv
                past_len = past_k.size(-2)
                with recorder.record("attn.cache_concat"):
                    k = torch.cat([past_k, k], dim=-2)
                    v = torch.cat([past_v, v], dim=-2)

            scale = 1.0 / math.sqrt(self.head_dim)
            with recorder.record("attn.matmul_qk"):
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            total_k_len = k.size(-2)
            if seq_len > 1 or past_len > 0:
                key_pos = torch.arange(total_k_len, device=x.device)
                query_pos = past_len + torch.arange(seq_len, device=x.device)
                causal_mask = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)
                if causal_mask.any():
                    with recorder.record("attn.apply_causal_mask"):
                        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            with recorder.record("attn.softmax"):
                attn_weights = F.softmax(attn_scores, dim=-1)
            with recorder.record("attn.weighted_sum"):
                attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.d_out)
            with recorder.record("attn.out_projection"):
                out = self.out_proj(attn_out)
            present = (k, v) if use_cache else None
            return out, present

    class FeedForward(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_model, 4 * d_model)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(4 * d_model, d_model)

        def forward(self, x):
            with recorder.record("ff.linear1"):
                x = self.linear1(x)
            with recorder.record("ff.gelu"):
                x = self.act(x)
            with recorder.record("ff.linear2"):
                return self.linear2(x)

    class TransformerBlock(nn.Module):
        def __init__(self, cfg: dict[str, object]) -> None:
            super().__init__()
            d_model = int(cfg["d_model"])
            self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
            self.attn = MultiHeadAttention(
                d_model,
                d_model,
                int(cfg["num_heads"]),
                qkv_bias=bool(cfg["qkv_bias"]),
            )
            self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
            self.ff = FeedForward(d_model)

        def forward(self, x, past_kv: Optional[PastKV] = None, use_cache: bool = False):
            residual = x
            with recorder.record("block.norm1"):
                x = self.norm1(x)
            x, present = self.attn(x, past_kv=past_kv, use_cache=use_cache)
            x = x + residual

            residual = x
            with recorder.record("block.norm2"):
                x = self.norm2(x)
            x = self.ff(x) + residual
            return x, present

    class GPTModel(nn.Module):
        def __init__(self, cfg: dict[str, object]) -> None:
            super().__init__()
            self.cfg = cfg
            self.token_embedding = nn.Embedding(int(cfg["vocab_size"]), int(cfg["d_model"]))
            self.position_embedding = nn.Embedding(int(cfg["context_length"]), int(cfg["d_model"]))
            self.drop = nn.Dropout(float(cfg["drop_rate"]))
            self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(int(cfg["num_layers"]))])
            self.final_norm = nn.LayerNorm(int(cfg["d_model"]), eps=1e-5)
            self.output_head = nn.Linear(int(cfg["d_model"]), int(cfg["vocab_size"]), bias=False)

        def forward(self, input_ids, past_kv: Optional[list[Optional[PastKV]]] = None, use_cache: bool = False):
            _, seq_len = input_ids.shape
            if past_kv is None:
                past_kv = [None] * len(self.blocks)
                past_len = 0
            else:
                past_len = next((entry[0].size(-2) for entry in past_kv if entry is not None), 0)

            total_len = past_len + seq_len
            if total_len > int(self.cfg["context_length"]):
                raise ValueError(f"sequence length {total_len} exceeds context length {self.cfg['context_length']}")

            pos_idx = torch.arange(past_len, total_len, device=input_ids.device, dtype=torch.long)
            x = self.drop(self.token_embedding(input_ids) + self.position_embedding(pos_idx).unsqueeze(0))

            next_kv = []
            for block, block_past in zip(self.blocks, past_kv):
                x, present = block(x, past_kv=block_past, use_cache=use_cache)
                if use_cache:
                    next_kv.append(present)

            x = self.final_norm(x)
            with recorder.record("model.output_head"):
                logits = self.output_head(x)
            return (logits, next_kv) if use_cache else logits

    return GPTModel


def split_prefill_decode(samples: dict[str, list[float]], prefill_counts: dict[str, int]):
    prefill: dict[str, list[float]] = {}
    decode: dict[str, list[float]] = {}
    for key, values in samples.items():
        boundary = prefill_counts.get(key, 0)
        prefill[key] = values[:boundary]
        decode[key] = values[boundary:]
    return prefill, decode


def rows_from_samples(
    samples: dict[str, list[float]],
    seq_len: int,
    phase: str,
    phase_tokens: int,
    repeats: int,
    device: str,
) -> list[dict[str, object]]:
    totals = {key: sum(values) for key, values in samples.items() if values}
    grand_total = sum(totals.values())
    keys = [key for key in ORDER if key in samples and samples[key]]
    keys.extend(sorted(key for key in samples if key not in keys and samples[key]))
    rows = []
    for key in keys:
        values = samples[key]
        if not values:
            continue
        total_ms = totals[key]
        row = {
            "architecture": "decoder",
            "model_family": "phi",
            "shape_name": PHI_CONFIG["shape_name"],
            "d_model": PHI_CONFIG["d_model"],
            "num_heads": PHI_CONFIG["num_heads"],
            "num_kv_heads": PHI_CONFIG["num_heads"],
            "head_dim": PHI_CONFIG["d_model"] // PHI_CONFIG["num_heads"],
            "num_layers": PHI_CONFIG["num_layers"],
            "d_ff": PHI_CONFIG["d_ff"],
            "vocab_size": PHI_CONFIG["vocab_size"],
            "batch_size": 1,
            "seq_len": seq_len,
            "phase": phase,
            "phase_tokens": phase_tokens,
            "decode_tokens": DECODE_TOKENS,
            "timed_repeats": repeats,
            "dtype": "cuda_autocast_float16" if device == "cuda" else "float32",
            "device": device,
            "name": key,
            "operation_key": key,
            "pretty_name": pretty_name(key),
            "group": component_group(key),
            "count": len(values),
            "total_ms": total_ms,
            "total": total_ms / 1000.0,
            "avg_ms": statistics.fmean(values),
            "avg": statistics.fmean(values) / 1000.0,
            "min_ms": min(values),
            "min": min(values) / 1000.0,
            "max_ms": max(values),
            "max": max(values) / 1000.0,
            "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "avg_total_ms_per_repeat": total_ms / max(repeats, 1),
            "avg_total_ms_per_token": total_ms / max(repeats * phase_tokens, 1),
            "pct_total": (100.0 * total_ms / grand_total) if grand_total else 0.0,
        }
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "architecture",
        "model_family",
        "shape_name",
        "d_model",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "num_layers",
        "d_ff",
        "vocab_size",
        "batch_size",
        "seq_len",
        "phase",
        "phase_tokens",
        "decode_tokens",
        "timed_repeats",
        "dtype",
        "device",
        "name",
        "operation_key",
        "pretty_name",
        "group",
        "count",
        "total_ms",
        "total",
        "avg_ms",
        "avg",
        "min_ms",
        "min",
        "max_ms",
        "max",
        "std_ms",
        "avg_total_ms_per_repeat",
        "avg_total_ms_per_token",
        "pct_total",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_one_sequence(args, seq_len: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    torch.manual_seed(args.seed + seq_len)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + seq_len)

    recorder = LatencyRecorder(use_cuda=(device == "cuda"), torch_module=torch)
    GPTModel = make_model_classes(torch, nn, F, recorder)
    model = GPTModel(PHI_CONFIG).to(device)
    model.eval()

    input_ids = torch.randint(0, PHI_CONFIG["vocab_size"], (1, seq_len), dtype=torch.long, device=device)
    decode_input_ids = [
        torch.randint(0, PHI_CONFIG["vocab_size"], (1, 1), dtype=torch.long, device=device)
        for _ in range(DECODE_TOKENS)
    ]

    def autocast_context():
        if device != "cuda":
            return contextlib.nullcontext()
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    combined_samples: dict[str, list[float]] = defaultdict(list)
    prefill_samples_all: dict[str, list[float]] = defaultdict(list)
    decode_samples_all: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for _ in range(args.warmups):
            with autocast_context():
                _, past_kv = model(input_ids, use_cache=True)
                for token_ids in decode_input_ids:
                    _, past_kv = model(token_ids, past_kv=past_kv, use_cache=True)
        if device == "cuda":
            torch.cuda.empty_cache()

        for _ in range(args.repeats):
            recorder.samples_ms.clear()
            with autocast_context():
                _, past_kv = model(input_ids, use_cache=True)
                prefill_counts = {key: len(values) for key, values in recorder.samples_ms.items()}
                for token_ids in decode_input_ids:
                    _, past_kv = model(token_ids, past_kv=past_kv, use_cache=True)
            pass_samples = {key: list(values) for key, values in recorder.samples_ms.items()}
            prefill_samples, decode_samples = split_prefill_decode(pass_samples, prefill_counts)
            for key, values in pass_samples.items():
                combined_samples[key].extend(values)
            for key, values in prefill_samples.items():
                prefill_samples_all[key].extend(values)
            for key, values in decode_samples.items():
                decode_samples_all[key].extend(values)

    combined_rows = rows_from_samples(
        combined_samples,
        seq_len,
        "prefill_plus_decode",
        seq_len + DECODE_TOKENS,
        args.repeats,
        device,
    )
    prefill_rows = rows_from_samples(prefill_samples_all, seq_len, "prefill", seq_len, args.repeats, device)
    decode_rows = rows_from_samples(decode_samples_all, seq_len, "decode", DECODE_TOKENS, args.repeats, device)
    return combined_rows, prefill_rows, decode_rows


def plot_pies(csv_paths: list[Path], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available; skipping pie charts")
        return

    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return

    all_components = sorted({name for frame in frames for name in frame["pretty_name"]})
    cmap = plt.cm.tab20
    component_colors = {component: cmap(idx % 20) for idx, component in enumerate(all_components)}

    for frame in frames:
        seq_len = int(frame["seq_len"].iloc[0])
        frame = frame.sort_values("total", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, _, autotexts = ax.pie(
            frame["total"],
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 1.8 else "",
            startangle=90,
            colors=[component_colors[name] for name in frame["pretty_name"]],
            pctdistance=0.7,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        for wedge, autotext in zip(wedges, autotexts):
            angle = (wedge.theta1 + wedge.theta2) / 2
            if 90 < angle < 270:
                angle += 180
            autotext.set_rotation(angle)
            autotext.set_rotation_mode("anchor")
            autotext.set_fontsize(35)
        fig.tight_layout()
        fig.savefig(out_dir / f"pie_chart_{seq_len}.png", dpi=800)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="s", color=component_colors[name], linestyle="", markersize=18)
        for name in all_components
    ]
    ax.legend(handles, all_components, title="Components", loc="center", fontsize=30, title_fontsize=22, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "component_legend.png", dpi=800)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate old Phi/GPT baseline pies with fused QKV and CUDA fp16 autocast.")
    parser.add_argument("--seq-lens", default=None, help="Comma-separated context lengths.")
    parser.add_argument("--output-dir", default="reproduce/phi_baseline_like")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    aligned_dir = out_dir / "old_aligned"
    raw_dir = out_dir / "raw_current"
    phase_dir = out_dir / "phase_split"
    aligned_paths: list[Path] = []

    print("Old-compatible Phi/GPT reproduction config:")
    for key, value in PHI_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"  decode_tokens: {DECODE_TOKENS}")
    print(f"  seq_lens: {parse_int_list(args.seq_lens)}")
    print("  cuda path uses torch.amp.autocast('cuda', dtype=torch.float16)")

    for seq_len in parse_int_list(args.seq_lens):
        print(f"\nRUN seq_len={seq_len}")
        combined_rows, prefill_rows, decode_rows = run_one_sequence(args, seq_len)
        aligned_path = aligned_dir / f"latency_seqlen_{seq_len}.csv"
        raw_path = raw_dir / f"latency_seqlen_{seq_len}.csv"
        prefill_path = phase_dir / f"prefill_seqlen_{seq_len}.csv"
        decode_path = phase_dir / f"decode10_seqlen_{seq_len}.csv"
        save_csv(aligned_path, combined_rows)
        save_csv(raw_path, combined_rows)
        save_csv(prefill_path, prefill_rows)
        save_csv(decode_path, decode_rows)
        aligned_paths.append(aligned_path)
        total_s = sum(float(row["total"]) for row in combined_rows)
        softmax = next((row for row in combined_rows if row["name"] == "attn.softmax"), None)
        softmax_pct = float(softmax["pct_total"]) if softmax else 0.0
        print(f"  saved {aligned_path} total={total_s:.4f}s softmax={softmax_pct:.2f}%")

    if not args.no_plots:
        plot_pies(aligned_paths, aligned_dir)
        print(f"\nSaved pie charts to {aligned_dir}")


if __name__ == "__main__":
    main()
