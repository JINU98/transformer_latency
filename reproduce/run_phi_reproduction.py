from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
DECODE_TOKENS = 10
VOCAB_SIZE = 30257

PHI_CONFIG = {
    "shape_name": "phi_old_baseline_like",
    "d_model": 3072,
    "num_heads": 32,
    "num_kv_heads": 32,
    "num_layers": 32,
    "d_ff": 12288,
    "vocab_size": VOCAB_SIZE,
}

NAME_MAP = {
    "attn.q_proj": "Q Projection",
    "attn.k_proj": "K Projection",
    "attn.v_proj": "V Projection",
    "attn.qkv_projection": "QKV Projection",
    "attn.matmul_qk": "QK^T MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.softmax": "Softmax",
    "attn.weighted_sum": "Attention Weighted Sum",
    "attn.out_projection": "Output Projection",
    "attn.cache_concat": "KV Cache Concat",
    "block.norm1": "LayerNorm (Pre-Attention)",
    "block.norm2": "LayerNorm (Pre-FFN)",
    "ff.linear1": "FFN Linear 1",
    "ff.gelu": "GELU Activation",
    "ff.linear2": "FFN Linear 2",
    "model.final_norm": "Final LayerNorm",
    "model.output_head": "Output Head Projection",
}

OLD_ALIGNED_ORDER = [
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

RAW_ORDER = [
    "block.norm1",
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.matmul_qk",
    "attn.apply_causal_mask",
    "attn.softmax",
    "attn.weighted_sum",
    "attn.out_projection",
    "block.norm2",
    "ff.linear1",
    "ff.gelu",
    "ff.linear2",
    "model.final_norm",
    "model.output_head",
    "attn.cache_concat",
]


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return SEQ_LENS
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dtype_from_name(torch, dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


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


def build_metadata(seq_len: int, phase: str, phase_tokens: int, repeats: int, dtype: str, device: str) -> dict[str, object]:
    return {
        "architecture": "decoder",
        "model_family": "phi",
        "shape_name": PHI_CONFIG["shape_name"],
        "d_model": PHI_CONFIG["d_model"],
        "num_heads": PHI_CONFIG["num_heads"],
        "num_kv_heads": PHI_CONFIG["num_kv_heads"],
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
        "dtype": dtype,
        "device": device,
    }


def append_samples(target: dict[str, list[float]], source: dict[str, list[float]]) -> None:
    for key, values in source.items():
        target[key].extend(values)


def split_prefill_decode(samples: dict[str, list[float]], prefill_lengths: dict[str, int]):
    prefill = {}
    decode = {}
    for key, values in samples.items():
        boundary = prefill_lengths.get(key, 0)
        prefill[key] = values[:boundary]
        decode[key] = values[boundary:]
    return prefill, decode


def rows_from_samples(samples: dict[str, list[float]], metadata: dict[str, object], preferred_order: list[str]) -> list[dict[str, object]]:
    totals = {key: sum(values) for key, values in samples.items() if values}
    grand_total = sum(totals.values())
    keys = [key for key in preferred_order if key in samples and samples[key]]
    keys.extend(sorted(key for key in samples if key not in keys and samples[key]))
    rows = []
    phase_tokens = max(int(metadata["phase_tokens"]), 1)
    repeats = max(int(metadata["timed_repeats"]), 1)
    for key in keys:
        values = samples[key]
        if not values:
            continue
        total_ms = totals[key]
        row = dict(metadata)
        row.update(
            {
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
                "avg_total_ms_per_repeat": total_ms / repeats,
                "avg_total_ms_per_token": total_ms / (repeats * phase_tokens),
                "pct_total": (100.0 * total_ms / grand_total) if grand_total else 0.0,
            }
        )
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
        "source_operations",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def old_aligned_samples(prefill_samples: dict[str, list[float]], decode_samples: dict[str, list[float]]) -> dict[str, list[float]]:
    aligned: dict[str, list[float]] = defaultdict(list)
    combined_keys = set(prefill_samples) | set(decode_samples)
    for key in combined_keys:
        if key == "model.final_norm":
            continue
        if key in {"attn.q_proj", "attn.k_proj", "attn.v_proj"}:
            aligned["attn.qkv_projection"].extend(prefill_samples.get(key, []))
            aligned["attn.qkv_projection"].extend(decode_samples.get(key, []))
            continue
        if key == "attn.apply_causal_mask":
            aligned[key].extend(prefill_samples.get(key, []))
            continue
        aligned[key].extend(prefill_samples.get(key, []))
        aligned[key].extend(decode_samples.get(key, []))
    return dict(aligned)


def adjust_qkv_group_row_counts(rows: list[dict[str, object]]) -> None:
    """Make grouped split Q/K/V counts comparable to the older fused QKV scope."""
    for row in rows:
        if row["name"] != "attn.qkv_projection":
            continue
        split_count = int(row["count"])
        fused_like_count = max(split_count // 3, 1)
        row["source_operations"] = "attn.q_proj+attn.k_proj+attn.v_proj"
        row["count"] = fused_like_count
        row["avg_ms"] = float(row["total_ms"]) / fused_like_count
        row["avg"] = float(row["total"]) / fused_like_count


def run_one_sequence(args, seq_len: int):
    from common.config import ModelShape
    from common.latency import LatencyRecorder
    from common.models import DecoderModel, require_torch

    torch, _, _ = require_torch()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = dtype_from_name(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    if args.seed is not None:
        torch.manual_seed(args.seed + seq_len)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + seq_len)

    shape = ModelShape(
        PHI_CONFIG["shape_name"],
        PHI_CONFIG["d_model"],
        PHI_CONFIG["num_heads"],
        PHI_CONFIG["num_layers"],
        PHI_CONFIG["d_ff"],
        num_kv_heads=PHI_CONFIG["num_kv_heads"],
        note="Old Phi/GPT baseline reproduction shape",
    )
    recorder = LatencyRecorder(sync_cuda=(device.type == "cuda"), torch_module=torch)
    model = DecoderModel(
        shape,
        recorder,
        ffn_kind="gelu",
        norm_kind="layernorm",
        vocab_size=PHI_CONFIG["vocab_size"],
    ).to(device=device, dtype=dtype)
    model.eval()

    prefill = torch.randn(1, seq_len, shape.d_model, device=device, dtype=dtype)
    decode_inputs = [
        torch.randn(1, 1, shape.d_model, device=device, dtype=dtype)
        for _ in range(DECODE_TOKENS)
    ]

    def run_pass(record: bool):
        if not record:
            context = recorder.disabled()
        else:
            context = nullcontext()
        with context:
            _, past_kv = model(prefill, None, full_logits=True)
            prefill_lengths = {key: len(values) for key, values in recorder.samples_ms.items()}
            for token in decode_inputs:
                _, past_kv = model(token, past_kv, full_logits=False)
        return prefill_lengths

    with torch.no_grad():
        for _ in range(args.warmups):
            run_pass(record=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        combined_samples: dict[str, list[float]] = defaultdict(list)
        prefill_samples_all: dict[str, list[float]] = defaultdict(list)
        decode_samples_all: dict[str, list[float]] = defaultdict(list)
        for _ in range(args.repeats):
            recorder.samples_ms.clear()
            prefill_lengths = run_pass(record=True)
            pass_samples = {key: list(values) for key, values in recorder.samples_ms.items()}
            prefill_samples, decode_samples = split_prefill_decode(pass_samples, prefill_lengths)
            append_samples(combined_samples, pass_samples)
            append_samples(prefill_samples_all, prefill_samples)
            append_samples(decode_samples_all, decode_samples)

    device_label = str(device)
    dtype_label = args.dtype if dtype != torch.float32 or args.dtype == "float32" else "float32"
    combined_meta = build_metadata(seq_len, "prefill_plus_decode", seq_len + DECODE_TOKENS, args.repeats, dtype_label, device_label)
    prefill_meta = build_metadata(seq_len, "prefill", seq_len, args.repeats, dtype_label, device_label)
    decode_meta = build_metadata(seq_len, "decode", DECODE_TOKENS, args.repeats, dtype_label, device_label)

    raw_rows = rows_from_samples(dict(combined_samples), combined_meta, RAW_ORDER)
    prefill_rows = rows_from_samples(dict(prefill_samples_all), prefill_meta, RAW_ORDER)
    decode_rows = rows_from_samples(dict(decode_samples_all), decode_meta, RAW_ORDER)
    aligned = old_aligned_samples(dict(prefill_samples_all), dict(decode_samples_all))
    aligned_rows = rows_from_samples(aligned, combined_meta, OLD_ALIGNED_ORDER)
    adjust_qkv_group_row_counts(aligned_rows)

    return raw_rows, aligned_rows, prefill_rows, decode_rows


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def plot_pies(aligned_csvs: list[Path], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available; CSVs were saved but plots were skipped")
        return

    frames = []
    for path in aligned_csvs:
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return
    all_components = sorted({name for df in frames for name in df["pretty_name"]})
    cmap = plt.cm.tab20
    colors = {component: cmap(idx % 20) for idx, component in enumerate(all_components)}

    for df in frames:
        seq_len = int(df["seq_len"].iloc[0])
        df = df.sort_values("total", ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        wedges, _, autotexts = ax.pie(
            df["total"],
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 1.8 else "",
            startangle=90,
            colors=[colors[name] for name in df["pretty_name"]],
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
        fig.savefig(out_dir / f"pie_chart_{seq_len}.png", dpi=400)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="s", color=colors[name], linestyle="", markersize=18)
        for name in all_components
    ]
    ax.legend(handles, all_components, title="Components", loc="center", fontsize=30, title_fontsize=22, frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "component_legend.png", dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate old Phi-style prefill+10 decode latency pies with current code.")
    parser.add_argument("--seq-lens", default=None, help="Comma-separated context lengths. Default: 256,512,1024,2048,4096,8192")
    parser.add_argument("--output-dir", default="reproduce/phi_baseline_like")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_current"
    aligned_dir = out_dir / "old_aligned"
    phase_dir = out_dir / "phase_split"
    aligned_csvs: list[Path] = []

    print("Phi-style reproduction config:")
    for key, value in PHI_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"  decode_tokens: {DECODE_TOKENS}")
    print(f"  seq_lens: {parse_int_list(args.seq_lens)}")

    for seq_len in parse_int_list(args.seq_lens):
        print(f"\nRUN seq_len={seq_len}")
        raw_rows, aligned_rows, prefill_rows, decode_rows = run_one_sequence(args, seq_len)
        raw_path = raw_dir / f"latency_seqlen_{seq_len}.csv"
        aligned_path = aligned_dir / f"latency_seqlen_{seq_len}.csv"
        prefill_path = phase_dir / f"prefill_seqlen_{seq_len}.csv"
        decode_path = phase_dir / f"decode10_seqlen_{seq_len}.csv"
        save_csv(raw_path, raw_rows)
        save_csv(aligned_path, aligned_rows)
        save_csv(prefill_path, prefill_rows)
        save_csv(decode_path, decode_rows)
        aligned_csvs.append(aligned_path)
        total_s = sum(float(row["total"]) for row in aligned_rows)
        print(f"  saved {aligned_path} total={total_s:.4f}s")

    if not args.no_plots:
        plot_pies(aligned_csvs, aligned_dir)
        print(f"\nSaved figures to {aligned_dir}")


if __name__ == "__main__":
    main()
