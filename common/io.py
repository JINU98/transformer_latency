from __future__ import annotations

import csv
from pathlib import Path


FIELDNAMES = [
    "architecture",
    "model_family",
    "shape_name",
    "d_model",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "num_layers",
    "d_ff",
    "batch_size",
    "seq_len",
    "encoder_seq_len",
    "decoder_seq_len",
    "dtype",
    "device",
    "operation_key",
    "count",
    "avg_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "total_ms",
    "pct_total",
]


def save_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def operation_group(operation_key: str) -> str:
    if operation_key.startswith("cross_attn."):
        return "cross_attention"
    if operation_key.startswith("attn."):
        return "self_attention"
    if operation_key.startswith("ff."):
        return "ffn"
    if operation_key.startswith("block."):
        return "norm_residual"
    if operation_key.startswith("model."):
        return "model_io"
    return "other"
