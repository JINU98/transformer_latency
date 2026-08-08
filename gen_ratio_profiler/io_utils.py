from __future__ import annotations

import csv
from pathlib import Path

# Raw per-(L) component rows: one full prefill pass + exactly one cached
# decode step, unscaled. Same shape as the base repo's CSV schema plus the
# usual metadata columns.
RAW_FIELDNAMES = [
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
    "phase",
    "phase_tokens",
    "timed_repeats",
    "dtype",
    "device",
    "operation_key",
    "count",
    "avg_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "total_ms",
    "avg_total_ms_per_repeat",
    "avg_total_ms_per_token",
    "pct_total",
]

# Scenario-expanded rows: every raw component row, tagged with which
# generation scenario it belongs to and its scaled contribution.
SCENARIO_FIELDNAMES = RAW_FIELDNAMES + [
    "gen_role",
    "tokens_generated",
    "token_scenario",
    "token_scenario_display",
    "scaled_total_ms",
]

# One row per (L, scenario): the headline prefill/decode split.
SUMMARY_FIELDNAMES = [
    "architecture",
    "shape_name",
    "d_model",
    "num_heads",
    "num_layers",
    "seq_len",
    "token_scenario",
    "token_scenario_display",
    "tokens_generated",
    "prefill_ms",
    "decode_ms",
    "total_ms",
    "prefill_pct",
    "decode_pct",
    "decode_ms_per_token",
]


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
