"""Prefill vs. decode latency-share sweep for a representative model.

Supports both decoder-only models (GPT-2/GPT-3/LLaMA style) and
encoder-decoder models (T5/BART style). This extends the repository's
existing prefill/decode profiling with a question the per-shape figures
don't answer directly: *for a fixed model, how does the split between
prefill time and decode time change as you generate more tokens?*

Method
------
For each sequence length L (the prompt / source length):

1. Benchmark one full "prefill" pass over L tokens:
   - decoder-only: a full causal forward over L tokens to fill the KV cache.
   - encoder-decoder: encode a length-L source, then a causal, teacher-forced
     decoder pass over L decoder tokens (with cross-attention to the encoded
     source) to fill the decoder KV cache. Matches
     ``encoder_decoder_profiler``'s prefill phase.
2. Benchmark exactly ONE cached decode step taken right after that L-token
   prefill (i.e. ``--decode-tokens 1`` in the existing profilers). For
   encoder-decoder this is one decoder step: self-attention over the growing
   cache plus cross-attention to the (already encoded, now fixed) source.
   This gives the marginal cost of a single decode step at context length L,
   broken down by component (QKV projections, cross-attention, KV-cache
   concat, attention matmuls, FFN, norms, output head, ...).
3. For each "tokens generated" scenario (e.g. 1 token, 10 tokens, 20% of L,
   50% of L, 100% of L, ...), scale that single decode step's per-component
   latency linearly by the number of tokens N in the scenario. This is the
   approximation the assignment calls for: "for each sequence length you can
   have one decode step and scale it up for multiple decode steps." It is a
   good approximation as long as N stays a modest fraction of L, since the
   dominant per-step costs (QKV/FFN/output-head/cross-attention projections)
   don't depend on cache length at all, and the part that does (self-attention
   matmul + softmax over the growing decoder cache) grows slowly step to step
   for N << L.
4. Combine the (unscaled) prefill cost with the scaled decode cost to get a
   total "prefill share" and "decode share" of end-to-end generation latency
   for that (L, N) pair, plus a full component breakdown of the combined
   total.

Outputs
-------
- ``latency_results/<architecture>/<shape>/raw_l<L>.csv``: unscaled prefill +
  single-decode-step component latencies, one file per L.
- ``latency_results/<architecture>/<shape>/scenario_components.csv``: every
  (L, scenario, component) row after scaling, all L values and scenarios in
  one file.
- ``latency_results/<architecture>/<shape>/scenario_summary.csv``: one row
  per (L, scenario) with total prefill ms, total decode ms, and each phase's
  percent share.
- ``figures/gen_ratio/<architecture>/prefill_decode_share_<shape>.png``:
  stacked bar chart, prefill % vs decode % of total latency, grouped by L
  then by scenario. Same visual language as
  ``figures/decoder/*/model_family_component_share.png`` and
  ``figures/encoder_decoder/*/model_family_component_share.png``.
- ``figures/gen_ratio/<architecture>/component_share_<shape>.png``:
  fine-grained component share (QKV projection, FFN, KV-cache concat,
  cross-attention, output head, ...) of the *combined* prefill+decode total
  for each (L, scenario), in the same stacked-bar style as the repository's
  existing per-phase component-share figures.
- ``figures/gen_ratio/<architecture>/pie_charts/pie_<shape>_l<L>_<scenario>.png``:
  one pie chart per (L, scenario) combination, same style as the existing
  per-config pies.

Usage
-----
    cd gen_ratio_profiler

    # Decoder-only (default)
    python run_and_plot.py --architecture decoder --shape-name llama_7b \\
        --preset standard --token-scenarios 1,10,20%,50%,100%

    # Encoder-decoder
    python run_and_plot.py --architecture encoder_decoder --shape-name t5_base \\
        --preset standard --token-scenarios 1,10,20%,50%,100%
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import MODEL_SHAPES, SEQ_LEN_PRESETS, dtype_bytes, estimate_attention_gb, parse_int_list
from common.runner import metadata_for_shape, resolve_device_and_dtype
from gen_ratio_profiler.bench import ARCHITECTURES
from gen_ratio_profiler.scenarios import DEFAULT_SCENARIOS, parse_token_scenarios
from gen_ratio_profiler.io_utils import (
    RAW_FIELDNAMES,
    SCENARIO_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    save_csv,
)
from gen_ratio_profiler import gen_plots


def run_shape(args, architecture: str, shape, torch_module, device, dtype) -> tuple[list[dict], list[dict], list[dict]]:
    seq_lens = parse_int_list(args.seq_lens, SEQ_LEN_PRESETS[args.preset])
    scenarios = parse_token_scenarios(args.token_scenarios)
    spec = ARCHITECTURES[architecture]
    benchmark_fn = spec["benchmark_fn"]
    attn_multiplier = spec["attn_multiplier"]

    raw_rows: list[dict] = []
    scenario_rows: list[dict] = []
    summary_rows: list[dict] = []

    for seq_len in seq_lens:
        est_gb = estimate_attention_gb(
            args.batch_size, shape.num_heads, seq_len, seq_len, dtype_bytes(args.dtype), multiplier=attn_multiplier
        )
        if args.max_attn_gb > 0 and est_gb > args.max_attn_gb:
            print(f"SKIP {shape.name} L={seq_len}: estimated attention buffers {est_gb:.2f} GB > {args.max_attn_gb:.2f} GB")
            continue

        print(f"RUN {architecture} {shape.name} L={seq_len} device={device} (prefill + 1 decode step)")
        prefill_recorder, decode_recorder = benchmark_fn(
            shape, seq_len, args.batch_size, args.warmups, args.repeats, device, dtype, torch_module
        )

        prefill_meta = metadata_for_shape(args, architecture, architecture, shape, seq_len, device, "prefill", seq_len)
        decode_step_meta = metadata_for_shape(args, architecture, architecture, shape, seq_len, device, "decode_step", 1)
        prefill_component_rows = prefill_recorder.rows(prefill_meta, timed_repeats=args.repeats, phase_tokens=seq_len)
        decode_step_component_rows = decode_recorder.rows(decode_step_meta, timed_repeats=args.repeats, phase_tokens=1)
        raw_rows.extend(prefill_component_rows)
        raw_rows.extend(decode_step_component_rows)

        prefill_total_ms = sum(row["avg_total_ms_per_repeat"] for row in prefill_component_rows)
        decode_step_total_ms = sum(row["avg_total_ms_per_repeat"] for row in decode_step_component_rows)

        for scenario in scenarios:
            n_tokens = scenario.resolve(seq_len)
            decode_total_ms = decode_step_total_ms * n_tokens

            for row in prefill_component_rows:
                scenario_rows.append(
                    _scenario_row(row, "prefill", n_tokens, scenario, row["avg_total_ms_per_repeat"])
                )
            for row in decode_step_component_rows:
                scenario_rows.append(
                    _scenario_row(row, "decode", n_tokens, scenario, row["avg_total_ms_per_repeat"] * n_tokens)
                )

            total_ms = prefill_total_ms + decode_total_ms
            summary_rows.append(
                {
                    "architecture": architecture,
                    "shape_name": shape.name,
                    "d_model": shape.d_model,
                    "num_heads": shape.num_heads,
                    "num_layers": shape.num_layers,
                    "seq_len": seq_len,
                    "token_scenario": scenario.label,
                    "token_scenario_display": scenario.display,
                    "tokens_generated": n_tokens,
                    "prefill_ms": round(prefill_total_ms, 5),
                    "decode_ms": round(decode_total_ms, 5),
                    "total_ms": round(total_ms, 5),
                    "prefill_pct": round(100.0 * prefill_total_ms / total_ms, 3) if total_ms else 0.0,
                    "decode_pct": round(100.0 * decode_total_ms / total_ms, 3) if total_ms else 0.0,
                    "decode_ms_per_token": round(decode_step_total_ms, 5),
                }
            )

    return raw_rows, scenario_rows, summary_rows


def _scenario_row(component_row: dict, gen_role: str, n_tokens: int, scenario, scaled_total_ms: float) -> dict:
    row = dict(component_row)
    row["gen_role"] = gen_role
    row["tokens_generated"] = n_tokens
    row["token_scenario"] = scenario.label
    row["token_scenario_display"] = scenario.display
    row["scaled_total_ms"] = round(scaled_total_ms, 6)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep sequence length and number of generated tokens for one representative "
            "model (decoder-only or encoder-decoder), and chart the prefill-vs-decode "
            "latency split plus a full component breakdown, matching the style of "
            "figures/*/model_family_component_share.png."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=list(ARCHITECTURES),
        default="decoder",
        help="Which model family to profile: decoder-only (GPT-2/GPT-3/LLaMA style) or encoder_decoder (T5/BART style).",
    )
    parser.add_argument(
        "--shape-name",
        default=None,
        help=(
            "Representative model shape to profile. Defaults to gpt3_2p7b for --architecture decoder "
            "and bart_large for --architecture encoder_decoder. Available shapes: " + ",".join(MODEL_SHAPES)
        ),
    )
    parser.add_argument("--preset", choices=list(SEQ_LEN_PRESETS), default="quick")
    parser.add_argument("--seq-lens", default=None, help="Comma-separated context lengths L, e.g. 512,1024,2048")
    parser.add_argument(
        "--token-scenarios",
        default=DEFAULT_SCENARIOS,
        help=(
            "Comma-separated tokens-generated scenarios. Each item is either an absolute "
            "token count (e.g. '1', '10') or a percentage of L (e.g. '20%%', '50%%'). "
            f"Default: {DEFAULT_SCENARIOS}"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--max-attn-gb", type=float, default=8.0, help="Skip L values whose estimated attention buffers are too large. <=0 disables."
    )
    parser.add_argument("--output-dir", default="latency_results")
    parser.add_argument(
        "--figures-dir",
        default="../figures",
        help="Where to write figures. Defaults to the repository's shared figures/ folder (sibling to this script's directory), under gen_ratio/<architecture>/, matching figures/encoder, figures/decoder, figures/encoder_decoder.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    spec = ARCHITECTURES[args.architecture]
    shape_name = args.shape_name or spec["default_shape"]
    if shape_name not in MODEL_SHAPES:
        raise SystemExit(f"Unknown shape '{shape_name}'. Available: {', '.join(MODEL_SHAPES)}")
    if shape_name not in spec["shape_names"]:
        print(
            f"NOTE: '{shape_name}' is not one of the usual {args.architecture} shapes "
            f"({', '.join(spec['shape_names'])}), but proceeding since it's a known shape."
        )
    shape = MODEL_SHAPES[shape_name]

    from common.models import require_torch

    torch_module, _, _ = require_torch()
    device, dtype = resolve_device_and_dtype(torch_module, args.device, args.dtype)

    raw_rows, scenario_rows, summary_rows = run_shape(args, args.architecture, shape, torch_module, device, dtype)
    if not summary_rows:
        print("No configurations were run (all skipped by --max-attn-gb?).")
        return

    out_dir = Path(args.output_dir) / args.architecture / shape.name
    raw_by_l: dict[int, list[dict]] = {}
    for row in raw_rows:
        raw_by_l.setdefault(int(row["seq_len"]), []).append(row)
    for seq_len, rows in raw_by_l.items():
        save_csv(out_dir / f"raw_l{seq_len}.csv", rows, RAW_FIELDNAMES)

    scenario_csv = out_dir / "scenario_components.csv"
    summary_csv = out_dir / "scenario_summary.csv"
    save_csv(scenario_csv, scenario_rows, SCENARIO_FIELDNAMES)
    save_csv(summary_csv, summary_rows, SUMMARY_FIELDNAMES)
    print(f"Wrote {scenario_csv}")
    print(f"Wrote {summary_csv}")

    if not args.no_plots:
        figures_dir = Path(args.figures_dir) / "gen_ratio" / args.architecture
        gen_plots.make_all_plots(scenario_rows, summary_rows, shape.name, args.architecture, figures_dir)


if __name__ == "__main__":
    main()
