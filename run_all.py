from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common.config import SEQ_LEN_PRESETS


REPO_ROOT = Path(__file__).resolve().parent

ENCODER_SHAPES = ["bert_base", "bert_large"]
DECODER_SHAPES = ["gpt2_medium", "gpt3_2p7b", "llama_7b"]
ENCODER_DECODER_SHAPES = ["t5_base", "t5_large", "bart_large"]
EXTREME_DECODER_SHAPES = ["llama2_70b_gqa"]
FAMILIES = ["bert", "gpt", "llama", "t5", "bart"]

TIER_SEQ_LENS = {
    "smoke": [128],
    "standard": SEQ_LEN_PRESETS["standard"],
    "full": sorted({seq for values in SEQ_LEN_PRESETS.values() for seq in values}),
    "exhaustive": sorted({seq for values in SEQ_LEN_PRESETS.values() for seq in values}),
}


def csv(values: list[object]) -> str:
    return ",".join(str(value) for value in values)


def add_shared_args(command: list[str], args: argparse.Namespace, seq_lens: list[int]) -> list[str]:
    command.extend(
        [
            "--seq-lens",
            csv(seq_lens),
            "--batch-size",
            str(args.batch_size),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-attn-gb",
            str(args.max_attn_gb),
        ]
    )
    if args.no_plots:
        command.append("--no-plots")
    return command


def run_command(command: list[str], cwd: Path, args: argparse.Namespace, failures: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n$ cd {cwd.relative_to(REPO_ROOT)} && {printable}")
    if args.dry_run:
        return
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        label = f"{cwd.relative_to(REPO_ROOT)} :: {printable}"
        failures.append(label)
        if not args.keep_going:
            raise SystemExit(result.returncode)


def clean_outputs() -> None:
    paths = [
        REPO_ROOT / "encoder_profiler" / "latency_results",
        REPO_ROOT / "decoder_profiler" / "latency_results",
        REPO_ROOT / "encoder_decoder_profiler" / "latency_results",
        REPO_ROOT / "model_family_profiler" / "latency_results",
        REPO_ROOT / "figures",
    ]
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path.relative_to(REPO_ROOT)}")


def shapes_for_tier(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str]]:
    if args.tier == "smoke":
        return ["tiny_debug"], ["tiny_debug"], ["tiny_debug"], ["tiny_debug"]

    encoder_shapes = list(ENCODER_SHAPES)
    decoder_shapes = list(DECODER_SHAPES)
    encoder_decoder_shapes = list(ENCODER_DECODER_SHAPES)
    attention_shapes = encoder_shapes + decoder_shapes + encoder_decoder_shapes

    if args.tier == "exhaustive":
        decoder_shapes.extend(EXTREME_DECODER_SHAPES)
        attention_shapes.extend(EXTREME_DECODER_SHAPES)

    return encoder_shapes, decoder_shapes, encoder_decoder_shapes, attention_shapes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Transformer latency profiler sweeps, attention microbenchmarks, "
            "and summary figure generation."
        )
    )
    parser.add_argument("--tier", choices=["smoke", "standard", "full", "exhaustive"], default="full")
    parser.add_argument("--seq-lens", default=None, help="Override tier sequence lengths, e.g. 128,512,1024")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-attn-gb", type=float, default=8.0)
    parser.add_argument("--no-plots", action="store_true", help="Run CSV sweeps only and skip all plot/figure commands.")
    parser.add_argument("--skip-attention-bench", action="store_true")
    parser.add_argument("--skip-model-family", action="store_true")
    parser.add_argument("--skip-summary-figures", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete generated latency_results/ and figures/ before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed command and report failures at the end.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seq_lens = [int(part.strip()) for part in args.seq_lens.split(",")] if args.seq_lens else TIER_SEQ_LENS[args.tier]
    encoder_shapes, decoder_shapes, encoder_decoder_shapes, attention_shapes = shapes_for_tier(args)
    failures: list[str] = []

    if args.clean and not args.dry_run:
        clean_outputs()

    py = sys.executable

    commands: list[tuple[list[str], Path]] = [
        (
            add_shared_args([py, "run_and_plot.py", "--shape-names", csv(encoder_shapes)], args, seq_lens),
            REPO_ROOT / "encoder_profiler",
        ),
        (
            add_shared_args([py, "run_and_plot.py", "--shape-names", csv(decoder_shapes)], args, seq_lens),
            REPO_ROOT / "decoder_profiler",
        ),
        (
            add_shared_args([py, "run_and_plot.py", "--shape-names", csv(encoder_decoder_shapes)], args, seq_lens),
            REPO_ROOT / "encoder_decoder_profiler",
        ),
    ]

    if not args.skip_model_family:
        family_command = [py, "run_and_plot.py", "--families", csv(FAMILIES)]
        if args.tier == "smoke":
            family_command.extend(["--shape-names", "tiny_debug"])
        commands.append(
            (
                add_shared_args(family_command, args, seq_lens),
                REPO_ROOT / "model_family_profiler",
            )
        )

    if not args.skip_attention_bench:
        commands.append(
            (
                add_shared_args(
                    [py, "simple_attn_bench.py", "--kind", "both", "--shape-names", csv(attention_shapes)],
                    args,
                    seq_lens,
                ),
                REPO_ROOT / "model_family_profiler",
            )
        )

    if not args.no_plots:
        commands.extend(
            [
                ([py, "plot_from_csv.py"], REPO_ROOT / "encoder_profiler"),
                ([py, "plot_from_csv.py"], REPO_ROOT / "decoder_profiler"),
                ([py, "plot_from_csv.py"], REPO_ROOT / "encoder_decoder_profiler"),
            ]
        )
        if not args.skip_model_family:
            commands.append(([py, "heatmap.py"], REPO_ROOT / "model_family_profiler"))
        if not args.skip_summary_figures:
            commands.append(([py, "figures.py"], REPO_ROOT))

    for command, cwd in commands:
        run_command(command, cwd, args, failures)

    if failures:
        print("\nFailed command(s):")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nAll requested profiler commands completed.")


if __name__ == "__main__":
    main()
