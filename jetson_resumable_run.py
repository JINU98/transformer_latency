#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192]
ENCODER_SHAPES = ["bert_base", "bert_large"]
DECODER_SHAPES = ["gpt2_medium", "gpt3_2p7b", "llama_7b"]
ENCODER_DECODER_SHAPES = ["t5_base", "t5_large", "bart_large"]
FAMILY_TO_SHAPE = {
    "bert": "bert_base",
    "gpt": "gpt2_medium",
    "llama": "llama_7b",
    "t5": "t5_base",
    "bart": "bart_large",
}
ATTN_SHAPES = ENCODER_SHAPES + DECODER_SHAPES + ENCODER_DECODER_SHAPES
EXPECTED_TOTAL = (
    len(ENCODER_SHAPES) * len(SEQ_LENS)
    + len(DECODER_SHAPES) * len(SEQ_LENS)
    + len(ENCODER_DECODER_SHAPES) * len(SEQ_LENS)
    + len(FAMILY_TO_SHAPE) * len(SEQ_LENS)
    + len(ATTN_SHAPES) * len(SEQ_LENS) * 2
)

COMMON = [
    "--batch-size", "1",
    "--warmups", "2",
    "--repeats", "5",
    "--device", "cuda",
    "--dtype", "float16",
    "--max-attn-gb", "16",
    "--no-plots",
]


def shape_meta(name: str):
    from common.config import MODEL_SHAPES
    s = MODEL_SHAPES[name]
    return s.d_model, s.num_heads


def csv_path(cwd: Path, family: str, shape: str, seq_len: int) -> Path:
    d_model, heads = shape_meta(shape)
    return cwd / "latency_results" / family / f"latency_{shape}_d{d_model}_h{heads}_l{seq_len}.csv"


def attn_paths(shape: str, seq_len: int) -> list[Path]:
    d_model, heads = shape_meta(shape)
    return [
        ROOT / "model_family_profiler" / "latency_results" / "attention_microbench" / kind / f"latency_{shape}_d{d_model}_h{heads}_l{seq_len}.csv"
        for kind in ["self", "cross"]
    ]


def run(label: str, cwd: Path, command: list[str], expected: list[Path] | None, failures: list[str]) -> None:
    if expected and all(path.exists() for path in expected):
        print(f"[SKIP_EXISTS] {label}", flush=True)
        return
    print(f"\n[RUN_START] {time.strftime('%Y-%m-%d %H:%M:%S')} {label}", flush=True)
    print("[CMD] cd", cwd, "&&", " ".join(command), flush=True)
    start = time.time()
    result = subprocess.run(command, cwd=cwd, check=False)
    elapsed = time.time() - start
    print(f"[RUN_END] {time.strftime('%Y-%m-%d %H:%M:%S')} {label} status={result.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    missing = [str(path.relative_to(ROOT)) for path in expected or [] if not path.exists()]
    if result.returncode != 0 or missing:
        if missing:
            print(f"[MISSING] {label}: {missing}", flush=True)
        failures.append(f"{label} status={result.returncode} missing={missing}")


def clean_outputs() -> None:
    for path in [
        ROOT / "encoder_profiler" / "latency_results",
        ROOT / "decoder_profiler" / "latency_results",
        ROOT / "encoder_decoder_profiler" / "latency_results",
        ROOT / "model_family_profiler" / "latency_results",
        ROOT / "figures",
    ]:
        if path.exists():
            shutil.rmtree(path)
            print(f"[CLEAN] removed {path.relative_to(ROOT)}", flush=True)


def count_csvs() -> int:
    roots = [
        ROOT / "encoder_profiler" / "latency_results",
        ROOT / "decoder_profiler" / "latency_results",
        ROOT / "encoder_decoder_profiler" / "latency_results",
        ROOT / "model_family_profiler" / "latency_results",
    ]
    total = 0
    for root in roots:
        count = len(list(root.rglob("*.csv"))) if root.exists() else 0
        total += count
        print(f"[CSV_COUNT] {root.relative_to(ROOT)} {count}", flush=True)
    print(f"[CSV_COUNT] total {total} expected {EXPECTED_TOTAL}", flush=True)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    if args.clean:
        clean_outputs()

    failures: list[str] = []
    py = sys.executable

    print(f"[JETSON_RUN_START] {time.strftime('%Y-%m-%d %H:%M:%S')} expected_csv={EXPECTED_TOTAL}", flush=True)
    print(f"[PYTHON] {py}", flush=True)
    subprocess.run([py, "-c", "import torch; print('[TORCH]', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')"], cwd=ROOT, check=False)

    for shape in ENCODER_SHAPES:
        for seq_len in SEQ_LENS:
            cwd = ROOT / "encoder_profiler"
            run(f"encoder {shape} L={seq_len}", cwd, [py, "run_and_plot.py", "--shape-names", shape, "--seq-lens", str(seq_len), *COMMON], [csv_path(cwd, "encoder", shape, seq_len)], failures)
            count_csvs()

    for shape in DECODER_SHAPES:
        for seq_len in SEQ_LENS:
            cwd = ROOT / "decoder_profiler"
            run(f"decoder {shape} L={seq_len}", cwd, [py, "run_and_plot.py", "--shape-names", shape, "--seq-lens", str(seq_len), *COMMON], [csv_path(cwd, "decoder", shape, seq_len)], failures)
            count_csvs()

    for shape in ENCODER_DECODER_SHAPES:
        for seq_len in SEQ_LENS:
            cwd = ROOT / "encoder_decoder_profiler"
            run(f"encoder_decoder {shape} L={seq_len}", cwd, [py, "run_and_plot.py", "--shape-names", shape, "--seq-lens", str(seq_len), *COMMON], [csv_path(cwd, "encoder_decoder", shape, seq_len)], failures)
            count_csvs()

    for family, shape in FAMILY_TO_SHAPE.items():
        for seq_len in SEQ_LENS:
            cwd = ROOT / "model_family_profiler"
            run(f"family {family} L={seq_len}", cwd, [py, "run_and_plot.py", "--families", family, "--seq-lens", str(seq_len), *COMMON], [csv_path(cwd, family, shape, seq_len)], failures)
            count_csvs()

    for shape in ATTN_SHAPES:
        for seq_len in SEQ_LENS:
            cwd = ROOT / "model_family_profiler"
            run(f"attention_microbench {shape} L={seq_len}", cwd, [py, "simple_attn_bench.py", "--kind", "both", "--shape-names", shape, "--seq-lens", str(seq_len), *COMMON], attn_paths(shape, seq_len), failures)
            count_csvs()

    if not args.skip_plots:
        for label, cwd, command in [
            ("plot encoder", ROOT / "encoder_profiler", [py, "plot_from_csv.py"]),
            ("plot decoder", ROOT / "decoder_profiler", [py, "plot_from_csv.py"]),
            ("plot encoder_decoder", ROOT / "encoder_decoder_profiler", [py, "plot_from_csv.py"]),
            ("heatmap", ROOT / "model_family_profiler", [py, "heatmap.py"]),
            ("summary figures", ROOT, [py, "figures.py"]),
        ]:
            run(label, cwd, command, None, failures)

    total = count_csvs()
    print(f"[JETSON_RUN_END] {time.strftime('%Y-%m-%d %H:%M:%S')} failures={len(failures)} csv_total={total} expected_csv={EXPECTED_TOTAL}", flush=True)
    if failures:
        print("[FAILURES]", flush=True)
        for item in failures:
            print("-", item, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
