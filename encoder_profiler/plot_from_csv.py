from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import plot_comparisons, plot_single_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate encoder profiler plots from CSV files.")
    parser.add_argument("--input-dir", default="latency_results/encoder")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    csv_paths = sorted(input_dir.glob("*.csv"))
    for csv_path in csv_paths:
        plot_single_csv(csv_path, output_dir)
    plot_comparisons(csv_paths, output_dir, "comparison")


if __name__ == "__main__":
    main()
