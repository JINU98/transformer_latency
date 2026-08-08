"""Regenerate the prefill/decode-share and component-share figures from the
CSVs already written by run_and_plot.py, without re-running any benchmarks.

Usage:
    cd gen_ratio_profiler
    python plot_from_csv.py --architecture decoder --shape-name llama_7b
    python plot_from_csv.py --architecture encoder_decoder --shape-name t5_base
    python plot_from_csv.py --architecture encoder_decoder --shape-name t5_base \\
        --results-dir latency_results --figures-dir ../figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_ratio_profiler.bench import ARCHITECTURES
from gen_ratio_profiler.io_utils import read_csv
from gen_ratio_profiler import gen_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=list(ARCHITECTURES), default="decoder")
    parser.add_argument("--shape-name", required=True, help="Shape name used in the earlier run_and_plot.py call")
    parser.add_argument("--results-dir", default="latency_results")
    parser.add_argument("--figures-dir", default="../figures")
    args = parser.parse_args()

    shape_dir = Path(args.results_dir) / args.architecture / args.shape_name
    scenario_csv = shape_dir / "scenario_components.csv"
    summary_csv = shape_dir / "scenario_summary.csv"
    if not scenario_csv.exists() or not summary_csv.exists():
        raise SystemExit(
            f"Missing {scenario_csv} or {summary_csv}. Run "
            f"run_and_plot.py --architecture {args.architecture} --shape-name {args.shape_name} first."
        )

    scenario_rows = read_csv(scenario_csv)
    summary_rows = read_csv(summary_csv)
    out_dir = Path(args.figures_dir) / "gen_ratio" / args.architecture
    gen_plots.make_all_plots(scenario_rows, summary_rows, args.shape_name, args.architecture, out_dir)


if __name__ == "__main__":
    main()
