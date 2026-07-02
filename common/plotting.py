from __future__ import annotations

from pathlib import Path


def phase_metric(phase: str, df) -> tuple[str, str]:
    if phase == "decode" and "avg_total_ms_per_token" in df.columns:
        return "avg_total_ms_per_token", "Average component latency (ms per decoded token, log scale)"
    if "avg_total_ms_per_repeat" in df.columns:
        return "avg_total_ms_per_repeat", "Average component latency (ms per phase run, log scale)"
    return "avg_ms", "Average latency per timed scope (ms, log scale)"


def require_plotting():
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires pandas and matplotlib. Install with: pip install -r requirements.txt"
        ) from exc
    return pd, plt


def plot_single_csv(csv_path: Path, out_dir: Path) -> None:
    pd, plt = require_plotting()
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    label = csv_path.stem
    phase_groups = [("", df)]
    if "phase" in df.columns:
        phase_groups = [(str(phase), sub.copy()) for phase, sub in df.groupby("phase")]

    for phase, phase_df in phase_groups:
        metric_col, xlabel = phase_metric(phase, phase_df)
        bar_df = phase_df.sort_values(metric_col, ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(bar_df["operation_key"], bar_df[metric_col])
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        title = label if not phase else f"{label} ({phase})"
        suffix = "" if not phase else f"_{phase}"
        ax.set_title(title)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / f"bar_{label}{suffix}.png", dpi=180)
        plt.close(fig)


def plot_comparisons(csv_paths: list[Path], out_dir: Path, prefix: str) -> None:
    return
