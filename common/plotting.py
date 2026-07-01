from __future__ import annotations

from pathlib import Path

from common.io import operation_group


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

    bar_df = df.sort_values("total_ms", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(bar_df["operation_key"], bar_df["avg_ms"], xerr=bar_df["std_ms"])
    ax.set_xscale("log")
    ax.set_xlabel("Average latency per timed scope (ms, log scale)")
    ax.set_title(label)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{label}.png", dpi=180)
    plt.close(fig)

    pie_df = df.copy()
    pie_df["group"] = pie_df["operation_key"].map(operation_group)
    grouped = pie_df.groupby("group", as_index=False)["total_ms"].sum()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(grouped["total_ms"], labels=grouped["group"], autopct="%1.1f%%")
    ax.set_title(f"Latency share: {label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"pie_{label}.png", dpi=180)
    plt.close(fig)


def plot_comparisons(csv_paths: list[Path], out_dir: Path, prefix: str) -> None:
    pd, plt = require_plotting()
    frames = [pd.read_csv(path) for path in csv_paths if path.exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = (
        df.groupby(["shape_name", "d_model", "num_heads", "seq_len"], as_index=False)["total_ms"]
        .sum()
        .sort_values(["d_model", "num_heads", "seq_len"])
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for shape, sub in total.groupby("shape_name"):
        ax.plot(sub["seq_len"], sub["total_ms"], marker="o", label=shape)
    ax.set_xlabel("Context length L")
    ax.set_ylabel("Total timed latency (ms)")
    ax.set_title("Total latency vs context length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_avg_latency.png", dpi=180)
    plt.close(fig)

    df["group"] = df["operation_key"].map(operation_group)
    pct = df.groupby(["seq_len", "group"], as_index=False)["total_ms"].sum()
    pct["pct"] = pct.groupby("seq_len")["total_ms"].transform(lambda s: 100 * s / s.sum())
    pivot = pct.pivot(index="seq_len", columns="group", values="pct").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("% of timed latency")
    ax.set_title("Latency share by component group")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_pct_latency.png", dpi=180)
    plt.close(fig)
