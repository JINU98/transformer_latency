from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.io import operation_group
from common.plotting import require_plotting


def collect_csvs(paths: list[str]) -> list[Path]:
    csvs: list[Path] = []
    for path in paths:
        root = Path(path)
        if root.is_file() and root.suffix == ".csv":
            csvs.append(root)
        elif root.exists():
            csvs.extend(sorted(root.rglob("*.csv")))
    return sorted(set(csvs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce summary figures from profiler CSV outputs.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "results",
            "encoder_profiler/latency_results",
            "decoder_profiler/latency_results",
            "encoder_decoder_profiler/latency_results",
            "model_family_profiler/latency_results",
        ],
    )
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    pd, plt = require_plotting()
    csv_paths = collect_csvs(args.inputs)
    if not csv_paths:
        print("No CSV inputs found. Run one of the profiler scripts first.")
        return

    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df["group"] = df["operation_key"].map(operation_group)

    totals = df.groupby(["model_family", "shape_name", "seq_len"], as_index=False)["total_ms"].sum()
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, sub in totals.groupby(["model_family", "shape_name"]):
        sub = sub.sort_values("seq_len")
        ax.plot(sub["seq_len"], sub["total_ms"], marker="o", label=" / ".join(label))
    ax.set_xlabel("Context length L")
    ax.set_ylabel("Total timed latency (ms)")
    ax.set_title("Runtime scaling across context lengths")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_scaling.png", dpi=180)
    plt.close(fig)

    group_share = df.groupby(["seq_len", "group"], as_index=False)["total_ms"].sum()
    pivot = group_share.pivot_table(index="seq_len", columns="group", values="total_ms", aggfunc="sum").fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Total timed latency (ms)")
    ax.set_title("Latency contribution by profiled component")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_grouped_bar.png", dpi=180)
    plt.close(fig)

    heat = totals.pivot_table(index=["model_family", "shape_name"], columns="seq_len", values="total_ms", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(heat.index))))
    image = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), [str(col) for col in heat.columns])
    ax.set_yticks(range(len(heat.index)), [" / ".join(idx) for idx in heat.index])
    ax.set_xlabel("Context length L")
    ax.set_title("Latency heatmap")
    fig.colorbar(image, ax=ax, label="Total latency (ms)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=180)
    plt.close(fig)

    cross = df[df["group"].isin(["self_attention", "cross_attention"])]
    if not cross.empty:
        share = cross.groupby(["model_family", "group"], as_index=False)["total_ms"].sum()
        share["pct"] = share.groupby("model_family")["total_ms"].transform(lambda values: 100 * values / values.sum())
        pivot = share.pivot(index="model_family", columns="group", values="pct").fillna(0.0)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("% of attention latency")
        ax.set_title("Self-attention vs cross-attention share")
        fig.tight_layout()
        fig.savefig(out_dir / "cross_attn_overhead.png", dpi=180)
        plt.close(fig)

    print(f"Wrote summary figures to {out_dir}")


if __name__ == "__main__":
    main()
