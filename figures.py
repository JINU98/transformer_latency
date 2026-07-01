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


def save_extended_figures(df, out_dir: Path, pd, plt) -> None:
    extended_dir = out_dir / "extended"
    extended_dir.mkdir(parents=True, exist_ok=True)

    profile_df = df[df["architecture"] != "attention_microbench"].copy()
    attn_df = df[df["architecture"] == "attention_microbench"].copy()

    if not profile_df.empty:
        totals = profile_df.groupby(
            ["architecture", "model_family", "shape_name", "seq_len"],
            as_index=False,
        )["total_ms"].sum()
        totals["label"] = totals["model_family"] + " / " + totals["shape_name"]

        fig, ax = plt.subplots(figsize=(12, 7))
        for label, sub in totals.groupby("label"):
            sub = sub.sort_values("seq_len")
            ax.plot(sub["seq_len"], sub["total_ms"], marker="o", label=label)
        ax.set_xlabel("Context length L")
        ax.set_ylabel("Total timed latency (ms)")
        ax.set_title("Per-model latency scaling")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(extended_dir / "per_model_latency_scaling_log.png", dpi=180)
        plt.close(fig)

        max_l = int(totals["seq_len"].max())
        max_l_totals = totals[totals["seq_len"] == max_l].sort_values("total_ms", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(max_l_totals["label"], max_l_totals["total_ms"])
        ax.set_xlabel("Total timed latency (ms)")
        ax.set_title(f"Total latency at L={max_l}")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(extended_dir / "total_latency_at_max_context.png", dpi=180)
        plt.close(fig)

        profile_df["group"] = profile_df["operation_key"].map(operation_group)
        group_totals = profile_df.groupby(["architecture", "group"], as_index=False)["total_ms"].sum()
        group_totals["pct"] = group_totals.groupby("architecture")["total_ms"].transform(
            lambda values: 100 * values / values.sum()
        )
        pivot = group_totals.pivot(index="architecture", columns="group", values="pct").fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("% of timed latency")
        ax.set_title("Component share by architecture")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(extended_dir / "component_share_by_architecture.png", dpi=180)
        plt.close(fig)

        op_totals = (
            profile_df.groupby("operation_key", as_index=False)["total_ms"]
            .sum()
            .sort_values("total_ms", ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(op_totals["operation_key"], op_totals["total_ms"])
        ax.set_xlabel("Total timed latency (ms)")
        ax.set_title("Top operation scopes across full run")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(extended_dir / "top_operation_scopes.png", dpi=180)
        plt.close(fig)

        layer_df = totals[totals["seq_len"] == max_l].copy()
        layer_df["num_layers"] = pd.to_numeric(layer_df["num_layers"], errors="coerce")
        fig, ax = plt.subplots(figsize=(9, 6))
        for architecture, sub in layer_df.groupby("architecture"):
            ax.scatter(sub["num_layers"], sub["total_ms"], label=architecture, s=80)
            for _, row in sub.iterrows():
                ax.annotate(row["shape_name"], (row["num_layers"], row["total_ms"]), fontsize=7)
        ax.set_xlabel("Number of layers")
        ax.set_ylabel(f"Total timed latency at L={max_l} (ms)")
        ax.set_title("Layer count vs latency")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(extended_dir / "layer_count_vs_latency.png", dpi=180)
        plt.close(fig)

    if not attn_df.empty:
        attn_totals = attn_df.groupby(["model_family", "shape_name", "seq_len"], as_index=False)["total_ms"].sum()
        fig, ax = plt.subplots(figsize=(11, 6))
        for (kind, shape), sub in attn_totals.groupby(["model_family", "shape_name"]):
            sub = sub.sort_values("seq_len")
            ax.plot(sub["seq_len"], sub["total_ms"], marker="o", label=f"{shape} / {kind}")
        ax.set_xlabel("Context length L")
        ax.set_ylabel("Total attention microbench latency (ms)")
        ax.set_title("Self-attention and cross-attention microbench scaling")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(extended_dir / "attention_microbench_scaling.png", dpi=180)
        plt.close(fig)

        max_l = int(attn_totals["seq_len"].max())
        attn_max = attn_totals[attn_totals["seq_len"] == max_l].copy()
        attn_max["label"] = attn_max["shape_name"] + " / " + attn_max["model_family"]
        attn_max = attn_max.sort_values("total_ms", ascending=False)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(attn_max["label"], attn_max["total_ms"])
        ax.set_xlabel("Total attention microbench latency (ms)")
        ax.set_title(f"Attention microbench latency at L={max_l}")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(extended_dir / "attention_microbench_at_max_context.png", dpi=180)
        plt.close(fig)


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

    save_extended_figures(df, out_dir, pd, plt)
    print(f"Wrote summary figures to {out_dir}")


if __name__ == "__main__":
    main()
