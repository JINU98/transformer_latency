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


def save_by_architecture_figures(df, out_dir: Path, pd, plt) -> None:
    base_dir = out_dir / "by_architecture"
    architecture_families = {
        "encoder": "encoder",
        "decoder": "decoder",
        "encoder_decoder": "encoder_decoder",
    }

    numeric_cols = ["d_model", "num_heads", "num_layers", "d_ff", "seq_len", "total_ms"]
    profile_df = df[df["architecture"].isin(architecture_families)].copy()
    for col in numeric_cols:
        profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce")

    for architecture, profiler_family in architecture_families.items():
        arch_df = profile_df[
            (profile_df["architecture"] == architecture)
            & (profile_df["model_family"] == profiler_family)
        ].copy()
        if arch_df.empty:
            continue

        arch_dir = base_dir / architecture
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_df["group"] = arch_df["operation_key"].map(operation_group)

        totals = arch_df.groupby(
            ["shape_name", "d_model", "num_heads", "num_layers", "d_ff", "seq_len"],
            as_index=False,
        )["total_ms"].sum()
        totals["shape_label"] = (
            totals["shape_name"]
            + " (d="
            + totals["d_model"].astype(int).astype(str)
            + ", h="
            + totals["num_heads"].astype(int).astype(str)
            + ", layers="
            + totals["num_layers"].astype(int).astype(str)
            + ")"
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        for label, sub in totals.groupby("shape_label"):
            sub = sub.sort_values("seq_len")
            ax.plot(sub["seq_len"], sub["total_ms"], marker="o", label=label)
        ax.set_xlabel("Context length L")
        ax.set_ylabel("Total timed latency (ms)")
        ax.set_title(f"{architecture}: latency vs L")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(arch_dir / "latency_vs_L_by_shape.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        for seq_len, sub in totals.groupby("seq_len"):
            sub = sub.sort_values(["d_model", "shape_name"])
            ax.plot(sub["d_model"], sub["total_ms"], marker="o", label=f"L={int(seq_len)}")
        ax.set_xlabel("Hidden dimension d")
        ax.set_ylabel("Total timed latency (ms)")
        ax.set_title(f"{architecture}: latency vs d")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(arch_dir / "latency_vs_d_by_L.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        for seq_len, sub in totals.groupby("seq_len"):
            sub = sub.sort_values(["num_heads", "shape_name"])
            ax.plot(sub["num_heads"], sub["total_ms"], marker="o", label=f"L={int(seq_len)}")
        ax.set_xlabel("Number of attention heads h")
        ax.set_ylabel("Total timed latency (ms)")
        ax.set_title(f"{architecture}: latency vs h")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(arch_dir / "latency_vs_h_by_L.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        for seq_len, sub in totals.groupby("seq_len"):
            sub = sub.sort_values(["num_layers", "shape_name"])
            ax.plot(sub["num_layers"], sub["total_ms"], marker="o", label=f"L={int(seq_len)}")
        ax.set_xlabel("Number of layers")
        ax.set_ylabel("Total timed latency (ms)")
        ax.set_title(f"{architecture}: latency vs layer count")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(arch_dir / "latency_vs_layers_by_L.png", dpi=180)
        plt.close(fig)

        heat = totals.pivot_table(
            index="shape_label",
            columns="seq_len",
            values="total_ms",
            aggfunc="mean",
        ).fillna(0.0)
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.6 * len(heat.index))))
        image = ax.imshow(heat.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(heat.columns)), [str(int(col)) for col in heat.columns])
        ax.set_yticks(range(len(heat.index)), heat.index)
        ax.set_xlabel("Context length L")
        ax.set_title(f"{architecture}: latency heatmap by shape and L")
        fig.colorbar(image, ax=ax, label="Total timed latency (ms)")
        fig.tight_layout()
        fig.savefig(arch_dir / "shape_L_latency_heatmap.png", dpi=180)
        plt.close(fig)

        group_totals = arch_df.groupby(["seq_len", "group"], as_index=False)["total_ms"].sum()
        group_totals["pct"] = group_totals.groupby("seq_len")["total_ms"].transform(
            lambda values: 100 * values / values.sum()
        )
        pivot = group_totals.pivot(index="seq_len", columns="group", values="pct").fillna(0.0)
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Context length L")
        ax.set_ylabel("% of timed latency")
        ax.set_title(f"{architecture}: component share by L")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(arch_dir / "component_share_by_L.png", dpi=180)
        plt.close(fig)

        max_l = int(totals["seq_len"].max())
        max_l_totals = totals[totals["seq_len"] == max_l].copy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = 120 + 420 * max_l_totals["total_ms"] / max_l_totals["total_ms"].max()
        scatter = ax.scatter(
            max_l_totals["d_model"],
            max_l_totals["num_heads"],
            s=sizes,
            c=max_l_totals["total_ms"],
            cmap="viridis",
            alpha=0.85,
        )
        for _, row in max_l_totals.iterrows():
            ax.annotate(row["shape_name"], (row["d_model"], row["num_heads"]), fontsize=8)
        ax.set_xlabel("Hidden dimension d")
        ax.set_ylabel("Number of attention heads h")
        ax.set_title(f"{architecture}: d/h grid at L={max_l}")
        fig.colorbar(scatter, ax=ax, label="Total timed latency (ms)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(arch_dir / "d_h_latency_grid_at_max_L.png", dpi=180)
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

    save_by_architecture_figures(df, out_dir, pd, plt)
    print(f"Wrote summary figures to {out_dir}")


if __name__ == "__main__":
    main()
