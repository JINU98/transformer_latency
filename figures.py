from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.plotting import require_plotting


PROFILE_ARCHITECTURES = ("encoder", "decoder", "encoder_decoder")

ARCHITECTURE_TITLES = {
    "encoder": "Encoder-only",
    "decoder": "Decoder-only",
    "encoder_decoder": "Encoder-decoder",
}

SHAPE_TITLES = {
    "tiny_debug": "Tiny",
    "bert_base": "BERT-base",
    "bert_large": "BERT-large",
    "gpt2_medium": "GPT-2 medium",
    "gpt3_2p7b": "GPT-3 2.7B",
    "llama_7b": "LLaMA-7B",
    "t5_base": "T5-base",
    "t5_large": "T5-large",
    "bart_large": "BART-large",
}

COMPONENT_COLORS = {
    "Self Attention Weighted Sum": "#1f77b4",
    "Cross Attention Weighted Sum": "#0f4c81",
    "Self Causal Mask": "#aec7e8",
    "Self Padding Mask": "#6baed6",
    "FFN Linear 1": "#ff7f0e",
    "FFN Linear 2": "#ffbb78",
    "FFN Activation": "#2ca02c",
    "Self KV Cache Concat": "#98df8a",
    "Self GQA KV Expand": "#17becf",
    "LayerNorm (Pre-Attention)": "#d62728",
    "LayerNorm (Pre-FFN)": "#ff9896",
    "LayerNorm (Pre-Cross-Attention)": "#f7b6d2",
    "Output Head Projection": "#9467bd",
    "Self Output Projection": "#c5b0d5",
    "Cross Output Projection": "#6a51a3",
    "Self QKV Projection": "#8c564b",
    "Cross QKV Projection": "#a56752",
    "Self QKT MatMul": "#c49c94",
    "Cross QKT MatMul": "#7f5f5b",
    "Self Softmax": "#e377c2",
    "Cross Softmax": "#b84ea0",
    "Final Norm": "#7f7f7f",
    "Unmapped": "#c7c7c7",
}

STACK_COLORS = COMPONENT_COLORS


def collect_csvs(paths: list[str]) -> list[Path]:
    csvs: list[Path] = []
    for path in paths:
        root = Path(path)
        if root.is_file() and root.suffix == ".csv":
            csvs.append(root)
        elif root.exists():
            csvs.extend(sorted(root.rglob("*.csv")))
    return sorted(set(csvs))


def display_shape_name(shape_name: str) -> str:
    return SHAPE_TITLES.get(shape_name, shape_name.replace("_", " "))


def component_for_operation(operation_key: str) -> str:
    suffix = operation_key.split(".", 1)[-1]
    prefix = "Cross" if operation_key.startswith("cross_attn.") else "Self"
    if suffix in {"q_proj", "k_proj", "v_proj"}:
        return f"{prefix} QKV Projection"
    if suffix == "matmul_qk":
        return f"{prefix} QKT MatMul"
    if suffix == "softmax":
        return f"{prefix} Softmax"
    if suffix == "weighted_sum":
        return f"{prefix} Attention Weighted Sum"
    if suffix == "out_projection":
        return f"{prefix} Output Projection"
    if suffix == "apply_causal_mask":
        return "Self Causal Mask"
    if suffix == "apply_padding_mask":
        return "Self Padding Mask"
    if suffix == "cache_concat":
        return "Self KV Cache Concat"
    if suffix == "gqa_expand":
        return "Self GQA KV Expand"
    if operation_key == "ff.linear1" or operation_key in {"ff.w1_gate", "ff.w2_up"}:
        return "FFN Linear 1"
    if operation_key == "ff.linear2" or operation_key == "ff.w3_down":
        return "FFN Linear 2"
    if operation_key in {"ff.gelu", "ff.swiglu_act"}:
        return "FFN Activation"
    if operation_key == "block.norm1":
        return "LayerNorm (Pre-Attention)"
    if operation_key == "block.norm2":
        return "LayerNorm (Pre-FFN)"
    if operation_key == "block.norm3":
        return "LayerNorm (Pre-Cross-Attention)"
    if operation_key == "model.output_head":
        return "Output Head Projection"
    if operation_key == "model.final_norm":
        return "Final Norm"
    return "Unmapped"


def stack_group_for_operation(operation_key: str) -> str:
    return component_for_operation(operation_key)


def normalize_profile_df(df, pd):
    profile_df = df[df["architecture"].isin(PROFILE_ARCHITECTURES)].copy()
    if "phase" not in profile_df.columns:
        profile_df["phase"] = "prefill"
    profile_df["phase"] = profile_df["phase"].fillna("prefill")
    numeric_cols = [
        "d_model",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "num_layers",
        "d_ff",
        "batch_size",
        "seq_len",
        "phase_tokens",
        "total_ms",
        "avg_total_ms_per_repeat",
        "avg_total_ms_per_token",
    ]
    for col in numeric_cols:
        if col in profile_df.columns:
            profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce")
    if "phase_tokens" not in profile_df.columns:
        profile_df["phase_tokens"] = profile_df["seq_len"]
    if "avg_total_ms_per_repeat" not in profile_df.columns:
        profile_df["avg_total_ms_per_repeat"] = profile_df["total_ms"]
    if "avg_total_ms_per_token" not in profile_df.columns:
        profile_df["avg_total_ms_per_token"] = profile_df["total_ms"]
    profile_df = profile_df.dropna(subset=["d_model", "num_heads", "num_layers", "seq_len", "total_ms"])
    profile_df["phase_latency_ms"] = profile_df["avg_total_ms_per_repeat"]
    decode_mask = profile_df["phase"].astype(str).eq("decode")
    profile_df.loc[decode_mask, "phase_latency_ms"] = profile_df.loc[decode_mask, "avg_total_ms_per_token"]
    for col in ["d_model", "num_heads", "num_layers", "seq_len"]:
        profile_df[col] = profile_df[col].astype(int)
    profile_df["component"] = profile_df["operation_key"].map(component_for_operation)
    profile_df["stack_group"] = profile_df["operation_key"].map(stack_group_for_operation)
    return profile_df


def architecture_df(profile_df, architecture: str):
    arch_df = profile_df[profile_df["architecture"] == architecture].copy()
    direct_df = arch_df[arch_df["model_family"] == architecture].copy()
    return direct_df if not direct_df.empty else arch_df


def ordered_present(values, order: list[str]) -> list[str]:
    present = set(values)
    return [value for value in order if value in present]


def save_component_legend(profile_df, out_dir: Path, plt) -> None:
    components = ordered_present(list(profile_df["component"].unique()), list(COMPONENT_COLORS))
    if not components:
        return

    height = max(5.0, 0.55 * len(components) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, height))
    ax.axis("off")
    ax.set_title("Components", fontsize=24, pad=12)

    y_start = len(components) - 1
    for idx, component in enumerate(components):
        y = y_start - idx
        ax.scatter(
            0.06,
            y,
            marker="s",
            s=240,
            color=COMPONENT_COLORS[component],
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.text(
            0.14,
            y,
            component,
            va="center",
            ha="left",
            fontsize=20,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_ylim(-0.8, len(components) - 0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "component_legend.png", dpi=220)
    plt.close(fig)


def phase_title(phase: str) -> str:
    if phase == "decode":
        return "Decode avg/token"
    if phase == "prefill":
        return "Prefill"
    return phase.replace("_", " ").title()


def save_stacked_share_chart(arch_df, architecture: str, phase: str, out_dir: Path, plt) -> None:
    import matplotlib.patheffects as path_effects

    if arch_df.empty:
        return

    group_cols = ["seq_len", "shape_name", "d_model", "num_heads", "num_layers", "stack_group"]
    grouped = arch_df.groupby(group_cols, as_index=False)["phase_latency_ms"].sum()
    total_cols = ["seq_len", "shape_name", "d_model", "num_heads", "num_layers"]
    grouped["pct"] = grouped.groupby(total_cols)["phase_latency_ms"].transform(lambda values: 100 * values / values.sum())

    configs = (
        grouped[total_cols]
        .drop_duplicates()
        .sort_values(["seq_len", "d_model", "num_heads", "num_layers", "shape_name"])
        .to_dict("records")
    )
    seq_lens = sorted(grouped["seq_len"].unique())
    if not configs:
        return

    bars_per_l = {seq_len: [cfg for cfg in configs if cfg["seq_len"] == seq_len] for seq_len in seq_lens}
    gap = 1.8
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_spans: list[tuple[float, float, float, int]] = []
    cursor = 0.0

    for seq_len in seq_lens:
        current = bars_per_l[seq_len]
        start = cursor
        for cfg in current:
            x_positions.append(cursor)
            x_labels.append(display_shape_name(str(cfg["shape_name"])))
            cursor += 1.0
        end = cursor - 1.0
        center = (start + end) / 2 if current else start
        group_spans.append((start - 0.5, end + 0.5, center, int(seq_len)))
        cursor += gap

    width = max(13.5, 0.72 * len(x_positions) + 3.8)
    fig, ax = plt.subplots(figsize=(width, 7.4))

    for start, end, _, _ in group_spans:
        ax.axvspan(start, end, color="#f7f7f7", zorder=0)
    for start, _, _, _ in group_spans[1:]:
        ax.axvline(start - gap / 2, color="#bdbdbd", linestyle="--", linewidth=1.0, zorder=1)

    lookup = {
        (
            int(row["seq_len"]),
            str(row["shape_name"]),
            int(row["d_model"]),
            int(row["num_heads"]),
            int(row["num_layers"]),
            str(row["stack_group"]),
        ): float(row["pct"])
        for row in grouped.to_dict("records")
    }
    bottoms = [0.0 for _ in x_positions]
    stack_order = ordered_present(list(grouped["stack_group"].unique()), list(STACK_COLORS))

    for stack_group in stack_order:
        heights: list[float] = []
        for cfg in configs:
            key = (
                int(cfg["seq_len"]),
                str(cfg["shape_name"]),
                int(cfg["d_model"]),
                int(cfg["num_heads"]),
                int(cfg["num_layers"]),
                stack_group,
            )
            heights.append(lookup.get(key, 0.0))
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.78,
            color=STACK_COLORS[stack_group],
            edgecolor="white",
            linewidth=0.8,
            label=stack_group,
            zorder=2,
        )
        for idx, height in enumerate(heights):
            if height < 5.0:
                continue
            text = ax.text(
                x_positions[idx],
                bottoms[idx] + height / 2,
                f"{height:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="#1f1f1f",
                zorder=3,
            )
            text.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    for _, _, center, seq_len in group_spans:
        ax.text(
            center,
            -0.19,
            f"L = {seq_len}",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=13,
            weight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Latency share (%)", fontsize=15)
    ax.set_title(
        f"{ARCHITECTURE_TITLES[architecture]} {phase_title(phase)} component share by model shape and L",
        fontsize=18,
    )
    ax.set_xticks(x_positions, x_labels, rotation=38, ha="right")
    ax.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=12)
    fig.subplots_adjust(bottom=0.22, right=0.84)
    fig.savefig(out_dir / "model_family_component_share.png", dpi=220)
    plt.close(fig)


def autopct_label(min_pct: float):
    def format_pct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= min_pct else ""

    return format_pct


def pie_group_columns(index_mode: str) -> list[str]:
    if index_mode == "d_l":
        return ["d_model", "seq_len"]
    if index_mode == "d_h_l":
        return ["d_model", "num_heads", "seq_len"]
    return ["d_model", "num_heads", "num_layers", "seq_len"]


def pie_filename(key: dict[str, int], index_mode: str) -> str:
    if index_mode == "d_l":
        return f"pie_d{key['d_model']}_l{key['seq_len']}.png"
    if index_mode == "d_h_l":
        return f"pie_d{key['d_model']}_h{key['num_heads']}_l{key['seq_len']}.png"
    return f"pie_d{key['d_model']}_h{key['num_heads']}_layers{key['num_layers']}_l{key['seq_len']}.png"


def pie_title(architecture: str, key: dict[str, int], config_df, index_mode: str) -> str:
    shape_names = sorted(display_shape_name(str(value)) for value in config_df["shape_name"].unique())
    heads = sorted(int(value) for value in config_df["num_heads"].unique())
    layers = sorted(int(value) for value in config_df["num_layers"].unique())
    pieces = [
        ARCHITECTURE_TITLES[architecture],
        phase_title(str(config_df["phase"].iloc[0])),
        f"d={key['d_model']}",
        f"L={key['seq_len']}",
    ]
    if index_mode != "d_l" or len(heads) == 1:
        pieces.insert(2, "h=" + ",".join(str(value) for value in heads))
    if index_mode == "d_h_layers_l" or len(layers) == 1:
        pieces.append("layers=" + ",".join(str(value) for value in layers))
    title = " | ".join(pieces)
    if len(shape_names) <= 3:
        title += "\n" + ", ".join(shape_names)
    return title


def save_pie_charts(arch_df, architecture: str, out_dir: Path, plt, index_mode: str) -> int:
    if arch_df.empty:
        return 0

    pie_dir = out_dir / "pie_charts"
    pie_dir.mkdir(parents=True, exist_ok=True)
    group_cols = pie_group_columns(index_mode)
    count = 0

    for raw_key, config_df in arch_df.groupby(group_cols):
        if not isinstance(raw_key, tuple):
            raw_key = (raw_key,)
        key = {col: int(value) for col, value in zip(group_cols, raw_key)}
        component_totals = (
            config_df.groupby("component", as_index=False)["phase_latency_ms"]
            .sum()
            .sort_values("phase_latency_ms", ascending=False)
        )
        component_totals = component_totals[component_totals["phase_latency_ms"] > 0]
        if component_totals.empty:
            continue

        order = ordered_present(list(component_totals["component"]), list(COMPONENT_COLORS))
        component_totals = component_totals.set_index("component").loc[order].reset_index()
        colors = [COMPONENT_COLORS[component] for component in component_totals["component"]]

        fig, ax = plt.subplots(figsize=(9.2, 9.2))
        wedges, _, autotexts = ax.pie(
            component_totals["phase_latency_ms"],
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=autopct_label(1.5),
            pctdistance=0.72,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 16, "color": "black"},
        )
        for autotext in autotexts:
            autotext.set_weight("bold")
        ax.set_title(pie_title(architecture, key, config_df, index_mode), fontsize=17, pad=18)
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(pie_dir / pie_filename(key, index_mode), dpi=220)
        plt.close(fig)
        count += 1

    return count


def save_figures(profile_df, out_dir: Path, plt, index_mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_component_legend(profile_df, out_dir, plt)

    for architecture in PROFILE_ARCHITECTURES:
        arch_df = architecture_df(profile_df, architecture)
        if arch_df.empty:
            continue
        for phase, phase_df in arch_df.groupby("phase"):
            phase_dir = out_dir / architecture / str(phase)
            phase_dir.mkdir(parents=True, exist_ok=True)
            save_stacked_share_chart(phase_df, architecture, str(phase), phase_dir, plt)
            pie_count = save_pie_charts(phase_df, architecture, phase_dir, plt, index_mode)

            dh_pairs = phase_df[["d_model", "num_heads"]].drop_duplicates().shape[0]
            d_values = phase_df["d_model"].nunique()
            l_values = phase_df["seq_len"].nunique()
            print(
                f"{architecture}/{phase}: wrote {pie_count} pie chart(s) "
                f"({dh_pairs} d/h setting(s), {d_values} d value(s), {l_values} L value(s))."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create latency figures from profiler CSV outputs."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "encoder_profiler/latency_results",
            "decoder_profiler/latency_results",
            "encoder_decoder_profiler/latency_results",
        ],
        help="CSV files or directories containing profiler CSVs.",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--pie-index",
        choices=["d_l", "d_h_l", "d_h_layers_l"],
        default="d_h_l",
        help=(
            "Controls how pie charts are split. The default includes h in the file name; "
            "with the built-in real-shape presets, h is tied to d, so this is d x L."
        ),
    )
    args = parser.parse_args()

    pd, plt = require_plotting()
    csv_paths = collect_csvs(args.inputs)
    if not csv_paths:
        print("No CSV inputs found. Run one of the profiler scripts first.")
        return

    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    profile_df = normalize_profile_df(df, pd)
    if profile_df.empty:
        print("No encoder, decoder, or encoder-decoder profiler rows found.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figures(profile_df, out_dir, plt, args.pie_index)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
