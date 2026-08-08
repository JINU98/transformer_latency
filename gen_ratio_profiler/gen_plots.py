from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures import COMPONENT_COLORS, ARCHITECTURE_TITLES, component_for_operation, ordered_present, display_shape_name

PHASE_COLORS = {"Prefill": "#4C78A8", "Decode": "#F58518"}


def _arch_title(architecture: str) -> str:
    return ARCHITECTURE_TITLES.get(architecture, architecture.replace("_", " ").title())


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
    except ImportError as exc:
        raise SystemExit("Plotting requires matplotlib. Install with: pip install -r ../requirements.txt") from exc
    return plt, path_effects


def _ordered_scenarios(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        label = str(row["token_scenario"])
        if label not in seen:
            seen.append(label)
    return seen


def _x_layout(seq_lens: list[int], scenario_labels: list[str], gap: float = 1.8):
    x_positions: list[float] = []
    x_labels: list[tuple[int, str]] = []
    group_spans: list[tuple[float, float, float, int]] = []
    cursor = 0.0
    for seq_len in seq_lens:
        start = cursor
        for label in scenario_labels:
            x_positions.append(cursor)
            x_labels.append((seq_len, label))
            cursor += 1.0
        end = cursor - 1.0
        group_spans.append((start - 0.5, end + 0.5, (start + end) / 2, seq_len))
        cursor += gap
    return x_positions, x_labels, group_spans


def _draw_group_backgrounds(ax, group_spans):
    for start, end, _, _ in group_spans:
        ax.axvspan(start, end, color="#f7f7f7", zorder=0)
    for start, _, _, _ in group_spans[1:]:
        ax.axvline(start - 0.9, color="#bdbdbd", linestyle="--", linewidth=1.0, zorder=1)


def _draw_group_labels(ax, group_spans):
    for _, _, center, seq_len in group_spans:
        ax.text(
            center,
            -0.22,
            f"L = {seq_len}",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=13,
            weight="bold",
        )


def plot_prefill_decode_share(summary_rows: list[dict], shape_name: str, architecture: str, out_dir: Path) -> Path | None:
    """Stacked bar: prefill % vs decode % of total latency, per (L, scenario)."""
    if not summary_rows:
        return None
    plt, path_effects = _require_matplotlib()

    display_by_label = {str(r["token_scenario"]): str(r["token_scenario_display"]) for r in summary_rows}
    scenario_labels = _ordered_scenarios(summary_rows)
    seq_lens = sorted({int(r["seq_len"]) for r in summary_rows})

    lookup = {
        (int(r["seq_len"]), str(r["token_scenario"])): (float(r["prefill_pct"]), float(r["decode_pct"]))
        for r in summary_rows
    }

    x_positions, x_labels, group_spans = _x_layout(seq_lens, scenario_labels)
    width = max(13.5, 0.9 * len(x_positions) + 3.8)
    fig, ax = plt.subplots(figsize=(width, 6.6))
    _draw_group_backgrounds(ax, group_spans)

    for phase_idx, (phase_name, key_idx) in enumerate([("Prefill", 0), ("Decode", 1)]):
        heights = []
        for seq_len, label in x_labels:
            pct = lookup.get((seq_len, label), (0.0, 0.0))[key_idx]
            heights.append(pct)
        bottoms = [0.0] * len(x_positions) if phase_idx == 0 else [
            lookup.get((seq_len, label), (0.0, 0.0))[0] for seq_len, label in x_labels
        ]
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.78,
            color=PHASE_COLORS[phase_name],
            edgecolor="white",
            linewidth=0.8,
            label=phase_name,
            zorder=2,
        )
        for idx, height in enumerate(heights):
            if height < 4.0:
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

    _draw_group_labels(ax, group_spans)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Latency share (%)", fontsize=15)
    ax.set_title(
        f"{_arch_title(architecture)} — {display_shape_name(shape_name)}: prefill vs. decode latency share "
        "by sequence length and tokens generated",
        fontsize=16,
    )
    ax.set_xticks(x_positions, [display_by_label[label] for _, label in x_labels], rotation=38, ha="right")
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=12)
    fig.subplots_adjust(bottom=0.24, right=0.88)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prefill_decode_share_{shape_name}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_component_share(scenario_rows: list[dict], shape_name: str, architecture: str, out_dir: Path) -> Path | None:
    """Full component breakdown (QKV, FFN, KV-cache concat, ...) of combined
    prefill+decode latency, per (L, scenario). Same visual language as
    figures/decoder/*/model_family_component_share.png."""
    if not scenario_rows:
        return None
    plt, path_effects = _require_matplotlib()

    for row in scenario_rows:
        row.setdefault("component", component_for_operation(str(row["operation_key"])))

    display_by_label = {str(r["token_scenario"]): str(r["token_scenario_display"]) for r in scenario_rows}
    scenario_labels = _ordered_scenarios(scenario_rows)
    seq_lens = sorted({int(r["seq_len"]) for r in scenario_rows})

    totals: dict[tuple[int, str], float] = {}
    per_component: dict[tuple[int, str, str], float] = {}
    for row in scenario_rows:
        key2 = (int(row["seq_len"]), str(row["token_scenario"]))
        component = str(row.get("component") or component_for_operation(str(row["operation_key"])))
        value = float(row["scaled_total_ms"])
        totals[key2] = totals.get(key2, 0.0) + value
        key3 = (*key2, component)
        per_component[key3] = per_component.get(key3, 0.0) + value

    components_present = ordered_present({k[2] for k in per_component}, list(COMPONENT_COLORS))

    x_positions, x_labels, group_spans = _x_layout(seq_lens, scenario_labels)
    width = max(13.5, 0.9 * len(x_positions) + 4.2)
    fig, ax = plt.subplots(figsize=(width, 7.4))
    _draw_group_backgrounds(ax, group_spans)

    bottoms = [0.0] * len(x_positions)
    for component in components_present:
        heights = []
        for seq_len, label in x_labels:
            total = totals.get((seq_len, label), 0.0)
            value = per_component.get((seq_len, label, component), 0.0)
            heights.append(100.0 * value / total if total else 0.0)
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.78,
            color=COMPONENT_COLORS[component],
            edgecolor="white",
            linewidth=0.8,
            label=component,
            zorder=2,
        )
        for idx, height in enumerate(heights):
            if height >= 5.0:
                text = ax.text(
                    x_positions[idx],
                    bottoms[idx] + height / 2,
                    f"{height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    color="#1f1f1f",
                    zorder=3,
                )
                text.set_path_effects([path_effects.withStroke(linewidth=2.0, foreground="white")])
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    _draw_group_labels(ax, group_spans)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Latency share of combined prefill+decode total (%)", fontsize=13)
    ax.set_title(
        f"{_arch_title(architecture)} — {display_shape_name(shape_name)}: component share of total generation "
        "latency by L and tokens generated",
        fontsize=15,
    )
    ax.set_xticks(x_positions, [display_by_label[label] for _, label in x_labels], rotation=38, ha="right")
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=11)
    fig.subplots_adjust(bottom=0.24, right=0.82)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"component_share_{shape_name}.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_pie_charts(scenario_rows: list[dict], shape_name: str, architecture: str, out_dir: Path) -> int:
    """One pie chart per (L, scenario): component share of the combined
    prefill+decode total, same style as figures/*/pie_charts/pie_*.png."""
    if not scenario_rows:
        return 0
    plt, _ = _require_matplotlib()

    for row in scenario_rows:
        row.setdefault("component", component_for_operation(str(row["operation_key"])))

    pie_dir = out_dir / "pie_charts"
    pie_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[int, str], dict[str, float]] = {}
    display_by_label = {}
    for row in scenario_rows:
        key = (int(row["seq_len"]), str(row["token_scenario"]))
        display_by_label[str(row["token_scenario"])] = str(row["token_scenario_display"])
        component = str(row["component"])
        groups.setdefault(key, {})
        groups[key][component] = groups[key].get(component, 0.0) + float(row["scaled_total_ms"])

    count = 0
    for (seq_len, label), component_totals in sorted(groups.items()):
        items = sorted(component_totals.items(), key=lambda kv: kv[1], reverse=True)
        items = [(name, value) for name, value in items if value > 0]
        if not items:
            continue
        order = ordered_present([name for name, _ in items], list(COMPONENT_COLORS))
        by_name = dict(items)
        names = [name for name in order if name in by_name]
        values = [by_name[name] for name in names]
        colors = [COMPONENT_COLORS[name] for name in names]

        fig, ax = plt.subplots(figsize=(8.6, 8.6))

        def format_pct(pct):
            return f"{pct:.1f}%" if pct >= 1.5 else ""

        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            autopct=format_pct,
            pctdistance=0.72,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 15, "color": "black"},
        )
        for autotext in autotexts:
            autotext.set_weight("bold")
        ax.set_title(
            f"{_arch_title(architecture)} — {display_shape_name(shape_name)} | L={seq_len} | "
            f"{display_by_label[label]} generated\nComponent share of total prefill+decode latency",
            fontsize=14,
            pad=16,
        )
        ax.axis("equal")
        fig.tight_layout()
        fig.savefig(pie_dir / f"pie_{shape_name}_l{seq_len}_{label}.png", dpi=210)
        plt.close(fig)
        count += 1
    return count


def make_all_plots(scenario_rows: list[dict], summary_rows: list[dict], shape_name: str, architecture: str, out_dir: Path) -> None:
    share_path = plot_prefill_decode_share(summary_rows, shape_name, architecture, out_dir)
    component_path = plot_component_share(scenario_rows, shape_name, architecture, out_dir)
    pie_count = plot_pie_charts(scenario_rows, shape_name, architecture, out_dir)
    if share_path:
        print(f"Wrote {share_path}")
    if component_path:
        print(f"Wrote {component_path}")
    print(f"Wrote {pie_count} pie chart(s) under {out_dir / 'pie_charts'}")
