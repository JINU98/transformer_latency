from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


COLORS = {
    "Self-Attention": "#4C78A8",
    "Cross-Attention": "#8E6C8A",
    "Feed Forward Network": "#F58518",
    "Layer Normalization": "#54A24B",
    "Model Output": "#B279A2",
    "Other": "#9E9E9E",
}

NAME_MAP = {
    "attn.q_proj": "Q Projection",
    "attn.k_proj": "K Projection",
    "attn.v_proj": "V Projection",
    "attn.qkv_projection": "QKV Projection",
    "attn.gqa_expand": "GQA Expand",
    "attn.matmul_qk": "QK^T MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.apply_padding_mask": "Padding Mask",
    "attn.softmax": "Softmax",
    "attn.weighted_sum": "Attention Weighted Sum",
    "attn.out_projection": "Output Projection",
    "attn.cache_concat": "KV Cache Concat",
    "cross_attn.q_proj": "Cross Q Projection",
    "cross_attn.k_proj": "Cross K Projection",
    "cross_attn.v_proj": "Cross V Projection",
    "cross_attn.matmul_qk": "Cross QK^T MatMul",
    "cross_attn.softmax": "Cross Softmax",
    "cross_attn.weighted_sum": "Cross Attention Weighted Sum",
    "cross_attn.out_projection": "Cross Output Projection",
    "block.norm1": "LayerNorm (Pre-Attention)",
    "block.norm2": "LayerNorm (Pre-FFN)",
    "block.norm3": "LayerNorm (Pre-Cross-Attention)",
    "ff.linear1": "FFN Linear 1",
    "ff.gelu": "GELU Activation",
    "ff.linear2": "FFN Linear 2",
    "ff.w1_gate": "SwiGLU Gate",
    "ff.w2_up": "SwiGLU Up",
    "ff.swiglu_act": "SwiGLU Activation",
    "ff.w3_down": "SwiGLU Down",
    "model.final_norm": "Final LayerNorm",
    "model.output_head": "Output Head Projection",
}


def require_plotting():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Plotting requires Pillow. Install with: pip install -r requirements.txt") from exc
    return Image, ImageDraw, ImageFont


def phase_metric(phase: str, columns: set[str]) -> tuple[str, str]:
    if phase == "decode" and "avg_total_ms_per_token" in columns:
        return "avg_total_ms_per_token", "Average latency per decoded token (ms)"
    if "avg_total_ms_per_repeat" in columns:
        return "avg_total_ms_per_repeat", "Average latency per phase run (ms)"
    if "total_ms" in columns:
        return "total_ms", "Total latency (ms)"
    return "avg_ms", "Average latency per timed scope (ms)"


def operation_group(operation_key: str) -> str:
    if operation_key.startswith("cross_attn."):
        return "Cross-Attention"
    if operation_key.startswith("attn."):
        return "Self-Attention"
    if operation_key.startswith("ff."):
        return "Feed Forward Network"
    if operation_key.startswith("block."):
        return "Layer Normalization"
    if operation_key.startswith("model."):
        return "Model Output"
    return "Other"


def pretty_name(operation_key: str) -> str:
    return NAME_MAP.get(operation_key, operation_key.replace(".", " ").replace("_", " ").title())


def load_font(ImageFont, size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def format_title_token(value: str) -> str:
    if not value:
        return value
    return value.replace("_", " ").replace("-", " ").title()


def phase_label(phase: str, phase_tokens: str) -> str:
    if phase == "decode":
        token_label = phase_tokens if phase_tokens else "N"
        return f"Cached Decode ({token_label} Tokens)"
    if phase:
        return phase.replace("_", " ").title()
    return "Components"


def metric_value_label(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_phase(rows: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    if not rows or "phase" not in rows[0]:
        return [("", rows)]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("phase", ""))].append(row)
    phase_order = ["prefill", "decode"]
    ordered = [(phase, grouped.pop(phase)) for phase in phase_order if phase in grouped]
    ordered.extend(sorted(grouped.items()))
    return ordered


def draw_legend(draw, x: int, y: int, font, groups: list[str]) -> None:
    cursor = x
    for group in groups:
        color = hex_to_rgb(COLORS.get(group, COLORS["Other"]))
        draw.rounded_rectangle([cursor, y + 3, cursor + 18, y + 21], radius=3, fill=color)
        draw.text((cursor + 26, y), group, fill=(55, 55, 55), font=font)
        cursor += 26 + int(draw.textlength(group, font=font)) + 32


def title_for_phase(label: str, rows: list[dict[str, str]], phase: str) -> str:
    first = rows[0] if rows else {}
    shape = str(first.get("shape_name") or label)
    architecture = str(first.get("architecture") or first.get("model_family") or "")
    seq_len = str(first.get("seq_len") or first.get("decoder_seq_len") or first.get("encoder_seq_len") or "")
    phase_tokens = str(first.get("phase_tokens") or "")

    prefix = format_title_token(shape)
    if architecture and architecture != shape:
        prefix = f"{prefix} {format_title_token(architecture)}"
    title = f"{prefix} {phase_label(phase, phase_tokens)} Components"
    if seq_len:
        title += f", L={seq_len}"
    return title


def draw_bar_plot(
    rows: list[dict[str, str]],
    out_path: Path,
    metric_col: str,
    metric_label: str,
    title: str,
) -> None:
    Image, ImageDraw, ImageFont = require_plotting()
    plot_rows = []
    for row in rows:
        try:
            value = float(row.get(metric_col, ""))
        except ValueError:
            continue
        operation_key = str(row.get("operation_key") or row.get("name") or "")
        group = str(row.get("group") or operation_group(operation_key))
        label = str(row.get("pretty_name") or pretty_name(operation_key))
        plot_rows.append({"value": value, "group": group, "label": label})

    if not plot_rows:
        return

    plot_rows = sorted(plot_rows, key=lambda row: row["value"], reverse=True)[:30]
    groups = []
    for row in plot_rows:
        if row["group"] not in groups:
            groups.append(row["group"])

    width = 1800
    row_h = 58
    top = 155
    bottom = 70
    left_label = 48
    bar_left = 560
    bar_right = width - 210
    bar_width = bar_right - bar_left
    height = top + len(plot_rows) * row_h + bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(ImageFont, 42, bold=True)
    subtitle_font = load_font(ImageFont, 24)
    label_font = load_font(ImageFont, 25)
    small_font = load_font(ImageFont, 20)
    value_font = load_font(ImageFont, 22, bold=True)

    draw.text((left_label, 38), title, fill=(20, 20, 20), font=title_font)
    draw.text((left_label, 94), metric_label, fill=(75, 75, 75), font=subtitle_font)
    draw_legend(draw, bar_left, 94, small_font, groups)

    max_value = max(float(row["value"]) for row in plot_rows)
    max_value = max(max_value, 1.0)
    axis_y = top + len(plot_rows) * row_h + 16

    for tick in range(6):
        fraction = tick / 5
        x = bar_left + int(bar_width * fraction)
        value = max_value * fraction
        draw.line([x, top - 16, x, axis_y - 8], fill=(232, 232, 232), width=1)
        draw.text((x - 18, axis_y), metric_value_label(value), fill=(95, 95, 95), font=small_font)

    for idx, row in enumerate(plot_rows):
        y = top + idx * row_h
        value = float(row["value"])
        bar_len = max(2, int(bar_width * value / max_value))
        color = hex_to_rgb(COLORS.get(row["group"], COLORS["Other"]))
        draw.text((left_label, y + 10), str(row["label"]), fill=(35, 35, 35), font=label_font)
        draw.rounded_rectangle(
            [bar_left, y + 8, bar_left + bar_len, y + 39],
            radius=5,
            fill=color,
        )
        value_x = min(bar_left + bar_len + 12, width - 150)
        draw.text((value_x, y + 10), metric_value_label(value), fill=(35, 35, 35), font=value_font)

    draw.line([bar_left, axis_y - 8, bar_right, axis_y - 8], fill=(180, 180, 180), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def plot_single_csv(csv_path: Path, out_dir: Path) -> None:
    rows = read_csv_rows(csv_path)
    if not rows:
        return
    label = csv_path.stem
    columns = set(rows[0])
    for phase, phase_rows in group_by_phase(rows):
        metric_col, metric_label = phase_metric(phase, columns)
        suffix = "" if not phase else f"_{phase}"
        out_path = out_dir / f"bar_{label}{suffix}.png"
        draw_bar_plot(
            rows=phase_rows,
            out_path=out_path,
            metric_col=metric_col,
            metric_label=metric_label,
            title=title_for_phase(label, phase_rows, phase),
        )


def plot_comparisons(csv_paths: list[Path], out_dir: Path, prefix: str) -> None:
    return
