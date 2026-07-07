from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLORS = {
    "Self-Attention": "#4C78A8",
    "Feed Forward Network": "#F58518",
    "Layer Normalization": "#54A24B",
    "Model Output": "#B279A2",
    "Other": "#9E9E9E",
}


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Bar plots require Pillow. Use the bundled Codex Python runtime.") from exc
    return Image, ImageDraw, ImageFont


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


def csv_sort_key(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def phase_label(phase: str) -> str:
    return "Cached Decode (10 Tokens)" if phase == "decode" else "Prefill"


def metric_label(metric: str) -> str:
    return {
        "total_ms": "Total latency (ms)",
        "avg_total_ms_per_token": "Latency per token (ms)",
        "pct_total": "Share of phase latency (%)",
    }[metric]


def format_value(value: float, metric: str) -> str:
    if metric == "pct_total":
        return f"{value:.1f}%"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def read_rows(csv_path: Path, metric: str) -> list[dict[str, object]]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_metric"] = float(row[metric])
    return sorted(rows, key=lambda row: float(row["_metric"]), reverse=True)


def draw_legend(draw, x: int, y: int, font, groups: list[str]) -> None:
    cursor = x
    for group in groups:
        color = hex_to_rgb(COLORS.get(group, COLORS["Other"]))
        draw.rounded_rectangle([cursor, y + 3, cursor + 18, y + 21], radius=3, fill=color)
        draw.text((cursor + 26, y), group, fill=(55, 55, 55), font=font)
        cursor += 26 + int(draw.textlength(group, font=font)) + 32


def plot_bar(csv_path: Path, out_path: Path, metric: str) -> None:
    Image, ImageDraw, ImageFont = require_pillow()
    rows = read_rows(csv_path, metric)
    if not rows:
        return

    seq_len = int(rows[0]["seq_len"])
    phase = str(rows[0]["phase"])
    title = f"Phi-Style {phase_label(phase)} Components, L={seq_len}"
    subtitle = metric_label(metric)
    groups = []
    for row in rows:
        group = str(row["group"])
        if group not in groups:
            groups.append(group)

    width = 1800
    row_h = 58
    top = 155
    bottom = 70
    left_label = 48
    bar_left = 520
    bar_right = width - 210
    bar_width = bar_right - bar_left
    height = top + len(rows) * row_h + bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(ImageFont, 42, bold=True)
    subtitle_font = load_font(ImageFont, 24)
    label_font = load_font(ImageFont, 25)
    small_font = load_font(ImageFont, 20)
    value_font = load_font(ImageFont, 22, bold=True)

    draw.text((left_label, 38), title, fill=(20, 20, 20), font=title_font)
    draw.text((left_label, 94), subtitle, fill=(75, 75, 75), font=subtitle_font)
    draw_legend(draw, bar_left, 94, small_font, groups)

    max_value = max(float(row["_metric"]) for row in rows)
    max_value = max(max_value, 1.0)
    axis_y = top + len(rows) * row_h + 16

    for tick in range(6):
        fraction = tick / 5
        x = bar_left + int(bar_width * fraction)
        value = max_value * fraction
        draw.line([x, top - 16, x, axis_y - 8], fill=(232, 232, 232), width=1)
        draw.text((x - 18, axis_y), format_value(value, metric), fill=(95, 95, 95), font=small_font)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = str(row["pretty_name"])
        value = float(row["_metric"])
        group = str(row["group"])
        bar_len = max(2, int(bar_width * value / max_value))
        color = hex_to_rgb(COLORS.get(group, COLORS["Other"]))

        draw.text((left_label, y + 10), label, fill=(35, 35, 35), font=label_font)
        draw.rounded_rectangle(
            [bar_left, y + 8, bar_left + bar_len, y + 39],
            radius=5,
            fill=color,
        )
        value_label = format_value(value, metric)
        value_x = min(bar_left + bar_len + 12, width - 150)
        draw.text((value_x, y + 10), value_label, fill=(35, 35, 35), font=value_font)

    draw.line([bar_left, axis_y - 8, bar_right, axis_y - 8], fill=(180, 180, 180), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create prefill and decode bar plots for Phi reproduction CSVs.")
    parser.add_argument("--phase-dir", default="reproduce/phi_baseline_like/phase_split")
    parser.add_argument("--output-dir", default="reproduce/phi_baseline_like/bar_plots")
    parser.add_argument(
        "--metric",
        choices=["total_ms", "avg_total_ms_per_token", "pct_total"],
        default="total_ms",
        help="Metric to plot for each component.",
    )
    args = parser.parse_args()

    phase_dir = Path(args.phase_dir)
    output_dir = Path(args.output_dir)
    prefill_csvs = sorted(phase_dir.glob("prefill_seqlen_*.csv"), key=csv_sort_key)
    decode_csvs = sorted(phase_dir.glob("decode10_seqlen_*.csv"), key=csv_sort_key)
    if not prefill_csvs and not decode_csvs:
        raise SystemExit(f"No phase-split CSVs found in {phase_dir}")

    for csv_path in prefill_csvs:
        seq_len = csv_sort_key(csv_path)
        out_path = output_dir / "prefill" / f"bar_prefill_seqlen_{seq_len}.png"
        plot_bar(csv_path, out_path, args.metric)
        print(out_path)

    for csv_path in decode_csvs:
        seq_len = csv_sort_key(csv_path)
        out_path = output_dir / "decode" / f"bar_decode10_seqlen_{seq_len}.png"
        plot_bar(csv_path, out_path, args.metric)
        print(out_path)


if __name__ == "__main__":
    main()
