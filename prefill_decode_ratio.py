from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math


ARCHITECTURE_TITLES = {
    "decoder": "Decoder-only",
    "encoder_decoder": "Encoder-decoder",
}

SHAPE_TITLES = {
    "gpt2_medium": "GPT-2 medium",
    "gpt3_2p7b": "GPT-3 2.7B",
    "llama_7b": "LLaMA-7B",
    "t5_base": "T5-base",
    "t5_large": "T5-large",
    "bart_large": "BART-large",
}

SHAPE_ORDER = {
    "decoder": ["gpt2_medium", "gpt3_2p7b", "llama_7b"],
    "encoder_decoder": ["t5_base", "bart_large", "t5_large"],
}

SHAPE_COLORS = {
    "gpt2_medium": "#4C78A8",
    "gpt3_2p7b": "#F58518",
    "llama_7b": "#54A24B",
    "t5_base": "#4C78A8",
    "bart_large": "#8E6C8A",
    "t5_large": "#F58518",
}

REPRESENTATIVE_SHAPES = {
    "decoder": "gpt2_medium",
    "encoder_decoder": "t5_base",
}

GENERATED_TOKEN_FRACTIONS = (None, 0.10, 0.25, 0.50, 0.75, 1.00)

TOKEN_SWEEP_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
)


@dataclass(frozen=True)
class RatioRecord:
    architecture: str
    shape_name: str
    seq_len: int
    prefill_ms: float
    decode_ms_per_token: float

    @property
    def ratio(self) -> float:
        return self.prefill_ms / self.decode_ms_per_token


@dataclass(frozen=True)
class TokenSweepRecord:
    architecture: str
    shape_name: str
    seq_len: int
    token_case: int
    generated_tokens: int
    prefill_ms: float
    decode_ms_per_token: float

    @property
    def total_decode_ms(self) -> float:
        return self.decode_ms_per_token * self.generated_tokens

    @property
    def prefill_to_decode_ratio(self) -> float:
        return self.prefill_ms / self.total_decode_ms


def require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Ratio plotting requires Pillow. Install with: pip install pillow") from exc
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


def csv_inputs(root: Path) -> list[Path]:
    return sorted(
        list((root / "decoder_profiler" / "latency_results" / "decoder").glob("*.csv"))
        + list((root / "encoder_decoder_profiler" / "latency_results" / "encoder_decoder").glob("*.csv"))
    )


def metric_value(row: dict[str, str], phase: str) -> float | None:
    metric = "avg_total_ms_per_token" if phase == "decode" else "avg_total_ms_per_repeat"
    try:
        return float(row[metric])
    except (KeyError, TypeError, ValueError):
        return None


def load_ratio_records(paths: list[Path]) -> list[RatioRecord]:
    totals: defaultdict[tuple[str, str, int, str], float] = defaultdict(float)
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                architecture = str(row.get("architecture") or "")
                if architecture not in ARCHITECTURE_TITLES:
                    continue
                phase = str(row.get("phase") or "prefill")
                if phase not in {"prefill", "decode"}:
                    continue
                value = metric_value(row, phase)
                if value is None:
                    continue
                try:
                    seq_len = int(float(str(row.get("seq_len") or "")))
                except ValueError:
                    continue
                shape_name = str(row.get("shape_name") or "")
                totals[(architecture, shape_name, seq_len, phase)] += value

    records: list[RatioRecord] = []
    for architecture, shape_name, seq_len, phase in sorted(totals):
        if phase != "prefill":
            continue
        decode_key = (architecture, shape_name, seq_len, "decode")
        decode_ms = totals.get(decode_key, 0.0)
        prefill_ms = totals[(architecture, shape_name, seq_len, phase)]
        if prefill_ms > 0 and decode_ms > 0:
            records.append(RatioRecord(architecture, shape_name, seq_len, prefill_ms, decode_ms))
    return records


def display_shape(shape_name: str) -> str:
    return SHAPE_TITLES.get(shape_name, shape_name.replace("_", " "))


def parse_representatives(value: str) -> dict[str, str]:
    representatives = dict(REPRESENTATIVE_SHAPES)
    if not value:
        return representatives
    for part in value.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError(f"Representative must be architecture:shape, got {part!r}")
        architecture, shape_name = [piece.strip() for piece in part.split(":", 1)]
        if architecture not in ARCHITECTURE_TITLES:
            raise ValueError(f"Unsupported architecture for ratio sweep: {architecture}")
        representatives[architecture] = shape_name
    return representatives


def generated_token_count(seq_len: int, fraction: float | None) -> int:
    if fraction is None:
        return 1
    return max(1, int(round(seq_len * fraction)))


def format_token_count(value: int) -> str:
    return f"{value} token" if value == 1 else f"{value} tokens"


def build_token_sweep_records(
    records: list[RatioRecord],
    representatives: dict[str, str],
) -> list[TokenSweepRecord]:
    sweep_records: list[TokenSweepRecord] = []
    for record in records:
        if representatives.get(record.architecture) != record.shape_name:
            continue
        for token_case, fraction in enumerate(GENERATED_TOKEN_FRACTIONS):
            generated_tokens = generated_token_count(record.seq_len, fraction)
            sweep_records.append(
                TokenSweepRecord(
                    architecture=record.architecture,
                    shape_name=record.shape_name,
                    seq_len=record.seq_len,
                    token_case=token_case,
                    generated_tokens=generated_tokens,
                    prefill_ms=record.prefill_ms,
                    decode_ms_per_token=record.decode_ms_per_token,
                )
            )
    return sweep_records


def format_ratio(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}x"
    if value >= 0.1:
        return f"{value:.2f}x"
    return f"{value:.3f}x"


def draw_centered(draw, xy: tuple[float, float], text: str, font, fill) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def nice_y_max(value: float) -> int:
    if value <= 10:
        return 10
    return int(math.ceil(value / 10.0) * 10)


def save_ratio_chart(records: list[RatioRecord], architecture: str, output: Path) -> None:
    Image, ImageDraw, ImageFont = require_pillow()
    arch_records = [record for record in records if record.architecture == architecture]
    if not arch_records:
        return

    seq_lens = sorted({record.seq_len for record in arch_records})
    present_shapes = {record.shape_name for record in arch_records}
    shapes = [shape for shape in SHAPE_ORDER[architecture] if shape in present_shapes]
    shapes.extend(sorted(present_shapes - set(shapes)))
    ratio_lookup = {
        (record.shape_name, record.seq_len): record.ratio
        for record in arch_records
    }

    width = 1800
    height = 980
    margin_l = 135
    margin_r = 70
    margin_t = 150
    margin_b = 185
    plot_l = margin_l
    plot_r = width - margin_r
    plot_t = margin_t
    plot_b = height - margin_b
    plot_w = plot_r - plot_l
    plot_h = plot_b - plot_t
    y_max = nice_y_max(max(record.ratio for record in arch_records) * 1.08)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(ImageFont, 42, bold=True)
    subtitle_font = load_font(ImageFont, 24)
    axis_font = load_font(ImageFont, 22, bold=True)
    tick_font = load_font(ImageFont, 20)
    label_font = load_font(ImageFont, 18, bold=True)
    legend_font = load_font(ImageFont, 21)

    title = f"{ARCHITECTURE_TITLES[architecture]} prefill-to-decode latency ratio"
    subtitle = "Full-context prefill latency divided by cached-decode latency per generated token"
    draw.text((margin_l, 42), title, fill=(22, 22, 22), font=title_font)
    draw.text((margin_l, 96), subtitle, fill=(78, 78, 78), font=subtitle_font)

    legend_x = width - 560
    legend_y = 48
    for idx, shape_name in enumerate(shapes):
        x = legend_x + idx * 180
        color = hex_to_rgb(SHAPE_COLORS.get(shape_name, "#999999"))
        draw.rounded_rectangle([x, legend_y + 6, x + 24, legend_y + 30], radius=4, fill=color)
        draw.text((x + 34, legend_y + 2), display_shape(shape_name), fill=(45, 45, 45), font=legend_font)

    for tick in range(0, y_max + 1, 10):
        y = plot_b - (tick / y_max) * plot_h
        color = (210, 210, 210) if tick else (85, 85, 85)
        draw.line([(plot_l, y), (plot_r, y)], fill=color, width=1)
        label = f"{tick}x"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_l - 18 - (bbox[2] - bbox[0]), y - 11), label, fill=(65, 65, 65), font=tick_font)

    draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=(70, 70, 70), width=2)
    draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=(70, 70, 70), width=2)

    group_w = plot_w / len(seq_lens)
    bar_gap = 10
    bar_w = min(58, (group_w - 56) / max(1, len(shapes)) - bar_gap)
    for group_idx, seq_len in enumerate(seq_lens):
        group_center = plot_l + group_w * (group_idx + 0.5)
        total_bar_w = len(shapes) * bar_w + (len(shapes) - 1) * bar_gap
        start_x = group_center - total_bar_w / 2
        for shape_idx, shape_name in enumerate(shapes):
            ratio = ratio_lookup.get((shape_name, seq_len))
            if ratio is None:
                continue
            x0 = start_x + shape_idx * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y1 = plot_b
            y0 = plot_b - (ratio / y_max) * plot_h
            color = hex_to_rgb(SHAPE_COLORS.get(shape_name, "#999999"))
            draw.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=color)
            label = f"{ratio:.1f}x"
            bbox = draw.textbbox((0, 0), label, font=label_font)
            lx = x0 + (bar_w - (bbox[2] - bbox[0])) / 2
            ly = y0 - 28
            draw.text((lx, ly), label, fill=(35, 35, 35), font=label_font)
        draw_centered(draw, (group_center, plot_b + 24), f"L = {seq_len}", tick_font, (35, 35, 35))

    draw_centered(draw, ((plot_l + plot_r) / 2, height - 68), "Context length", axis_font, (35, 35, 35))
    axis_label = "Prefill / decode-token latency"
    rotated = Image.new("RGBA", (360, 40), (255, 255, 255, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((0, 0), axis_label, fill=(35, 35, 35), font=axis_font)
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (28, int((plot_t + plot_b) / 2 - rotated.height / 2)), rotated)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def log_axis_bounds(values: list[float]) -> tuple[float, float]:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.01, 10.0
    lower_candidates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    upper_candidates = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    lower = 0.001
    for candidate in lower_candidates:
        if candidate <= min(positive):
            lower = candidate
    upper = upper_candidates[-1]
    for candidate in upper_candidates:
        if candidate >= max(positive):
            upper = candidate
            break
    if lower >= upper:
        lower = upper / 100.0
    return lower, upper


def log_y(value: float, y_min: float, y_max: float, plot_t: int, plot_b: int) -> float:
    clamped = min(max(value, y_min), y_max)
    span = math.log10(y_max) - math.log10(y_min)
    position = (math.log10(clamped) - math.log10(y_min)) / span
    return plot_b - position * (plot_b - plot_t)


def save_token_sweep_chart(records: list[TokenSweepRecord], architecture: str, shape_name: str, output: Path) -> None:
    Image, ImageDraw, ImageFont = require_pillow()
    arch_records = [
        record
        for record in records
        if record.architecture == architecture and record.shape_name == shape_name
    ]
    if not arch_records:
        return

    seq_lens = sorted({record.seq_len for record in arch_records})
    token_cases = sorted({record.token_case for record in arch_records})
    ratio_lookup = {
        (record.seq_len, record.token_case): record.prefill_to_decode_ratio
        for record in arch_records
    }
    token_lookup = {
        (record.seq_len, record.token_case): record.generated_tokens
        for record in arch_records
    }

    width = 2200
    height = 1120
    margin_l = 150
    margin_r = 80
    margin_t = 170
    margin_b = 270
    plot_l = margin_l
    plot_r = width - margin_r
    plot_t = margin_t
    plot_b = height - margin_b
    plot_w = plot_r - plot_l
    plot_h = plot_b - plot_t
    y_min, y_max = log_axis_bounds([record.prefill_to_decode_ratio for record in arch_records])

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(ImageFont, 44, bold=True)
    subtitle_font = load_font(ImageFont, 24)
    axis_font = load_font(ImageFont, 23, bold=True)
    tick_font = load_font(ImageFont, 20)
    token_font = load_font(ImageFont, 17)
    label_font = load_font(ImageFont, 16, bold=True)
    legend_font = load_font(ImageFont, 21)

    title = f"{display_shape(shape_name)} generated-token prefill/decode sweep"
    subtitle = (
        "Prefill latency divided by total decode latency; total decode scales "
        "one measured decode step by generated-token count"
    )
    draw.text((margin_l, 44), title, fill=(22, 22, 22), font=title_font)
    draw.text((margin_l, 100), subtitle, fill=(78, 78, 78), font=subtitle_font)

    draw.text(
        (margin_l, 135),
        "Generated-token count G is printed below each bar.",
        fill=(45, 45, 45),
        font=legend_font,
    )

    ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    for tick in ticks:
        if tick < y_min or tick > y_max:
            continue
        y = log_y(tick, y_min, y_max, plot_t, plot_b)
        is_one = abs(tick - 1.0) < 1e-9
        color = (82, 82, 82) if is_one else (214, 214, 214)
        draw.line([(plot_l, y), (plot_r, y)], fill=color, width=2 if is_one else 1)
        label = format_ratio(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_l - 18 - (bbox[2] - bbox[0]), y - 11), label, fill=(65, 65, 65), font=tick_font)

    draw.line([(plot_l, plot_t), (plot_l, plot_b)], fill=(70, 70, 70), width=2)
    draw.line([(plot_l, plot_b), (plot_r, plot_b)], fill=(70, 70, 70), width=2)

    group_w = plot_w / len(seq_lens)
    bar_gap = 8
    bar_w = min(38, (group_w - 68) / max(1, len(token_cases)) - bar_gap)
    baseline = log_y(y_min, y_min, y_max, plot_t, plot_b)
    labeled_cases = {token_cases[0], token_cases[1], token_cases[-1]}

    for group_idx, seq_len in enumerate(seq_lens):
        group_center = plot_l + group_w * (group_idx + 0.5)
        total_bar_w = len(token_cases) * bar_w + (len(token_cases) - 1) * bar_gap
        start_x = group_center - total_bar_w / 2
        for token_idx, token_case in enumerate(token_cases):
            ratio = ratio_lookup.get((seq_len, token_case))
            if ratio is None:
                continue
            x0 = start_x + token_idx * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = log_y(ratio, y_min, y_max, plot_t, plot_b)
            color = hex_to_rgb(TOKEN_SWEEP_COLORS[token_case])
            draw.rounded_rectangle([x0, y0, x1, baseline], radius=5, fill=color)
            if token_case in labeled_cases:
                label = format_ratio(ratio)
                bbox = draw.textbbox((0, 0), label, font=label_font)
                lx = x0 + (bar_w - (bbox[2] - bbox[0])) / 2
                ly = max(plot_t + 4, y0 - 24)
                draw.text((lx, ly), label, fill=(35, 35, 35), font=label_font)
            token_count = token_lookup[(seq_len, token_case)]
            token_label = str(token_count)
            bbox = draw.textbbox((0, 0), token_label, font=token_font)
            label_w = bbox[2] - bbox[0] + 4
            label_h = bbox[3] - bbox[1] + 4
            label_image = Image.new("RGBA", (label_w, label_h), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((2, 2), token_label, fill=(35, 35, 35), font=token_font)
            label_image = label_image.rotate(90, expand=True)
            lx = int(x0 + bar_w / 2 - label_image.width / 2)
            image.paste(label_image, (lx, plot_b + 18), label_image)
        draw_centered(draw, (group_center, plot_b + 82), f"L = {seq_len}", tick_font, (35, 35, 35))

    draw_centered(
        draw,
        ((plot_l + plot_r) / 2, height - 86),
        "Generated tokens G within each prompt length L",
        axis_font,
        (35, 35, 35),
    )
    axis_label = "Prefill / total decode latency (log scale)"
    rotated = Image.new("RGBA", (520, 42), (255, 255, 255, 0))
    rdraw = ImageDraw.Draw(rotated)
    rdraw.text((0, 0), axis_label, fill=(35, 35, 35), font=axis_font)
    rotated = rotated.rotate(90, expand=True)
    image.paste(rotated, (30, int((plot_t + plot_b) / 2 - rotated.height / 2)), rotated)

    note = "Numbers below bars are generated-token counts; the same G values are written to prefill_decode_token_sweep.csv."
    draw.text((plot_l, height - 36), note, fill=(85, 85, 85), font=tick_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def write_summary_csv(records: list[RatioRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "shape_name",
                "seq_len",
                "prefill_ms",
                "decode_ms_per_token",
                "prefill_to_decode_ratio",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row.architecture, row.seq_len, row.shape_name)):
            writer.writerow(
                {
                    "architecture": record.architecture,
                    "shape_name": record.shape_name,
                    "seq_len": record.seq_len,
                    "prefill_ms": f"{record.prefill_ms:.6f}",
                    "decode_ms_per_token": f"{record.decode_ms_per_token:.6f}",
                    "prefill_to_decode_ratio": f"{record.ratio:.6f}",
                }
            )


def write_token_sweep_csv(records: list[TokenSweepRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "shape_name",
                "seq_len",
                "generated_token_label",
                "generated_tokens",
                "prefill_ms",
                "decode_ms_per_token",
                "total_decode_ms",
                "prefill_to_total_decode_ratio",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda row: (row.architecture, row.shape_name, row.seq_len, row.generated_tokens)):
            writer.writerow(
                {
                    "architecture": record.architecture,
                    "shape_name": record.shape_name,
                    "seq_len": record.seq_len,
                    "generated_token_label": format_token_count(record.generated_tokens),
                    "generated_tokens": record.generated_tokens,
                    "prefill_ms": f"{record.prefill_ms:.6f}",
                    "decode_ms_per_token": f"{record.decode_ms_per_token:.6f}",
                    "total_decode_ms": f"{record.total_decode_ms:.6f}",
                    "prefill_to_total_decode_ratio": f"{record.prefill_to_decode_ratio:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create prefill-to-decode latency ratio charts from profiler CSV outputs."
    )
    parser.add_argument("--root", default=".", help="Transformer latency repository root.")
    parser.add_argument("--output-dir", default="figures", help="Directory for generated figures.")
    parser.add_argument(
        "--representatives",
        default="decoder:gpt2_medium,encoder_decoder:t5_base",
        help=(
            "Representative shapes for generated-token sweeps, formatted as "
            "architecture:shape_name pairs separated by commas."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    records = load_ratio_records(csv_inputs(root))
    if not records:
        raise SystemExit("No paired prefill/decode CSV records found.")

    output_dir = Path(args.output_dir)
    representatives = parse_representatives(args.representatives)
    for architecture in ARCHITECTURE_TITLES:
        save_ratio_chart(
            records,
            architecture,
            output_dir / architecture / "prefill_decode_ratio.png",
        )
    sweep_records = build_token_sweep_records(records, representatives)
    for architecture, shape_name in representatives.items():
        save_token_sweep_chart(
            sweep_records,
            architecture,
            shape_name,
            output_dir / architecture / "prefill_decode_token_sweep.png",
        )
    write_summary_csv(records, output_dir / "prefill_decode_ratio.csv")
    write_token_sweep_csv(sweep_records, output_dir / "prefill_decode_token_sweep.csv")
    print(f"Wrote prefill/decode ratio charts, token sweeps, and CSVs to {output_dir}")


if __name__ == "__main__":
    main()
