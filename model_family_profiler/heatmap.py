from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.plotting import require_plotting


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a model-family latency heatmap from CSV outputs.")
    parser.add_argument("--input-dir", default="latency_results")
    parser.add_argument("--output", default="latency_results/model_family_heatmap.png")
    args = parser.parse_args()

    pd, plt = require_plotting()
    csv_paths = sorted(Path(args.input_dir).glob("*/*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {args.input_dir}")

    df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    total = df.groupby(["model_family", "shape_name", "seq_len"], as_index=False)["total_ms"].sum()
    total["row"] = total["model_family"] + " / " + total["shape_name"]
    heat = total.pivot_table(index="row", columns="seq_len", values="total_ms", aggfunc="mean").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(heat.index))))
    image = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), [str(col) for col in heat.columns])
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xlabel("Context length L")
    ax.set_title("Total timed latency by model family")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Total latency (ms)")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
