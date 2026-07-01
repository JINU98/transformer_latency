from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (
    MODEL_SHAPES,
    SEQ_LEN_PRESETS,
    dtype_bytes,
    estimate_attention_gb,
    make_shape_overrides,
    parse_int_list,
    parse_shape_names,
)
from common.io import save_rows_csv
from common.latency import LatencyRecorder, benchmark_forward
from common.models import ProfiledCrossAttention, ProfiledSelfAttention, require_torch, torch
from common.plotting import plot_comparisons, plot_single_csv
from common.runner import add_common_args, resolve_device_and_dtype


class SelfAttentionBench(torch.nn.Module):
    def __init__(self, shape, recorder):
        super().__init__()
        self.recorder = recorder
        self.attn = ProfiledSelfAttention(
            shape.d_model,
            shape.num_heads,
            recorder,
            causal=True,
            num_kv_heads=shape.num_kv_heads,
        )

    def forward(self, x):
        return self.attn(x)


class CrossAttentionBench(torch.nn.Module):
    def __init__(self, shape, recorder):
        super().__init__()
        self.recorder = recorder
        self.attn = ProfiledCrossAttention(shape.d_model, shape.num_heads, recorder)

    def forward(self, x, encoder_states):
        return self.attn(x, encoder_states)


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark self-attention and cross-attention only.")
    add_common_args(parser, "model_family")
    parser.add_argument("--kind", choices=["self", "cross", "both"], default="both")
    args = parser.parse_args()

    torch_mod, _, _ = require_torch()
    device, dtype = resolve_device_and_dtype(torch_mod, args.device, args.dtype)
    seq_lens = parse_int_list(args.seq_lens, SEQ_LEN_PRESETS[args.preset])
    shape_names = parse_shape_names(args.shape_names, "model_family")
    shapes = [
        make_shape_overrides(MODEL_SHAPES[name], args.d_model, args.num_heads, args.num_layers, args.d_ff, args.num_kv_heads)
        for name in shape_names
    ]
    written = []

    for shape in shapes:
        for seq_len in seq_lens:
            est_gb = estimate_attention_gb(
                args.batch_size,
                shape.num_heads,
                seq_len,
                seq_len,
                dtype_bytes(args.dtype),
                multiplier=3.0,
            )
            if args.max_attn_gb > 0 and est_gb > args.max_attn_gb:
                print(f"SKIP {shape.name} L={seq_len}: estimated attention buffers {est_gb:.2f} GB")
                continue

            for kind in ["self", "cross"]:
                if args.kind not in {kind, "both"}:
                    continue
                recorder = LatencyRecorder(sync_cuda=(device.type == "cuda"), torch_module=torch_mod)
                module_cls = SelfAttentionBench if kind == "self" else CrossAttentionBench

                def make_model(module_cls=module_cls, recorder=recorder):
                    model = module_cls(shape, recorder).to(device=device, dtype=dtype)
                    model.recorder = recorder
                    return model

                def make_inputs(kind=kind):
                    x = torch_mod.randn(args.batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
                    if kind == "cross":
                        enc = torch_mod.randn(args.batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
                        return x, enc
                    return (x,)

                _, recorder = benchmark_forward(
                    make_model=make_model,
                    make_inputs=make_inputs,
                    forward=lambda model, inputs: model(*inputs),
                    warmups=args.warmups,
                    repeats=args.repeats,
                    torch_module=torch_mod,
                )
                rows = recorder.rows(
                    {
                        "architecture": "attention_microbench",
                        "model_family": kind,
                        "shape_name": shape.name,
                        "d_model": shape.d_model,
                        "num_heads": shape.num_heads,
                        "num_kv_heads": shape.kv_heads,
                        "head_dim": shape.head_dim,
                        "num_layers": 1,
                        "d_ff": shape.d_ff,
                        "batch_size": args.batch_size,
                        "seq_len": seq_len,
                        "encoder_seq_len": seq_len if kind == "cross" else "",
                        "decoder_seq_len": seq_len,
                        "dtype": args.dtype,
                        "device": str(device),
                    }
                )
                csv_path = (
                    Path(args.output_dir)
                    / "attention_microbench"
                    / kind
                    / f"latency_{shape.name}_d{shape.d_model}_h{shape.num_heads}_l{seq_len}.csv"
                )
                save_rows_csv(csv_path, rows)
                written.append(csv_path)
                if not args.no_plots:
                    plot_single_csv(csv_path, csv_path.parent)

    if written and not args.no_plots:
        plot_comparisons(written, Path(args.output_dir) / "attention_microbench", "attention")
    print(f"Wrote {len(written)} CSV file(s).")


if __name__ == "__main__":
    main()
