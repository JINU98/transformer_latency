from __future__ import annotations

import argparse
from pathlib import Path

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
from common.plotting import plot_comparisons, plot_single_csv
from common.models import require_torch


def add_common_args(parser: argparse.ArgumentParser, architecture: str) -> None:
    parser.add_argument("--preset", choices=SEQ_LEN_PRESETS, default="quick")
    parser.add_argument("--seq-lens", default=None, help="Comma-separated context lengths L, e.g. 512,1024,2048")
    parser.add_argument("--shape-names", default=None, help=f"Comma-separated model-shape names. Available: {','.join(MODEL_SHAPES)}")
    parser.add_argument("--d-model", type=int, default=None, help="Override hidden dimension d for a custom run")
    parser.add_argument("--num-heads", type=int, default=None, help="Override attention heads h for a custom run")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="Override KV heads for GQA decoder models")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-tokens", type=int, default=10, help="Number of one-token decode steps to average after prefill")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-attn-gb", type=float, default=8.0, help="Skip runs whose estimated attention buffers exceed this many GB. Use <=0 to disable.")
    parser.add_argument("--output-dir", default="latency_results")
    parser.add_argument("--no-plots", action="store_true")
    parser.set_defaults(architecture=architecture)


def resolve_device_and_dtype(torch, device_name: str, dtype_name: str):
    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if device == "auto":
        device = "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    return torch.device(device), dtype


def benchmark_phase(make_model, run_once, warmups: int, repeats: int, torch_module):
    model = make_model()
    model.eval()
    recorder = getattr(model, "recorder")
    with torch_module.no_grad():
        for _ in range(warmups):
            run_once(model)
        recorder.samples_ms.clear()
        for _ in range(repeats):
            run_once(model)
    return model, recorder


def metadata_for_shape(
    args,
    architecture: str,
    model_family: str,
    shape,
    seq_len: int,
    device,
    phase: str,
    phase_tokens: int,
) -> dict[str, object]:
    return {
        "architecture": architecture,
        "model_family": model_family,
        "shape_name": shape.name,
        "d_model": shape.d_model,
        "num_heads": shape.num_heads,
        "num_kv_heads": shape.kv_heads,
        "head_dim": shape.head_dim,
        "num_layers": shape.num_layers,
        "d_ff": shape.d_ff,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "encoder_seq_len": seq_len if architecture in {"encoder", "encoder_decoder"} else "",
        "decoder_seq_len": seq_len if architecture in {"decoder", "encoder_decoder"} else "",
        "phase": phase,
        "phase_tokens": phase_tokens,
        "timed_repeats": args.repeats,
        "dtype": args.dtype,
        "device": str(device),
    }


def run_sweep(args, architecture: str, model_family: str, model_builder, input_builder, encoder_decoder: bool = False) -> list[Path]:
    torch, _, _ = require_torch()
    device, dtype = resolve_device_and_dtype(torch, args.device, args.dtype)
    seq_lens = parse_int_list(args.seq_lens, SEQ_LEN_PRESETS[args.preset])
    shape_names = parse_shape_names(args.shape_names, architecture if architecture != "model_family" else "model_family")
    shapes = [
        make_shape_overrides(MODEL_SHAPES[name], args.d_model, args.num_heads, args.num_layers, args.d_ff, args.num_kv_heads)
        for name in shape_names
    ]
    out_dir = Path(args.output_dir)
    written: list[Path] = []
    for shape in shapes:
        for seq_len in seq_lens:
            est_gb = estimate_attention_gb(
                args.batch_size,
                shape.num_heads,
                seq_len,
                seq_len,
                dtype_bytes(args.dtype),
                multiplier=5.0 if encoder_decoder else 3.0,
            )
            if args.max_attn_gb > 0 and est_gb > args.max_attn_gb:
                print(f"SKIP {shape.name} L={seq_len}: estimated attention buffers {est_gb:.2f} GB > {args.max_attn_gb:.2f} GB")
                continue
            print(f"RUN {model_family} shape={shape.name} d={shape.d_model} h={shape.num_heads} L={seq_len} device={device}")

            def make_model():
                return model_builder(shape, device=device, dtype=dtype, sync_cuda=(device.type == "cuda"))

            if architecture == "decoder":
                rows = run_decoder_prefill_decode(args, make_model, shape, seq_len, model_family, device, dtype, torch)
            elif architecture == "encoder_decoder":
                rows = run_encoder_decoder_prefill_decode(
                    args, make_model, shape, seq_len, model_family, device, dtype, torch
                )
            else:
                inputs = input_builder(shape, seq_len, args.batch_size, device, dtype)

                def run_prefill(model, inputs=inputs):
                    return model(*inputs)

                _, recorder = benchmark_phase(
                    make_model=make_model,
                    run_once=run_prefill,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    torch_module=torch,
                )
                metadata = metadata_for_shape(
                    args, architecture, model_family, shape, seq_len, device, "prefill", seq_len
                )
                rows = recorder.rows(metadata, timed_repeats=args.repeats, phase_tokens=seq_len)

            csv_path = (
                out_dir
                / f"{model_family}"
                / f"latency_{shape.name}_d{shape.d_model}_h{shape.num_heads}_l{seq_len}.csv"
            )
            save_rows_csv(csv_path, rows)
            written.append(csv_path)
            if not args.no_plots:
                plot_single_csv(csv_path, csv_path.parent)
    if written and not args.no_plots:
        plot_comparisons(written, out_dir / model_family, "comparison")
    return written


def run_decoder_prefill_decode(args, make_model, shape, seq_len: int, model_family: str, device, dtype, torch_module):
    prefill = torch_module.randn(args.batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decode_inputs = [
        torch_module.randn(args.batch_size, 1, shape.d_model, device=device, dtype=dtype)
        for _ in range(args.decode_tokens)
    ]

    def run_prefill(model):
        return model(prefill, None, False)

    _, prefill_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_prefill,
        warmups=args.warmups,
        repeats=args.repeats,
        torch_module=torch_module,
    )
    prefill_metadata = metadata_for_shape(
        args, "decoder", model_family, shape, seq_len, device, "prefill", seq_len
    )
    rows = prefill_recorder.rows(prefill_metadata, timed_repeats=args.repeats, phase_tokens=seq_len)

    def run_decode(model):
        recorder = getattr(model, "recorder")
        with recorder.disabled():
            _, past_kv = model(prefill, None, False)
        for token in decode_inputs:
            _, past_kv = model(token, past_kv, False)
        return past_kv

    _, decode_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_decode,
        warmups=args.warmups,
        repeats=args.repeats,
        torch_module=torch_module,
    )
    decode_metadata = metadata_for_shape(
        args, "decoder", model_family, shape, seq_len, device, "decode", args.decode_tokens
    )
    rows.extend(
        decode_recorder.rows(decode_metadata, timed_repeats=args.repeats, phase_tokens=args.decode_tokens)
    )
    return rows


def run_encoder_decoder_prefill_decode(args, make_model, shape, seq_len: int, model_family: str, device, dtype, torch_module):
    encoder_x = torch_module.randn(args.batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decoder_prefill = torch_module.randn(args.batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decode_inputs = [
        torch_module.randn(args.batch_size, 1, shape.d_model, device=device, dtype=dtype)
        for _ in range(args.decode_tokens)
    ]

    def run_prefill(model):
        return model(encoder_x, decoder_prefill, None, False)

    _, prefill_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_prefill,
        warmups=args.warmups,
        repeats=args.repeats,
        torch_module=torch_module,
    )
    prefill_metadata = metadata_for_shape(
        args, "encoder_decoder", model_family, shape, seq_len, device, "prefill", seq_len
    )
    rows = prefill_recorder.rows(prefill_metadata, timed_repeats=args.repeats, phase_tokens=seq_len)

    def run_decode(model):
        recorder = getattr(model, "recorder")
        with recorder.disabled():
            encoder_states = model.encode(encoder_x)
            _, past_kv = model.decode(decoder_prefill, encoder_states, None, False)
        for token in decode_inputs:
            _, past_kv = model.decode(token, encoder_states, past_kv, False)
        return past_kv

    _, decode_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_decode,
        warmups=args.warmups,
        repeats=args.repeats,
        torch_module=torch_module,
    )
    decode_metadata = metadata_for_shape(
        args, "encoder_decoder", model_family, shape, seq_len, device, "decode", args.decode_tokens
    )
    rows.extend(
        decode_recorder.rows(decode_metadata, timed_repeats=args.repeats, phase_tokens=args.decode_tokens)
    )
    return rows


def random_hidden_inputs(shape, seq_len: int, batch_size: int, device, dtype):
    from common.models import torch

    return (torch.randn(batch_size, seq_len, shape.d_model, device=device, dtype=dtype),)


def random_decoder_inputs(shape, seq_len: int, batch_size: int, device, dtype):
    from common.models import torch

    query = torch.randn(batch_size, 1, shape.d_model, device=device, dtype=dtype)
    past_len = max(seq_len - 1, 0)
    if past_len == 0:
        return (query, None)
    past_kv = [
        (
            torch.randn(batch_size, shape.kv_heads, past_len, shape.head_dim, device=device, dtype=dtype),
            torch.randn(batch_size, shape.kv_heads, past_len, shape.head_dim, device=device, dtype=dtype),
        )
        for _ in range(shape.num_layers)
    ]
    return query, past_kv


def random_encoder_decoder_inputs(shape, seq_len: int, batch_size: int, device, dtype):
    from common.models import torch

    enc = torch.randn(batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    dec = torch.randn(batch_size, 1, shape.d_model, device=device, dtype=dtype)
    past_len = max(seq_len - 1, 0)
    if past_len == 0:
        return enc, dec, None
    past_kv = [
        (
            torch.randn(batch_size, shape.kv_heads, past_len, shape.head_dim, device=device, dtype=dtype),
            torch.randn(batch_size, shape.kv_heads, past_len, shape.head_dim, device=device, dtype=dtype),
        )
        for _ in range(shape.num_layers)
    ]
    return enc, dec, past_kv
