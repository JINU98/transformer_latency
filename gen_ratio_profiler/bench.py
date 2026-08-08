from __future__ import annotations

from common.runner import benchmark_phase
from decoder_profiler.dec_kv import build_decoder_model
from encoder_decoder_profiler.enc_dec_kv import build_encoder_decoder_model


def benchmark_decoder_prefill_and_decode_step(
    shape, seq_len, batch_size, warmups, repeats, device, dtype, torch_module
):
    """Decoder-only: one full causal prefill over L tokens, then exactly one
    cached decode step at context length L."""
    prefill_input = torch_module.randn(batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decode_input = torch_module.randn(batch_size, 1, shape.d_model, device=device, dtype=dtype)

    def make_model():
        return build_decoder_model(shape, device=device, dtype=dtype, sync_cuda=(device.type == "cuda"))

    def run_prefill(model):
        return model(prefill_input, None, False)

    _, prefill_recorder = benchmark_phase(
        make_model=make_model, run_once=run_prefill, warmups=warmups, repeats=repeats, torch_module=torch_module
    )

    def run_one_decode_step(model):
        recorder = getattr(model, "recorder")
        with recorder.disabled():
            _, past_kv = model(prefill_input, None, False)
        return model(decode_input, past_kv, False)

    _, decode_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_one_decode_step,
        warmups=warmups,
        repeats=repeats,
        torch_module=torch_module,
    )
    return prefill_recorder, decode_recorder


def benchmark_encoder_decoder_prefill_and_decode_step(
    shape, seq_len, batch_size, warmups, repeats, device, dtype, torch_module
):
    """Encoder-decoder: one full pass (encode source of length L, then a
    causal teacher-forced decoder pass with cross-attention over L decoder
    tokens) counts as prefill. Decode is exactly one cached decoder step
    (self-attention over the growing cache + cross-attention to the fixed,
    already-encoded source) at decoder context length L."""
    encoder_x = torch_module.randn(batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decoder_prefill = torch_module.randn(batch_size, seq_len, shape.d_model, device=device, dtype=dtype)
    decode_input = torch_module.randn(batch_size, 1, shape.d_model, device=device, dtype=dtype)

    def make_model():
        return build_encoder_decoder_model(shape, device=device, dtype=dtype, sync_cuda=(device.type == "cuda"))

    def run_prefill(model):
        return model(encoder_x, decoder_prefill, None, False)

    _, prefill_recorder = benchmark_phase(
        make_model=make_model, run_once=run_prefill, warmups=warmups, repeats=repeats, torch_module=torch_module
    )

    def run_one_decode_step(model):
        recorder = getattr(model, "recorder")
        with recorder.disabled():
            encoder_states = model.encode(encoder_x)
            _, past_kv = model.decode(decoder_prefill, encoder_states, None, False)
        return model.decode(decode_input, encoder_states, past_kv, False)

    _, decode_recorder = benchmark_phase(
        make_model=make_model,
        run_once=run_one_decode_step,
        warmups=warmups,
        repeats=repeats,
        torch_module=torch_module,
    )
    return prefill_recorder, decode_recorder


# Architecture -> (benchmark fn, attention-buffer multiplier used for --max-attn-gb
# skip logic, matching the multiplier used for the same architecture elsewhere in
# this repo, e.g. common/runner.py's run_sweep), default representative shape,
# and the valid shape names for that architecture.
ARCHITECTURES = {
    "decoder": {
        "benchmark_fn": benchmark_decoder_prefill_and_decode_step,
        "attn_multiplier": 3.0,
        "default_shape": "gpt3_2p7b",
        "shape_names": ["gpt2_medium", "gpt3_2p7b", "llama_3b", "llama_7b"],
    },
    "encoder_decoder": {
        "benchmark_fn": benchmark_encoder_decoder_prefill_and_decode_step,
        "attn_multiplier": 5.0,
        "default_shape": "bart_large",
        "shape_names": ["t5_base", "t5_large", "t5_3b", "bart_large"],
    },
}
