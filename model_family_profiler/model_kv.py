from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.latency import LatencyRecorder
from common.models import DecoderModel, EncoderDecoderModel, EncoderModel, torch


FAMILY_DEFAULT_SHAPES = {
    "bert": "bert_base",
    "gpt": "gpt2_medium",
    "llama": "llama_7b",
    "t5": "t5_base",
    "bart": "bart_large",
}

FAMILY_ARCHITECTURE = {
    "bert": "encoder",
    "gpt": "decoder",
    "llama": "decoder",
    "t5": "encoder_decoder",
    "bart": "encoder_decoder",
}


def build_bert_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = EncoderModel(shape, recorder).to(device=device, dtype=dtype)
    model.recorder = recorder
    return model


def build_gpt_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = DecoderModel(shape, recorder, ffn_kind="gelu", norm_kind="layernorm").to(device=device, dtype=dtype)
    model.recorder = recorder
    return model


def build_llama_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = DecoderModel(shape, recorder, ffn_kind="swiglu", norm_kind="rmsnorm").to(device=device, dtype=dtype)
    model.recorder = recorder
    return model


def build_t5_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = EncoderDecoderModel(shape, recorder, ffn_kind="gelu").to(device=device, dtype=dtype)
    model.recorder = recorder
    return model


def build_bart_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = EncoderDecoderModel(shape, recorder, ffn_kind="gelu").to(device=device, dtype=dtype)
    model.recorder = recorder
    return model


MODEL_BUILDERS = {
    "bert": build_bert_model,
    "gpt": build_gpt_model,
    "llama": build_llama_model,
    "t5": build_t5_model,
    "bart": build_bart_model,
}
