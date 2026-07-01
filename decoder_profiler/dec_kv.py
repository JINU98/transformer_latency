from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.latency import LatencyRecorder
from common.models import DecoderModel, torch


def build_decoder_model(shape, device, dtype, sync_cuda: bool = True):
    is_llama = "llama" in shape.name.lower()
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = DecoderModel(
        shape,
        recorder,
        ffn_kind="swiglu" if is_llama else "gelu",
        norm_kind="rmsnorm" if is_llama else "layernorm",
    ).to(device=device, dtype=dtype)
    model.recorder = recorder
    return model
