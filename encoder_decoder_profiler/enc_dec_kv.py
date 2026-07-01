from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.latency import LatencyRecorder
from common.models import EncoderDecoderModel, torch


def build_encoder_decoder_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = EncoderDecoderModel(shape, recorder, ffn_kind="gelu").to(device=device, dtype=dtype)
    model.recorder = recorder
    return model
