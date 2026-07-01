from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.latency import LatencyRecorder
from common.models import EncoderModel, torch


def build_encoder_model(shape, device, dtype, sync_cuda: bool = True):
    recorder = LatencyRecorder(sync_cuda=sync_cuda, torch_module=torch)
    model = EncoderModel(shape, recorder).to(device=device, dtype=dtype)
    model.recorder = recorder
    return model
