from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.runner import add_common_args, random_encoder_decoder_inputs, run_sweep
from enc_dec_kv import build_encoder_decoder_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile encoder-decoder Transformer latency across d, L, and h."
    )
    add_common_args(parser, "encoder_decoder")
    args = parser.parse_args()
    run_sweep(
        args,
        "encoder_decoder",
        "encoder_decoder",
        build_encoder_decoder_model,
        random_encoder_decoder_inputs,
        encoder_decoder=True,
    )


if __name__ == "__main__":
    main()
