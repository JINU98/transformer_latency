from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.runner import add_common_args, random_hidden_inputs, run_sweep
from enc_kv import build_encoder_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile encoder-only Transformer latency across d, L, and h."
    )
    add_common_args(parser, "encoder")
    args = parser.parse_args()
    run_sweep(args, "encoder", "encoder", build_encoder_model, random_hidden_inputs)


if __name__ == "__main__":
    main()
