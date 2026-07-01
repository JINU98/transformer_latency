from __future__ import annotations

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.runner import add_common_args, random_decoder_inputs, run_sweep
from dec_kv import build_decoder_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile decoder-only Transformer latency across d, L, h, and KV heads."
    )
    add_common_args(parser, "decoder")
    args = parser.parse_args()
    run_sweep(args, "decoder", "decoder", build_decoder_model, random_decoder_inputs)


if __name__ == "__main__":
    main()
