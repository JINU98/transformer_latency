from __future__ import annotations

from pathlib import Path
import argparse
import copy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.runner import (
    add_common_args,
    random_decoder_inputs,
    random_encoder_decoder_inputs,
    random_hidden_inputs,
    run_sweep,
)
from model_kv import FAMILY_ARCHITECTURE, FAMILY_DEFAULT_SHAPES, MODEL_BUILDERS


def parse_families(value: str) -> list[str]:
    families = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [family for family in families if family not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown model family/families: {', '.join(unknown)}")
    return families


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile BERT, GPT, LLaMA, T5, and BART style models across d, L, and h."
    )
    add_common_args(parser, "model_family")
    parser.add_argument("--families", default="bert,gpt,llama,t5,bart")
    args = parser.parse_args()

    all_written = []
    for family in parse_families(args.families):
        architecture = FAMILY_ARCHITECTURE[family]
        local_args = copy.copy(args)
        if args.shape_names is None:
            local_args.shape_names = FAMILY_DEFAULT_SHAPES[family]
        if architecture == "encoder_decoder":
            input_builder = random_encoder_decoder_inputs
        elif architecture == "decoder":
            input_builder = random_decoder_inputs
        else:
            input_builder = random_hidden_inputs
        written = run_sweep(
            local_args,
            architecture,
            family,
            MODEL_BUILDERS[family],
            input_builder,
            encoder_decoder=(architecture == "encoder_decoder"),
        )
        all_written.extend(written)

    if all_written:
        print(f"Wrote {len(all_written)} CSV file(s).")


if __name__ == "__main__":
    main()
