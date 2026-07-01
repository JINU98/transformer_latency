from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class ModelShape:
    name: str
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    num_kv_heads: int | None = None
    note: str = ""

    @property
    def head_dim(self) -> int:
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"{self.name}: d_model must be divisible by num_heads")
        return self.d_model // self.num_heads

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_heads


# Close-to-real dimensions. These are not checkpoint loaders; they are synthetic
# model shapes chosen to mirror common public model families.
MODEL_SHAPES: dict[str, ModelShape] = {
    "tiny_debug": ModelShape("tiny_debug", 256, 4, 2, 1024, note="fast CPU smoke test"),
    "bert_base": ModelShape("bert_base", 768, 12, 12, 3072, note="BERT-base style"),
    "bert_large": ModelShape("bert_large", 1024, 16, 24, 4096, note="BERT-large style"),
    "gpt2_medium": ModelShape("gpt2_medium", 1024, 16, 24, 4096, note="GPT-2 medium style"),
    "gpt3_2p7b": ModelShape("gpt3_2p7b", 2560, 32, 32, 10240, note="GPT-3 2.7B style"),
    "llama_7b": ModelShape("llama_7b", 4096, 32, 32, 11008, num_kv_heads=32, note="LLaMA-7B style"),
    "llama2_70b_gqa": ModelShape("llama2_70b_gqa", 8192, 64, 80, 28672, num_kv_heads=8, note="LLaMA-2-70B GQA style"),
    "t5_base": ModelShape("t5_base", 768, 12, 12, 3072, note="T5-base style"),
    "t5_large": ModelShape("t5_large", 1024, 16, 24, 4096, note="T5-large style"),
    "bart_large": ModelShape("bart_large", 1024, 16, 12, 4096, note="BART-large style"),
}


ARCH_DEFAULT_SHAPES: dict[str, list[str]] = {
    "encoder": ["bert_base", "bert_large"],
    "decoder": ["gpt2_medium", "gpt3_2p7b", "llama_7b"],
    "encoder_decoder": ["t5_base", "t5_large", "bart_large"],
    "model_family": ["bert_base", "gpt2_medium", "llama_7b", "t5_base", "bart_large"],
}


SEQ_LEN_PRESETS: dict[str, list[int]] = {
    "quick": [128, 256],
    "standard": [512, 1024, 2048, 4096],
    "long": [512, 1024, 2048, 4096, 8192],
}


def parse_int_list(value: str | None, fallback: list[int]) -> list[int]:
    if not value:
        return fallback
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_shape_names(value: str | None, architecture: str) -> list[str]:
    if value:
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = ARCH_DEFAULT_SHAPES[architecture]
    unknown = [name for name in names if name not in MODEL_SHAPES]
    if unknown:
        raise ValueError(f"Unknown model shape(s): {', '.join(unknown)}")
    return names


def make_shape_overrides(
    base: ModelShape,
    d_model: int | None,
    num_heads: int | None,
    num_layers: int | None,
    d_ff: int | None,
    num_kv_heads: int | None,
) -> ModelShape:
    if not any(v is not None for v in [d_model, num_heads, num_layers, d_ff, num_kv_heads]):
        return base
    d_model = d_model or base.d_model
    num_heads = num_heads or base.num_heads
    d_ff = d_ff or max(4 * d_model, base.d_ff if d_model == base.d_model else 0)
    return replace(
        base,
        name=f"custom_d{d_model}_h{num_heads}_layers{num_layers or base.num_layers}",
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers or base.num_layers,
        d_ff=d_ff,
        num_kv_heads=num_kv_heads if num_kv_heads is not None else base.num_kv_heads,
        note="custom override",
    )


def estimate_attention_gb(
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    dtype_bytes: int,
    multiplier: float = 3.0,
) -> float:
    # Scores, probabilities, and temporary buffers dominate full attention.
    return batch_size * num_heads * query_len * key_len * dtype_bytes * multiplier / (1024**3)


def dtype_bytes(dtype_name: str) -> int:
    if dtype_name in {"float16", "bfloat16", "fp16", "bf16"}:
        return 2
    return 4


def iter_shapes(names: Iterable[str]) -> list[ModelShape]:
    return [MODEL_SHAPES[name] for name in names]
