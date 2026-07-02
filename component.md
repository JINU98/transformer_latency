# Model-Family Figure Components

The model-family figures group profiled operation keys into human-readable
components. Each color in the stacked bars and pie charts corresponds to one
component below.

The figures are phase-specific:

- `prefill`: full context pass over `L` tokens.
- `decode`: after filling the `L`-token KV cache, averaged per generated token
  over 10 decode tokens.
- `attention_microbench`: isolated self-attention or cross-attention modules,
  without the full model stack.

## Component List

| Component | What it means | When it happens |
|---|---|---|
| `Self QKV Projection` | Linear projections that create self-attention queries, keys, and values. | In every self-attention block: encoder, decoder prefill, decoder decode, and encoder-decoder decoder self-attention. |
| `Self QKT MatMul` | Matrix multiply between self-attention queries and keys to form attention scores. | After Q/K projection in self-attention. |
| `Self Causal Mask` | Mask that prevents a decoder token from attending to future tokens. | Decoder and encoder-decoder decoder self-attention, in both prefill and decode. |
| `Self Softmax` | Softmax over self-attention scores. | After self-attention scoring and masking. |
| `Self Attention Weighted Sum` | Matrix multiply between self-attention probabilities and values. | After self-attention softmax. |
| `Self Output Projection` | Final linear projection after self-attention output. | End of each self-attention module. |
| `Self KV Cache Concat` | Concatenates cached past keys/values with the new token's keys/values. | Decode only, after the `L`-token prefill has populated the KV cache. |
| `Cross QKV Projection` | Decoder query projection plus encoder-state key/value projections. | Encoder-decoder models only, inside decoder cross-attention. |
| `Cross QKT MatMul` | Decoder queries attend over encoder keys. | Encoder-decoder cross-attention. |
| `Cross Softmax` | Softmax over encoder-source attention scores. | Encoder-decoder cross-attention. |
| `Cross Attention Weighted Sum` | Cross-attention probabilities multiplied by encoder values. | Encoder-decoder cross-attention. |
| `Cross Output Projection` | Final projection after cross-attention output. | Encoder-decoder cross-attention. |
| `LayerNorm (Pre-Attention)` | Normalization before self-attention. RMSNorm is grouped here for LLaMA-style models. | Start of each encoder or decoder block before self-attention. |
| `LayerNorm (Pre-FFN)` | Normalization before the feed-forward network. | After self-attention residual, before FFN. |
| `LayerNorm (Pre-Cross-Attention)` | Normalization before cross-attention. | Encoder-decoder decoder blocks only. |
| `FFN Linear 1` | First FFN projection. For SwiGLU, this includes gate/up projections. | After attention and pre-FFN normalization. |
| `FFN Activation` | GELU or SwiGLU activation. | Middle of the FFN. |
| `FFN Linear 2` | Down/output FFN projection back to `d_model`. | End of the FFN. |
| `Final Norm` | Final model-level normalization after all layers. | End of encoder, decoder, or encoder-decoder decoder stack. |
| `Output Head Projection` | Projection from hidden state to vocabulary logits. | Decoder and encoder-decoder models, used for next-token output. |

## Not Present In Current Figures

These labels are supported by the plotting code but do not appear in the
current regenerated model-family figures:

| Component | Why it is absent |
|---|---|
| `Self Padding Mask` | Only appears if an encoder padding mask is supplied; the current synthetic sweeps do not use one. |
| `Self GQA KV Expand` | Only appears when query heads differ from KV heads; the current selected shapes use equal query and KV head counts. |
| `Unmapped` | Appears only if a profiler operation key has no component label. The current CSVs are fully mapped. |

## Figure Locations

The root summary figures are organized by architecture and phase:

```text
figures/encoder/prefill/
figures/decoder/prefill/
figures/decoder/decode/
figures/encoder_decoder/prefill/
figures/encoder_decoder/decode/
```

The per-run bar plots are stored beside the CSVs under each profiler's
`latency_results/` directory. Decoder-style runs have separate `_prefill` and
`_decode` bar plots.
