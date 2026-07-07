# Differences From The Old Transformer Latency Analysis

This document compares the old latency analysis in
`/Users/jinendramalekar/Downloads/LLM_profiler` with the current
`transformer_latency` repository.

The main point: the old analysis and the current analysis are not measuring the
same thing by default. The old baseline combines one full prefill pass plus 10
cached decode steps into one CSV/plot. The current main profiler separates
prefill and decode, supports more architectures, and uses a different synthetic
model path. For old-style Phi/GPT pie charts, use the reproduction path under
`reproduce/phi_baseline_like/`.

## Source Files Compared

Old analysis:

| Area | File |
|---|---|
| Baseline GPT/Phi-style profiler | `/Users/jinendramalekar/Downloads/LLM_profiler/llm_profiler/llm_kv.py` |
| Baseline sweep and plots | `/Users/jinendramalekar/Downloads/LLM_profiler/llm_profiler/run_and_plot.py` |
| Old model-family profiler | `/Users/jinendramalekar/Downloads/LLM_profiler/model_family_profiler/llm_kv.py` |
| Old model-family sweep | `/Users/jinendramalekar/Downloads/LLM_profiler/model_family_profiler/run_and_plot.py` |
| Old baseline CSVs | `/Users/jinendramalekar/Downloads/LLM_profiler/results/baseline/latency_seqlen_*.csv` |

Current analysis:

| Area | File |
|---|---|
| Shared profiled modules | `common/models.py` |
| Shared runner | `common/runner.py` |
| Shared latency recorder | `common/latency.py` |
| Shape presets | `common/config.py` |
| Encoder runner | `encoder_profiler/run_and_plot.py` |
| Decoder runner | `decoder_profiler/run_and_plot.py` |
| Encoder-decoder runner | `encoder_decoder_profiler/run_and_plot.py` |
| Current model-family runner | `model_family_profiler/run_and_plot.py` |
| Old-compatible reproduction runner | `reproduce/run_phi_reproduction.py` |

## High-Level Difference

| Topic | Old analysis | Current analysis |
|---|---|---|
| Main target | Decoder-only GPT/Phi-style latency analysis, plus decoder-only model-family comparison. | Encoder-only, decoder-only, encoder-decoder, model-family, attention microbench, and old-style reproduction. |
| Default benchmark unit | One full prefill over `L` tokens, then 10 cached one-token decode steps, combined into one latency report. | Prefill and cached decode are measured as separate phases in the main profiler. |
| Input type | Token IDs through token embeddings and position embeddings. | Random hidden states in the main profiler. The old-compatible reproduction uses token IDs and embeddings. |
| Default dtype on CUDA | CUDA autocast float16 in generation. | Main profiler defaults to `float32` unless `--dtype float16` or `--dtype bfloat16` is passed. Reproduction uses CUDA autocast float16. |
| Repeats | Single measured run in the old baseline scripts. | Warmups and repeated runs are configurable; defaults are `--warmups 2 --repeats 5`. |
| Output format | Mostly compact CSVs: `name, pretty_name, count, total, avg, min, max`. Timings are seconds. | Metadata-rich CSVs with architecture, family, shape, `d`, `h`, layers, phase, dtype, device, `total_ms`, per-repeat and per-token metrics. |
| Plot interpretation | Old pies are combined prefill + 10 decode percentages. | Current figures are phase-specific: prefill and decode are plotted separately where relevant. |

## Measurement Phase Differences

### Old Baseline

The old baseline path runs:

1. Build a GPT-style decoder model.
2. Generate a random token sequence of length `L`.
3. Run one full prefill pass over those `L` tokens with KV cache enabled.
4. Generate 10 cached decode tokens one at a time.
5. Save one combined component report for prefill plus decode.

The old baseline percentages are therefore a mixture of:

- full-context prefill costs, where attention score tensors are `L x L`;
- cached decode costs, where each generated token attends to the existing cache;
- output-head work over the full prefill sequence plus one token per decode step.

### Current Main Profiler

The current main decoder runner separates the same conceptual workload into:

- `prefill`: full causal forward over `L` tokens;
- `decode`: first fills the `L`-token cache without recording, then records
  cached one-token decode averaged over `--decode-tokens`, default `10`.

This means a current `decode` CSV should not be compared directly to an old
combined pie chart. The current `prefill` and `decode` rows need to be combined
only if the goal is to reproduce the older combined view.

### Old-Compatible Reproduction

`reproduce/run_phi_reproduction.py` exists to recreate the old combined view.
It uses:

- token IDs;
- token and position embeddings;
- fused QKV projection;
- one prefill plus 10 cached decode tokens;
- CUDA autocast float16 on AGX;
- old-style component labels in `reproduce/phi_baseline_like/old_aligned/`.

## Model Implementation Differences

| Component | Old baseline GPT path | Current main profiler | Old-compatible reproduction |
|---|---|---|---|
| QKV projection | Fused `nn.Linear(d, 3d)`, recorded as `attn.qkv_projection`. | Split `q_proj`, `k_proj`, `v_proj`, recorded separately. | Fused `nn.Linear(d, 3d)`, recorded as `attn.qkv_projection`. |
| QKV bias | `qkv_bias=False`. | Split projections use `bias=False`. | `qkv_bias=False`. |
| Attention output projection bias | Old baseline `nn.Linear(d, d)` default bias enabled. | Current main `out_proj` uses `bias=False`. | Bias enabled to match old baseline. |
| Input pathway | Token embedding plus position embedding. | Random hidden states, no token/position embedding cost. | Token embedding plus position embedding. |
| Final norm timing | Final norm exists but is not timed in the old baseline CSVs. | `model.final_norm` is timed. | Final norm exists but is not included in old-aligned timing. |
| Output head timing | Full hidden sequence is projected in prefill; one token is projected during each decode step. | Current decoder runner records only the last-token output head when `full_logits=False`. | Full hidden sequence is projected in prefill; one token is projected during each decode step. |
| Causal mask timing | The old baseline records the mask only when there is an actual future-token mask to apply. One-token decode with no future positions is not recorded as mask work. | Current main causal attention records the causal-mask scope whenever causal attention runs, including decode scopes. | Matches the old baseline behavior. |
| KV cache concat | Timed during cached decode. | Timed during cached decode. | Timed during cached decode. |

## Attention And Component Label Differences

Old baseline component keys:

| Old key | Meaning |
|---|---|
| `attn.qkv_projection` | Fused Q/K/V projection. |
| `attn.matmul_qk` | Query-key score matmul. |
| `attn.apply_causal_mask` | Causal mask application. |
| `attn.softmax` | Attention softmax. |
| `attn.weighted_sum` | Attention probabilities times values. |
| `attn.out_projection` | Attention output projection. |
| `attn.cache_concat` | KV-cache concatenation during decode. |
| `block.norm1`, `block.norm2` | Pre-attention and pre-FFN normalization. |
| `ff.linear1`, `ff.gelu`, `ff.linear2` | GELU feed-forward network. |
| `model.output_head` | Vocabulary output projection. |

Current main profiler adds or changes:

| Current key | Difference |
|---|---|
| `attn.q_proj`, `attn.k_proj`, `attn.v_proj` | Split Q/K/V instead of fused QKV. |
| `attn.gqa_expand` | Explicit grouped-query KV expansion, if KV heads differ from query heads. |
| `attn.apply_padding_mask` | Encoder padding-mask scope, if a padding mask is supplied. |
| `cross_attn.*` | Encoder-decoder cross-attention scopes. |
| `block.norm3` | Pre-cross-attention normalization in encoder-decoder models. |
| `ff.w1_gate`, `ff.w2_up`, `ff.swiglu_act`, `ff.w3_down` | SwiGLU FFN scopes for LLaMA-style models. |
| `model.final_norm` | Final model normalization is timed in the main profiler. |

## Model-Family Coverage Differences

| Topic | Old model-family profiler | Current model-family profiler |
|---|---|---|
| Families | `gpt`, `phi`, `llama`, `gemma`, `opt`. | `bert`, `gpt`, `llama`, `t5`, `bart`. |
| Architecture type | All compared models are decoder-only. | Families span encoder-only, decoder-only, and encoder-decoder. |
| Dimension control | Shared `d=3072`, `h=32`, `layers=32` so model differences mostly come from architecture choices. | Realistic shape presets vary `d`, `h`, layers, and FFN size by family. |
| GQA | Old LLaMA/Gemma use `num_kv_heads=8` with GQA expansion. | Current default `llama_7b` shape uses equal query and KV heads unless overridden with `--num-kv-heads`. |
| OPT extra projections | Old model-family path includes `model.proj_in` and `model.proj_out` for OPT. | Current family set does not include OPT by default. |
| Cross-attention | Not present. | Present for T5/BART-style encoder-decoder models. |

## Shape And Sweep Differences

Old baseline shape:

| Field | Value |
|---|---:|
| `vocab_size` | 30257 |
| `context_length` | 16348 |
| `d_model` / `emb_dim` | 3072 |
| `num_heads` | 32 |
| `num_layers` | 32 |
| `d_ff` | 12288 |
| decode tokens | 10 |

Current shape presets include:

| Shape | `d` | `h` | Layers | FFN dim |
|---|---:|---:|---:|---:|
| `bert_base` | 768 | 12 | 12 | 3072 |
| `bert_large` | 1024 | 16 | 24 | 4096 |
| `gpt2_medium` | 1024 | 16 | 24 | 4096 |
| `gpt3_2p7b` | 2560 | 32 | 32 | 10240 |
| `llama_7b` | 4096 | 32 | 32 | 11008 |
| `t5_base` | 768 | 12 | 12 | 3072 |
| `t5_large` | 1024 | 16 | 24 | 4096 |
| `bart_large` | 1024 | 16 | 12 | 4096 |

Default sequence-length handling also changed:

| Old scripts | Current scripts |
|---|---|
| Baseline script lists `512, 1024, 2048, 4096, 8192`; saved old baseline results also include `256`. | Presets are `quick = 128,256`, `standard = 512,1024,2048,4096`, and `long = 512,1024,2048,4096,8192`. |

## CSV Schema Differences

Old saved baseline CSV header:

```text
name,pretty_name,count,total,avg,min,max
```

Current main CSVs include metadata and millisecond timing columns:

```text
architecture,model_family,shape_name,d_model,num_heads,num_kv_heads,head_dim,
num_layers,d_ff,batch_size,seq_len,encoder_seq_len,decoder_seq_len,phase,
phase_tokens,timed_repeats,dtype,device,operation_key,count,avg_ms,min_ms,
max_ms,std_ms,total_ms,avg_total_ms_per_repeat,avg_total_ms_per_token,pct_total
```

The old-compatible reproduction CSVs include both old-friendly columns
(`name`, `pretty_name`, `total`, `avg`) and current metadata columns
(`phase`, `d_model`, `num_heads`, `total_ms`, `pct_total`, etc.).

## Plot Differences

Old plots:

- per-sequence bar charts with average latency and min/max;
- grouped pie charts by broad component group;
- cross-sequence average and percent line plots;
- model-family comparison bars for decoder-only families.

Current plots:

- per-run bar plots are phase-specific;
- decoder and encoder-decoder plots are split into `prefill` and `decode`;
- root `figures.py` keeps encoder, decoder, and encoder-decoder figures in
  separate folders;
- component-share plots quantify detailed components directly instead of
  relying on a large `Other` bucket;
- cross-attention components are separately labeled where they exist.

## Why The Old 8192 Softmax Did Not Match The Earlier Current Plot

The earlier incorrect reproduction used the current main profiler path. That
path used split Q/K/V projections and a different dtype/measurement scope, so
`pie_chart_8192.png` showed softmax around `7.9%`.

The corrected reproduction uses the old-compatible path. The AGX rerun now
matches the old combined workload much more closely:

| Sequence length | Old total (s) | Reproduced total (s) | Old softmax share | Reproduced softmax share |
|---:|---:|---:|---:|---:|
| 256 | 8.7867 | 6.0809 | 1.65% | 1.64% |
| 512 | 5.4042 | 4.7604 | 1.49% | 4.86% |
| 1024 | 5.6054 | 5.7693 | 3.07% | 11.88% |
| 2048 | 8.0357 | 7.6047 | 17.40% | 31.88% |
| 4096 | 13.7426 | 14.2076 | 21.07% | 33.54% |
| 8192 | 35.1445 | 34.8562 | 25.24% | 30.02% |

The important fix is that the 8192 result is no longer in the wrong regime:
softmax is again one of the dominant components, and total runtime is very
close to the old baseline. Remaining percentage differences are expected from
rerun variance, runtime/library differences, and the fact that the current
reproduction is a clean reimplementation of the old path rather than the exact
old source file.

## Which File To Use For Which Comparison

| Goal | Use |
|---|---|
| Recreate the old combined Phi/GPT pie charts | `reproduce/run_phi_reproduction.py` and `reproduce/phi_baseline_like/old_aligned/` |
| Compare current decoder prefill vs decode | `decoder_profiler/latency_results/decoder/*.csv` |
| Compare encoder-only behavior | `encoder_profiler/latency_results/encoder/*.csv` |
| Compare encoder-decoder self-attention and cross-attention | `encoder_decoder_profiler/latency_results/encoder_decoder/*.csv` |
| Compare the current architecture families | `model_family_profiler/latency_results/` |
| Understand component labels in current figures | `component.md` |

## Interpretation Notes

- Old combined percentages should not be compared directly against current
  phase-specific prefill or decode percentages.
- Old fused `attn.qkv_projection` should not be compared one-to-one against a
  single current split projection. Compare it against
  `attn.q_proj + attn.k_proj + attn.v_proj` if using current main CSVs.
- Old baseline output-head timing includes full prefill logits. Current main
  decoder output-head timing uses last-token logits when `full_logits=False`.
- Current encoder-decoder results contain `cross_attn.*` components that do
  not exist in the old decoder-only analysis.
- Current default family shapes vary `d`, `h`, layers, and FFN size together.
  They are realistic model-style presets, not a controlled architecture-only
  comparison with identical dimensions.
