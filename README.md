# Transformer Architecture Profiler

Fine-grained, operation-level latency profiler for **encoder-only**,
**decoder-only**, and **encoder-decoder** transformer models. Every major
sub-operation (Q/K/V projection, self-attention matmul, cross-attention
matmul, softmax, masking, KV-cache concat, FFN layers, etc.) is individually
timed using a lightweight context-manager wrapper, with optional CUDA
synchronisation so that GPU kernel time is correctly attributed.

Because all three architecture families are profiled with the same harness
and the same hidden dimension / layer count / head count, latency
differences observed across runs reflect **architecture alone** — not
parameter count or hardware noise.

---

## Repository Structure

```
Transformer_Profiler/
│
├── figures.py                          # ★ Reproduce all paper figures from CSVs
│
├── encoder_profiler/                    # Encoder-only profiler (BERT-style)
│   ├── enc_kv.py                        # Encoder model + LatencyRecorder
│   ├── run_and_plot.py                  # Sequence-length sweep driver + plots
│   └── plot_from_csv.py                 # Regenerate comparison plots from saved CSVs
│
├── decoder_profiler/                    # Decoder-only profiler (GPT-style)
│   ├── dec_kv.py                        # Decoder model + LatencyRecorder
│   ├── run_and_plot.py                  # Sequence-length sweep driver + plots
│   └── plot_from_csv.py                 # Regenerate comparison plots from saved CSVs
│
├── encoder_decoder_profiler/            # Encoder-decoder profiler (T5/BART-style)
│   ├── enc_dec_kv.py                    # Encoder-decoder model + LatencyRecorder
│   ├── run_and_plot.py                  # Sequence-length sweep driver + plots
│   └── plot_from_csv.py                 # Regenerate comparison plots from saved CSVs
│
├── model_family_profiler/               # Cross-architecture, multi-model profiler
│   ├── model_kv.py                      # BERT / GPT / LLaMA / T5 / BART models
│   ├── run_and_plot.py                  # Model × sequence-length sweep driver + plots
│   ├── heatmap.py                       # Cross-model latency heatmap
│   └── simple_attn_bench.py             # Micro-benchmark for self- and cross-attention
│
├── results/
│   ├── encoder/                         # Pre-generated CSVs — encoder-only sweep
│   │   └── latency_seqlen_*.csv
│   ├── decoder/                         # Pre-generated CSVs — decoder-only sweep
│   │   └── latency_seqlen_*.csv
│   ├── encoder_decoder/                 # Pre-generated CSVs — encoder-decoder sweep
│   │   └── latency_seqlen_*.csv
│   └── model_family/                    # Pre-generated CSVs — cross-architecture sweep
│       ├── bert/latency_seq*.csv
│       ├── gpt/latency_seq*.csv
│       ├── llama/latency_seq*.csv
│       ├── t5/latency_seq*.csv
│       └── bart/latency_seq*.csv
│
├── figures/                             # Generated output (created by figures.py)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Profiled Operations

Operation coverage differs slightly by architecture, since encoders have no
causal mask or KV cache, and encoder-decoders add a cross-attention block.

### Shared operations (all architectures)

| Key | Label |
|-----|-------|
| `attn.q_proj` | Q Projection |
| `attn.k_proj` | K Projection |
| `attn.v_proj` | V Projection |
| `attn.gqa_expand` | GQA Expand *(GQA models only)* |
| `attn.matmul_qk` | QKᵀ MatMul |
| `attn.softmax` | Softmax |
| `attn.weighted_sum` | Attention Weighted Sum |
| `attn.out_projection` | Output Projection |
| `block.norm1` | LayerNorm / RMSNorm (Pre-Attention) |
| `block.norm2` | LayerNorm / RMSNorm (Pre-FFN) |
| `ff.linear1` / `ff.w1_gate` | FFN Linear 1 / SwiGLU Gate |
| `ff.gelu` / `ff.swiglu_act` | GELU / SwiGLU Activation |
| `ff.linear2` / `ff.w3_down` | FFN Linear 2 / SwiGLU Down |
| `model.output_head` | Output Head Projection |

### Decoder-only and decoder-side of encoder-decoder

| Key | Label |
|-----|-------|
| `attn.apply_causal_mask` | Causal Mask |
| `attn.cache_concat` | KV Cache Concat |

### Encoder-decoder only (cross-attention block)

| Key | Label |
|-----|-------|
| `cross_attn.q_proj` | Cross-Attn Q Projection (decoder states) |
| `cross_attn.k_proj` | Cross-Attn K Projection (encoder states) |
| `cross_attn.v_proj` | Cross-Attn V Projection (encoder states) |
| `cross_attn.matmul_qk` | Cross-Attn QKᵀ MatMul |
| `cross_attn.softmax` | Cross-Attn Softmax |
| `cross_attn.weighted_sum` | Cross-Attn Weighted Sum |
| `cross_attn.out_projection` | Cross-Attn Output Projection |
| `block.norm3` | LayerNorm / RMSNorm (Pre-Cross-Attn) |

### Encoder-only only

| Key | Label |
|-----|-------|
| `attn.apply_padding_mask` | Padding Mask *(no causal mask)* |

---

## Supported Model Families

| Family | Architecture | Attention | FFN | Extra Blocks |
|--------|--------------|-----------|-----|--------------|
| BERT / RoBERTa | Encoder-only | MHA, bidirectional | GELU | — |
| GPT | Decoder-only | MHA, causal | GELU | — |
| LLaMA | Decoder-only | GQA, causal | SwiGLU | — |
| T5 | Encoder-Decoder | MHA + cross-attn | ReLU/GEGLU | Cross-attention |
| BART | Encoder-Decoder | MHA + cross-attn | GELU | Cross-attention |

All families share the same hidden dimension (3072), layer count (32),
and query heads (32) per stack, so latency differences reflect
architecture alone.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/Transformer_Profiler.git
cd Transformer_Profiler

# Install dependencies (Python ≥ 3.10 recommended)
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The profiler falls back to CPU
automatically but runtimes will be substantially longer.

---

## Quick Start

### 1 — Encoder-only sweep (BERT-style)

Profiles a single bidirectional encoder across sequence lengths
[512, 1024, 2048, 4096, 8192]. No causal mask or KV cache is profiled.

```bash
cd encoder_profiler
python run_and_plot.py
```

**Outputs** (in `encoder_profiler/latency_results/`):
- `latency_seqlen_<N>.csv` — per-operation stats for each sequence length
- `bar_seqlen_<N>.png` — log-scale bar chart with min/max error bars
- `pie_seqlen_<N>.png` — latency share by transformer component group
- `comparison_avg_latency.png` — avg latency vs sequence length
- `comparison_pct_latency.png` — % share vs sequence length

### 2 — Decoder-only sweep (GPT-style)

Profiles a single causal decoder with KV-cache across the same sequence
lengths.

```bash
cd decoder_profiler
python run_and_plot.py
```

**Outputs** (in `decoder_profiler/latency_results/`): same file set as above,
plus causal-mask and cache-concat timings.

### 3 — Encoder-decoder sweep (T5/BART-style)

Profiles a full encoder-decoder stack, including cross-attention between
decoder queries and encoder key/value states.

```bash
cd encoder_decoder_profiler
python run_and_plot.py
```

**Outputs** (in `encoder_decoder_profiler/latency_results/`): same file set
as above, plus a `cross_attn_share_seq<N>.png` chart isolating
cross-attention's contribution to total latency.

### 4 — Cross-architecture model-family sweep

Profiles BERT, GPT, LLaMA, T5, and BART across the same sequence lengths so
architectures can be compared directly.

```bash
cd model_family_profiler
python run_and_plot.py
```

**Outputs** (in `model_family_profiler/latency_results/`):
- `<model>/latency_seq<N>.csv` — per-operation stats for each model/seq pair
- `<model>/bar_seq<N>.png`, `<model>/pie_seq<N>.png` — per-run charts
- `<model>/comparison_avg_latency.png` — cross-sequence comparison per model
- `<model>/comparison_pct_latency.png` — % share comparison per model
- `model_comparison_seq<N>.png` — total latency bar chart across all five models
- `architecture_comparison_seq<N>.png` — grouped bar chart by architecture class (encoder-only vs decoder-only vs encoder-decoder)

### 5 — Heatmap

```bash
cd model_family_profiler
python heatmap.py          # requires model family CSVs to already exist
```

### 6 — Regenerate plots from existing CSVs

```bash
cd decoder_profiler
python plot_from_csv.py    # reads latency_results/*.csv, writes comparison PNGs
```

### 7 — Attention micro-benchmark

Benchmarks self-attention and cross-attention (where applicable) in
isolation, independent of the full model forward pass.

```bash
cd model_family_profiler
python simple_attn_bench.py
```

---

## Reproducing Paper Figures

All paper figures are generated by a single script from the pre-computed
CSVs already checked into `results/`:

```bash
# From the repo root — generates all paper figures into figures/
python figures.py
```

| Output file | Description |
|---|---|
| `figures/pie_<N>.png` | Per-sequence-length runtime pie chart |
| `figures/shared_legend.png` | Standalone legend for the pie charts |
| `figures/runtime_scaling.png` | % runtime vs sequence length line plot |
| `figures/heatmap.png` | Cross-model latency heatmap |
| `figures/linegraph_all_seqlens.png` | Per-model line graphs, grouped by architecture |
| `figures/latency_grouped_bar.png` | Grouped stacked bar chart |
| `figures/cross_attn_overhead.png` | Cross-attention overhead, encoder-decoder models only |

Individual figures can also be generated selectively:

```python
from figures import pie_charts_baseline, runtime_scaling, heatmap
from figures import linegraph_all_seqlens, latency_grouped_bar, cross_attn_overhead

runtime_scaling()          # just the scaling line plot
heatmap(seq_len=8192)      # heatmap at a different sequence length
cross_attn_overhead()      # cross-attention cost isolated for T5/BART
```

To fully re-run the profiler from scratch on your own hardware and then
regenerate figures, run the sweep scripts first:

```bash
cd encoder_profiler         && python run_and_plot.py   # encoder-only
cd ../decoder_profiler      && python run_and_plot.py   # decoder-only
cd ../encoder_decoder_profiler && python run_and_plot.py # encoder-decoder
cd ../model_family_profiler && python run_and_plot.py   # cross-architecture
cd .. && python figures.py                               # figures
```

---

## Customisation

### Change sequence lengths

Edit `SEQ_LENGTHS` at the top of any `run_and_plot.py`:

```python
SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
```

### Change model families

Edit `MODEL_TYPES` in `model_family_profiler/run_and_plot.py`:

```python
MODEL_TYPES = ["bert", "gpt", "t5"]
```

### Run a single configuration

```python
import model_kv
model_kv.MODEL_TYPE = "t5"
model_kv.SEQ_LEN    = 4096
model_kv.main()
print(model_kv.LATENCY.report())
```

### Enable / disable encoder-decoder cross-attention profiling

For encoder-decoder models, cross-attention timing can be toggled off if
you only care about self-attention cost:

```python
enc_dec_kv.PROFILE_CROSS_ATTENTION = False
```

---

## Notes on Architecture-Specific Behaviour

- **Encoder-only** models process the full sequence bidirectionally in a
  single forward pass — there is no autoregressive decoding loop and no
  KV cache to profile. Latency scales with sequence length but each run is
  a single timed forward pass.
- **Decoder-only** models are profiled in an autoregressive loop with
  KV-cache growth, so `attn.cache_concat` and `attn.apply_causal_mask`
  become significant contributors at longer sequence lengths.
- **Encoder-decoder** models are profiled as an encoder forward pass
  followed by an autoregressive decoder loop; cross-attention key/value
  projections are computed once against the encoder output and reused
  across decoding steps, which the profiler accounts for separately from
  the decoder's own self-attention.

---

## License

This project is released for academic reproducibility. See `LICENSE` for
details.
