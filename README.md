# Transformer Latency Profiler

Fine-grained latency profiling scripts for encoder-only, decoder-only, and
encoder-decoder Transformer models. The goal is to measure where latency is
spent inside each model class while sweeping realistic values for:

- `d`: model hidden dimension
- `L`: context length
- `h`: number of query attention heads

The models are synthetic, randomly initialized PyTorch modules. They do not
load pretrained checkpoints. Their dimensions are chosen to mirror common
public model families so the profiling plan stays close to real systems while
remaining easy to run and modify.

## Repository Structure

```text
transformer_latency/
├── common/
│   ├── config.py             # Realistic model-shape presets and sweep helpers
│   ├── io.py                 # CSV schema and operation grouping
│   ├── latency.py            # LatencyRecorder and benchmark loop
│   ├── models.py             # Profiled Transformer blocks
│   ├── plotting.py           # Shared plotting utilities
│   └── runner.py             # Shared CLI and sweep runner
├── encoder_profiler/
│   ├── enc_kv.py             # Encoder-only model builder
│   ├── run_and_plot.py       # Encoder sweep
│   └── plot_from_csv.py      # Rebuild encoder plots from CSVs
├── decoder_profiler/
│   ├── dec_kv.py             # Decoder-only model builder
│   ├── run_and_plot.py       # Decoder sweep with synthetic KV cache
│   └── plot_from_csv.py      # Rebuild decoder plots from CSVs
├── encoder_decoder_profiler/
│   ├── enc_dec_kv.py         # Encoder-decoder model builder
│   ├── run_and_plot.py       # Encoder-decoder sweep
│   └── plot_from_csv.py      # Rebuild encoder-decoder plots from CSVs
├── model_family_profiler/
│   ├── model_kv.py           # BERT, GPT, LLaMA, T5, and BART builders
│   ├── run_and_plot.py       # Cross-family sweep
│   ├── heatmap.py            # Cross-family heatmap
│   └── simple_attn_bench.py  # Isolated self/cross-attention benchmark
├── figures.py                # Summary figures from saved CSVs
├── requirements.txt
└── README.md
```

## Profiled Components

The recorder times each operation scope independently and writes one CSV row
per component.

| Component key | What it measures |
|---|---|
| `attn.q_proj` | Self-attention query projection |
| `attn.k_proj` | Self-attention key projection |
| `attn.v_proj` | Self-attention value projection |
| `attn.gqa_expand` | Grouped-query attention KV expansion |
| `attn.cache_concat` | KV-cache concatenation for decode-style runs |
| `attn.matmul_qk` | QK attention score matmul |
| `attn.apply_causal_mask` | Decoder causal mask |
| `attn.apply_padding_mask` | Encoder padding mask |
| `attn.softmax` | Attention softmax |
| `attn.weighted_sum` | Attention probability times value matmul |
| `attn.out_projection` | Attention output projection |
| `cross_attn.*` | Encoder-decoder cross-attention projections, matmul, softmax, weighted sum, and output projection |
| `block.norm1`, `block.norm2`, `block.norm3` | LayerNorm or RMSNorm scopes |
| `ff.linear1`, `ff.gelu`, `ff.linear2` | GELU feed-forward network |
| `ff.w1_gate`, `ff.w2_up`, `ff.swiglu_act`, `ff.w3_down` | SwiGLU feed-forward network |
| `model.final_norm` | Final model normalization |
| `model.output_head` | Decoder language-model output projection |

## Model Shapes

These presets live in `common/config.py`.

| Shape | Family style | `d` | `h` | KV heads | Layers | FFN dim |
|---|---:|---:|---:|---:|---:|---:|
| `tiny_debug` | smoke test | 256 | 4 | 4 | 2 | 1024 |
| `bert_base` | BERT-base | 768 | 12 | 12 | 12 | 3072 |
| `bert_large` | BERT-large | 1024 | 16 | 16 | 24 | 4096 |
| `gpt2_medium` | GPT-2 medium | 1024 | 16 | 16 | 24 | 4096 |
| `gpt3_2p7b` | GPT-3 2.7B style | 2560 | 32 | 32 | 32 | 10240 |
| `llama_7b` | LLaMA-7B style | 4096 | 32 | 32 | 32 | 11008 |
| `t5_base` | T5-base | 768 | 12 | 12 | 12 | 3072 |
| `t5_large` | T5-large | 1024 | 16 | 16 | 24 | 4096 |
| `bart_large` | BART-large | 1024 | 16 | 16 | 12 | 4096 |

## Sweep Dimensions

The main sweeps vary context length `L` directly and vary model dimension
parameters through the model-shape presets above. Each selected shape carries
its own realistic combination of:

- hidden dimension `d`
- query attention heads `h`
- KV heads for GQA/MQA-style models
- number of Transformer layers
- FFN/intermediate dimension

This means a run over `bert_base`, `bert_large`, `gpt3_2p7b`, and `llama_7b`
covers different `d`, `h`, and layer-count values while preserving real
model-like pairings. The scripts do not use an arbitrary Cartesian product
such as every `d` with every `h` and every layer count by default. Individual
scripts still expose `--d-model`, `--num-heads`, `--num-layers`,
`--num-kv-heads`, and `--d-ff` for controlled custom runs.

Default context-length presets:

| Preset | `L` values |
|---|---|
| `quick` | 128, 256 |
| `standard` | 512, 1024, 2048, 4096 |
| `long` | 512, 1024, 2048, 4096, 8192 |

## What `L` Means

- Encoder-only sweeps run a full bidirectional forward pass over `L` tokens.
- Decoder-only sweeps run a one-token decode step at total context `L`, using
  synthetic past KV cache of length `L - 1`. This exposes cache concat, causal
  masking, GQA expansion, and output-head latency.
- Encoder-decoder sweeps run an encoder source length of `L` plus a one-token
  decoder step with synthetic decoder KV cache of length `L - 1`. This exposes
  encoder latency, decoder self-attention latency, and cross-attention latency.
- The attention micro-benchmark isolates self-attention and cross-attention
  without the full model stack.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is recommended for meaningful GPU latency numbers. CPU runs work, but
large shapes and long contexts can be very slow.

## Running Experiments

All scripts support the same common controls:

```bash
--preset quick|standard|long
--seq-lens 512,1024,2048
--shape-names bert_base,bert_large
--d-model 1536
--num-heads 24
--num-kv-heads 8
--num-layers 24
--d-ff 6144
--batch-size 1
--device auto|cpu|cuda
--dtype float32|float16|bfloat16
--warmups 2
--repeats 5
--max-attn-gb 8
--no-plots
```

The `--max-attn-gb` guard skips runs whose estimated attention buffers are too
large. Set it to `0` to disable the guard.

For a single command that runs the complete profiling plan, see
[`RUN_ALL.md`](RUN_ALL.md).

### Encoder-only

```bash
cd encoder_profiler
python run_and_plot.py --preset standard --shape-names bert_base,bert_large
```

### Decoder-only

```bash
cd decoder_profiler
python run_and_plot.py --preset standard --shape-names gpt2_medium,gpt3_2p7b,llama_7b
```

### Encoder-decoder

```bash
cd encoder_decoder_profiler
python run_and_plot.py --preset standard --shape-names t5_base,t5_large,bart_large
```

### Cross-family comparison

```bash
cd model_family_profiler
python run_and_plot.py --preset standard --families bert,gpt,llama,t5,bart
python heatmap.py
```

### Custom `d`, `L`, and `h` run

```bash
cd decoder_profiler
python run_and_plot.py \
  --shape-names gpt2_medium \
  --seq-lens 512,1024,2048 \
  --d-model 1536 \
  --num-heads 24 \
  --num-layers 24 \
  --d-ff 6144
```

## Outputs

Each run writes CSVs under the script's local `latency_results/` directory:

```text
latency_results/<family>/latency_<shape_name>_d<d>_h<h>_l<L>.csv
```

Each CSV includes metadata columns for architecture, model family, shape name,
`d_model`, query heads, KV heads, head dimension, number of layers, FFN size,
batch size, context length, dtype, and device. Timing columns include count,
average, min, max, standard deviation, total latency, and percent of timed
latency.

When plotting is enabled, the scripts also write:

- `bar_latency_<shape_name>_d<d>_h<h>_l<L>.png`
- `pie_latency_<shape_name>_d<d>_h<h>_l<L>.png`
- `comparison_avg_latency.png`
- `comparison_pct_latency.png`
- `model_family_heatmap.png` from `model_family_profiler/heatmap.py`
- summary figures under `figures/` from root-level `figures.py`

Regenerate plots from existing CSVs:

```bash
cd encoder_profiler
python plot_from_csv.py

cd ../decoder_profiler
python plot_from_csv.py

cd ../encoder_decoder_profiler
python plot_from_csv.py
```

Build root-level summary figures after one or more sweeps:

```bash
cd ..
python figures.py
```
