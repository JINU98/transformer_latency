# Run All Profiler Guide

Use `run_all.py` from the repository root to launch the full profiling plan:
architecture sweeps, model-family sweeps, attention microbenchmarks, heatmaps,
and summary figures.

## Recommended Commands

Preview the full command plan without running anything:

```bash
python run_all.py --dry-run
```

Fast smoke test:

```bash
python run_all.py --tier smoke --warmups 0 --repeats 1 --no-plots
```

Standard GPU run:

```bash
python run_all.py \
  --tier standard \
  --device cuda \
  --dtype float16 \
  --max-attn-gb 8 \
  --keep-going
```

Full built-in realistic run:

```bash
python run_all.py \
  --tier full \
  --device cuda \
  --dtype float16 \
  --max-attn-gb 12 \
  --keep-going
```

## Tiers

| Tier | Context lengths `L` | Shapes |
|---|---|---|
| `smoke` | 128 | `tiny_debug` only |
| `standard` | 512, 1024, 2048, 4096 | Realistic default shapes |
| `full` | 128, 256, 512, 1024, 2048, 4096, 8192 | Realistic default shapes |

The script uses real model-like `(d, h)` pairs from `common/config.py` rather
than arbitrary Cartesian products. That keeps experiments close to actual BERT,
GPT, LLaMA, T5, and BART-style configurations.

## Parameter Coverage

`run_all.py` varies these dimensions:

| Dimension | How it varies |
|---|---|
| Context length `L` | Directly swept by tier or `--seq-lens` |
| Hidden dimension `d` | Swept through model-shape presets |
| Query heads `h` | Swept through model-shape presets |
| KV heads | Tracked through model-shape presets; custom GQA runs can use `--num-kv-heads` |
| Number of layers | Swept through model-shape presets |
| FFN/intermediate dimension | Swept through model-shape presets |

The built-in layer counts are:

| Shape | Layers |
|---|---:|
| `tiny_debug` | 2 |
| `bert_base` | 12 |
| `bert_large` | 24 |
| `gpt2_medium` | 24 |
| `gpt3_2p7b` | 32 |
| `llama_7b` | 32 |
| `t5_base` | 12 |
| `t5_large` | 24 |
| `bart_large` | 12 |

So the run-all workflow covers different `L`, `d`, `h`, KV-head, layer-count,
and FFN-size configurations. It does not independently run every possible
`d x h x layer-count` Cartesian combination.

## What Runs

`run_all.py` launches these steps in order:

1. `encoder_profiler/run_and_plot.py`
2. `decoder_profiler/run_and_plot.py`
3. `encoder_decoder_profiler/run_and_plot.py`
4. `model_family_profiler/run_and_plot.py`
5. `model_family_profiler/simple_attn_bench.py`
6. `encoder_profiler/plot_from_csv.py`
7. `decoder_profiler/plot_from_csv.py`
8. `encoder_decoder_profiler/plot_from_csv.py`
9. `model_family_profiler/heatmap.py`
10. `figures.py`

Use `--no-plots` to run only CSV-producing sweeps. Use
`--skip-attention-bench`, `--skip-model-family`, or `--skip-summary-figures`
to narrow the run.

## Useful Flags

| Flag | Purpose |
|---|---|
| `--clean` | Delete generated `latency_results/` and `figures/` before running |
| `--dry-run` | Print every command without executing |
| `--keep-going` | Continue after a failed command and summarize failures |
| `--seq-lens 512,1024` | Override the tier's context lengths |
| `--device auto|cpu|cuda` | Select execution device |
| `--dtype float32|float16|bfloat16` | Select tensor dtype |
| `--max-attn-gb N` | Skip runs whose estimated attention buffers exceed `N` GB |
| `--no-plots` | Skip per-run plots and summary figures |

## Outputs

CSV files are written under each profiler directory:

```text
encoder_profiler/latency_results/
decoder_profiler/latency_results/
encoder_decoder_profiler/latency_results/
model_family_profiler/latency_results/
```

Summary figures are written to:

```text
figures/
```

Each CSV filename includes the profiled hidden dimension, head count, and
context length:

```text
latency_<shape_name>_d<d>_h<h>_l<L>.csv
```
