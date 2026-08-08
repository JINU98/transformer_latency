# Prefill vs. Decode Ratio Profiler

Extends the repository's existing per-shape prefill/decode component figures
(`figures/decoder/prefill/model_family_component_share.png` and
`figures/decoder/decode/model_family_component_share.png`) with a sweep those
figures don't answer directly:

> For one representative model, how does the split between **prefill time**
> and **decode time** shift as you generate more tokens, at different prompt
> (sequence) lengths?

## Method

For a chosen representative decoder shape (default `llama_7b`) and each
sequence length `L`:

1. Run one full causal **prefill** pass over `L` tokens (identical to
   `decoder_profiler`'s prefill phase) and record its per-component latency.
2. Run exactly **one cached decode step** at context length `L` (i.e.
   `decoder_profiler --decode-tokens 1`), and record its per-component
   latency. This is the marginal cost of generating a single token once the
   `L`-token KV cache is filled.
3. For each "tokens generated" scenario `N` (e.g. `1`, `10`, `20%` of `L`,
   `50%` of `L`, `100%` of `L`, ...), **scale the single decode step's
   per-component latency linearly by `N`**. This is the approximation the
   assignment prescribes: measure one decode step and scale it up for
   multiple decode steps, rather than re-running `N` sequential (and
   increasingly cache-heavy) decode steps for every scenario.
4. Add the (unscaled) prefill cost to the scaled decode cost to get the
   total prefill/decode split and the full component breakdown of combined
   generation latency for that `(L, N)` pair.

This linear-scaling approximation is accurate as long as `N` stays a modest
fraction of `L`: most per-step costs (QKV/FFN/output-head projections) are
independent of cache length, and the part that isn't (attention matmul +
softmax + KV-cache concat over the growing cache) grows slowly step to step
when `N << L`. It becomes optimistic (undercounts true decode cost) once `N`
approaches or exceeds `L`, because a real run's attention/KV-cache-concat
cost keeps growing with every step instead of staying pinned at the L-token
level. The `component_share_*.png` figure makes this visible directly: watch
how `Self KV Cache Concat`'s share grows with `N` and with `L` in the example
figures below.

## Files

- `scenarios.py` — parses `--token-scenarios` strings like `"1,10,20%,50%"`
  into absolute token counts or percentages of `L`.
- `bench.py` — per-architecture benchmark functions (one full prefill + one
  cached decode step) and the `ARCHITECTURES` registry mapping
  `decoder`/`encoder_decoder` to their benchmark function, attention-buffer
  multiplier, default shape, and usual shape names.
- `io_utils.py` — CSV schema (`RAW_FIELDNAMES`, `SCENARIO_FIELDNAMES`,
  `SUMMARY_FIELDNAMES`) and save/load helpers.
- `run_and_plot.py` — runs the benchmarks and produces all figures.
- `gen_plots.py` — the actual matplotlib chart code, reusing the component
  names/colors from the repository's root `figures.py` so styling matches
  the rest of the repo.
- `plot_from_csv.py` — rebuild the figures from previously saved CSVs
  without re-running any benchmarks.

## Usage

```bash
cd gen_ratio_profiler

# Decoder-only: representative model = LLaMA-7B style, standard L sweep, five
# generation scenarios: 1 token, 10 tokens, 20% of L, 50% of L, 100% of L.
python run_and_plot.py --architecture decoder --shape-name llama_7b \
    --preset standard --token-scenarios "1,10,20%,50%,100%"

# Encoder-decoder: representative model = T5-base style.
python run_and_plot.py --architecture encoder_decoder --shape-name t5_base \
    --preset standard --token-scenarios "1,10,20%,50%,100%"

# Smaller/faster examples used to produce the figures checked into this repo:
python run_and_plot.py --architecture decoder --shape-name gpt2_medium \
    --seq-lens 128,256,512,1024 --token-scenarios "1,10,20%,50%,100%" \
    --repeats 3 --warmups 1 --device cpu

python run_and_plot.py --architecture encoder_decoder --shape-name t5_base \
    --seq-lens 128,256,512,1024 --token-scenarios "1,10,20%,50%,100%" \
    --repeats 3 --warmups 1 --device cpu
```

`--architecture` selects which model family and benchmark path to use:

| `--architecture` | Default `--shape-name` | Also accepts |
|---|---|---|
| `decoder` (default) | `llama_7b` | `gpt2_medium`, `gpt3_2p7b`, or any other shape in `common/config.py` |
| `encoder_decoder` | `t5_base` | `t5_large`, `bart_large`, or any other shape |

Common flags (see `--help` for the full list):

```
--architecture       decoder (default) or encoder_decoder
--shape-name         One representative shape from common/config.py
                     (default: llama_7b for decoder, t5_base for encoder_decoder)
--preset             quick|standard|long L presets, or use --seq-lens directly
--seq-lens           Comma-separated L values, e.g. 512,1024,2048
--token-scenarios   Comma-separated scenarios: absolute counts ("1", "10")
                     and/or percentages of L ("20%", "50%"). Default:
                     "1,10,20%,50%,100%"
--batch-size, --warmups, --repeats, --device, --dtype, --max-attn-gb
                     Same meaning as the other profilers in this repo.
--output-dir         Where CSVs are written
                     (default: latency_results/<architecture>/<shape>/)
--figures-dir         Where figures are written
                     (default: ../figures, i.e. the repository's shared
                     figures/ folder, under gen_ratio/<architecture>/)
--no-plots           Skip figure generation, just write CSVs
```

Rebuild figures later without re-benchmarking:

```bash
python plot_from_csv.py --architecture decoder --shape-name gpt2_medium
python plot_from_csv.py --architecture encoder_decoder --shape-name t5_base
```

## Outputs

```text
gen_ratio_profiler/latency_results/<architecture>/<shape>/
├── raw_l<L>.csv              # unscaled prefill + single-decode-step rows, one file per L
├── scenario_components.csv   # every (L, scenario, component) row, scaled
└── scenario_summary.csv      # one row per (L, scenario): prefill_ms, decode_ms, total_ms, prefill_pct, decode_pct

figures/gen_ratio/<architecture>/
├── prefill_decode_share_<shape>.png   # stacked bar: Prefill % vs Decode % by L and tokens generated
├── component_share_<shape>.png        # fine-grained component share of combined prefill+decode total
└── pie_charts/
    └── pie_<shape>_l<L>_<scenario>.png  # one pie per (L, scenario) combination
```

`<architecture>` is `decoder` or `encoder_decoder`, matching the `figures/decoder/`,
`figures/encoder/`, and `figures/encoder_decoder/` layout already used elsewhere
in this repo.

`prefill_decode_share_<shape>.png` is the direct answer to "prefill vs decode
ratio, sweeping sequence length and tokens generated": each group of bars is
one `L`, and within a group each bar is a tokens-generated scenario (1 token,
10 tokens, 20% of L, 50% of L, 100% of L). `component_share_<shape>.png`
is the "similar shared-components figure" requested for this analysis: same
visual language and component palette as
`figures/decoder/*/model_family_component_share.png` and
`figures/encoder_decoder/*/model_family_component_share.png`, but the x-axis
is `(L, tokens generated)` instead of `(L, model shape)`, and each bar's 100%
is the *combined* prefill+decode total for that scenario rather than a
single phase.

## A note on encoder-decoder cross-attention cost

The example `component_share_t5_base.png` figure shows `Cross QKV Projection`
growing to dominate decode-heavy scenarios (up to ~75% of total latency at
L=1024, 100% of L generated). That is a real property of this repository's
synthetic `EncoderDecoderModel` (`common/models.py`), not an artifact of the
linear-scaling approximation used here: its cross-attention layer
recomputes K/V projections from the encoder states on *every* decode step
instead of caching them once after encoding, which real T5/BART inference
does. Since prefill only pays that cost once but decode pays it on every
scaled step, it comes to dominate as the number of generated tokens grows.
Worth keeping in mind when comparing the encoder-decoder decode-share figures
to a production inference stack with proper cross-attention KV caching, which
would show a *smaller* decode share at large L/N than these figures do.

## A note on `figures.py`

`gen_plots.py` imports `COMPONENT_COLORS`, `component_for_operation`, and
`ordered_present` from the repository's root `figures.py` so the component
palette matches the rest of the repo. It does **not** call `figures.py`'s
own `main()`/`require_plotting()` path: as of this writing,
`common.plotting.require_plotting()` returns `(Image, ImageDraw, ImageFont)`
(a Pillow helper) but `figures.py`'s `main()` unpacks it as `pd, plt = ...`,
which raises `ValueError: too many values to unpack` if you run
`python figures.py` directly. That's a pre-existing issue in the base
repository, unrelated to this module — `gen_plots.py` imports matplotlib and
pandas directly instead of going through that helper.
