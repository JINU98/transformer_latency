# Phi-Style Reproduction

This folder recreates the old Phi/GPT-style pie-chart experiment with the
current profiler code.

The target run is:

- Decoder-only transformer
- Prefill over context length `L`
- Then 10 cached one-token decode steps
- `d_model = 3072`
- `num_heads = 32`
- `num_layers = 32`
- `d_ff = 12288`
- `vocab_size = 30257`
- Sequence lengths: `256, 512, 1024, 2048, 4096, 8192`

The script saves two views of each run:

- `raw_current/`: current profiler components exactly as recorded, including
  split `q_proj`, `k_proj`, `v_proj`, and `model.final_norm`.
- `old_aligned/`: comparison-friendly components where `q_proj + k_proj +
  v_proj` are grouped as `attn.qkv_projection`, decode-only no-op causal-mask
  timing is excluded, and `model.final_norm` is dropped to match the older
  baseline plotting setup.

Phase-split CSVs are also saved in `phase_split/` so prefill and cached decode
can be checked independently.

Run on the Jetson container from the repository root:

```bash
python3 reproduce/run_phi_reproduction.py
```

For a shorter smoke test:

```bash
python3 reproduce/run_phi_reproduction.py --seq-lens 256 --no-plots
```

The main comparison figures are written as:

```text
reproduce/phi_baseline_like/old_aligned/pie_chart_<L>.png
reproduce/phi_baseline_like/old_aligned/component_legend.png
```
