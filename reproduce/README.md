# Phi-Style Reproduction

This folder recreates the old Phi/GPT-style pie-chart experiment with an
old-compatible profiler path.

The target run is:

- Decoder-only transformer
- Prefill over context length `L`
- Then 10 cached one-token decode steps
- `d_model = 3072`
- `num_heads = 32`
- `num_layers = 32`
- `d_ff = 12288`
- `vocab_size = 30257`
- Fused `attn.qkv_projection`
- CUDA autocast `float16` on Jetson/AGX
- Sequence lengths: `256, 512, 1024, 2048, 4096, 8192`

The script saves:

- `old_aligned/`: combined prefill + 10-token decode CSVs and pie charts using
  the same component labels as the old baseline plots.
- `raw_current/`: a copy of the fused-QKV combined run for easy lookup.
- `phase_split/`: separate prefill and cached-decode CSVs.

Phase-split CSVs are also saved in `phase_split/` so prefill and cached decode
can be checked independently.

Phase-split bar plots are saved in:

- `phi_baseline_like/bar_plots/prefill/`
- `phi_baseline_like/bar_plots/decode/`

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

To regenerate the phase-split bar plots:

```bash
python3 reproduce/plot_phi_phase_bars.py
```
