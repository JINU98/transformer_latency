# Main Differences From The Old Transformer Latency Analysis

This short summary lists the five main differences between the old analysis and
the current repository.

## 1. Combined Run vs Phase-Split Run

The old analysis reports one combined measurement: full prefill over context
length `L` plus 10 cached one-token decode steps.

The current profiler separates latency into `prefill` and `decode` phases, so
old pie charts should not be compared directly against a single current
phase-specific CSV.

## 2. Fused QKV vs Split Q/K/V Projections

The old baseline uses one fused `attn.qkv_projection` operation.

The current main profiler records `attn.q_proj`, `attn.k_proj`, and
`attn.v_proj` separately. For old-style comparison, these split projections
must be grouped together or reproduced with the old-compatible runner.

## 3. Token Inputs vs Synthetic Hidden Inputs

The old baseline runs token IDs through token embeddings and position
embeddings.

The current main profiler usually starts from random hidden states so it can
profile many architecture shapes consistently. The reproduction runner restores
the old token/position embedding path.

## 4. Decoder-Only Old Scope vs Broader Current Scope

The old model-family analysis focuses on decoder-only GPT/Phi/LLaMA/Gemma/OPT
style models.

The current repository covers encoder-only, decoder-only, encoder-decoder,
model-family, and attention microbenchmark experiments. This adds components
such as `cross_attn.*`, `block.norm3`, and phase-specific decode metrics.

## 5. Old-Compatible Reproduction Is Required For Old Pie Charts

Use `reproduce/run_phi_reproduction.py` and
`reproduce/phi_baseline_like/old_aligned/` when recreating old Phi/GPT-style
combined pie charts. The corrected AGX reproduction gives an 8192 softmax share
of `30.02%`, close to the old baseline regime of `25.24%`, instead of the
incorrect low value from the split-QKV path.
