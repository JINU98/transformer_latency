# Technical Conclusion

This work shows that Transformer inference latency is not governed by one
stable bottleneck. The dominant component changes with architecture, sequence
length, and inference phase, so optimization decisions should be based on
phase-specific component profiles rather than aggregate runtime.

## Main Finding

Long-context prefill is dominated by operations over the full attention matrix.
For encoder-only models at `L = 8192`, BERT-base and BERT-large spend most of
their measured latency in self-attention softmax, QK score matmul, and the
attention weighted-value sum. BERT-large, for example, spends 43.7% in
self-attention softmax and 33.9% in self-attention QK matmul. At this context
length, feed-forward and projection layers remain measurable but no longer
control the latency profile.

Decoder-only prefill follows the same long-context transition, but causal
masking becomes a major additional term. At `L = 8192`, GPT-2 medium spends
36.4% in causal masking, 30.9% in softmax, and 20.0% in QK matmul. GPT-3
2.7B-style and LLaMA-7B-style shapes show the same qualitative behavior. This
means long-context decoder prefill should be treated primarily as an
attention-matrix optimization problem.

Cached decode exposes a different bottleneck. After the context cache is
filled, each decode step processes one new token, but the KV cache must still
be updated and traversed. At `L = 8192`, KV-cache concatenation becomes the
largest decoder-only component: 32.3% for GPT-2 medium, 39.3% for GPT-3
2.7B-style, and 36.8% for LLaMA-7B-style. This is a memory-layout and
cache-management problem, not the same problem as prefill attention.

Encoder-decoder models add a third latency path through cross-attention.
Long-context encoder-decoder prefill combines decoder self-attention and
cross-attention over encoder states. In T5-large at `L = 8192`, self-attention
softmax accounts for 26.7%, self-attention QK matmul for 18.7%, causal masking
for 16.9%, cross-attention softmax for 12.1%, and cross-attention QK matmul for
9.3%. During cached decode, cross-attention projection becomes especially
important: cross-attention QKV projection accounts for 32.3% of T5-large decode
latency at `L = 8192`.

## Regime Interpretation

The measured results imply three practical latency regimes:

1. Short-context regime: projection and feed-forward work remain prominent
   because dense linear layers dominate when attention matrices are small.
2. Long-context prefill regime: attention score computation, masking, softmax,
   and weighted value accumulation dominate because the full attention matrix is
   active.
3. Long-context decode regime: KV-cache operations and memory traffic dominate
   because the generated token is small but the cache grows with context.

The old-compatible Phi/GPT reproduction supports this interpretation. Its
combined `L = 8192` measurement reports softmax as the largest component at
30.0%, followed by weighted sum at 19.6%, causal masking at 17.0%, and QK
matmul at 15.9%. When split by phase, however, prefill is dominated by
attention-matrix operations while decode spends 40.1% in KV-cache
concatenation. Combined measurements therefore hide the actual optimization
target.

## Optimization Implications

The strongest optimization target depends on the serving phase:

- For prefill, prioritize attention kernels and layouts that reduce QK score
  matmul, mask application, softmax, and weighted-sum overhead.
- For cached decode, prioritize KV-cache layout, cache update strategy, memory
  movement, and avoiding avoidable cache concatenation.
- For encoder-decoder inference, optimize cross-attention explicitly rather
  than folding it into self-attention or an `Other` bucket.
- For reporting, always separate prefill and decode and keep component-level
  percentages alongside total latency and per-token latency.

Overall, the profiler demonstrates that accurate Transformer latency analysis
must be architecture-aware, phase-aware, and component-aware. A single
end-to-end latency number can be valid as a throughput metric, but it is not
specific enough to guide kernel, cache, or model-architecture optimization.
