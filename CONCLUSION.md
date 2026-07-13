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

Decoder-only prefill follows the same long-context transition, and in the
measurements causal masking becomes the single largest component. At
`L = 8192`, GPT-2 medium spends 36.4% in causal masking, 30.9% in softmax, and
20.0% in QK matmul. GPT-3 2.7B-style (32.9% mask) and LLaMA-7B-style (27.5%
mask) show the same ordering: in each decoder-only prefill figure at
`L = 8192`, causal masking is the top component, ahead of softmax and at
1.6-1.8x the cost of the QK score matmul. This means long-context decoder
prefill should be treated primarily as an attention-matrix optimization
problem.

## Experimental Results on Causal Masking

Across the measured configurations, causal mask application is the largest
single component of long-context decoder-only prefill. Its share grows
monotonically with sequence length: for GPT-2 medium it rises from 10.1% at
`L = 128` to 36.4% at `L = 8192`; for GPT-3 2.7B-style from 6.9% to 32.9%; for
LLaMA-7B-style from 3.7% to 27.5%. In absolute terms, GPT-2 medium mask time
grows from 6.9 ms to 2,300 ms per prefill as `L` goes from 128 to 8192, and
LLaMA-7B-style records 6,002 ms of mask time against 3,776 ms for the QK
matmul at `L = 8192`. In the profiled implementation, masking the full
`L x L` score matrix costs 1.6-1.8x more than computing it.

The measured cost is consistent with how the mask is implemented: a standalone
elementwise pass in which each layer constructs an `L x L` boolean triangular
tensor and applies `masked_fill` to the full `heads x L x L` score tensor.
This pass reads and writes the entire attention matrix once per layer while
performing trivial arithmetic, so it is memory-bandwidth-bound, which explains
why it exceeds the compute-bound QK matmul at long context. It also implies
that the entries being masked, roughly half the score matrix, were computed by
the QK matmul and passed through softmax before being discarded.

The results also show that the mask cost is phase-specific. During cached
decode the query is a single token whose score row is inherently causal: the
decode-phase mask share is small and decreasing with context length (8.0% for
GPT-2 medium, 5.7% for GPT-3 2.7B-style, 4.0% for LLaMA-7B-style at
`L = 8192`), and in the phase-split Phi/GPT reproduction the decode
measurement contains no mask cost at all. Encoder-decoder profiles show the
same pattern: decoder self-attention masking accounts for about 17% of
T5-large and BART-large prefill at `L = 8192`, while cross-attention, which
applies no causal mask, contributes no mask cost.

Phase aggregation changes how prominent this component appears. In the
combined Phi/GPT measurement at `L = 8192`, causal masking is 17.0% of
latency, third behind softmax; split by phase it is 18.7% of prefill and
absent from decode, and in the per-model decoder profiles it is the top
prefill component. These measurements indicate that fused attention kernels
that apply causality by construction, skipping masked blocks rather than
computing and then erasing them, would remove the standalone mask pass and
the associated discarded score and softmax work, without changing model
output.

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

- For decoder prefill, the measurements identify standalone causal-mask
  application as the largest component at long context; fusing causality into
  the attention kernel removes this pass along with the discarded
  upper-triangle score and softmax work. QK score matmul, softmax, and
  weighted-sum overhead are the next targets.
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
