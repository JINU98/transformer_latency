from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to run latency experiments. Install with: pip install -r requirements.txt"
        ) from exc
    return torch, nn, F


torch, nn, F = require_torch()


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class ProfiledSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, recorder, causal=False, num_kv_heads=None, prefix="attn"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.prefix = prefix
        self.recorder = recorder
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _shape_q(self, x):
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x):
        bsz, seqlen, _ = x.shape
        return x.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, padding_mask=None, past_kv=None):
        bsz, seqlen, _ = x.shape
        with self.recorder.record(f"{self.prefix}.q_proj"):
            q = self._shape_q(self.q_proj(x))
        with self.recorder.record(f"{self.prefix}.k_proj"):
            k = self._shape_kv(self.k_proj(x))
        with self.recorder.record(f"{self.prefix}.v_proj"):
            v = self._shape_kv(self.v_proj(x))
        if past_kv is not None:
            past_k, past_v = past_kv
            with self.recorder.record(f"{self.prefix}.cache_concat"):
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        cache_k, cache_v = k, v
        attn_k, attn_v = k, v
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            with self.recorder.record(f"{self.prefix}.gqa_expand"):
                attn_k = k.repeat_interleave(repeat, dim=1)
                attn_v = v.repeat_interleave(repeat, dim=1)
        with self.recorder.record(f"{self.prefix}.matmul_qk"):
            scores = torch.matmul(q, attn_k.transpose(-2, -1)) / (self.head_dim**0.5)
        if self.causal:
            with self.recorder.record(f"{self.prefix}.apply_causal_mask"):
                q_len, k_len = scores.shape[-2], scores.shape[-1]
                mask = torch.ones((q_len, k_len), device=scores.device, dtype=torch.bool).triu(k_len - q_len + 1)
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        elif padding_mask is not None:
            with self.recorder.record(f"{self.prefix}.apply_padding_mask"):
                scores = scores.masked_fill(~padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        with self.recorder.record(f"{self.prefix}.softmax"):
            probs = torch.softmax(scores, dim=-1)
        with self.recorder.record(f"{self.prefix}.weighted_sum"):
            y = torch.matmul(probs, attn_v)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        with self.recorder.record(f"{self.prefix}.out_projection"):
            return self.out_proj(y), (cache_k, cache_v)


class ProfiledCrossAttention(ProfiledSelfAttention):
    def __init__(self, d_model, num_heads, recorder):
        super().__init__(d_model, num_heads, recorder, causal=False, prefix="cross_attn")

    def forward(self, x, encoder_states):
        bsz, tgt_len, _ = x.shape
        src_len = encoder_states.shape[1]
        with self.recorder.record("cross_attn.q_proj"):
            q = self._shape_q(self.q_proj(x))
        with self.recorder.record("cross_attn.k_proj"):
            k = self._shape_kv(self.k_proj(encoder_states))
        with self.recorder.record("cross_attn.v_proj"):
            v = self._shape_kv(self.v_proj(encoder_states))
        with self.recorder.record("cross_attn.matmul_qk"):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        with self.recorder.record("cross_attn.softmax"):
            probs = torch.softmax(scores, dim=-1)
        with self.recorder.record("cross_attn.weighted_sum"):
            y = torch.matmul(probs, v)
        y = y.transpose(1, 2).contiguous().view(bsz, tgt_len, self.d_model)
        with self.recorder.record("cross_attn.out_projection"):
            return self.out_proj(y)


class ProfiledFFN(nn.Module):
    def __init__(self, d_model, d_ff, recorder, kind="gelu"):
        super().__init__()
        self.kind = kind
        self.recorder = recorder
        if kind == "swiglu":
            self.w1_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w2_up = nn.Linear(d_model, d_ff, bias=False)
            self.w3_down = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, d_ff, bias=False)
            self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            with self.recorder.record("ff.w1_gate"):
                gate = self.w1_gate(x)
            with self.recorder.record("ff.w2_up"):
                up = self.w2_up(x)
            with self.recorder.record("ff.swiglu_act"):
                hidden = F.silu(gate) * up
            with self.recorder.record("ff.w3_down"):
                return self.w3_down(hidden)
        with self.recorder.record("ff.linear1"):
            hidden = self.linear1(x)
        with self.recorder.record("ff.gelu"):
            hidden = F.gelu(hidden)
        with self.recorder.record("ff.linear2"):
            return self.linear2(hidden)


class EncoderBlock(nn.Module):
    def __init__(self, shape, recorder):
        super().__init__()
        self.recorder = recorder
        self.norm1 = nn.LayerNorm(shape.d_model)
        self.attn = ProfiledSelfAttention(shape.d_model, shape.num_heads, recorder, causal=False)
        self.norm2 = nn.LayerNorm(shape.d_model)
        self.ffn = ProfiledFFN(shape.d_model, shape.d_ff, recorder, kind="gelu")

    def forward(self, x, padding_mask=None):
        with self.recorder.record("block.norm1"):
            y = self.norm1(x)
        y, _ = self.attn(y, padding_mask=padding_mask)
        x = x + y
        with self.recorder.record("block.norm2"):
            y = self.norm2(x)
        return x + self.ffn(y)


class DecoderBlock(nn.Module):
    def __init__(self, shape, recorder, ffn_kind="gelu", norm_kind="layernorm"):
        super().__init__()
        self.recorder = recorder
        norm = RMSNorm if norm_kind == "rmsnorm" else nn.LayerNorm
        self.norm1 = norm(shape.d_model)
        self.attn = ProfiledSelfAttention(
            shape.d_model, shape.num_heads, recorder, causal=True, num_kv_heads=shape.num_kv_heads
        )
        self.norm2 = norm(shape.d_model)
        self.ffn = ProfiledFFN(shape.d_model, shape.d_ff, recorder, kind=ffn_kind)

    def forward(self, x, past_kv=None):
        with self.recorder.record("block.norm1"):
            y = self.norm1(x)
        y, kv = self.attn(y, past_kv=past_kv)
        x = x + y
        with self.recorder.record("block.norm2"):
            y = self.norm2(x)
        return x + self.ffn(y), kv


class EncoderDecoderBlock(nn.Module):
    def __init__(self, shape, recorder, ffn_kind="gelu"):
        super().__init__()
        self.recorder = recorder
        self.self_block = DecoderBlock(shape, recorder, ffn_kind=ffn_kind)
        self.norm3 = nn.LayerNorm(shape.d_model)
        self.cross_attn = ProfiledCrossAttention(shape.d_model, shape.num_heads, recorder)

    def forward(self, x, encoder_states, past_kv=None):
        x, kv = self.self_block(x, past_kv=past_kv)
        with self.recorder.record("block.norm3"):
            y = self.norm3(x)
        return x + self.cross_attn(y, encoder_states), kv


class EncoderModel(nn.Module):
    def __init__(self, shape, recorder):
        super().__init__()
        self.recorder = recorder
        self.layers = nn.ModuleList([EncoderBlock(shape, recorder) for _ in range(shape.num_layers)])
        self.final_norm = nn.LayerNorm(shape.d_model)

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        with self.recorder.record("model.final_norm"):
            return self.final_norm(x)


class DecoderModel(nn.Module):
    def __init__(self, shape, recorder, ffn_kind="gelu", norm_kind="layernorm", vocab_size=32000):
        super().__init__()
        self.recorder = recorder
        self.layers = nn.ModuleList([DecoderBlock(shape, recorder, ffn_kind, norm_kind) for _ in range(shape.num_layers)])
        self.final_norm = RMSNorm(shape.d_model) if norm_kind == "rmsnorm" else nn.LayerNorm(shape.d_model)
        self.output_head = nn.Linear(shape.d_model, vocab_size, bias=False)

    def forward(self, x, past_kv=None, full_logits=False):
        new_kv = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_kv[idx] if past_kv is not None else None
            x, kv = layer(x, past_kv=layer_past)
            new_kv.append(kv)
        with self.recorder.record("model.final_norm"):
            x = self.final_norm(x)
        logits_input = x if full_logits else x[:, -1:, :]
        with self.recorder.record("model.output_head"):
            _ = self.output_head(logits_input)
        return x, new_kv


class EncoderDecoderModel(nn.Module):
    def __init__(self, shape, recorder, ffn_kind="gelu", vocab_size=32000):
        super().__init__()
        self.recorder = recorder
        self.encoder = EncoderModel(shape, recorder)
        self.decoder = nn.ModuleList([EncoderDecoderBlock(shape, recorder, ffn_kind) for _ in range(shape.num_layers)])
        self.final_norm = nn.LayerNorm(shape.d_model)
        self.output_head = nn.Linear(shape.d_model, vocab_size, bias=False)

    def forward(self, encoder_x, decoder_x, past_kv=None, full_logits=False):
        enc = self.encoder(encoder_x)
        x = decoder_x
        new_kv = []
        for idx, layer in enumerate(self.decoder):
            layer_past = past_kv[idx] if past_kv is not None else None
            x, kv = layer(x, enc, past_kv=layer_past)
            new_kv.append(kv)
        with self.recorder.record("model.final_norm"):
            x = self.final_norm(x)
        logits_input = x if full_logits else x[:, -1:, :]
        with self.recorder.record("model.output_head"):
            _ = self.output_head(logits_input)
        return x, new_kv
