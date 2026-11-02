"""Neural-Engine form of the Gemma 4 / EmbeddingGemma 2 audio encoder, used by convert.py --split_audio.

The HF tower runs chunked local attention over 5-D blocked tensors built with unfold (a gather) and a relative-shift reshape: the
Neural Engine compiler takes 15+ minutes on a 10 s window and the result is slower than the GPU. The attention it computes is just
a causal band: query q sees keys q-11..q (chunk 12, left context 13, right 0), with a relative-position bias q.R(q-k) over the 13
rel_pos_enc entries. DenseAudioAttention computes exactly that as plain [T,T] matmuls with a constant additive band mask and 13
constant distance planes; everything else (feed-forwards, light conv, norms, output projection, multimodal embedder) is the HF
module unchanged.

Split export: AudioConvFront (mel -> conv subsampling, any compute unit) and DenseAudioStack (hidden -> audio token embeddings,
Neural Engine). The clamps (clippable linears at +-inf, gradient clipping at +-1e10) are dropped and the norms run without
the fp32 round trip: both are numerical no-ops, and both put ops on the CPU when left in. check_audio_dense.py verifies the
dense form against the HF forward (random weights, max abs diff about 1e-5); parity.py --split checks exported units.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_NEG = -30000.0  # fp16-safe stand-in for the tower's -1e9 invalid logit (exp underflows to 0 either way)
FP16_MAX = 65504.0


def fin(v):
    return v if math.isfinite(v) else None


def lin(cl, x):
    """Gemma4ClippableLinear with its bounds as constants. The checkpoint's bounds are finite (e.g. +-12.8 in, +-40 out); the
    same clamp with 0-d tensor bounds lands on the CPU, and an infinite bound is dropped (a clip with inf is a CPU op too)."""
    if not cl.use_clipped_linears:
        return cl.linear(x)
    lo, hi, olo, ohi = (float(b) for b in (cl.input_min, cl.input_max, cl.output_min, cl.output_max))
    if fin(lo) is not None or fin(hi) is not None:
        x = x.clamp(min=fin(lo), max=fin(hi))
    y = cl.linear(x)
    if fin(olo) is not None or fin(ohi) is not None:
        y = y.clamp(min=fin(olo), max=fin(ohi))
    return y


class ANERMSNorm(nn.Module):
    """Gemma4RMSNorm in fp16-safe form (rows scaled by their max before the mean of squares) and without the fp32 round trip"""

    def __init__(self, norm):
        super().__init__()
        self.eps, self.weight = norm.eps, (norm.weight if norm.with_scale else None)

    def forward(self, x):
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
        xs = x / s
        e = (self.eps**0.5) / s
        y = xs * torch.pow(xs.pow(2).mean(-1, keepdim=True) + e * e, -0.5)
        return y * self.weight if self.weight is not None else y


class DenseFeedForward(nn.Module):
    def __init__(self, ff):
        super().__init__()
        self.ff = ff
        self.pre, self.post = ANERMSNorm(ff.pre_layer_norm), ANERMSNorm(ff.post_layer_norm)

    def forward(
        self, x
    ):  # the tower's clamp(+-1e10) is a no-op at these magnitudes (and +-1e10 is inf in fp16, which is a CPU op)
        return (
            self.post(lin(self.ff.ffw_layer_2, self.ff.act_fn(lin(self.ff.ffw_layer_1, self.pre(x))))) * self.ff.post_layer_scale
            + x
        )


class DenseLightConv(nn.Module):
    def __init__(self, lc):
        super().__init__()
        self.lc = lc
        self.pre, self.cnorm = ANERMSNorm(lc.pre_layer_norm), ANERMSNorm(lc.conv_norm)

    def forward(self, x):
        h = F.glu(lin(self.lc.linear_start, self.pre(x)), dim=-1)
        h = self.lc.depthwise_conv1d(h.transpose(1, 2)).transpose(1, 2)  # causal: pads kernel-1 on the left
        return lin(self.lc.linear_end, self.lc.act_fn(self.cnorm(h))) + x


class DenseAudioLayer(nn.Module):
    def __init__(self, layer, T):
        super().__init__()
        self.ff1, self.ff2 = DenseFeedForward(layer.feed_forward1), DenseFeedForward(layer.feed_forward2)
        self.attn, self.lconv = DenseAudioAttention(layer.self_attn, T), DenseLightConv(layer.lconv1d)
        self.n_pre, self.n_post, self.n_out = (
            ANERMSNorm(layer.norm_pre_attn),
            ANERMSNorm(layer.norm_post_attn),
            ANERMSNorm(layer.norm_out),
        )

    def forward(self, x, pos):
        x = self.ff1(x)
        r = x
        x = self.n_post(self.attn(self.n_pre(x), pos)[0]) + r
        return self.n_out(self.ff2(self.lconv(x)))


class DenseAudioAttention(nn.Module):
    def __init__(self, attn, T):
        super().__init__()
        self.a, self.T = attn, T
        past = attn.max_past_horizon
        assert attn.max_future_horizon == 0, 'dense form written for left-context-only attention'
        n_rel = attn.context_size // 2 + 1  # rel_pos_enc entries: position ids n_rel-1 .. 0
        d = torch.arange(T)[:, None] - torch.arange(T)[None, :]  # q - k
        band = torch.where((d >= 0) & (d < past), 0.0, MASK_NEG)[None, None]  # HF: q-k < left-1 = 12
        self.register_buffer('band', band, persistent=False)
        # plane j selects the diagonal q-k == n_rel-1-j (rel_pos_enc index j is position id n_rel-1-j)
        planes = torch.stack([(d == n_rel - 1 - j).float() for j in range(n_rel)])
        # [n_rel,1,1,T,T]; used one plane at a time: rank-4 ops only
        self.register_buffer('planes', planes[:, None, None], persistent=False)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        a = self.a
        B, T, _ = hidden_states.shape
        shape = (B, T, a.num_heads, a.head_dim)
        q = lin(a.q_proj, hidden_states).view(shape) * (a.q_scale * F.softplus(a.per_dim_scale))
        k = lin(a.k_proj, hidden_states).view(shape) * a.k_scale
        v = lin(a.v_proj, hidden_states).view(shape)
        rk = a.relative_k_proj(position_embeddings).view(-1, a.num_heads, a.head_dim)  # [n_rel, H, hd]
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3)  # [B,H,T,hd] [B,H,hd,T] [B,H,T,hd]
        ac = q @ k  # [B,H,T,T]
        qr = q @ rk.permute(1, 2, 0)[None]  # [B,H,T,n_rel]
        bd = qr[..., 0:1] * self.planes[0]  # [B,H,T,T]: bd[q,k] = qr[q, j(q-k)], 13 rank-4 terms
        for j in range(1, self.planes.shape[0]):
            bd = bd + qr[..., j : j + 1] * self.planes[j]  # (a single rank-5 broadcast failed on the A16 ANE)
        w = torch.tanh((ac + bd) / a.softcap) * a.softcap + self.band
        w = F.softmax(w, dim=-1)
        out = (w @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return lin(a.post, out), None


class AudioConvFront(nn.Module):
    """log-mel features [1,F,128] + frame mask [1,F] -> subsampled hidden states [1,T,hidden] (two stride-2 convs)"""

    def __init__(self, tower):
        super().__init__()
        self.sub = tower.subsample_conv_projection

    def forward(self, features, mask):
        return self.sub(features, mask > 0.5)[0]


class DenseAudioStack(nn.Module):
    """subsampled hidden states [1,T,hidden] -> audio token embeddings [1,T,text_hidden] (conformer, output_proj, embed_audio)"""

    def __init__(self, tower, embed_audio, T):
        super().__init__()
        self.tower, self.T = tower, T
        self.layers = nn.ModuleList(DenseAudioLayer(l, T) for l in tower.layers[: tower.config.num_hidden_layers])
        self.out_proj, self.pre_norm, self.proj = (
            tower.output_proj,
            ANERMSNorm(embed_audio.embedding_pre_projection_norm),
            embed_audio.embedding_projection,
        )
        pos = tower.rel_pos_enc(torch.zeros(1, 1, tower.config.hidden_size))  # constant [n_rel, hidden]
        self.register_buffer('pos', pos, persistent=False)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden, self.pos)
        return self.proj(self.pre_norm(self.out_proj(hidden)))


def tokens_for_frames(frames):
    for _ in range(2):
        frames = (frames + 2 - 3) // 2 + 1
    return frames
