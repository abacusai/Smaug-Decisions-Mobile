"""Export the fast tier (abacusai/Smaug-Audio-Decisions-0.7B) to ONNX for ONNX Runtime Android, mirroring the iOS Core ML design:

    audio.onnx   log-mel features [1, F, 128] (only the clip's valid frames) -> audio token embeddings [1, T, 512], T = ceil(ceil(F/2)/2)
    trunk.onnx   inputs_embeds [1, L, 512] (one unpadded sequence)            -> hidden states [1, L, 768]
    embed_tokens_f16.bin   the scaled token-embedding table, raw little-endian fp16 [262144, 512], gathered in Kotlin
    head.json              marker head weight [768] and bias
Both graphs take dynamic lengths, so short prompts and short clips pay only for their own length (iOS uses fixed L512 / 10 s).

Why this is exact (checked by parity_onnx.py against the stock HF forward):
- Trunk: EmbeddingGemma 2 uses projection-only PLE (from inputs_embeds alone), so inputs_embeds is a complete input. One unpadded
  sequence needs no padding mask; the sliding layers use the bidirectional band |q - k| <= sliding_window, built in-graph from L.
- Audio: the conformer's local attention only looks left (q-11..q) and its light conv is causal, so the valid tokens of a clip
  do not depend on the padding the HF processor adds after them; the subsampling convs see zeros there either way. The chunked
  5-D attention is replaced by the same band as dense [T, T] attention with a gather for the 13 relative-position terms
  (dynamic T; same math as scripts/convert/coreml/audio_dense.py, which fixed T for the Neural Engine).

usage: export_onnx.py --model <audio model dir> --out <dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


MASK_NEG = -30000.0


def dense_audio_attention(self, hidden_states, position_embeddings, attention_mask=None):
    """Gemma4AudioAttention.forward as dense band attention with dynamic T (attention_mask ignored: the input has no padding)."""
    B, T, _ = hidden_states.shape
    shape = (B, T, self.num_heads, self.head_dim)
    q = self.q_proj(hidden_states).float().view(shape) * self.q_scale * F.softplus(self.per_dim_scale)
    k = self.k_proj(hidden_states).float().view(shape) * self.k_scale
    v = self.v_proj(hidden_states).float().view(shape)
    rk = self.relative_k_proj(position_embeddings).view(-1, self.num_heads, self.head_dim)  # [n_rel, H, hd], n_rel = 13
    n_rel = rk.shape[0]
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3)
    ac = q @ k                                                                            # [B,H,T,T]
    qr = q @ rk.permute(1, 2, 0)[None]                                                    # [B,H,T,n_rel]
    i = torch.arange(T, device=hidden_states.device)
    d = i[:, None] - i[None, :]                                                           # q - k
    idx = (n_rel - 1 - d).clamp(0, n_rel - 1)                                             # rel_pos_enc row j <-> distance n_rel-1-j
    bd = torch.gather(qr, 3, idx[None, None].expand(B, self.num_heads, T, T))
    band = torch.where((d >= 0) & (d < self.max_past_horizon), 0.0, MASK_NEG)
    w = torch.tanh((ac + bd) / self.softcap) * self.softcap + band
    w = F.softmax(w, dim=-1)
    out = (w @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
    return self.post(out.to(hidden_states.dtype)), None


class AudioWrap(torch.nn.Module):
    """valid log-mel frames -> audio token embeddings in the text embedding space (audio tower + embed_audio)."""

    def __init__(self, model):
        super().__init__()
        self.tower, self.embed = model.audio_tower, model.embed_audio

    def forward(self, features):
        t = self.tower
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        h, _ = t.subsample_conv_projection(features, mask)
        pos = t.rel_pos_enc(h)
        for layer in t.layers[: t.config.num_hidden_layers]:
            h = layer(h, attention_mask=None, position_embeddings=pos)
        return self.embed(inputs_embeds=t.output_proj(h))


class TrunkWrap(torch.nn.Module):
    """inputs_embeds of one unpadded sequence -> last_hidden_state (768-d, after embedding_projection)."""

    def __init__(self, model):
        super().__init__()
        self.lm = model.language_model
        self.window = self.lm.config.sliding_window

    def forward(self, inputs_embeds):
        L = inputs_embeds.shape[1]
        i = torch.arange(L, device=inputs_embeds.device)
        band = torch.where((i[:, None] - i[None, :]).abs() <= self.window, 0.0, MASK_NEG)[None, None]
        masks = {'full_attention': None, 'sliding_attention': band}
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=masks, position_ids=i[None]).last_hidden_state


def load(path):
    from transformers import AutoModel
    import transformers.models.gemma4.modeling_gemma4 as g4

    model = AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation='eager').eval()
    return model, g4


def patch_audio(g4):
    g4.Gemma4AudioAttention.forward = dense_audio_attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--opset', type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, g4 = load(a.model)
    patch_audio(g4)
    with torch.no_grad():
        feats = torch.randn(1, 301, 128) * 0.5 - 4.0
        torch.onnx.export(AudioWrap(model).eval(), (feats,), f'{a.out}/audio.onnx', input_names=['features'], output_names=['emb'],
                          dynamic_axes={'features': {1: 'F'}, 'emb': {1: 'T'}}, opset_version=a.opset, dynamo=False)
        print('saved audio.onnx', flush=True)
        x = torch.randn(1, 37, 512) * 0.5
        torch.onnx.export(TrunkWrap(model).eval(), (x,), f'{a.out}/trunk.onnx', input_names=['inputs_embeds'], output_names=['hidden'],
                          dynamic_axes={'inputs_embeds': {1: 'L'}, 'hidden': {1: 'L'}}, opset_version=a.opset, dynamo=False)
        print('saved trunk.onnx', flush=True)
        emb = model.language_model.embed_tokens
        table = emb(torch.arange(emb.weight.shape[0])).to(torch.float16).numpy()   # includes the sqrt(512) embedding scale
    table.astype('<f2').tofile(f'{a.out}/embed_tokens_f16.bin')
    head = torch.load(os.path.join(a.model, 'head.pt'), map_location='cpu')
    json.dump({'weight': head['weight'].float().reshape(-1).tolist(), 'bias': float(head['bias'].float().reshape(-1)[0])},
              open(f'{a.out}/head.json', 'w'))
    for f in sorted(os.listdir(a.out)):
        print(f, round(os.path.getsize(os.path.join(a.out, f)) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
