"""Export the fast tier (Smaug-Audio-Decisions-0.7B) as fixed-shape Core ML units for iOS and macOS.

    python scripts/convert/coreml/convert.py --model abacusai/Smaug-Audio-Decisions-0.7B --out build/coreml
                                    [--L 512,1024] [--audio 10,30] [--split_audio]

Units (all ML programs, fp16 compute, int8 weights unless noted; minimum deployment target iOS 18 / macOS 15):
    trunk_L{L}_int8         inputs_embeds [1,L,512] + additive mask [1,1,L,L] + position_ids [1,L] -> last_hidden_state [1,L,768]
    audio_{S}s_int8         log-mel features [1,F,128] + frame mask [1,F] -> audio token embeddings [1,T,512]
                            (F = 100 frames/s: 10 s = 999 frames; T = F through two stride-2 convolutions)
    --split_audio           the same audio tower as two units that the Neural Engine can compile (see audio_dense.py):
    audio_conv_{S}s_fp16    features + mask -> subsampled hidden states [1,T,1024]
    audio_stack_{S}s_int8   hidden states -> audio token embeddings [1,T,512]
Plus the constants the app needs: embed_tokens_fp16.bin (the scaled token-embedding table, raw little-endian fp16 [vocab, 512]),
head.json (the marker head's weight and bias) and readout.json (shapes, token ids and the mask constant).

Run on a Mac (coremltools converts on Linux too, but only macOS can load and test the result). Conversion of the trunk takes a few
minutes per sequence length; existing units in --out are kept, delete them to rebuild.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audio_dense import AudioConvFront, DenseAudioStack, tokens_for_frames  # noqa: E402
from patches import (  # noqa: E402  (applies the patches)
    MASK_NEG,
    freeze_audio_mask,
    register_coremltools_ops,
    unfreeze_audio_mask,
)


WEIGHT_FILES = ['*.json', '*.jinja', 'model*.safetensors', 'head.pt']


class TrunkWindowWrap(nn.Module):
    """The text trunk with an explicit additive mask and position ids as inputs. The sliding-attention layers attend only within
    |q - k| <= sliding_window; that band is a graph constant combined with the padding mask, so the inputs stay the same."""

    def __init__(self, model, L):
        super().__init__()
        self.lm = model.language_model
        cfg = model.config.text_config if hasattr(model.config, 'text_config') else model.config
        i = torch.arange(L)
        band = torch.where((i[:, None] - i[None, :]).abs() <= cfg.sliding_window, 0.0, MASK_NEG)[None, None]
        self.register_buffer('band', band, persistent=False)

    def forward(self, inputs_embeds, mask, position_ids):
        masks = {'full_attention': mask, 'sliding_attention': torch.minimum(mask, self.band)}
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=masks, position_ids=position_ids.long()).last_hidden_state


class AudioWrap(nn.Module):
    """log-mel features + frame mask -> audio token embeddings (the HF audio tower and its multimodal embedder)."""

    def __init__(self, model):
        super().__init__()
        self.tower, self.embed = model.audio_tower, model.embed_audio

    def forward(self, features, mask):
        h = self.tower(features, mask > 0.5, return_dict=True).last_hidden_state
        return self.embed(inputs_embeds=h)


def mel_frames(seconds):
    """Frames the Gemma 4 feature extractor makes: 160 zeros prepended, a 321-sample window, hop 160 (10 s -> 999 frames)."""
    n = int(16000 * seconds)
    return (n + 160 - 321) // 160 + 1


def dir_mb(path):
    if os.path.isfile(path):
        return round(os.path.getsize(path) / 1e6, 1)
    return round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='abacusai/Smaug-Audio-Decisions-0.7B', help='Hugging Face repo id or local checkpoint directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--L', default='512,1024', help='trunk sequence lengths')
    ap.add_argument('--audio', default='10,30', help='audio window lengths in seconds')
    ap.add_argument('--quant', default='int8', choices=['int8', 'fp16'])
    ap.add_argument('--split_audio', action='store_true', help='also export the Neural-Engine-friendly split audio units')
    ap.add_argument('--only_split', action='store_true', help='skip the trunk and single-unit audio exports')
    a = ap.parse_args()

    import coremltools as ct
    import coremltools.optimize.coreml as cto
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor

    register_coremltools_ops()

    path = a.model if os.path.isdir(a.model) else snapshot_download(a.model, allow_patterns=WEIGHT_FILES)
    os.makedirs(a.out, exist_ok=True)
    model = AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation='eager').eval()
    proc = AutoProcessor.from_pretrained(path)

    def quantize(m):
        if a.quant != 'int8':
            return m
        cfg = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8', granularity='per_channel')
        )
        return cto.linear_quantize_weights(m, cfg)

    def convert(traced, inputs, output):
        return ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType(name=output)],
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )

    def convert_trunk(L):
        out = f'{a.out}/trunk_L{L}_{a.quant}.mlpackage'
        if os.path.exists(out):
            print('kept', out)
            return
        wrap = TrunkWindowWrap(model, L).eval()
        ex = (torch.randn(1, L, 512) * 0.5, torch.zeros(1, 1, L, L), torch.arange(L, dtype=torch.int32)[None])
        with torch.no_grad():
            traced = torch.jit.trace(wrap, ex, check_trace=False)
        t = time.time()
        m = convert(
            traced,
            [
                ct.TensorType(name='inputs_embeds', shape=(1, L, 512), dtype=np.float32),
                ct.TensorType(name='mask', shape=(1, 1, L, L), dtype=np.float32),
                ct.TensorType(name='position_ids', shape=(1, L), dtype=np.int32),
            ],
            'out',
        )
        quantize(m).save(out)
        print('saved', out, f'{time.time() - t:.0f}s', flush=True)

    def convert_audio(seconds):
        frames, out = mel_frames(seconds), f'{a.out}/audio_{seconds}s_{a.quant}.mlpackage'
        if os.path.exists(out):
            print('kept', out)
            return
        wrap = AudioWrap(model).eval()
        feats, mask = torch.randn(1, frames, 128) * 0.5 - 4.0, torch.ones(1, frames)
        freeze_audio_mask(model.audio_tower, feats, mask)
        with torch.no_grad():
            traced = torch.jit.trace(wrap, (feats, mask), check_trace=False)
        t = time.time()
        m = convert(
            traced,
            [
                ct.TensorType(name='features', shape=(1, frames, 128), dtype=np.float32),
                ct.TensorType(name='mask', shape=(1, frames), dtype=np.float32),
            ],
            'out',
        )
        quantize(m).save(out)
        print('saved', out, f'{time.time() - t:.0f}s', flush=True)
        unfreeze_audio_mask(model.audio_tower)

    def convert_audio_split(seconds):
        frames = mel_frames(seconds)
        T, hid = tokens_for_frames(frames), model.audio_tower.config.hidden_size
        out_conv, out_stack = (
            f'{a.out}/audio_conv_{seconds}s_fp16.mlpackage',
            f'{a.out}/audio_stack_{seconds}s_{a.quant}.mlpackage',
        )
        if not os.path.exists(out_conv):
            front = AudioConvFront(model.audio_tower).eval()
            feats, mask = torch.randn(1, frames, 128) * 0.5 - 4.0, torch.ones(1, frames)
            with torch.no_grad():
                traced = torch.jit.trace(front, (feats, mask), check_trace=False)
            m = convert(
                traced,
                [
                    ct.TensorType(name='features', shape=(1, frames, 128), dtype=np.float32),
                    ct.TensorType(name='mask', shape=(1, frames), dtype=np.float32),
                ],
                'hidden',
            )
            m.save(out_conv)  # under a million weights: kept in fp16
            print('saved', out_conv, flush=True)
        else:
            print('kept', out_conv)
        if os.path.exists(out_stack):
            print('kept', out_stack)
            return
        stack = DenseAudioStack(copy.deepcopy(model.audio_tower), model.embed_audio, T).eval()
        with torch.no_grad():
            traced = torch.jit.trace(stack, (torch.randn(1, T, hid),), check_trace=False)
        t = time.time()
        m = convert(traced, [ct.TensorType(name='hidden', shape=(1, T, hid), dtype=np.float32)], 'out')
        quantize(m).save(out_stack)
        print('saved', out_stack, f'{time.time() - t:.0f}s', 'tokens', T, flush=True)

    seconds = [int(s) for s in a.audio.split(',')]
    if not a.only_split:
        for L in a.L.split(','):
            convert_trunk(int(L))
        for s in seconds:
            convert_audio(s)
    if a.split_audio:
        for s in seconds:
            convert_audio_split(s)

    # constants for the app: the scaled token embeddings the trunk expects, the marker head and the readout description
    with torch.no_grad():
        emb = model.language_model.embed_tokens
        table = emb(torch.arange(emb.weight.shape[0])).to(torch.float16).numpy()  # [vocab, 512], includes the embedding scale
    table.astype('<f2').tofile(f'{a.out}/embed_tokens_fp16.bin')
    head = torch.load(os.path.join(path, 'head.pt'), map_location='cpu', weights_only=True)
    with open(f'{a.out}/head.json', 'w') as f:
        json.dump(
            {'weight': head['weight'].float().numpy().reshape(-1).tolist(), 'bias': float(head['bias'].float().reshape(-1)[0])}, f
        )
    with open(os.path.join(path, 'readout.json')) as f:
        readout = json.load(f)
    readout = {k: v for k, v in readout.items() if k not in ('source_run', 'base_model')}  # keep only what a runtime needs
    tok = proc.tokenizer
    readout.update(
        {
            'audio_frames': {str(s): mel_frames(s) for s in seconds},
            'audio_tokens': {str(s): tokens_for_frames(mel_frames(s)) for s in seconds},
            'audio_hidden': model.audio_tower.config.hidden_size,
            'vocab': int(table.shape[0]),
            'embed_dim': int(table.shape[1]),
            'hidden_out': 768,
            'mask_neg': MASK_NEG,
            'audio_frames_per_s': 100,
            'audio_frames_per_token': 4,
            'audio_token': '<|audio|>',
            'audio_token_id': tok.convert_tokens_to_ids('<|audio|>'),
            'embed_table': 'embed_tokens_fp16.bin: raw little-endian float16, row-major [vocab, embed_dim]',
        }
    )
    with open(f'{a.out}/readout.json', 'w') as f:
        json.dump(readout, f, indent=1)
    print('sizes (MB):', {name: dir_mb(os.path.join(a.out, name)) for name in sorted(os.listdir(a.out))})


if __name__ == '__main__':
    main()
