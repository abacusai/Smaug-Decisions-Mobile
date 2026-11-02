"""Does the dense band attention (audio_dense.py) equal the HF chunked attention? A random-weight Gemma 4 audio tower (real config
shape, a few layers): HF forward vs AudioConvFront -> DenseAudioStack on the same features, with and without padding frames
(padding must not change the valid prefix: the attention is causal, so the app can mask the padded tail and read the first N
rows).
Needs no checkpoint. Expect max abs differences around 1e-5.

    python scripts/convert/coreml/check_audio_dense.py [layers]
"""

import copy
import os
import sys
import time

import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_dense import AudioConvFront, DenseAudioStack, tokens_for_frames  # noqa: E402
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4TextConfig  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder  # noqa: E402


torch.manual_seed(0)
if len(sys.argv) > 1 and not sys.argv[1].isdigit():
    sys.exit('usage: check_audio_dense.py [layers, default 3]')
layers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cfg = Gemma4AudioConfig(num_hidden_layers=layers)
tower = Gemma4AudioModel(cfg).eval()
with (
    torch.no_grad()
):  # non-trivial relative bias and clipped linears: default init leaves per_dim_scale 0, relative_k small, bounds +-inf
    for l in tower.layers:
        l.self_attn.per_dim_scale.normal_(0, 0.5)
        l.self_attn.relative_k_proj.weight.normal_(0, 0.05)
        for cl in (
            l.feed_forward1.ffw_layer_1,
            l.self_attn.q_proj,
            l.lconv1d.linear_end,
        ):  # finite bounds like the checkpoint's, tight enough to bite
            cl.input_min.fill_(-0.8)
            cl.input_max.fill_(0.9)
            cl.output_min.fill_(-0.5)
            cl.output_max.fill_(0.6)
embed = Gemma4MultimodalEmbedder(cfg, Gemma4TextConfig(hidden_size=512)).eval()
F = 999
T = tokens_for_frames(F)
feats = torch.randn(1, F, 128) * 0.5 - 4.0
for valid in (F, 611):
    mask = torch.zeros(1, F, dtype=torch.bool)
    mask[0, :valid] = True
    with torch.no_grad():
        t = time.time()
        ref = embed(tower(feats, mask, return_dict=True).last_hidden_state)
        t_ref = time.time() - t
        dense = DenseAudioStack(copy.deepcopy(tower), embed, T).eval()
        front = AudioConvFront(dense.tower)
        t = time.time()
        got = dense(front(feats, mask.float()))
        t_dense = time.time() - t
    n = tokens_for_frames(valid)
    print(
        f'valid frames {valid} ({n} of {T} tokens): max abs diff on valid rows {(ref[0, :n] - got[0, :n]).abs().max():.2e} '
        f'(ref abs max {ref.abs().max():.2f}) '
        f'| HF {t_ref * 1000:.0f} ms dense {t_dense * 1000:.0f} ms'
    )
    if valid == F:
        full = got
    else:
        print(
            f'  padded vs unpadded run: valid rows but the last two {(full[0, : n - 2] - got[0, : n - 2]).abs().max():.2e} '
            '(the convs see one zeroed frame past the valid ones)'
        )
