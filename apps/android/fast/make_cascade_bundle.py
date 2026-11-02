"""Cascade self-test bundle (app/src/main/assets/cascade_selftest): requests the app's Cascade must route and answer exactly like
the Python reference of the same on-device path.

For every audio request (a synthetic clip from make_clips.py x an app scenario) it records:
- p_torch: the audio model in PyTorch float32 (the fast tier's reference), and its confidence;
- p_onnx: the phone's fast-tier path replayed in Python (mel.py + the bundled ONNX units + fp16 table + head), and its confidence;
- route: the cascade decision on p_onnx with smaug_decide.cascade.confidence (imported from the package) at tau, i.e. what the app
  must decide; route_torch: the same decision on p_torch (flips between the two are reported, they come from int8 noise);
- for escalated requests: Smaug-Decisions through llama_decide.py (the same C core / library the app's JNI path links) with the
  scenario's question, options and descriptions: prompt, positions, p_big.
Text and image requests are routed to Smaug-Decisions (no fast pass) and get p_big the same way.
usage: DEC_NO_REPACK=1 LLAMA_DECIDE_LIB=<host libdecide_jni.so> make_cascade_bundle.py --fast_model <audio model dir> --onnx <dir> --scenarios <scenarios.json>
       --clips <make_clips dir> --images <dir with an image> --gguf X --mmproj Y --out <assets/cascade_selftest> [--tau 0.9]
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'llama.cpp'))
from mel import frame_count, log_mel  # noqa: E402

# (clip, scenario) pairs: every audio scenario on a spread of clips, plus extra pairs
CLIPS = ['ask_station.wav', 'done_turn.wav', 'unfinished.wav', 'complaint.wav', 'two_speakers.wav', 'music_only.wav', 'siren.wav',
         'quiet_room.wav', 'blip_short.wav', 'long_12s.wav', 'long_28s.wav', 'resample_44k_stereo.wav']
TEXTS = [('triage', None), ('triage', 'Thanks so much, the replacement works perfectly!')]


def tokens_for_frames(f):
    for _ in range(2):
        f = (f + 2 - 3) // 2 + 1
    return min(f, 750)


def main():
    ap = argparse.ArgumentParser()
    for k in ('fast_model', 'onnx', 'scenarios', 'clips', 'images', 'gguf', 'mmproj', 'out'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--tau', type=float, default=0.9); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--smaug-pkg', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')); ap.add_argument('--per-clip', type=int, default=0, help='scenarios per clip (default all)')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('sd_cascade', os.path.join(a.smaug_pkg, 'smaug_decide', 'cascade.py'))
    cmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmod); confidence = cmod.confidence
    sys.path.insert(0, a.fast_model)
    import torch
    import smaug_decide as sd   # the audio model's smaug_decide.py
    from llama_decide import LlamaDecide, load_audio as big_load_audio
    from PIL import Image

    torch.set_num_threads(a.threads)
    fast = sd.SmaugDecide.from_pretrained(a.fast_model, device='cpu')
    W, B = fast.w.numpy().reshape(-1).astype(np.float32), float(fast.b.reshape(-1)[0])
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sa = ort.InferenceSession(os.path.join(a.onnx, 'audio.onnx'), so); st = ort.InferenceSession(os.path.join(a.onnx, 'trunk.onnx'), so)
    table = np.fromfile(os.path.join(a.onnx, 'embed_tokens_f16.bin'), dtype='<f2').reshape(-1, 512)
    sc = json.load(open(a.scenarios)); scen = {s['id']: s for s in sc['scenarios']}; AUD = sc['audio_token_id']
    big = LlamaDecide(a.gguf, a.mmproj, readout=None, b=8.010554313659668, n_threads=a.threads)
    os.makedirs(a.out, exist_ok=True)

    def onnx_fast(w, s):
        F = frame_count(len(w)); n = tokens_for_frames(F)
        emb = sa.run(None, {'features': log_mel(w)[None]})[0][0]
        ids = s['pre_ids'] + [AUD] * n + s['post_ids']
        x = table[ids].astype(np.float32); x[len(s['pre_ids']): len(s['pre_ids']) + n] = emb[:n]
        h = st.run(None, {'inputs_embeds': x[None]})[0][0]
        marks = [len(s['pre_ids']) + n + o for o in s['marker_offsets_in_post']]
        z = (h[marks] @ W + B).astype(np.float64)
        return (1 / (1 + np.exp(-z)) if s['multi'] else np.exp(z - z.max()) / np.exp(z - z.max()).sum()), len(ids)

    def run_big(inp, s, kind):
        t0 = time.time()
        z, prompt = big.letter_logits(inp, s['question'], s['options'], s['descriptions'], s['multi'], s.get('rules'), None)
        p = 1 / (1 + np.exp(-(z - big.b))) if s['multi'] else np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        return {'p_big': p.tolist(), 'z_big': z.tolist(), 'prompt_big': prompt, 'positions_big': big.last_tokens, 'ms_big_py': 1000 * (time.time() - t0)}

    reqs = []
    audio_ids = [s['id'] for s in sc['scenarios'] if s['kind'] == 'audio']
    per = max(1, a.per_clip) if a.per_clip else len(audio_ids)
    # each clip gets `per` scenarios, rotated from clip to clip so every scenario is covered
    pairs = [(c, audio_ids[(i + k * (len(audio_ids) // per)) % len(audio_ids)]) for i, c in enumerate(CLIPS) for k in range(per)]
    for clip, sid in pairs:
        s = scen[sid]; src = os.path.join(a.clips, clip); shutil.copy(src, os.path.join(a.out, clip))
        w = sd.load_audio(src)
        assert np.array_equal(w, big_load_audio(src)), 'the audio model and Smaug-Decisions load_audio differ'
        pt = fast(src, s['question'], s['options'], s['descriptions'], s['multi'], s.get('rules'))
        p_torch = [pt[o] for o in s['options']]
        p_onnx, n_tok = onnx_fast(w, s)
        c_t, c_o = confidence(dict(enumerate(p_torch)), s['multi']), confidence(dict(enumerate(p_onnx.tolist())), s['multi'])
        r = {'id': f'{clip[:-4]}:{sid}', 'kind': 'audio', 'scenario': sid, 'input': clip, 'p_torch': p_torch, 'p_onnx': p_onnx.tolist(),
             'conf_torch': c_t, 'conf_onnx': c_o, 'fast_tokens': n_tok, 'route': 'fast' if c_o >= a.tau else 'escalated',
             'route_torch': 'fast' if c_t >= a.tau else 'escalated'}
        if r['route'] == 'escalated':
            r.update(run_big(src, s, 'audio'))
        reqs.append(r)
        print(r['id'], r['route'], f"conf onnx {c_o:.3f} torch {c_t:.3f}", 'big' if 'p_big' in r else '', flush=True)
    for sid, text in TEXTS:
        s = scen[sid]; text = text or s['text']
        r = {'id': f'text:{sid}:{len(reqs)}', 'kind': 'text', 'scenario': sid, 'input': text, 'route': 'routed'}
        r.update(run_big(text, s, 'text')); reqs.append(r); print(r['id'], 'routed', flush=True)
    for f in sorted(os.listdir(a.images))[:1]:
        s = scen['photo_people']; shutil.copy(os.path.join(a.images, f), os.path.join(a.out, f))
        r = {'id': f'image:{f}', 'kind': 'image', 'scenario': 'photo_people', 'input': f, 'route': 'routed'}
        r.update(run_big(os.path.join(a.out, f), s, 'image')); reqs.append(r); print(r['id'], 'routed', flush=True)
    meta = {'tau': a.tau, 'gguf': os.path.basename(a.gguf), 'mmproj': os.path.basename(a.mmproj), 'onnx': os.path.basename(os.path.normpath(a.onnx)),
            'b': big.b, 'threads': a.threads, 'dec_no_repack': os.environ.get('DEC_NO_REPACK', '0'), 'requests': reqs,
            'generated_by': 'fast/make_cascade_bundle.py: PyTorch Smaug-Audio-Decisions-0.7B + Python replay of the ONNX path + llama_decide.py (same C core)'}
    json.dump(meta, open(os.path.join(a.out, 'expected.json'), 'w'), ensure_ascii=False, indent=1)
    n_esc = sum(r['route'] == 'escalated' for r in reqs); flips = sum(r.get('route_torch', r['route']) != r['route'] for r in reqs)
    print(f'{len(reqs)} requests, {n_esc} escalated, {flips} route flips torch vs onnx ->', os.path.join(a.out, 'expected.json'))


if __name__ == '__main__':
    main()
