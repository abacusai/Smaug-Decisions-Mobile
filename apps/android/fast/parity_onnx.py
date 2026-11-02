"""Fast tier: ONNX Runtime (the Android path, replayed in Python) vs the PyTorch release model on your own clips.

For each case of a cases.jsonl (make_clips.py), the reference is the release's own SmaugDecide (the audio model's smaug_decide.py, float32,
CPU). The ONNX path is what the phone does: the clip's log-mel (mel.py, the spec of the Kotlin MelFrontEnd), audio.onnx on the
valid frames, the fp16 token-embedding table gathered for the prompt ids with the audio rows spliced in, trunk.onnx, the marker
head, softmax / sigmoid. Also reports the same path with the HF feature extractor's features (isolates the mel port) and the
mel difference itself. Writes per-case rows + a summary JSON.
usage: parity_onnx.py --model <audio model dir> --onnx <dir> [--onnx <dir2> ...] --cases clips/cases.jsonl --out parity.json [--threads 4]
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mel import frame_count, log_mel  # noqa: E402


def tokens_for_frames(f):
    for _ in range(2):
        f = (f + 2 - 3) // 2 + 1
    return f


def probs(z, multi):
    z = np.asarray(z, np.float64)
    if multi:
        return 1 / (1 + np.exp(-z))
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--onnx', action='append', required=True)
    ap.add_argument('--cases', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--features', default='kotlin_mel,hf_features', help='which front ends to run through the ONNX path')
    a = ap.parse_args()
    sys.path.insert(0, a.model)
    import torch
    import smaug_decide as sd  # the release file in the model dir

    torch.set_num_threads(a.threads)
    ref = sd.SmaugDecide.from_pretrained(a.model, device='cpu')
    proc, tok = ref.proc, ref.proc.tokenizer
    AUD = tok.convert_tokens_to_ids('<|audio|>')
    W, B = ref.w.numpy().reshape(-1).astype(np.float32), float(ref.b.reshape(-1)[0])
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    variants = []
    for d in a.onnx:
        table = np.fromfile(os.path.join(d, 'embed_tokens_f16.bin'), dtype='<f2').reshape(-1, 512)
        variants.append((os.path.basename(os.path.normpath(d)), ort.InferenceSession(os.path.join(d, 'audio.onnx'), so),
                         ort.InferenceSession(os.path.join(d, 'trunk.onnx'), so), table))
    base = os.path.dirname(os.path.abspath(a.cases))
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    if a.limit:
        cases = cases[: a.limit]
    rows = []
    for c in cases:
        path = os.path.join(base, c['input'])
        opts, multi = c['options'], bool(c.get('multi'))
        desc = c.get('descriptions') or [None] * len(opts)
        t0 = time.time(); p_ref = ref(path, c['question'], opts, desc, multi); ms_ref = 1000 * (time.time() - t0)
        w = sd.load_audio(path)
        # the exact prompt ids the reference used
        budget = int(max(300, (sd.MAX_TOKENS - (len(w) / sd.SAMPLE_RATE * sd.AUDIO_TOKENS_PER_SECOND + 2) - 60 - 8 * len(opts)) * 3.5))
        text = sd.render(c['question'], opts, desc, multi, '<|audio|>', None, None, None, budget)
        enc = proc(text=[text], audio=[w], return_tensors='np')
        ids = enc['input_ids'][0].tolist()
        n_aud = ids.count(AUD)
        F = frame_count(len(w)); T = min(tokens_for_frames(F), 750)
        assert n_aud == T, (c['id'], n_aud, T, F)
        marks = [i for i, t in enumerate(ids) if t == ref.marker_id]
        assert len(marks) == len(opts)
        mel_k = log_mel(w)
        hf = enc['input_features'][0][:F].astype(np.float32)
        mel_diff = float(np.abs(mel_k - hf).max())
        row = {'id': c['id'], 'seconds': round(len(w) / 16000, 2), 'tokens': len(ids), 'audio_tokens': T, 'mel_max_abs_diff': mel_diff,
               'p_ref': [p_ref[o] for o in opts], 'ms_ref': ms_ref, 'multi': multi, 'variants': {}}
        aud_pos = [i for i, t in enumerate(ids) if t == AUD]
        for name, sa, st, table in variants:
            out = {}
            for feat_name, feats in [(k, v) for k, v in (('kotlin_mel', mel_k), ('hf_features', hf)) if k in a.features.split(',')]:
                t1 = time.time(); emb = sa.run(None, {'features': feats[None]})[0][0]; ms_a = 1000 * (time.time() - t1)
                x = table[ids].astype(np.float32)
                x[aud_pos] = emb[:T]
                t2 = time.time(); h = st.run(None, {'inputs_embeds': x[None]})[0][0]; ms_t = 1000 * (time.time() - t2)
                p = probs(h[marks] @ W + B, multi)
                out[feat_name] = {'p': p.tolist(), 'max_dp': float(np.max(np.abs(p - np.array(row['p_ref'])))), 'ms_audio': ms_a, 'ms_trunk': ms_t,
                                  'argmax_agree': int(np.argmax(p) == int(np.argmax(row['p_ref'])))}
            row['variants'][name] = out
        rows.append(row)
        print(c['id'][:40].ljust(40), f"{row['seconds']:5.1f}s {len(ids):4d} tok mel {mel_diff:.1e} |",
              ' '.join(f"{n}: " + '/'.join(f"{x['max_dp']:.4f}" for x in v.values()) + f" ({next(iter(v.values()))['ms_audio'] + next(iter(v.values()))['ms_trunk']:.0f} ms)"
                       for n, v in row['variants'].items()), flush=True)
    summ = {'n': len(rows), 'mel_max_abs_diff': max(r['mel_max_abs_diff'] for r in rows), 'variants': {}}
    for name, *_ in variants:
        for fn in a.features.split(','):
            d = [r['variants'][name][fn]['max_dp'] for r in rows]
            ms = [r['variants'][name][fn]['ms_audio'] + r['variants'][name][fn]['ms_trunk'] for r in rows]
            agree = [r['variants'][name][fn]['argmax_agree'] for r in rows if not r['multi']]
            summ['variants'][f'{name}/{fn}'] = {'median_dp': statistics.median(d), 'p90_dp': float(np.percentile(d, 90)), 'max_dp': max(d),
                                                'argmax_agree_single': f'{sum(agree)}/{len(agree)}', 'median_ms': statistics.median(ms)}
    print(json.dumps(summ, indent=1))
    json.dump({'summary': summ, 'rows': rows}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
