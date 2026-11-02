"""Check exported Core ML units against the PyTorch fast tier on your own audio, and time them.

    python scripts/convert/coreml/parity.py --units build/coreml --cases cases.jsonl [--model abacusai/Smaug-Audio-Decisions-0.7B]
                                   [--L 512] [--audio 10] [--compute CPU_AND_NE] [--split] [--out parity.json]

cases.jsonl: one decision per line, {"id", "input": "<audio path>", "question", "options", "descriptions"?, "multi"?, "rules"?,
"context"?} (the same format as `python -m smaug_decide --cases`). Audio cases only; the clip is cut to its last --audio seconds,
because that is the window the device units see.

For each case it runs (1) the PyTorch model in float32 on the window, exactly as SmaugDecide does, and (2) the Core ML path the
app uses: zero-pad the window, mask the padding frames, run the audio unit(s), splice the audio embeddings into the token
embeddings, run the trunk and read the marker head. It prints the largest probability difference per case and the medians, and
writes everything (with the Core ML probabilities and timings) to --out, which make_scenarios.py can bundle as on-device
self-tests. Expect a median difference around 0.01 for int8 units; differences far above that point to a conversion problem.
Runs on macOS only (Core ML prediction).
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
from smaug_decide import SmaugDecide, build_prompt, option_probs  # noqa: E402
from smaug_decide.media import SAMPLE_RATE, load_audio  # noqa: E402


MASK_NEG = -30000.0
MAX_AUDIO_TOKENS = 750


def tokens_for_frames(frames):
    for _ in range(2):
        frames = (frames + 2 - 3) // 2 + 1
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--units', required=True, help='directory with the exported units (convert.py --out)')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--model', default='abacusai/Smaug-Audio-Decisions-0.7B', help='the PyTorch reference: repo id or local directory')
    ap.add_argument('--L', type=int, default=512)
    ap.add_argument('--audio', type=int, default=10, help='audio window in seconds (must match an exported audio unit)')
    ap.add_argument('--quant', default='int8')
    ap.add_argument(
        '--compute', default='CPU_AND_NE', help='compute units for the audio unit / stack: CPU_AND_NE, CPU_AND_GPU, ALL, CPU_ONLY'
    )
    ap.add_argument('--trunk_compute', default=None, help='compute units for the trunk (default: --compute)')
    ap.add_argument('--split', action='store_true', help='use audio_conv + audio_stack instead of the single audio unit')
    ap.add_argument('--conv_compute', default='CPU_AND_GPU')
    ap.add_argument('--out', default=None, help='output JSON (default: <units>/parity.json)')
    a = ap.parse_args()

    import coremltools as ct

    ref = SmaugDecide.from_pretrained(a.model, device='cpu')  # float32 on the CPU: the reference
    assert ref.readout == 'marker', 'the Core ML export is for the fast tier (marker readout)'
    proc, tok, marker_id = ref.proc, ref.proc.tokenizer, ref.marker_id
    W, b = ref.w.numpy().reshape(-1), float(ref.b.reshape(-1)[0])
    emb = ref.model.language_model.embed_tokens
    audio_id = tok.convert_tokens_to_ids('<|audio|>')
    frames_window = (a.audio * SAMPLE_RATE + 160 - 321) // 160 + 1

    def load(name, compute):
        t = time.time()
        m = ct.models.MLModel(os.path.join(a.units, name), compute_units=getattr(ct.ComputeUnit, compute))
        print(f'{name}: load/compile {time.time() - t:.0f}s on {compute}', flush=True)
        return m

    trunk = load(f'trunk_L{a.L}_{a.quant}.mlpackage', a.trunk_compute or a.compute)
    if a.split:
        conv = load(f'audio_conv_{a.audio}s_fp16.mlpackage', a.conv_compute)
        stack = load(f'audio_stack_{a.audio}s_{a.quant}.mlpackage', a.compute)
    else:
        audio_unit = load(f'audio_{a.audio}s_{a.quant}.mlpackage', a.compute)

    def coreml_probs(clip, c):
        k, multi = len(c['options']), c.get('multi', False)
        padded = np.pad(clip, (0, a.audio * SAMPLE_RATE - len(clip)))
        text, _, _ = build_prompt(
            'marker',
            padded,
            c['question'],
            c['options'],
            c.get('descriptions'),
            multi,
            c.get('rules'),
            c.get('context'),
            input_type='audio',
        )
        enc = proc(text=[text], audio=[padded], return_tensors='pt')
        feats = enc['input_features']
        assert feats.shape[1] == frames_window, (feats.shape, frames_window)
        valid = (len(clip) + 160 - 321) // 160 + 1
        fmask = torch.zeros(1, frames_window)
        fmask[0, :valid] = 1.0
        n_tok = min(tokens_for_frames(valid), MAX_AUDIO_TOKENS)
        ids = enc['input_ids'][0].tolist()  # shrink the processor's audio block (sized for the padded window) to the valid tokens
        i0, i1 = ids.index(audio_id), len(ids) - ids[::-1].index(audio_id)
        ids = ids[:i0] + [audio_id] * n_tok + ids[i1:]
        marks = [i for i, t in enumerate(ids) if t == marker_id]
        assert len(marks) == k, (len(marks), k)
        n = len(ids)
        assert n <= a.L, f'{n} tokens > L{a.L}: use a longer trunk or a shorter prompt'
        t0 = time.time()
        feed = {'features': feats.numpy().astype(np.float32), 'mask': fmask.numpy()}
        if a.split:
            hidden = conv.predict(feed)['hidden']
            au = stack.predict({'hidden': hidden.astype(np.float32)})['out'][0]
        else:
            au = audio_unit.predict(feed)['out'][0]
        ms_audio = 1000 * (time.time() - t0)
        with torch.no_grad():
            x = emb(torch.tensor(ids)).numpy().astype(np.float32)
        x[[i for i, t in enumerate(ids) if t == audio_id]] = au[:n_tok]
        xe = np.zeros((1, a.L, x.shape[1]), np.float32)
        xe[0, :n] = x
        mask = np.zeros((1, 1, a.L, a.L), np.float32)
        mask[..., n:] = MASK_NEG
        t1 = time.time()
        h = trunk.predict({'inputs_embeds': xe, 'mask': mask, 'position_ids': np.arange(a.L, dtype=np.int32)[None]})['out'][0]
        ms_trunk = 1000 * (time.time() - t1)
        z = (h[marks] @ W + b).tolist()
        return dict(zip(c['options'], option_probs(z, multi))), ms_audio, ms_trunk, n

    with open(a.cases) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    rows, diffs = [], []
    for i, c in enumerate(cases):
        cid = str(c.get('id', i))
        clip = load_audio(c['input'])[-a.audio * SAMPLE_RATE :]
        p_ref = ref(
            clip,
            c['question'],
            c['options'],
            c.get('descriptions'),
            c.get('multi', False),
            c.get('rules'),
            c.get('context'),
            input_type='audio',
        )
        p_cm, ms_audio, ms_trunk, n = coreml_probs(clip, c)
        d = max(abs(p_cm[o] - p_ref[o]) for o in p_ref)
        diffs.append(d)
        print(
            f'{cid[:40]:<40} {len(clip) / SAMPLE_RATE:4.1f}s {n:3d} tok | max |dp| {d:.4f} | '
            f'audio {ms_audio:.0f} ms trunk {ms_trunk:.0f} ms',
            flush=True,
        )
        rows.append(
            {
                **c,
                'id': cid,
                'window_s': len(clip) / SAMPLE_RATE,
                'tokens': n,
                'torch': p_ref,
                'coreml': p_cm,
                'max_abs_diff': d,
                'ms_audio': ms_audio,
                'ms_trunk': ms_trunk,
                'ms': ms_audio + ms_trunk,
            }
        )
    if not rows:
        sys.exit('no cases')
    print(
        f'\n{len(rows)} cases: max |dp| median {statistics.median(diffs):.4f}, max {max(diffs):.4f}; '
        f'median latency {statistics.median(r["ms"] for r in rows):.0f} ms (the first call includes warm-up)'
    )
    out = a.out or os.path.join(a.units, 'parity.json')
    with open(out, 'w') as f:
        json.dump(
            {'L': a.L, 'audio_s': a.audio, 'quant': a.quant, 'compute': a.compute, 'split': a.split, 'cases': rows}, f, indent=1
        )
    print('wrote', out)


if __name__ == '__main__':
    main()
