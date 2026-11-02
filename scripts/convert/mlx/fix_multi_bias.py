"""Correct the select-all threshold of a quantised MLX build, without labels.

Quantisation shifts the letter logits by a roughly constant amount. Softmax (one-answer questions) does not care, but the
select-all readout sigmoid(z - b) does. This script measures the mean logit shift of a quantised build against a reference build
(for example 4-bit against 8-bit) on your own audio clips, and writes the quantised build's readout.json with b corrected by that
shift (the original value is kept as multi_bias_b_ref).

    python scripts/convert/mlx/fix_multi_bias.py --ref build/Smaug-Decisions-mlx8 --quant build/Smaug-Decisions-mlx4 \
        --audio-dir my_clips/ [--n 40]

Any speech, music or ambient recordings work; the questions below are generic and their answers are not used.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity import MLXDecide  # noqa: E402


QUESTIONS = [
    ('Which of these are present in the audio?', ['speech', 'music', 'background noise', 'silence', 'a vehicle', 'an animal']),
    ('Which apply to the speaker?', ['one person', 'several people', 'male voice', 'female voice', 'child', 'shouting']),
    ('What is the recording environment?', ['indoors', 'outdoors', 'in a car', 'near water', 'a crowd', 'wind']),
]


def select_all_logits(model, files):
    """Raw letter logits (no softcap, no bias) for every clip x question."""
    import mlx.core as mx

    from smaug_decide import build_prompt

    out = []
    for f in files:
        for question, options in QUESTIONS:
            text, audio, _ = build_prompt('letters', f, question, options, multi=True, input_type='audio')
            enc = model.proc(text=[text], audio=[audio], return_tensors='np', add_special_tokens=False)
            extra = {k: mx.array(np.asarray(enc[k])) for k in enc if k != 'input_ids'}
            res = model.model(mx.array(np.asarray(enc['input_ids'])), **extra)
            logits = res.logits if hasattr(res, 'logits') else res
            out.append(np.array(logits[0, -1, model.letter_ids[: len(options)]].astype(mx.float32)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help='reference MLX build (e.g. 8-bit)')
    ap.add_argument('--quant', required=True, help='quantised MLX build to correct (e.g. 4-bit)')
    ap.add_argument('--audio-dir', required=True, help='directory of .wav / .flac clips')
    ap.add_argument('--n', type=int, default=40)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.audio_dir, '*.wav')) + glob.glob(os.path.join(a.audio_dir, '*.flac')))[: a.n]
    if not files:
        sys.exit(f'no .wav or .flac files in {a.audio_dir}')
    z_ref = select_all_logits(MLXDecide(a.ref), files)
    z_q = select_all_logits(MLXDecide(a.quant), files)
    d = np.concatenate([r - q for r, q in zip(z_ref, z_q)])
    print(f'{len(files)} clips x {len(QUESTIONS)} questions: mean shift (ref - quant) {d.mean():+.3f} logits, sd {d.std():.3f}')
    path = os.path.join(a.quant, 'readout.json')
    with open(path) as f:
        readout = json.load(f)
    readout.setdefault('multi_bias_b_ref', readout['multi_bias_b'])
    readout['multi_bias_b'] = float(readout['multi_bias_b_ref'] - d.mean())
    readout['logit_shift_vs_ref'] = float(d.mean())
    with open(path, 'w') as f:
        json.dump(readout, f, indent=1)
    print('wrote', path, 'multi_bias_b', readout['multi_bias_b'])


if __name__ == '__main__':
    main()
