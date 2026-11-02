"""Check an MLX build of Smaug-Decisions against reference probabilities from the PyTorch model, on your own cases.

    # 1. reference probabilities, from the PyTorch model (a CUDA box, or the same Mac with MPS)
    python -m smaug_decide --model abacusai/Smaug-Decisions --cases cases.jsonl --out reference.jsonl
    # 2. the MLX build on the Mac
    python scripts/convert/mlx/parity.py --mlx build/Smaug-Decisions-mlx4 --cases cases.jsonl --reference reference.jsonl

cases.jsonl: {"id", "input", "question", "options", "descriptions"?, "multi"?, "rules"?, "context"?} per line; audio and text
inputs. The script renders each case exactly as SmaugDecide does, runs the MLX model once, reads the letter logits at the last
position and applies the same readout. It prints the largest probability difference per case and the median; without
--reference it just prints the MLX probabilities. A 4-bit build differs more than an 8-bit one; for select-all questions see
fix_multi_bias.py.
"""

import argparse
import inspect
import json
import os
import statistics
import sys
import time

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
from smaug_decide import LETTERS, build_prompt, option_probs  # noqa: E402


class MLXDecide:
    """The letter readout on an MLX build: same prompt, same positions, same math as SmaugDecide."""

    def __init__(self, path, cap='auto'):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model, self.proc = load(path)
        tok = self.proc.tokenizer
        ids = [tok(L, add_special_tokens=False)['input_ids'] for L in LETTERS]
        assert all(len(i) == 1 for i in ids), 'every option letter must be a single token'
        self.letter_ids = [i[0] for i in ids]
        with open(os.path.join(path, 'readout.json')) as f:
            readout = json.load(f)
        self.b = float(readout['multi_bias_b'])
        # the softcap must be applied exactly once: skip it here when the MLX language model already applies it to its logits
        source = inspect.getsource(sys.modules[type(self.model.language_model).__module__])
        applied_by_model = 'final_logit_softcapping' in source
        if cap == 'auto':
            self.cap = None if applied_by_model else readout.get('final_logit_softcapping')
        else:
            self.cap = None if cap == 'none' else float(cap)
        print(f'{path}: softcap applied by the model: {applied_by_model}; applied here: {self.cap}', flush=True)

    def __call__(self, inp, question, options, descriptions=None, multi=False, rules=None, context=None):
        mx = self.mx
        text, audio, image = build_prompt('letters', inp, question, options, descriptions, multi, rules, context)
        if image is not None:
            raise NotImplementedError('this harness checks audio and text inputs')
        kw = {'audio': [audio]} if audio is not None else {}
        enc = self.proc(text=[text], return_tensors='np', add_special_tokens=False, **kw)
        extra = {k: mx.array(np.asarray(enc[k])) for k in enc if k != 'input_ids'}
        out = self.model(mx.array(np.asarray(enc['input_ids'])), **extra)
        logits = out.logits if hasattr(out, 'logits') else out
        z = np.array(logits[0, -1, self.letter_ids[: len(options)]].astype(mx.float32)).tolist()
        return dict(zip(options, option_probs(z, multi, self.cap, self.b))), int(enc['input_ids'].shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mlx', required=True, help='MLX build directory (convert.sh output, with readout.json)')
    ap.add_argument('--cases', required=True)
    ap.add_argument('--reference', default=None, help='JSONL from `python -m smaug_decide --cases` on the PyTorch model')
    ap.add_argument('--cap', default='auto', help='auto | none | <float>: softcap for the letter logits')
    ap.add_argument('--out', default=None, help='write per-case results as JSON')
    a = ap.parse_args()

    model = MLXDecide(a.mlx, a.cap)
    ref = {}
    if a.reference:
        with open(a.reference) as f:
            ref = {str(r['id']): r['probabilities'] for r in (json.loads(line) for line in f if line.strip())}
    with open(a.cases) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    diffs, rows = [], []
    for i, c in enumerate(cases):
        cid = str(c.get('id', i))
        t0 = time.time()
        p, n = model(
            c['input'],
            c['question'],
            c.get('options', ['Yes', 'No']),
            c.get('descriptions'),
            c.get('multi', False),
            c.get('rules'),
            c.get('context'),
        )
        ms = 1000 * (time.time() - t0)
        row = {'id': cid, 'mlx': p, 'tokens': n, 'ms': ms}
        if cid in ref:
            row['max_abs_diff'] = max(abs(p[o] - ref[cid][o]) for o in p)
            diffs.append(row['max_abs_diff'])
        print(
            f'{cid[:40]:<40} {n:5d} tok {ms:6.0f} ms' + (f' | max |dp| {row["max_abs_diff"]:.4f}' if cid in ref else f' | {p}'),
            flush=True,
        )
        rows.append(row)
    if diffs:
        print(
            f'\n{len(diffs)} cases: max |dp| median {statistics.median(diffs):.4f}, max {max(diffs):.4f} '
            '(the first call includes warm-up)'
        )
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
