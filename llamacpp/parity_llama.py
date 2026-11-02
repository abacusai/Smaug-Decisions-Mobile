# llama.cpp vs Transformers parity on the parity set: run LlamaDecide (GGUF + mmproj) on every case of a ref_hf.py output and
# compare probabilities (|dp| per option, max per case), the mean letter-logit shift (z_ref - z_gguf; the select-all readout
# sigmoid(z - b) is not shift-invariant), prompt lengths, and the rendered prompt itself (asserted identical).
# usage: parity_llama.py --ref ref.json --cases inputs/cases.json --gguf X.gguf [--mmproj mm.gguf] [--ngl 0|-1] [--threads N]
#        [--repeat 3] --out p.json
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_decide import LlamaDecide  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument('--ref', required=True)
ap.add_argument('--cases', required=True)
ap.add_argument('--gguf', required=True)
ap.add_argument('--mmproj')
ap.add_argument('--ngl', type=int, default=0)
ap.add_argument('--threads', type=int)
ap.add_argument('--out', required=True)
ap.add_argument('--readout', help='readout json whose multi_bias_b replaces the reference b (to test a corrected threshold)')
ap.add_argument('--repeat', type=int, default=1, help='timing: run each case this many times, report the median')
a = ap.parse_args()
ref = json.load(open(a.ref))
cases = {c['id']: c for c in json.load(open(a.cases))}
base = os.path.dirname(os.path.abspath(a.cases))
b = json.load(open(a.readout))['multi_bias_b'] if a.readout else ref['b']
t0 = time.time()
m = LlamaDecide(a.gguf, a.mmproj, b=b, n_gpu_layers=a.ngl, n_threads=a.threads)
load_s = time.time() - t0
rows, dps, shifts = [], {'text': [], 'image': [], 'audio': []}, []
for r in ref['rows']:
    c = cases[r['id']]
    if c['kind'] != 'text' and not a.mmproj:
        continue
    inp = os.path.join(base, c['input']) if c['kind'] != 'text' else c['input']
    kw = dict(descriptions=c.get('descriptions'), multi=bool(c.get('multi')), rules=c.get('rules'), context=c.get('context'))
    ms = []
    for _ in range(a.repeat):
        z, text = m.letter_logits(inp, c['question'], c['options'], **kw)
        ms.append(m.last_ms)
    assert text == r['prompt'], f'{r["id"]}: rendered prompt differs from smaug_decide.render'
    p = 1 / (1 + np.exp(-(z - b))) if kw['multi'] else np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    dp = float(np.max(np.abs(p - np.array(r['p']))))
    dps[c['kind']].append(dp)
    sh = float(np.mean(np.array(r['z']) - z))
    shifts.append(sh)
    rows.append(
        {
            'id': r['id'],
            'kind': c['kind'],
            'multi': kw['multi'],
            'max_dp': dp,
            'z_shift': sh,
            'p': p.tolist(),
            'p_ref': r['p'],
            'z': z.tolist(),
            'z_ref': r['z'],
            'tokens': m.last_tokens,
            'tokens_ref': r['tokens'],
            'ms': statistics.median(ms),
            'argmax_agree': int(np.argmax(p) == np.argmax(r['p'])),
        }
    )
    print(
        f'{r["id"]} {c["kind"]:<5} max|dp|={dp:.4f} shift={sh:+.3f} tok {m.last_tokens}/{r["tokens"]} '
        f'{statistics.median(ms):7.0f} ms',
        flush=True,
    )
alld = [x for v in dps.values() for x in v]
summ = {
    'gguf': os.path.basename(a.gguf),
    'b': b,
    'mmproj': os.path.basename(a.mmproj or ''),
    'ngl': a.ngl,
    'n': len(alld),
    'load_s': load_s,
    'median_dp': statistics.median(alld),
    'max_dp': max(alld),
    'mean_z_shift': statistics.mean(shifts),
    'argmax_agree': sum(r['argmax_agree'] for r in rows) / len(rows),
    'per_kind': {
        k: {
            'n': len(v),
            'median_dp': statistics.median(v),
            'max_dp': max(v),
            'median_ms': statistics.median([r['ms'] for r in rows if r['kind'] == k]),
        }
        for k, v in dps.items()
        if v
    },
}
print(json.dumps(summ, indent=1))
json.dump({'summary': summ, 'rows': rows}, open(a.out, 'w'), indent=1)
