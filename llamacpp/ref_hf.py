# Transformers reference for the parity set: smaug_decide.SmaugDecide on the published checkpoint, recording per case the letter
# logits (softcapped), the probabilities, the prompt length, the rendered prompt and the latency. parity_llama.py compares a GGUF
# against this file. The probabilities are spot-checked against SmaugDecide.__call__.
# usage: ref_hf.py --cases inputs/cases.json --out ref.json [--model abacusai/Smaug-Decisions | <local dir>] [--device cuda|mps|cpu]
#   run from the repository root or with the package installed (pip install -e .)
import argparse
import json
import os
import sys
import time

import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from smaug_decide import SmaugDecide, build_prompt, option_probs  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--model', default='abacusai/Smaug-Decisions', help='Hugging Face repo id or a local checkpoint dir')
ap.add_argument('--device', help='default: cuda, then mps, then cpu')
a = ap.parse_args()

m = SmaugDecide.from_pretrained(a.model, device=a.device)
assert m.readout == 'letters', 'the llama.cpp path covers the letter-readout model (Smaug-Decisions)'
cases = json.load(open(a.cases))
base = os.path.dirname(os.path.abspath(a.cases))


def sync():
    if m.device == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def letter_z(inp, kind, question, options, descriptions=None, multi=False, rules=None, context=None):
    """SmaugDecide.logits plus the softcap, also returning the rendered prompt and its length in tokens."""
    text, audio, image = build_prompt('letters', inp, question, options, descriptions, multi, rules, context, input_type=kind)
    kw = {}
    if audio is not None:
        kw['audio'] = [audio]
    if image is not None:
        kw['images'] = [image]
    enc = m.proc(text=[text], return_tensors='pt', add_special_tokens=False, **kw).to(m.device)
    h = m.model.model(**enc, use_cache=False).last_hidden_state[0, -1].float()
    z = h @ m.model.lm_head.weight[m.letter_ids[: len(options)]].float().T
    z = torch.tanh(z / m.cap) * m.cap
    return z.tolist(), text, int(enc['input_ids'].shape[1])


rows = []
for i, c in enumerate(cases):
    inp = os.path.join(base, c['input']) if c['kind'] != 'text' else c['input']
    kw = {k: c.get(k) for k in ('descriptions', 'multi', 'rules', 'context')}
    kw['multi'] = bool(kw['multi'])
    sync()
    t0 = time.time()
    z, text, n = letter_z(inp, c['kind'], c['question'], c['options'], **kw)
    sync()
    ms = (time.time() - t0) * 1000
    p = option_probs(z, kw['multi'], None, m.b)  # z is already softcapped
    if i < 3 or c['kind'] != 'text' and i % 4 == 0:  # spot-check against the package's own __call__
        ref = m(inp, c['question'], c['options'], input_type=c['kind'], **kw)
        d = max(abs(ref[o] - q) for o, q in zip(c['options'], p))  # bf16 kernels are not bit-deterministic across calls
        assert d < 1e-3, (c['id'], ref, p)
        print(f'  check vs SmaugDecide.__call__: max|dp| {d:.2e}', flush=True)
    rows.append(
        {
            'id': c['id'],
            'kind': c['kind'],
            'multi': kw['multi'],
            'options': c['options'],
            'z': z,
            'p': p,
            'tokens': n,
            'ms': ms,
            'prompt': text,
        }
    )
    print(f'{c["id"]} {c["kind"]:<5} {n:>4} tok {ms:7.0f} ms  p={[round(x, 3) for x in p]}', flush=True)
json.dump({'device': m.device, 'b': m.b, 'rows': rows}, open(a.out, 'w'), indent=1)
print('wrote', a.out)
