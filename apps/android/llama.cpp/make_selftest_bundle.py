# Self-test bundle for the Android app / host JVM test (android/app/src/main/assets/selftest): the parity cases + a few extra
# self-made unicode text cases, their media, and the Python path's expected output (llama_decide.py through the same C core:
# rendered prompt, positions, letter logits z, probabilities p), plus the HF bf16 reference p when given.
# JPEG inputs are also stored as a lossless PNG of PIL's decode (the parity row), and kept as JPEG in a "decoder row" that measures
# only the platform's JPEG decoder (BitmapFactory / ImageIO vs PIL); decoder rows are excluded from the parity summary.
# usage: LLAMA_DECIDE_LIB=<libdecide.so or libdecide_jni.so> make_selftest_bundle.py --cases inputs/cases.json --gguf X.gguf
#        --mmproj mm.gguf --out <assets/selftest> [--hf-ref ref_cpu_bf16.json] [--threads 4]
import argparse, json, os, platform, shutil, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_decide import LlamaDecide  # noqa: E402

EXTRA = [  # self-made; emoji / CJK exercise the UTF-8 path into llama.cpp (JNI must not pass modified UTF-8)
    {'id': 'u00', 'kind': 'text', 'input': 'Loved the color 😍😍 but the box arrived crushed 😡📦', 'question': 'What is the overall sentiment?',
     'options': ['positive', 'negative', 'mixed']},
    {'id': 'u01', 'kind': 'text', 'input': '明日の会議は午後3時に変更になりました。資料を持ってきてください。', 'question': 'Which fields are present?',
     'options': ['date', 'time', 'location', 'attachment'], 'multi': True},
    {'id': 'u02', 'kind': 'text', 'input': 'Ça marche 👍🏽, à demain!', 'question': 'Does this message need a reply?', 'options': ['Yes', 'No'],
     'rules': 'Short acknowledgements never need a reply.'},
]

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True); ap.add_argument('--gguf', required=True); ap.add_argument('--mmproj', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--hf-ref'); ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--b', type=float, default=8.010554313659668)
a = ap.parse_args()
base = os.path.dirname(os.path.abspath(a.cases)); os.makedirs(a.out, exist_ok=True)
hf = {r['id']: r for r in json.load(open(a.hf_ref))['rows']} if a.hf_ref else {}
cases = []
for c in json.load(open(a.cases)) + EXTRA:
    c = dict(c); row = None
    if c['kind'] != 'text':
        src = os.path.join(base, c['input']); shutil.copy(src, os.path.join(a.out, c['input']))
        if c['input'].lower().endswith(('.jpg', '.jpeg')):
            row = dict(c, id=c['id'] + 'j', decoder_row=True)   # same expected values, JPEG decoded by the platform
            c['input'] = os.path.splitext(c['input'])[0] + '.png'
            Image.open(src).convert('RGB').save(os.path.join(a.out, c['input']))
    cases += [c] + ([row] if row else [])
m = LlamaDecide(a.gguf, a.mmproj, b=a.b, n_threads=a.threads)
byid = {}
for c in cases:
    if c.get('decoder_row'):
        continue
    inp = os.path.join(a.out, c['input']) if c['kind'] != 'text' else c['input']
    t0 = time.time()
    z, text = m.letter_logits(inp, c['question'], c['options'], c.get('descriptions'), bool(c.get('multi')), c.get('rules'), c.get('context'))
    p = 1 / (1 + np.exp(-(z - a.b))) if c.get('multi') else np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    c.update(prompt=text, tokens=m.last_tokens, z=z.tolist(), p=p.tolist(), ms=m.last_ms)
    if c['id'] in hf:
        assert hf[c['id']]['prompt'] == text, c['id']
        c['p_hf'] = hf[c['id']]['p']
    byid[c['id']] = c
    print(f"{c['id']} {c['kind']:<5} tok {m.last_tokens} {m.last_ms:7.0f} ms p={np.round(p, 4).tolist()}", flush=True)
for c in cases:
    if c.get('decoder_row'):
        ref = byid[c['id'][:-1]]
        c.update({k: ref[k] for k in ('prompt', 'tokens', 'z', 'p', 'ms') if k in ref}); c.update({'p_hf': ref['p_hf']} if 'p_hf' in ref else {})
meta = {'gguf': os.path.basename(a.gguf), 'mmproj': os.path.basename(a.mmproj), 'b': a.b, 'threads': a.threads,
        'generated_by': f'llama_decide.py (Python ctypes, same decide_core) on {platform.machine()} CPU, {a.threads} threads',
        'hf_reference': os.path.basename(a.hf_ref) if a.hf_ref else None, 'cases': cases}
json.dump(meta, open(os.path.join(a.out, 'expected.json'), 'w'), ensure_ascii=False, indent=1)
print(len(cases), 'cases ->', os.path.join(a.out, 'expected.json'))
