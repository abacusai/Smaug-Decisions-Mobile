"""Goldens for the Cascade logic tests (no model needed):

    python tools/make_goldens.py --smaug <Smaug-Decisions repository root> --big_decide <abacusai/Smaug-Decisions smaug_decide.py>

cascade_goldens.json:
  confidence: probability vectors with smaug_decide.cascade.confidence (the rule the Swift Router ports), one-answer and select-all
  big_prompts: the Smaug-Decisions prompt of every built-in case and scenario, rendered by the Smaug-Decisions reference smaug_decide.py
  marker_texts: the fast-tier prompt texts in cascade.json (rendered by smaug_decide.prompt.render_marker), for FastTokenizer
"""

import argparse
import ast
import json
import os
import random
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'Cascade', 'Cascade', 'Resources')
OUT = os.path.join(HERE, '..', 'Cascade', 'CascadeTests', 'goldens', 'cascade_goldens.json')
WANT = {'LETTERS', 'SAMPLE_RATE', 'MAX_TOKENS', 'AUDIO_TOKENS_PER_SECOND', 'IMAGE_TOKENS', 'PREAMBLE', 'trim', 'render'}


def load_big(path):
    keep = []
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            names = [t.id] if isinstance(t, ast.Name) else [e.id for e in t.elts if isinstance(e, ast.Name)]
            if WANT & set(names):
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WANT:
            keep.append(node)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, 'exec'), ns)
    return ns


def samples_of(media):
    p = os.path.join(RES, media)
    if media.endswith('.pcm'):
        return os.path.getsize(p) // 4
    import soundfile as sf

    return sf.info(p).frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smaug', required=True)
    ap.add_argument('--big_decide', required=True)
    ap.add_argument('--tokenizer', default=None, help='Smaug-Decisions tokenizer.json: record and check the token counts of the device-budget prompts')
    a = ap.parse_args()
    sys.path.insert(0, a.smaug)
    from smaug_decide.cascade import confidence
    from smaug_decide.prompt import char_budget, render_marker

    ns = load_big(a.big_decide)
    rnd = random.Random(0)
    conf = []
    for _ in range(60):
        k, multi = rnd.randint(2, 8), rnd.random() < 0.4
        if multi:
            p = [rnd.random() for _ in range(k)]
        else:
            w = [rnd.random() ** rnd.choice([1, 4, 12]) for _ in range(k)]
            p = [x / sum(w) for x in w]
        conf.append(dict(p=p, multi=multi, confidence=confidence(dict(enumerate(p)), multi)))
    conf += [dict(p=[0.9, 0.1], multi=False, confidence=0.9), dict(p=[0.5, 0.5], multi=False, confidence=0.5),
             dict(p=[0.95, 0.04, 0.99], multi=True, confidence=confidence(dict(enumerate([0.95, 0.04, 0.99])), True))]

    c = json.load(open(os.path.join(RES, 'cascade.json')))
    prompts, markers = [], []
    items = [dict(s, kind='audio', media=None) for s in c['scenarios']] + c['cases']
    for x in items:
        opts, desc = x['options'], x['descriptions']
        n = samples_of(x['media']) if x.get('media') and x['kind'] == 'audio' else (16000 * 10 if x['kind'] == 'audio' else 0)
        n = min(n, 30 * 16000)
        mt = n / ns['SAMPLE_RATE'] * ns['AUDIO_TOKENS_PER_SECOND'] + 2 if x['kind'] == 'audio' else (ns['IMAGE_TOKENS'] + 2 if x['kind'] == 'image' else 0)
        budget = int(max(400, (ns['MAX_TOKENS'] - mt - 80 - 10 * len(opts)) * 3.5))
        tag = {'audio': '<|audio|>', 'image': '<|image|>'}.get(x['kind'])
        prompt = ns['render'](x['question'], opts, desc, x['multi'], tag, x.get('rules'), None, x.get('text') if not tag else None, budget)
        prompts.append(dict(id=x['id'], kind=x['kind'], samples=n, question=x['question'], options=opts, descriptions=desc,
                            multi=x['multi'], rules=x.get('rules'), text=x.get('text'), prompt=prompt))
        if x.get('fast'):
            text = render_marker(x['question'], opts, desc, x['multi'], '<|audio|>', x.get('rules'), None, None,
                                 char_budget('marker', 10 * 25 + 2, len(opts)))
            assert text == x['fast']['text'], x['id']
            markers.append(dict(id=x['id'], question=x['question'], options=opts, descriptions=desc, multi=x['multi'],
                                rules=x.get('rules'), text=text))
    # On-device budget: the Smaug-Decisions-LiteRT builds hold 2048 tokens, so the app renders at MAX_TOKENS 2000 (same formula and trim).
    device = []
    long_text = ' '.join(f'Line {i}: the customer wrote again about order {1000 + i} and asked for a refund or a replacement.' for i in range(400))
    long_rules = 'Escalate anything that mentions money, threats or account access. ' * 60
    for cid, kind, n, text, rules, img in [('long_text', 'text', 0, long_text, None, 560), ('long_text_rules', 'text', 0, long_text, long_rules, 560),
                                          ('image_rules_560', 'image', 0, None, long_rules, 560), ('image_rules_280', 'image', 0, None, long_rules, 280),
                                          ('audio30_rules', 'audio', 30 * 16000, None, long_rules, 560), ('short_text', 'text', 0, 'Is this spam?', None, 560)]:
        opts = ['Yes', 'No', 'Unsure']
        mt = n / ns['SAMPLE_RATE'] * ns['AUDIO_TOKENS_PER_SECOND'] + 2 if kind == 'audio' else (img + 2 if kind == 'image' else 0)
        tag = {'audio': '<|audio|>', 'image': '<|image|>'}.get(kind)

        def render(m):
            budget = int(max(400, (m - mt - 80 - 10 * len(opts)) * 3.5))
            return ns['render']('Which team should handle this?', opts, [None] * 3, False, tag, rules, None, text, budget)

        row = dict(id=cid, kind=kind, samples=n, image_tokens=img, question='Which team should handle this?', options=opts,
                   descriptions=[None] * 3, multi=False, rules=rules, text=text, prompt=render(2000))
        if a.tokenizer:
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(a.tokenizer)
            # BigEngine.promptTokens: text tokens without the leading <bos> + 1, media tag replaced by the media estimate
            # (audio 25/s + 2; image budget * 1.1 + 2)
            media = (mt - 1) if kind == 'audio' else ((img * 1.1 + 2 - 1) if kind == 'image' else 0)
            count = lambda p: len(tk.encode(p[len('<bos>'):], add_special_tokens=False).ids) + 1 + int(-(-media // 1))  # noqa: E731
            m, p = 2000, render(2000)
            steps = {p: count(p)}
            for _ in range(8):
                if steps[p] <= 2000:
                    break
                m -= (steps[p] - 2000) + 16
                p = render(m)
                steps[p] = count(p)
            row.update(counts=steps, fitted_max_tokens=m, fitted_tokens=steps[p], fitted_prompt=p)
            assert steps[p] <= 2000, cid
        device.append(row)
    json.dump(dict(confidence=conf, big_prompts=prompts, marker_texts=markers, device_budget=device), open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('device budget:', [(r['id'], len(r['prompt']), r.get('counts', {}).get(r['prompt']), r.get('fitted_max_tokens'), r.get('fitted_tokens')) for r in device])
    print('wrote', OUT, len(conf), len(prompts), len(markers))


if __name__ == '__main__':
    main()
