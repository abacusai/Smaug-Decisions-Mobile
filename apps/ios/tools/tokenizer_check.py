"""Is Smaug-Audio-Decisions-0.7B's tokenizer the same as Smaug-Decisions' (Gemma 4)? And can Smaug-Decisions' tokenizer build the audio model's prompt ids?

    hf download abacusai/Smaug-Audio-Decisions-0.7B tokenizer.json --local-dir Smaug-Audio-Decisions-0.7B
    hf download abacusai/Smaug-Decisions tokenizer.json --local-dir Smaug-Decisions
    python tools/tokenizer_check.py Smaug-Audio-Decisions-0.7B/tokenizer.json Smaug-Decisions/tokenizer.json Cascade/Cascade/Resources/cascade.json

Compares the BPE model (vocab, merges), normalizer, pre-tokenizer, decoder, added tokens and post-processor, then rebuilds every
pre-tokenised fast-tier prompt in cascade.json with Smaug-Decisions' tokenizer by splitting at <|audio|> / <unused0> (FastTokenizer.swift).
"""

import json
import sys

from tokenizers import Tokenizer


def main(fast_path, big_path, cascade_path):
    a, b = json.load(open(fast_path)), json.load(open(big_path))
    for k in ('normalizer', 'pre_tokenizer', 'decoder', 'post_processor'):
        print(f'{k:15s}', 'same' if a[k] == b[k] else 'DIFFERENT')
    ma, mb = a['model'], b['model']
    print('model          ', ma['type'], mb['type'], 'vocab', len(ma['vocab']), len(mb['vocab']), 'same' if ma['vocab'] == mb['vocab'] else 'DIFFERENT',
          '| merges', len(ma['merges']), len(mb['merges']), 'same' if ma['merges'] == mb['merges'] else 'DIFFERENT')
    aa = {t['id']: t['content'] for t in a['added_tokens']}
    ab = {t['id']: t['content'] for t in b['added_tokens']}
    print('added tokens   ', len(aa), len(ab), 'only in the audio model:', {i: aa[i] for i in aa if ab.get(i) != aa[i]},
          'only in Smaug-Decisions:', {i: ab[i] for i in ab if aa.get(i) != ab[i]})
    print('post_processor  the audio model wraps in <bos> ... <eos>' if 'SpecialToken' in json.dumps(a['post_processor']) else '', '| Smaug-Decisions adds nothing'
          if 'SpecialToken' not in json.dumps(b['post_processor']) else '')
    te = Tokenizer.from_file(big_path)
    c = json.load(open(cascade_path))
    enc = lambda s: te.encode(s, add_special_tokens=False).ids if s else []  # noqa: E731
    items = [(s['id'], s['fast']) for s in c['scenarios']] + [(x['id'], x['fast']) for x in c['cases'] if x.get('fast')]
    ok = 0
    for i, f in items:
        before, after = f['text'].split('<|audio|>')
        pre = [2] + enc(before) + [c['audio_bos_id']]
        parts = after.split('<unused0>')
        post = [c['audio_eos_id']] + enc(parts[0])
        for p in parts[1:]:
            post += [c['marker_id']] + enc(p)
        post += [1]
        same = pre == f['pre_ids'] and post == f['post_ids']
        ok += same
        if not same:
            print('  differs:', i)
    print(f'{ok}/{len(items)} fast-tier prompts rebuilt exactly with the Smaug-Decisions tokenizer (segmented at <|audio|> / <unused0>)')
    print('literal "<unused0>" with the Smaug-Decisions tokenizer ->', te.encode('<unused0>', add_special_tokens=False).ids, '(not id 6: split first)')


if __name__ == '__main__':
    main(*sys.argv[1:4])
