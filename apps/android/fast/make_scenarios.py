"""Pre-tokenise the Android app's scenarios (fast/scenarios.py) for the fast tier, so the phone needs no tokenizer for the audio model; mirrors
the earlier Core ML scenario script. For an audio scenario the audio model's processor runs on the rendered marker prompt with
dummy clips and the ids are split around the audio block: pre (up to and including <audio_bos>) and post (from <audio_eos>), plus
the <unused0> marker offsets in post. Checked: pre + [<|audio|>] * N + post reproduces the processor's ids for 0.4 s, 3 s, 10 s
and 28 s clips. Every scenario also keeps question / options / descriptions / rules / text for the Smaug-Decisions prompt.
usage: make_scenarios.py --model <audio model dir> --out app/src/main/assets/fast/scenarios.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='abacusai/Smaug-Audio-Decisions-0.7B', help='recorded in the file')
    a = ap.parse_args()
    sys.path.insert(0, a.model)
    import smaug_decide as sd  # the audio model's file
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    tok.add_tokens([sd.MARKER], special_tokens=True)
    ids_of = tok.convert_tokens_to_ids
    MARK, AUD = ids_of(sd.MARKER), ids_of('<|audio|>')
    BOA = ids_of(getattr(tok, 'boa_token', None) or '<audio_bos>'); EOA = ids_of(getattr(tok, 'eoa_token', None) or '<audio_eos>')
    assert tok.unk_token_id not in (MARK, AUD, BOA, EOA), (MARK, AUD, BOA, EOA)

    def text_for(s, seconds):
        opts, desc = [o for o, _ in s['options']], [d for _, d in s['options']]
        media = seconds * sd.AUDIO_TOKENS_PER_SECOND + 2
        budget = int(max(300, (sd.MAX_TOKENS - media - 60 - 8 * len(opts)) * 3.5))
        return sd.render(s['question'], opts, desc, s['multi'], '<|audio|>', s.get('rules'), None, None, budget)

    out = {'marker_id': MARK, 'audio_token_id': AUD, 'audio_bos_id': BOA, 'audio_eos_id': EOA, 'source': a.source,
           'scenarios': []}
    for s in SCENARIOS:
        opts, desc = [o for o, _ in s['options']], [d for _, d in s['options']]
        row = {'id': s['id'], 'title': s['title'], 'kind': s['kind'], 'question': s['question'], 'multi': s['multi'], 'options': opts,
               'descriptions': desc, 'rules': s.get('rules'), 'text': s.get('text')}
        if s['kind'] == 'audio':
            text = text_for(s, 10)
            ids = proc(text=[text], audio=[np.zeros(10 * 16000, np.float32)], return_tensors='np')['input_ids'][0].tolist()
            i0, i1 = ids.index(BOA), ids.index(EOA)
            pre, post = ids[: i0 + 1], ids[i1:]
            assert all(t == AUD for t in ids[i0 + 1: i1])
            for sec in (0.4, 3.0, 28.0):
                assert text_for(s, sec) == text, 'the char budget changed the prompt text'
                ids2 = proc(text=[text], audio=[np.zeros(int(sec * 16000), np.float32)], return_tensors='np')['input_ids'][0].tolist()
                assert ids2 == pre + [AUD] * ids2.count(AUD) + post, (s['id'], sec)
            marks = [i for i, t in enumerate(post) if t == MARK]
            assert len(marks) == len(opts)
            row.update({'pre_ids': pre, 'post_ids': post, 'marker_offsets_in_post': marks, 'fast_prompt': text})
        out['scenarios'].append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out, [(r['id'], len(r.get('pre_ids', [])) + len(r.get('post_ids', []))) for r in out['scenarios']])


if __name__ == '__main__':
    main()
