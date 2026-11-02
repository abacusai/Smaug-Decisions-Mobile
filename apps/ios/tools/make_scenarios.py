"""Pre-tokenise the iOS app's scenarios, so the phone needs no tokenizer.

    python tools/make_scenarios.py --out build/scenarios.json [--model abacusai/Smaug-Audio-Decisions-0.7B]
                                           [--selftest build/coreml/parity.json --selftest_n 3]

For each audio scenario the processor runs once on the rendered prompt with a dummy clip, and the token ids are split around the
audio block into pre (up to and including <audio_bos>) and post (from <audio_eos>). The app inserts N audio tokens for the clip it
recorded (N from the valid mel frames through the two stride-2 convolutions, at most 750) and reads the head at the <unused0>
markers in post. Text scenarios are written as full token ids. Edit SCENARIOS to change what the app asks.

--selftest takes a parity.py output and bundles up to --selftest_n of its cases (the clip's last 10 s as raw 16 kHz float32, next
to scenarios.json) with the Mac's Core ML probabilities, which the app's Self-test button must reproduce on the phone.
Needs only the processor files, not the weights.
"""

import argparse
import json
import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
from smaug_decide.media import SAMPLE_RATE, load_audio  # noqa: E402
from smaug_decide.prompt import AUDIO_TAG, AUDIO_TOKENS_PER_SECOND, MARKER, char_budget, render_marker  # noqa: E402


WINDOW_S = 10
SCENARIOS = [
    {
        'id': 'turn_complete',
        'title': 'Done talking?',
        'kind': 'audio',
        'multi': False,
        'question': 'At the end of this audio, has the person completed their speaking turn?',
        'options': [
            ('incomplete', 'The speaker has not completed their turn; wait for them to continue'),
            ('complete', 'The speaker has completed their turn; a response can naturally begin'),
        ],
    },
    {
        'id': 'speech_activity',
        'title': 'Anyone speaking?',
        'kind': 'audio',
        'multi': False,
        'question': 'Is someone speaking at the very end of this audio?',
        'options': [('not_speaking', 'Silence, noise or music only at the end'), ('speaking', 'A person is speaking at the end')],
    },
    {
        'id': 'speaker_count',
        'title': 'How many speakers?',
        'kind': 'audio',
        'multi': False,
        'question': 'How many different people speak in this audio?',
        'options': [('0', None), ('1', None), ('2', None), ('3', None), ('4 or more', None)],
    },
    {
        'id': 'sound_present',
        'title': 'What is in the audio?',
        'kind': 'audio',
        'multi': True,
        'question': 'Which of these can be heard in the audio?',
        'options': [
            ('speech', None),
            ('music', None),
            ('a vehicle', None),
            ('an animal', None),
            ('water', None),
            ('wind', None),
            ('an alarm or siren', None),
            ('tools or machinery', None),
        ],
    },
    {
        'id': 'music_genre',
        'title': 'Music genre',
        'kind': 'audio',
        'multi': False,
        'question': 'What is the genre of this music?',
        'options': [
            ('rock', None),
            ('pop', None),
            ('hip hop', None),
            ('electronic', None),
            ('classical', None),
            ('jazz', None),
            ('folk or country', None),
            ('not music', 'There is no music in the clip'),
        ],
    },
    {
        'id': 'intent',
        'title': 'What does the speaker want?',
        'kind': 'audio',
        'multi': False,
        'question': 'What is the speaker asking for?',
        'options': [
            ('information', 'A question or request for facts'),
            ('an action', 'Asking someone to do something'),
            ('small talk', 'Greeting or social chat, no request'),
            ('a complaint', 'Expressing a problem or dissatisfaction'),
        ],
    },
    {
        'id': 'triage',
        'title': 'Support inbox',
        'kind': 'text',
        'multi': False,
        'rules': 'You moderate a support inbox.',
        'text': 'My order arrived broken and nobody answers the phone. I want my money back now.',
        'question': 'Which team should handle this?',
        'options': [
            ('billing', 'Refunds and payments'),
            ('shipping', 'Delivery problems'),
            ('sales', None),
            ('abuse', 'Threats or harassment'),
        ],
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='abacusai/Smaug-Audio-Decisions-0.7B', help='repo id or local directory (processor files only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--selftest', default=None, help='parity.json from parity.py')
    ap.add_argument('--selftest_n', type=int, default=3)
    a = ap.parse_args()

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    tok.add_tokens([MARKER], special_tokens=True)
    ids_of = tok.convert_tokens_to_ids
    MARK, AUD = ids_of(MARKER), ids_of(AUDIO_TAG)
    BOA = ids_of(getattr(tok, 'boa_token', None) or '<audio_bos>')
    EOA = ids_of(getattr(tok, 'eoa_token', None) or '<audio_eos>')
    assert None not in (MARK, AUD, BOA, EOA) and tok.unk_token_id not in (MARK, AUD, BOA, EOA), (MARK, AUD, BOA, EOA)

    def audio_split(question, options, descriptions, multi, rules=None, context=None):
        """(pre ids, post ids, marker offsets in post, rendered text) for an audio prompt."""
        k = len(options)
        text = render_marker(
            question,
            options,
            descriptions,
            multi,
            AUDIO_TAG,
            rules,
            context,
            None,
            char_budget('marker', WINDOW_S * AUDIO_TOKENS_PER_SECOND + 2, k),
        )
        ids = proc(text=[text], audio=[np.zeros(WINDOW_S * SAMPLE_RATE, np.float32)], return_tensors='np')['input_ids'][
            0
        ].tolist()
        i0, i1 = ids.index(BOA), ids.index(EOA)
        pre, post = ids[: i0 + 1], ids[i1:]
        assert all(t == AUD for t in ids[i0 + 1 : i1]), 'the audio block must be contiguous'
        ids2 = proc(text=[text], audio=[np.zeros(3 * SAMPLE_RATE, np.float32)], return_tensors='np')['input_ids'][0].tolist()
        assert ids2 == pre + [AUD] * ids2.count(AUD) + post, 'the pre/post split does not reconstruct a shorter clip'
        marks = [i for i, t in enumerate(post) if t == MARK]
        assert len(marks) == k, (len(marks), k)
        return pre, post, marks, text, i1 - i0 - 1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {'marker_id': MARK, 'audio_token_id': AUD, 'audio_bos_id': BOA, 'audio_eos_id': EOA, 'scenarios': []}
    for s in SCENARIOS:
        options, descriptions = [o for o, _ in s['options']], [d for _, d in s['options']]
        base = {
            'id': s['id'],
            'title': s['title'],
            'kind': s['kind'],
            'question': s['question'],
            'multi': s['multi'],
            'options': options,
        }
        if s['kind'] == 'audio':
            pre, post, marks, text, n10 = audio_split(s['question'], options, descriptions, s['multi'], s.get('rules'))
            out['scenarios'].append(
                {**base, 'pre_ids': pre, 'post_ids': post, 'marker_offsets_in_post': marks, 'audio_tokens_10s': n10, 'text': text}
            )
        else:
            text = render_marker(
                s['question'],
                options,
                descriptions,
                s['multi'],
                None,
                s.get('rules'),
                None,
                s['text'],
                char_budget('marker', 0, len(options)),
            )
            ids = tok(text)['input_ids']
            marks = [i for i, t in enumerate(ids) if t == MARK]
            assert len(marks) == len(options)
            out['scenarios'].append({**base, 'ids': ids, 'marker_offsets': marks, 'text': text})

    if a.selftest:
        with open(a.selftest) as f:
            par = json.load(f)
        assert par['audio_s'] == WINDOW_S, f'the app uses a {WINDOW_S} s window; run parity.py with --audio {WINDOW_S}'
        out['selftests'] = []
        for c in par['cases'][: a.selftest_n]:
            options = c['options']
            pre, post, marks, text, _ = audio_split(
                c['question'],
                options,
                c.get('descriptions') or [None] * len(options),
                c.get('multi', False),
                c.get('rules'),
                c.get('context'),
            )
            clip = load_audio(c['input'])[-WINDOW_S * SAMPLE_RATE :]
            fname = f'selftest_{len(out["selftests"])}.pcm'
            clip.astype('<f4').tofile(os.path.join(os.path.dirname(os.path.abspath(a.out)), fname))
            out['selftests'].append(
                {
                    'id': c['id'],
                    'title': 'self-test: ' + c['id'],
                    'kind': 'audio',
                    'question': c['question'],
                    'multi': c.get('multi', False),
                    'options': options,
                    'pre_ids': pre,
                    'post_ids': post,
                    'marker_offsets_in_post': marks,
                    'text': text,
                    'audio_file': fname,
                    'expected': c['coreml'],
                    'expected_ms_mac': c['ms'],
                }
            )
        print('self-tests:', [(t['id'], t['audio_file']) for t in out['selftests']])
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(
        'wrote',
        a.out,
        [(s['id'], len(s.get('pre_ids', s.get('ids', []))) + len(s.get('post_ids', []))) for s in out['scenarios']],
    )


if __name__ == '__main__':
    main()
