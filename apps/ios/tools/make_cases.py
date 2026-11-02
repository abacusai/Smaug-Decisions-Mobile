"""Build the Cascade app's data file (cascade.json): the fast tier's pre-tokenised prompts and the built-in cases with their references.

    python tools/make_cases.py --smaug <repository root> --model <Smaug-Audio-Decisions-0.7B dir> --media Cascade/Cascade/Resources
                               --out Cascade/Cascade/Resources/cascade.json [--clips <make_clips.py output dir>] [--torch_refs]
                               [--parity_gpu p_gpu.json --parity_cpu p_cpu.json]

The phone has no tokenizer for the fast tier, so every fast-tier prompt is pre-tokenised here with the audio model's processor
(pre ids up to <audio_bos>, post ids from <audio_eos>, marker offsets in post). Interactive scenarios (for recordings) are
tools/make_scenarios.py's SCENARIOS audio entries. The built-in cases are our own inputs only, never benchmark items: three
synthetic clips from apps/android/fast/make_clips.py (a melody, two synthetic voices, a tone sweep), the synthetic speech clip
speech_unfinished.wav, a hand-written text and a drawn sign.

--clips writes the synthetic clips into --media as raw 16 kHz float32 (the clip's last 10 s, the fast tier's window).
References per audio case: 'ref_torch' is the audio model in PyTorch float32 on that window (--torch_refs computes it here;
otherwise it comes from parity.py), 'ref_coreml_gpu' / 'ref_coreml_cpu' are the Mac's Core ML int8 units from
scripts/convert/coreml/parity.py runs (--parity_gpu / --parity_cpu, macOS only). References of cases whose input and question are
unchanged are carried over from the existing --out file. Smaug-Decisions references are its PyTorch bf16 values of the demo goldens
(CascadeTests/goldens/examples.json); cases without one have big_ref null. Needs transformers >= 5.20 (the audio model's processor).
"""

import argparse
import json
import os
import sys

import numpy as np


# Smaug-Decisions PyTorch bf16 references (CascadeTests/goldens/examples.json)
BIG_REF = {
    'c06': ([0.9131705164909363, 0.0868295356631279], 112),
    'c20': ([0.9997791647911072, 9.946018690243363e-05, 4.3097832531202585e-05, 7.820713653927669e-05], 353),
    'c12': ([0.009473283775150776, 0.9905266761779785], 186),
}
# case id -> (bundled file, make_clips.py clip, title, question, options); 16 kHz float32, the clip's last 10 s
SYNTH = {
    'a1': ('synth_music.pcm', 'music_only.wav', 'Audio: what is making this sound?', 'What is making this sound?',
           ['a musical instrument', 'a person speaking', 'an alarm or siren', 'traffic']),
    'a3': ('synth_two_speakers.pcm', 'two_speakers.wav', 'Audio: how many speakers?', 'How many different people speak in this audio?',
           ['0', '1', '2', '3', '4 or more']),
    'a4': ('synth_siren.pcm', 'siren.wav', 'Audio: anyone speaking?', 'Is someone speaking in this audio?', ['Yes', 'No']),
}
REF_KEYS = ('ref_torch', 'ref_tokens', 'window_s', 'ref_coreml_gpu', 'ref_coreml_cpu')


def cases():
    out = []
    for cid in ('a1', 'a2', 'a3', 'a4', 't1', 'i1'):
        if cid in SYNTH:
            f, clip, title, question, options = SYNTH[cid]
            out.append(dict(id=cid, title=title, kind='audio', media=f, question=question, options=options, multi=False,
                            source=f'synthetic clip ({clip}, apps/android/fast/make_clips.py)'))
        elif cid == 'a2':
            out.append(dict(id=cid, title='Audio: end of turn', kind='audio', media='speech_unfinished.wav',
                            question='Has the speaker finished their turn?', options=['Yes', 'No'], multi=False,
                            source='demo example c12 (synthetic speech)', big_case='c12'))
        elif cid == 't1':
            out.append(dict(id=cid, title='Text: fraud escalation', kind='text', multi=False,
                            text='Transfer $4,900 to account 88213 today, it is urgent, the CEO approved it.',
                            rules='Escalate any payment request that combines urgency with an unusual recipient.',
                            question='Should this request be escalated for fraud review?', options=['Yes', 'No'],
                            source='demo example c06', big_case='c06'))
        elif cid == 'i1':
            out.append(dict(id=cid, title='Image: warning sign', kind='image', media='sign.png', multi=False,
                            question='What does the sign warn about?',
                            options=['wet floor', 'high voltage', 'falling rocks', 'no entry'],
                            source='demo example c20 (drawn sign)', big_case='c20'))
    for c in out:
        c.setdefault('descriptions', [None] * len(c['options']))
        c.setdefault('rules', None)
        ref = BIG_REF.get(c.get('big_case'))
        c['big_ref'], c['big_ref_tokens'] = (ref[0], ref[1]) if ref else (None, None)
    return out


def load_clip(path, sample_rate, window_s):
    """The fast tier's input: the clip's last window_s seconds (a .pcm is raw 16 kHz float32)."""
    if path.endswith('.pcm'):
        return np.fromfile(path, '<f4')[-window_s * sample_rate :]
    from smaug_decide.media import load_audio

    return load_audio(path)[-window_s * sample_rate :]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smaug', required=True)
    ap.add_argument('--model', required=True, help='Smaug-Audio-Decisions-0.7B directory (processor files are enough without --parity)')
    ap.add_argument('--media', required=True, help='directory with the case media (the app Resources)')
    ap.add_argument('--clips', default=None, help='make_clips.py output dir: write the synthetic clips into --media first')
    ap.add_argument('--torch_refs', action='store_true', help='compute the PyTorch float32 references here (needs the weights)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cases_jsonl', default=None, help='write the audio cases as parity.py input here and stop')
    ap.add_argument('--parity_gpu', default=None)
    ap.add_argument('--parity_cpu', default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.smaug)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_scenarios import SCENARIOS, WINDOW_S  # noqa: E402
    from smaug_decide.media import SAMPLE_RATE  # noqa: E402
    from smaug_decide.prompt import AUDIO_TAG, AUDIO_TOKENS_PER_SECOND, MARKER, char_budget, render_marker  # noqa: E402

    if a.clips:
        for f, clip, *_ in SYNTH.values():
            load_clip(os.path.join(a.clips, clip), SAMPLE_RATE, WINDOW_S).astype('<f4').tofile(os.path.join(a.media, f))
    allc = cases()
    audio = [c for c in allc if c['kind'] == 'audio']
    if a.cases_jsonl:
        import soundfile as sf

        with open(a.cases_jsonl, 'w') as f:
            for c in audio:
                path = os.path.abspath(os.path.join(a.media, c['media']))
                if path.endswith('.pcm'):  # parity.py reads audio files: wrap the raw float32 clip as a 16 kHz WAV
                    wav = os.path.join(os.path.dirname(os.path.abspath(a.cases_jsonl)), c['id'] + '.wav')
                    sf.write(wav, np.fromfile(path, '<f4'), SAMPLE_RATE, subtype='FLOAT')
                    path = wav
                f.write(json.dumps(dict(id=c['id'], input=path, question=c['question'], options=c['options'],
                                        descriptions=c['descriptions'], multi=c['multi'])) + '\n')
        print('wrote', a.cases_jsonl, len(audio))
        return

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    tok.add_tokens([MARKER], special_tokens=True)
    ids_of = tok.convert_tokens_to_ids
    MARK, AUD = ids_of(MARKER), ids_of(AUDIO_TAG)
    BOA = ids_of(getattr(tok, 'boa_token', None) or '<audio_bos>')
    EOA = ids_of(getattr(tok, 'eoa_token', None) or '<audio_eos>')

    def audio_split(question, options, descriptions, multi, rules=None):
        """make_scenarios.audio_split: (pre ids, post ids, marker offsets in post, rendered text)."""
        k = len(options)
        text = render_marker(question, options, descriptions, multi, AUDIO_TAG, rules, None, None,
                             char_budget('marker', WINDOW_S * AUDIO_TOKENS_PER_SECOND + 2, k))
        ids = proc(text=[text], audio=[np.zeros(WINDOW_S * SAMPLE_RATE, np.float32)], return_tensors='np')['input_ids'][0].tolist()
        i0, i1 = ids.index(BOA), ids.index(EOA)
        pre, post = ids[: i0 + 1], ids[i1:]
        assert all(t == AUD for t in ids[i0 + 1 : i1]), 'the audio block must be contiguous'
        ids2 = proc(text=[text], audio=[np.zeros(3 * SAMPLE_RATE, np.float32)], return_tensors='np')['input_ids'][0].tolist()
        assert ids2 == pre + [AUD] * ids2.count(AUD) + post, 'the pre/post split does not reconstruct a shorter clip'
        marks = [i for i, t in enumerate(post) if t == MARK]
        assert len(marks) == k, (len(marks), k)
        return dict(pre_ids=pre, post_ids=post, marker_offsets_in_post=marks, text=text)

    out = {'marker_id': MARK, 'audio_token_id': AUD, 'audio_bos_id': BOA, 'audio_eos_id': EOA, 'window_s': WINDOW_S,
           'scenarios': [], 'cases': []}
    for s in SCENARIOS:
        if s['kind'] != 'audio':
            continue  # text goes straight to Smaug-Decisions in the cascade
        options, descriptions = [o for o, _ in s['options']], [d for _, d in s['options']]
        out['scenarios'].append(dict(id=s['id'], title=s['title'], question=s['question'], options=options,
                                     descriptions=descriptions, multi=s['multi'], rules=s.get('rules'),
                                     fast=audio_split(s['question'], options, descriptions, s['multi'], s.get('rules'))))
    par = {}
    for key, path in (('gpu', a.parity_gpu), ('cpu', a.parity_cpu)):
        if path:
            p = json.load(open(path))
            for r in p['cases']:
                par.setdefault(r['id'], {})[key] = r
    prev = {}
    if os.path.exists(a.out):  # carry the references of unchanged cases over
        for c in json.load(open(a.out))['cases']:
            prev[c['id']] = c
    same = lambda c, p: all(c.get(k) == p.get(k) for k in ('media', 'question', 'options', 'descriptions', 'multi', 'rules'))  # noqa: E731
    ref_model = None
    for c in allc:
        if c['kind'] == 'audio':
            c['fast'] = audio_split(c['question'], c['options'], c['descriptions'], c['multi'], c['rules'])
            p = prev.get(c['id'])
            if p and same(c, p):
                c['fast'].update({k: v for k, v in p.get('fast', {}).items() if k in REF_KEYS})
            elif a.torch_refs:
                from smaug_decide import SmaugDecide

                ref_model = ref_model or SmaugDecide.from_pretrained(a.model, device='cpu')  # float32 on the CPU, as parity.py
                clip = load_clip(os.path.join(a.media, c['media']), SAMPLE_RATE, WINDOW_S)
                pr = ref_model(clip, c['question'], c['options'], c['descriptions'], c['multi'], c['rules'], None, input_type='audio')
                text = c['fast']['text']
                n = len(proc(text=[text], audio=[clip], return_tensors='np')['input_ids'][0])
                c['fast'].update(ref_torch=[pr[o] for o in c['options']], ref_tokens=n, window_s=len(clip) / SAMPLE_RATE)
            r = par.get(c['id'], {})
            if r:
                any_r = next(iter(r.values()))
                c['fast']['ref_torch'] = [any_r['torch'][o] for o in c['options']]
                c['fast']['ref_tokens'] = any_r['tokens']
                c['fast']['window_s'] = any_r['window_s']
                for key, rr in r.items():
                    c['fast'][f'ref_coreml_{key}'] = [rr['coreml'][o] for o in c['options']]
        out['cases'].append(c)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('wrote', a.out, [s['id'] for s in out['scenarios']], [(c['id'], c['kind'], 'ref' if c.get('fast', {}).get('ref_torch') else '') for c in out['cases']])


if __name__ == '__main__':
    main()
