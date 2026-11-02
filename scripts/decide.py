#!/usr/bin/env python3
"""One decision from the command line: an audio file, an image or a piece of text, a question and its options.

    python scripts/decide.py call.wav "Has the speaker finished their turn?"
    python scripts/decide.py song.wav "Which genre is playing?" --options rock jazz classical
    python scripts/decide.py room.wav "Which sounds are present?" --options speech music alarm --multi
    python scripts/decide.py "The order never arrived." "How upset is the customer?" --options calm annoyed angry --type text
    python scripts/decide.py call.wav "Has the speaker finished their turn?" --model cascade --threshold 0.9
    python scripts/decide.py call.wav "Has the speaker finished their turn?" --dry-run     # show the prompt, load nothing

--model is decisions, audio, cascade (the audio model first for audio, Smaug-Decisions for the rest and for unsure audio), a Hugging Face repo id
or a local checkpoint directory. Prints JSON: the probabilities, the top answer (or the selected options), the tier that
answered and the time taken.
"""

import argparse
import json
import os
import sys
import time


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # run from a checkout without installing

from smaug_decide import Cascade, build_prompt, readout_kind  # noqa: E402
from smaug_decide.media import classify_input  # noqa: E402


REPOS = {'decisions': 'abacusai/Smaug-Decisions', 'audio': 'abacusai/Smaug-Audio-Decisions-0.7B'}


def readout_of(model):
    """'marker' (Smaug-Audio-Decisions-0.7B) or 'letters' (Smaug-Decisions) without loading weights: from readout.json, else from the name."""
    path = os.path.join(model, 'readout.json')
    if os.path.isfile(path):
        with open(path) as f:
            return readout_kind(json.load(f))
    return 'marker' if 'audio' in model.lower() else 'letters'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='audio or image file, or the text itself')
    ap.add_argument('question')
    ap.add_argument('--options', nargs='+', default=['Yes', 'No'])
    ap.add_argument('--descriptions', nargs='+', help='one description per option ("" for none)')
    ap.add_argument('--multi', action='store_true', help='select all that apply (one independent probability per option)')
    ap.add_argument('--rules', help='standing instructions, e.g. "Judge the tone, not the facts."')
    ap.add_argument('--context', help='facts about this input, e.g. "A caller on a support line."')
    ap.add_argument('--type', choices=['audio', 'image', 'text'], help='input type (default: detected from the input)')
    ap.add_argument('--model', default='decisions', help='decisions, audio, cascade, a repo id or a local directory')
    ap.add_argument('--threshold', type=float, default=0.9, help='cascade only: the audio model answers at or above this confidence')
    ap.add_argument('--long-audio', choices=['last', 'mean', 'any'], default='last', help='clips over 30 s (single model only)')
    ap.add_argument('--device', help='cuda, mps or cpu (default: picked automatically)')
    ap.add_argument('--dry-run', action='store_true', help='render the prompt(s) and the route, without loading a model')
    a = ap.parse_args()

    descriptions = [d or None for d in a.descriptions] if a.descriptions else None
    if descriptions and len(descriptions) != len(a.options):
        ap.error('--descriptions needs one entry per option')
    kind = a.type or classify_input(a.input)
    if kind != 'text' and not os.path.exists(a.input):
        ap.error(f'{a.input}: no such file (pass --type text for a text input)')
    args = (a.input, a.question, a.options, descriptions, a.multi, a.rules, a.context, kind)

    if a.dry_run:
        if a.model == 'cascade':
            route = Cascade('audio', 'decisions', a.threshold).route(a.input, kind)
            tiers = ['marker', 'letters'] if route == 'fast' else ['letters']
            plan = (
                f'Smaug-Audio-Decisions-0.7B first; answer if confidence >= {a.threshold}, else Smaug-Decisions'
                if route == 'fast'
                else f'{kind} input: routed straight to Smaug-Decisions'
            )
        else:
            tiers = [readout_of(a.model)]
            plan = f'{REPOS.get(a.model, a.model)} only'
        out = {'input_type': kind, 'plan': plan, 'prompts': {}}
        for readout in tiers:
            text, audio, _ = build_prompt(readout, *args)
            name = 'Smaug-Audio-Decisions-0.7B' if readout == 'marker' else 'Smaug-Decisions'
            out['prompts'][name] = text
            if audio is not None:
                out['audio_seconds'] = round(len(audio) / 16000, 2)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return

    from smaug_decide import SmaugDecide

    t0 = time.time()
    if a.model == 'cascade':
        model = Cascade(REPOS['audio'], REPOS['decisions'], threshold=a.threshold, device=a.device)
    else:
        model = SmaugDecide.from_pretrained(REPOS.get(a.model, a.model), device=a.device)
    load_s = time.time() - t0
    t0 = time.time()
    if a.model == 'cascade':
        r = model(*args)
        probs, tier, conf = r.probabilities, r.tier, r.fast_confidence
    else:
        probs, tier, conf = model(*args, long_audio=a.long_audio), a.model, None
    ms = 1000 * (time.time() - t0)
    out = {'probabilities': {k: round(v, 4) for k, v in probs.items()}}
    if a.multi:
        out['selected'] = [o for o, p in probs.items() if p > 0.5]
    else:
        out['answer'] = max(probs, key=probs.get)
    out.update({'tier': tier, 'ms': round(ms, 1), 'load_s': round(load_s, 1)})
    if conf is not None:
        out['fast_confidence'] = round(conf, 4)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
