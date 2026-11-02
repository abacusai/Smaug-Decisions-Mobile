#!/usr/bin/env python3
"""Pick the cascade threshold (tau) on your own labeled examples.

    python scripts/calibrate.py examples.jsonl --out answers.jsonl          # run both models once, then sweep tau
    python scripts/calibrate.py --answers answers.jsonl                     # sweep again from saved answers (no models)
    python scripts/calibrate.py examples.jsonl --dry-run                    # check the file, load nothing

examples.jsonl has one labeled question per line ("descriptions", "multi", "rules", "context" are optional):

    {"input": "clips/0001.wav", "question": "Has the speaker finished their turn?", "options": ["Yes", "No"], "label": "Yes"}
    {"input": "clips/0002.wav", "question": "Which sounds are present?", "options": ["speech", "music", "alarm"],
     "multi": true, "label": ["speech", "alarm"]}

Smaug-Audio-Decisions-0.7B and Smaug-Decisions both answer every audio example once and the answers (with their latencies) are saved to --out. Then
every tau from 0.50 to 0.99 is scored: accuracy, the share escalated to Smaug-Decisions, and the mean latency per decision on this
machine. The suggestion is the lowest-escalation tau whose accuracy is within --tolerance of Smaug-Decisions alone. Text and image
examples are skipped: the cascade sends them straight to Smaug-Decisions, so tau does not affect them. A few hundred examples that look
like your traffic give a usable estimate; see docs/choosing-a-threshold.md.
"""

import argparse
import json
import os
import sys
import time


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from smaug_decide.calibration import DEFAULT_TAUS, suggest, sweep  # noqa: E402
from smaug_decide.media import classify_input  # noqa: E402


def read_examples(path):
    base, out = os.path.dirname(os.path.abspath(path)), []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            e = json.loads(line)
            for k in ('input', 'question', 'options', 'label'):
                if k not in e:
                    raise SystemExit(f'{path}:{n}: missing "{k}"')
            labels = [e['label']] if isinstance(e['label'], str) else e['label']
            if not e.get('multi') and not isinstance(e['label'], str):
                raise SystemExit(f'{path}:{n}: a one-answer question needs a single "label" (or set "multi": true)')
            if any(lb not in e['options'] for lb in labels):
                raise SystemExit(f'{path}:{n}: label {e["label"]!r} is not one of the options')
            kind = e.get('type') or classify_input(e['input'])
            if kind != 'text' and not os.path.isabs(e['input']):
                e['input'] = os.path.join(base, e['input'])  # media paths are relative to the examples file
            if kind != 'text' and not os.path.exists(e['input']):
                raise SystemExit(f'{path}:{n}: {e["input"]} does not exist')
            out.append({**e, 'type': kind, 'line': n})
    return out


def run_models(examples, fast_repo, big_repo, device, out_path):
    """Both tiers answer every example once (one model in memory at a time); returns the calibration records."""
    from smaug_decide import SmaugDecide

    records = [{'line': e['line'], 'label': e['label'], 'multi': e.get('multi', False)} for e in examples]
    for tier, repo in (('fast', fast_repo), ('big', big_repo)):
        model = SmaugDecide.from_pretrained(repo, device=device)
        for i, e in enumerate(examples):
            t0 = time.time()
            p = model(
                e['input'],
                e['question'],
                e['options'],
                e.get('descriptions'),
                e.get('multi', False),
                e.get('rules'),
                e.get('context'),
                input_type='audio',
            )
            records[i].update({tier: p, f'{tier}_ms': 1000 * (time.time() - t0)})
            print(f'\r{tier}: {i + 1}/{len(examples)}', end='', file=sys.stderr, flush=True)
        print(file=sys.stderr)
        del model
    if out_path:
        with open(out_path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
    return records


def print_table(rows, best):
    has_ms = 'mean_ms' in rows[0]
    print(f'{"tau":>10}  {"accuracy":>8}  {"escalated":>9}' + (f'  {"mean ms":>8}' if has_ms else ''))
    for r in rows:
        name = '0.7B audio model only' if r['tau'] == 0 else 'Smaug-Decisions' if r['tau'] == float('inf') else f'{r["tau"]:.2f}'
        mark = '  <- suggested' if r is best else ''
        ms = f'  {r["mean_ms"]:8.1f}' if has_ms else ''
        print(f'{name:>10}  {r["accuracy"]:8.3f}  {r["escalation_rate"]:9.1%}{ms}{mark}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('examples', nargs='?', help='labeled examples (JSONL)')
    ap.add_argument('--answers', help='saved answers from an earlier --out: sweep without running the models')
    ap.add_argument('--out', help="save both models' answers here (JSONL) for later sweeps")
    ap.add_argument('--fast', default='abacusai/Smaug-Audio-Decisions-0.7B', help='fast tier repo id or local directory')
    ap.add_argument('--big', default='abacusai/Smaug-Decisions', help='big model repo id or local directory')
    ap.add_argument('--device', help='cuda, mps or cpu (default: picked automatically)')
    ap.add_argument('--tolerance', type=float, default=0.005, help='accuracy you accept to give up versus Smaug-Decisions alone')
    ap.add_argument('--json', help='also write the sweep and the suggestion here')
    ap.add_argument('--dry-run', action='store_true', help='validate the examples and print the plan, load nothing')
    a = ap.parse_args()
    if bool(a.examples) == bool(a.answers):
        ap.error('give an examples file or --answers, not both')

    if a.answers:
        with open(a.answers) as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        examples = read_examples(a.examples)
        audio = [e for e in examples if e['type'] == 'audio']
        print(
            f'{len(examples)} examples: {len(audio)} audio, {len(examples) - len(audio)} text/image (skipped: routed to Smaug-Decisions)',
            file=sys.stderr,
        )
        if not audio:
            raise SystemExit('no audio examples: the threshold only affects audio questions')
        if a.dry_run:
            print(f'would run {a.fast} and {a.big} on {len(audio)} audio examples, then sweep {len(DEFAULT_TAUS)} values of tau')
            return
        records = run_models(audio, a.fast, a.big, a.device, a.out)

    rows = sweep(records)
    best = suggest(rows, a.tolerance)
    print_table(rows, best)
    if best['tau'] == float('inf'):
        print(f'\nNo threshold stays within {a.tolerance:.3f} of Smaug-Decisions alone on these examples: send everything to Smaug-Decisions.')
    else:
        print(
            f'\nSuggested tau = {best["tau"]:.2f}: accuracy {best["accuracy"]:.3f} (Smaug-Decisions alone {rows[-1]["accuracy"]:.3f}), '
            f'{best["escalation_rate"]:.0%} escalated, on {len(records)} examples.'
        )
    if a.json:
        clean = [{**r, 'tau': 'inf' if r['tau'] == float('inf') else r['tau']} for r in rows]
        with open(a.json, 'w') as f:
            json.dump({'rows': clean, 'suggested_tau': None if best['tau'] == float('inf') else best['tau']}, f, indent=1)


if __name__ == '__main__':
    main()
