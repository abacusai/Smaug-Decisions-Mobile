"""Command line: one decision, or a batch of decisions from a JSONL file.

    python -m smaug_decide --model abacusai/Smaug-Decisions --input call.wav --question "Has the speaker finished their turn?"
    python -m smaug_decide --model abacusai/Smaug-Decisions --input song.wav --question "Which genre?" --options rock jazz classical
    python -m smaug_decide --model abacusai/Smaug-Decisions --cases cases.jsonl --out results.jsonl

A cases file has one JSON object per line:
    {"id", "input", "question", "options", "descriptions"?, "multi"?, "rules"?, "context"?}
"input" is a file path (audio or image) or the text itself. The output adds "probabilities" and "ms" to each case; it is also the
reference format the export parity scripts read.
"""

import argparse
import json
import sys
import time


def main():
    ap = argparse.ArgumentParser(
        prog='python -m smaug_decide', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--model', default='abacusai/Smaug-Decisions', help='Hugging Face repo id or local checkpoint directory')
    ap.add_argument('--device', default=None, help='cuda, mps or cpu (default: picked automatically)')
    ap.add_argument('--input', help='audio or image path, or the text input itself')
    ap.add_argument('--question')
    ap.add_argument('--options', nargs='+', default=['Yes', 'No'])
    ap.add_argument('--descriptions', nargs='+', default=None)
    ap.add_argument('--multi', action='store_true', help='select all that apply')
    ap.add_argument('--rules', default=None)
    ap.add_argument('--context', default=None)
    ap.add_argument('--cases', default=None, help='JSONL file of decisions to run')
    ap.add_argument('--out', default=None, help='output JSONL for --cases (default: stdout)')
    a = ap.parse_args()
    if not a.cases and (a.input is None or a.question is None):
        ap.error('give --input and --question, or --cases')

    from .model import SmaugDecide

    model = SmaugDecide.from_pretrained(a.model, device=a.device)
    if not a.cases:
        print(json.dumps(model(a.input, a.question, a.options, a.descriptions, a.multi, a.rules, a.context), indent=1))
        return
    out = open(a.out, 'w') if a.out else sys.stdout
    with open(a.cases) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            t0 = time.time()
            p = model(
                c['input'],
                c['question'],
                c.get('options', ['Yes', 'No']),
                c.get('descriptions'),
                c.get('multi', False),
                c.get('rules'),
                c.get('context'),
            )
            out.write(json.dumps({**c, 'probabilities': p, 'ms': round(1000 * (time.time() - t0), 1)}) + '\n')
            out.flush()


if __name__ == '__main__':
    main()
