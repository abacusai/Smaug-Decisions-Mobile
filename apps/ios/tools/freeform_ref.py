"""Reference for the app's free-form fast-tier question (autorun --freeform): the audio model's processor's token ids for the same question
(must equal the ids the phone built with Smaug-Decisions' tokenizer) and PyTorch / Mac Core ML probabilities via parity.py.

    python tools/freeform_ref.py --smaug <Smaug-Decisions repository root> --model <audio model dir> --autorun autorun_freeform.json --clip a1.wav --work <dir>
then: python <repo>/scripts/convert/coreml/parity.py --units <units> --cases <work>/freeform.jsonl --model <audio model dir> --compute CPU_ONLY
"""
import argparse
import json
import os
import sys

import numpy as np


ap = argparse.ArgumentParser()
for k in ('--smaug', '--model', '--autorun', '--clip', '--work'):
    ap.add_argument(k, required=True)
a = ap.parse_args()
sys.path.insert(0, a.smaug)
from smaug_decide.prompt import AUDIO_TAG, MARKER, char_budget, render_marker  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

r = json.load(open(a.autorun))
opts = r['options']
# the question is not stored in the row; it is the one passed on the command line, recover it from the rendered ids instead
proc = AutoProcessor.from_pretrained(a.model)
tok = proc.tokenizer
tok.add_tokens([MARKER], special_tokens=True)
question = os.environ.get('FREEFORM_QUESTION')
text = render_marker(question, opts, [None] * len(opts), False, AUDIO_TAG, None, None, None, char_budget('marker', 252, len(opts)))
ids = proc(text=[text], audio=[np.zeros(160000, np.float32)], return_tensors='np')['input_ids'][0].tolist()
i0, i1 = ids.index(tok.convert_tokens_to_ids('<|audio>')), ids.index(tok.convert_tokens_to_ids('<audio|>'))
pre, post = ids[: i0 + 1], ids[i1:]
print('pre ids equal:', pre == r['fast_pre_ids'], '| post ids equal:', post == r['fast_post_ids'], '| lengths', len(pre), len(post))
with open(os.path.join(a.work, 'freeform.jsonl'), 'w') as f:
    f.write(json.dumps(dict(id='freeform', input=a.clip, question=question, options=opts, multi=False)) + '\n')
print('wrote', os.path.join(a.work, 'freeform.jsonl'))
