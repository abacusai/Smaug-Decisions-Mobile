# Calibration text for llama-imatrix (the importance matrix that guides 4-bit quantisation): rendered decision prompts in the
# smaug_decide prompt format, built from random synthetic templates written for this purpose (no evaluation data), each followed
# by an answer letter. usage: make_calib.py out.txt [n]   (300 prompts, about 30k tokens, by default)
import random
import sys

from llama_decide import render


rng = random.Random(1)
SUBJ = ['the customer', 'the user', 'the patient', 'the driver', 'the student', 'the caller', 'the reviewer', 'the employee']
ACTS = [
    'wants a refund for a late delivery',
    'asks how to reset a password',
    'reports a crash when exporting a file',
    'is happy with the service',
    'complains about a noisy neighbour',
    'asks whether the store is open on Sunday',
    'describes chest pain after running',
    'needs a ride to the airport',
    'says the invoice amount is wrong',
    'wants to upgrade the plan',
    'is asking for directions to the museum',
    'shares a recipe for lentil soup',
]
QS = [
    ('How urgent is this?', ['low', 'medium', 'high']),
    ('What does the message want?', ['information', 'action', 'complaint', 'praise', 'other']),
    ('Is this message safe to auto-reply?', ['Yes', 'No']),
    ('Which department should handle it?', ['billing', 'technical support', 'sales', 'medical', 'general']),
    ('Which apply?', ['mentions money', 'mentions time', 'mentions a place', 'mentions a person', 'is a question']),
    ('What is the sentiment?', ['negative', 'neutral', 'positive']),
]
if len(sys.argv) not in (2, 3) or sys.argv[1].startswith('-'):
    sys.exit('usage: make_calib.py <out.txt> [n prompts, default 300]')
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
with open(sys.argv[1], 'w') as f:
    for i in range(n):
        q, opts = rng.choice(QS)
        multi = q == 'Which apply?'
        text = f'{rng.choice(SUBJ).capitalize()} {rng.choice(ACTS)}. ' * rng.randint(1, 3)
        rules = rng.choice([None, None, 'Escalate anything involving health or money.', 'Never auto-reply to complaints.'])
        ctx = rng.choice([None, None, 'The account is two years old.', 'It is Sunday evening.'])
        tag = rng.choice([None, None, None, '<|audio|>', '<|image|>'])
        f.write(
            render(q, opts, [None] * len(opts), multi, tag, rules, ctx, None if tag else text, 3000).replace('<bos>', '')
            + rng.choice(['A', 'B', 'C'])
            + '\n'
        )
print('wrote', sys.argv[1])
