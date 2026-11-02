"""Cascade: Smaug-Audio-Decisions-0.7B (744M) answers audio questions when it is confident, everything else goes to Smaug-Decisions.

    python examples/cascade.py call.wav [--threshold 0.9]

The big model is loaded only on the first escalation. Set the threshold on your own data: higher escalates more often.
"""

import argparse

from smaug_decide import Cascade


ap = argparse.ArgumentParser()
ap.add_argument('audio')
ap.add_argument('--threshold', type=float, default=0.9)
a = ap.parse_args()

cascade = Cascade('abacusai/Smaug-Audio-Decisions-0.7B', 'abacusai/Smaug-Decisions', threshold=a.threshold)
questions = [
    ('Is anyone talking in this clip?', ['Yes', 'No'], False),
    ('How many different people speak in this audio?', ['0', '1', '2', '3', '4 or more'], False),
    ('Which of these can be heard in the audio?', ['speech', 'music', 'a vehicle', 'an animal'], True),
]
for question, options, multi in questions:
    r = cascade(a.audio, question, options, multi=multi)
    best = max(r.probabilities, key=r.probabilities.get)
    conf = 'routed' if r.routed else f'fast-tier confidence {r.fast_confidence:.2f}'
    print(f'{question}\n  answered by the {r.tier} tier ({conf}): {best}  {r.probabilities}')
