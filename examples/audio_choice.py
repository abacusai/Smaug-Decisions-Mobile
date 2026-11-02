"""One answer out of several, with optional descriptions: probabilities sum to 1.

python examples/audio_choice.py song.wav
"""

import sys

from smaug_decide import SmaugDecide


model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
p = model(
    sys.argv[1],
    'What genre is this music?',
    ['pop', 'rock', 'hip-hop', 'electronic', 'classical', 'jazz', 'not music'],
    descriptions=[None, None, None, None, None, None, 'There is no music in the clip'],
)
for option, prob in sorted(p.items(), key=lambda kv: -kv[1]):
    print(f'{option:<12} {prob:.3f}')
