"""Yes / No about an audio clip: has the speaker finished their turn?

python examples/audio_predicate.py call.wav
"""

import sys

from smaug_decide import SmaugDecide


model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
p = model(sys.argv[1], 'At the end of this audio, has the person completed their speaking turn?')
print(p)  # {'Yes': 0.93, 'No': 0.07}
print('respond now' if p['Yes'] >= 0.8 else 'keep listening')
