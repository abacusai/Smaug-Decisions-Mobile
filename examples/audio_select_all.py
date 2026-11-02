"""Select all that apply: one independent probability per option (they do not sum to 1).

python examples/audio_select_all.py street.wav
"""

import sys

from smaug_decide import SmaugDecide


model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
p = model(
    sys.argv[1],
    'Which of these can be heard in the audio?',
    ['speech', 'music', 'a vehicle', 'an animal', 'water', 'an alarm or siren'],
    multi=True,
)
print('present:', [option for option, prob in p.items() if prob > 0.5])
print(p)
