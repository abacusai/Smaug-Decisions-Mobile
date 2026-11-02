"""Questions about an image or a screenshot.

python examples/image.py photo.jpg
"""

import sys

from smaug_decide import SmaugDecide


model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
print(model(sys.argv[1], 'Does this image show a person?'))
print(
    model(
        sys.argv[1],
        'What is wrong with this photo, if anything?',
        ['looks fine', 'blurry', 'too dark', 'overexposed', 'sideways or upside down'],
    )
)
