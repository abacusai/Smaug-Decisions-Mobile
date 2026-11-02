"""smaug_decide: calibrated typed decisions over audio, images and text from Abacus.AI's on-device decision models.

from smaug_decide import SmaugDecide
model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
model('clip.wav', 'Has the speaker finished their turn?')                                  # -> {'Yes': 0.93, 'No': 0.07}
model('clip.wav', 'Which genre is playing?', ['rock', 'jazz', 'classical'])                # one answer: sums to 1
model('clip.wav', 'Which sounds are present?', ['speech', 'music', 'alarm'], multi=True)   # select-all: one probability each
"""

from .cascade import Cascade, CascadeResult, confidence
from .prompt import LETTERS, MARKER, build_prompt, render_letters, render_marker
from .readout import option_probs, readout_kind


__version__ = '0.1.0'
__all__ = [
    'SmaugDecide',
    'Cascade',
    'CascadeResult',
    'confidence',
    'build_prompt',
    'render_letters',
    'render_marker',
    'LETTERS',
    'MARKER',
    'option_probs',
    'readout_kind',
]


def __getattr__(name):  # torch and transformers load only when the model is used, so prompts and the cascade import without them
    if name == 'SmaugDecide':
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
