"""Confidence-gated cascade: a fast tier answers when it is sure, a bigger model answers the rest.

Confidence is the top probability for one-answer questions and, for select-all questions, the least certain option's
max(p, 1 - p). Below the threshold the question is escalated to the big model. The threshold sets the trade between latency and
accuracy; 0.9 is a reasonable starting point, but set it on your own data (scripts/calibrate.py, docs/choosing-a-threshold.md).

By default only audio goes to the fast tier. Smaug-Audio-Decisions-0.7B is an audio specialist and close to chance on text, so text and image
questions are routed straight to the big model (tier 'big', fast_confidence None) instead of paying for a fast pass that would
escalate anyway. Pass fast_inputs=None to send every input to the fast tier first.
"""

from dataclasses import dataclass

from .media import classify_input


def confidence(probs, multi=False):
    """probs: {option: probability} as returned by SmaugDecide. Returns a confidence in [0, 1]."""
    v = list(probs.values())
    if multi:
        return min(max(p, 1.0 - p) for p in v)
    return max(v) / max(sum(v), 1e-9)


@dataclass
class CascadeResult:
    probabilities: dict
    tier: str  # 'fast' (the fast tier answered) or 'big' (escalated, or routed straight to the big model)
    fast_confidence: float | None  # None when the input was routed straight to the big model

    @property
    def routed(self):
        """True when the fast tier was skipped because of the input type."""
        return self.fast_confidence is None


class Cascade:
    """fast, big: SmaugDecide instances, or Hugging Face repo ids / local paths (loaded on first use, so a cascade that never
    escalates never loads the big model).

    fast_inputs: input types ('audio', 'image', 'text') that try the fast tier first; the others go straight to the big model.
    None sends everything to the fast tier first."""

    def __init__(self, fast='abacusai/Smaug-Audio-Decisions-0.7B', big='abacusai/Smaug-Decisions', threshold=0.9, device=None, fast_inputs=('audio',)):
        self._fast, self._big, self.threshold, self.device = fast, big, threshold, device
        self.fast_inputs = None if fast_inputs is None else tuple(fast_inputs)

    @staticmethod
    def _load(m, device):
        if isinstance(m, str):
            from .model import SmaugDecide

            return SmaugDecide.from_pretrained(m, device=device)
        return m

    @property
    def fast(self):
        self._fast = self._load(self._fast, self.device)
        return self._fast

    @property
    def big(self):
        self._big = self._load(self._big, self.device)
        return self._big

    def route(self, inp, input_type=None):
        """'fast' when the input tries the fast tier first, 'big' when it goes straight to the big model."""
        if self.fast_inputs is None:
            return 'fast'
        return 'fast' if (input_type or classify_input(inp)) in self.fast_inputs else 'big'

    def __call__(
        self, inp, question, options=('Yes', 'No'), descriptions=None, multi=False, rules=None, context=None, input_type=None
    ):
        args = (inp, question, options, descriptions, multi, rules, context, input_type)
        if self.route(inp, input_type) == 'big':
            return CascadeResult(self.big(*args), 'big', None)
        p = self.fast(*args)
        c = confidence(p, multi)
        if c >= self.threshold:
            return CascadeResult(p, 'fast', c)
        return CascadeResult(self.big(*args), 'big', c)
