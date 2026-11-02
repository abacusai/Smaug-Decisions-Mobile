"""Turn gate for a voice agent, replayed over a recording: at every pause in the speech, ask whether the speaker has finished.
Above the threshold the agent would start replying at once; below it, it waits for more speech or a fixed hold.

    python examples/turn_gate.py conversation.wav [--threshold 0.8] [--pause-ms 350] [--model abacusai/Smaug-Decisions]

The pause detector is a plain energy threshold; use a real voice activity detector in production. Each decision sees the last
8 s of audio up to the pause, exactly what a live agent would have at that moment.
"""

import argparse

import numpy as np

from smaug_decide import SmaugDecide
from smaug_decide.media import SAMPLE_RATE, load_audio


QUESTION = 'At the end of this audio, has the person completed their speaking turn?'
OPTIONS = ['incomplete', 'complete']
DESCRIPTIONS = [
    'The speaker has not completed their turn; wait for them to continue',
    'The speaker has completed their turn; a response can naturally begin',
]


def pauses(audio, pause_ms, frame_ms=20, rms_threshold=0.015):
    """Sample offsets where speech is followed by at least pause_ms of low energy."""
    n = int(SAMPLE_RATE * frame_ms / 1000)
    rms = np.sqrt(np.mean(audio[: len(audio) // n * n].reshape(-1, n) ** 2, axis=1))
    out, speaking, quiet = [], False, 0
    for i, r in enumerate(rms):
        if r > rms_threshold:
            speaking, quiet = True, 0
        elif speaking:
            quiet += frame_ms
            if quiet >= pause_ms:
                out.append((i + 1) * n)
                speaking, quiet = False, 0
    return out


ap = argparse.ArgumentParser()
ap.add_argument('audio')
ap.add_argument('--model', default='abacusai/Smaug-Decisions')
ap.add_argument('--threshold', type=float, default=0.8)
ap.add_argument('--pause-ms', type=int, default=350)
a = ap.parse_args()

audio = load_audio(a.audio, max_seconds=float('inf'))
model = SmaugDecide.from_pretrained(a.model)
for end in pauses(audio, a.pause_ms):
    window = audio[max(0, end - 8 * SAMPLE_RATE) : end]
    p = model(window, QUESTION, OPTIONS, DESCRIPTIONS)['complete']
    print(f'{end / SAMPLE_RATE:7.2f} s  P(complete) {p:.2f}  ->  {"reply now" if p >= a.threshold else "wait"}')
