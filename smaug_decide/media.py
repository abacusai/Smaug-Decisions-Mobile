"""Input handling: decide whether an input is audio, an image or text, and load audio as 16 kHz mono float32."""

import os

import numpy as np


SAMPLE_RATE = 16000  # the models hear 16 kHz mono
MAX_AUDIO_SECONDS = 30.0  # longer clips keep their last 30 s: the decision is about how the clip ends
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')
AUDIO_SUFFIXES = ('.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aiff', '.aif')


def load_audio(x, max_seconds=MAX_AUDIO_SECONDS):
    """A path, a (samples, sample_rate) pair, or a 1-D float array at 16 kHz -> mono float32 at 16 kHz, the last `max_seconds`
    (None: all of it)."""
    if isinstance(x, (str, os.PathLike)):
        import soundfile as sf

        w, sr = sf.read(os.fspath(x), dtype='float32')
    elif isinstance(x, tuple):
        w, sr = x
    else:
        w, sr = x, SAMPLE_RATE
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SAMPLE_RATE:  # linear resampling is enough for a reference implementation; use a proper resampler in production
        n = int(round(len(w) * SAMPLE_RATE / sr))
        w = np.interp(np.linspace(0.0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)
    return w if max_seconds is None else w[-int(max_seconds * SAMPLE_RATE) :]


def audio_chunks(w, seconds=MAX_AUDIO_SECONDS, min_seconds=5.0):
    """Split a 16 kHz clip into windows of at most `seconds`, aligned to the end of the clip (the last window is always the final
    `seconds`). A leftover head shorter than `min_seconds` is replaced by the clip's first full window, which overlaps the next
    one."""
    n, m = int(seconds * SAMPLE_RATE), int(min_seconds * SAMPLE_RATE)
    if len(w) <= n:
        return [w]
    chunks = [w[max(0, e - n) : e] for e in range(len(w), 0, -n)]
    if len(chunks[-1]) < m:
        chunks[-1] = w[:n]
    return chunks[::-1]


def classify_input(inp):
    """'image', 'audio' or 'text' for an input as accepted by SmaugDecide.__call__."""
    try:
        from PIL import Image

        if isinstance(inp, Image.Image):
            return 'image'
    except ImportError:
        pass
    if isinstance(inp, os.PathLike):
        inp = os.fspath(inp)
    if isinstance(inp, str):
        low = inp.lower()
        if low.endswith(IMAGE_SUFFIXES):
            return 'image'
        if low.endswith(AUDIO_SUFFIXES) or os.path.exists(inp):
            return 'audio'
        return 'text'
    return 'audio'  # arrays and (samples, sample_rate) pairs


def load_image(inp):
    from PIL import Image

    if isinstance(inp, os.PathLike):
        inp = os.fspath(inp)
    return (inp if isinstance(inp, Image.Image) else Image.open(inp)).convert('RGB')
