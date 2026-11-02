"""The log-mel front end of the audio towers (Smaug-Decisions and Smaug-Audio-Decisions-0.7B) in plain numpy.

The phone apps have no Python, so they compute these features themselves (Swift on iOS, Kotlin on Android) before the audio
encoder runs. This module is the specification those ports follow, and a test checks it against the Transformers feature
extractor. The recipe: 16 kHz mono, 160 zeros prepended, frames of 320 samples every 160 samples, a periodic Hann window,
|rFFT 512|, 128 HTK mel filters over 0-8 kHz without normalisation, then log(mel + 1e-3). Only valid frames are returned:
frame i is valid when its 321-sample window lies inside the padded audio, so frames = (n + 160 - 321) // 160 + 1.
"""

import numpy as np


MEL_SAMPLE_RATE, MEL_FRAME, MEL_HOP, MEL_NFFT, MEL_BINS, MEL_FLOOR = 16000, 320, 160, 512, 128, 1e-3


def frame_count(n_samples):
    """Number of valid feature frames for a clip of n_samples at 16 kHz (100 frames per second)."""
    return max(0, (n_samples + MEL_FRAME // 2 - (MEL_FRAME + 1)) // MEL_HOP + 1)


def htk_filters(n_freq=MEL_NFFT // 2 + 1, n_mels=MEL_BINS, sr=MEL_SAMPLE_RATE, fmin=0.0, fmax=8000.0):
    """Triangular HTK-scale mel filters, [n_freq, n_mels], unnormalised."""

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.arange(n_freq) * sr / (2 * (n_freq - 1))
    lo, c, hi = pts[:-2], pts[1:-1], pts[2:]
    up = (freqs[:, None] - lo[None]) / (c - lo)[None]
    down = (hi[None] - freqs[:, None]) / (hi - c)[None]
    return np.maximum(0.0, np.minimum(up, down))


def log_mel(samples):
    """float 16 kHz mono samples -> [frames, 128] float32 log-mel features (valid frames only)."""
    x = np.concatenate([np.zeros(MEL_FRAME // 2), np.asarray(samples, np.float64)])
    n = frame_count(len(samples))
    if n == 0:
        return np.zeros((0, MEL_BINS), np.float32)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(MEL_FRAME) / MEL_FRAME)
    idx = np.arange(n)[:, None] * MEL_HOP + np.arange(MEL_FRAME)[None]
    mag = np.abs(np.fft.rfft(x[idx] * window, n=MEL_NFFT, axis=1))
    return np.log(mag @ htk_filters() + MEL_FLOOR).astype(np.float32)
