"""Log-mel front end of the Gemma 4 / EmbeddingGemma 2 audio tower in numpy, written as the spec the Kotlin port (MelFrontEnd.kt)
follows line for line (and the Swift one in the iOS app's MelFrontEnd.swift): 16 kHz mono, 160 zeros prepended,
frames of 320 samples every 160 samples, periodic Hann, |rFFT 512|, 128 HTK mel filters 0-8 kHz without normalisation,
log(mel + 1e-3). Only valid frames are returned: frame i is valid when its 321-sample unfold window lies inside the audio,
i.e. frames = (n + 160 - 321) // 160 + 1."""
import numpy as np

SR, FRAME, HOP, NFFT, NMELS, FLOOR = 16000, 320, 160, 512, 128, 1e-3


def frame_count(n):
    return max(0, (n + FRAME // 2 - (FRAME + 1)) // HOP + 1)


def htk_filters(n_freq=NFFT // 2 + 1, n_mels=NMELS, sr=SR, fmin=0.0, fmax=8000.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.arange(n_freq) * sr / (2 * (n_freq - 1))
    lo, c, hi = pts[:-2], pts[1:-1], pts[2:]
    up = (freqs[:, None] - lo[None]) / (c - lo)[None]
    down = (hi[None] - freqs[:, None]) / (hi - c)[None]
    return np.maximum(0, np.minimum(up, down))          # [n_freq, n_mels]


WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)
FILTERS = htk_filters()


def log_mel(samples):
    """float32 16 kHz mono -> [frames, 128] float32 (valid frames only)."""
    x = np.concatenate([np.zeros(FRAME // 2, np.float64), np.asarray(samples, np.float64)])
    n = frame_count(len(samples))
    out = np.zeros((n, NMELS), np.float32)
    for i in range(n):
        fr = np.zeros(NFFT)
        fr[:FRAME] = x[i * HOP: i * HOP + FRAME] * WINDOW
        mag = np.abs(np.fft.rfft(fr))
        out[i] = np.log(mag @ FILTERS + FLOOR)
    return out
