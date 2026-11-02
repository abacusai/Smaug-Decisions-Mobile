"""The readout as plain math, free of torch, so any runtime (MLX, Core ML, llama.cpp, a server) can reproduce it exactly."""

import math


def readout_kind(readout):
    """'marker' for the fast tier's marker head, 'letters' for the letter readout."""
    return 'marker' if 'marker' in readout or 'multi_bias_b' not in readout else 'letters'


def option_probs(z, multi, cap=None, b=0.0):
    """z: option logits. Softcap (letter readout), then softmax for one answer or sigmoid(z - b) for select-all."""
    if cap:
        z = [math.tanh(v / cap) * cap for v in z]
    if multi:
        return [1.0 / (1.0 + math.exp(-(v - b))) for v in z]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def combine_windows(ps, mode):
    """Combine per-window {option: probability} dicts: 'mean' averages them, 'any' takes each option's maximum."""
    keys = list(ps[0])
    if mode == 'mean':
        return {k: sum(p[k] for p in ps) / len(ps) for k in keys}
    return {k: max(p[k] for p in ps) for k in keys}
