#!/usr/bin/env python3
"""Smaug-Audio-Decisions-0.7B token-embedding table: fp16 [262144, 512] -> int8 with one fp32 scale per row (symmetric, absmax / 127).

Writes, next to each other:
  embed_tokens_int8.bin         int8 [vocab, 512], row-major            (134,217,728 bytes)
  embed_tokens_int8_scales.bin  float32 little-endian [vocab]           (1,048,576 bytes)
A row is dequantised as q[j] * scale (the iOS app with vDSP, the Android app in Kotlin). Also prints the table-level error.

usage: quantize_embed.py <embed_tokens_fp16.bin> <out dir>
"""
import json
import os
import sys

import numpy as np

DIM = 512


def quantize(t: np.ndarray):
    t = t.astype(np.float32)
    amax = np.abs(t).max(axis=1)
    scale = (amax / 127.0).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0).astype(np.float32)
    q = np.clip(np.rint(t / safe[:, None]), -127, 127).astype(np.int8)
    return q, scale


def main():
    src, out = sys.argv[1], sys.argv[2]
    t = np.fromfile(src, dtype='<f2').reshape(-1, DIM)
    q, scale = quantize(t)
    os.makedirs(out, exist_ok=True)
    q.tofile(os.path.join(out, 'embed_tokens_int8.bin'))
    scale.astype('<f4').tofile(os.path.join(out, 'embed_tokens_int8_scales.bin'))
    f = t.astype(np.float32)
    deq = q.astype(np.float32) * scale[:, None]
    err = np.abs(deq - f)
    rel = np.linalg.norm(deq - f, axis=1) / np.maximum(np.linalg.norm(f, axis=1), 1e-12)
    stats = {'vocab': int(t.shape[0]), 'dim': DIM, 'zero_rows': int((scale == 0).sum()),
             'max_abs_err': float(err.max()), 'mean_abs_err': float(err.mean()),
             'max_abs_value': float(np.abs(f).max()), 'row_rel_l2_err_median': float(np.median(rel)),
             'row_rel_l2_err_max': float(rel.max()),
             'bytes_int8': int(q.nbytes), 'bytes_scales': int(scale.nbytes), 'bytes_fp16': int(t.nbytes)}
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
