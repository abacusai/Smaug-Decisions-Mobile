#!/usr/bin/env python3
"""Measure decision latency and memory on this machine with the PyTorch package.

    python scripts/benchmark.py                                # Smaug-Audio-Decisions-0.7B and Smaug-Decisions: audio, text and image
    python scripts/benchmark.py --model audio --kinds audio    # one model, one input type
    python scripts/benchmark.py --n 20 --seconds 10 --json bench.json
    python scripts/benchmark.py --dry-run                      # device info and the plan, load nothing

Inputs are synthetic (a tone with noise, a short message, a drawn image), so no files are needed. Latency is the median and p90 of
--n decisions after --warmup untimed ones, with the model loaded. Memory is the process's peak resident set (and the CUDA peak on
a GPU). For phones, see docs/devices.md: these numbers come from the PyTorch path, not the on-device runtimes.
"""

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

REPOS = {'audio': 'abacusai/Smaug-Audio-Decisions-0.7B', 'decisions': 'abacusai/Smaug-Decisions'}
QUESTIONS = {
    'audio': ('Which of these can be heard?', ['speech', 'music', 'a tone', 'silence'], True),
    'text': ('Which team should handle this message?', ['billing', 'shipping', 'technical support'], False),
    'image': ('What is the main shape in the picture?', ['circle', 'square', 'triangle'], False),
}
TEXT = 'Hi, I was charged twice for my order last week and the second charge still has not been refunded.'


def device_info(device):
    import torch

    info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cpu_count': os.cpu_count(),
        'python': platform.python_version(),
        'torch': torch.__version__,
    }
    try:
        info['ram_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1)
    except (ValueError, OSError, AttributeError):  # not available on every OS; the benchmark still runs
        pass
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    info['device'] = device
    if device == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


def make_input(kind, seconds):
    import numpy as np

    if kind == 'audio':
        t = np.arange(int(seconds * 16000)) / 16000
        rng = np.random.default_rng(0)
        return (0.2 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    if kind == 'image':
        from PIL import Image, ImageDraw

        img = Image.new('RGB', (640, 480), 'white')
        ImageDraw.Draw(img).ellipse((200, 120, 440, 360), fill='navy')
        return img
    return TEXT


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (2**30 if sys.platform == 'darwin' else 2**20), 2)  # bytes on macOS, KiB on Linux


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['audio', 'decisions', 'both'], default='both')
    ap.add_argument('--kinds', default='audio,text,image', help='comma-separated: audio, text, image')
    ap.add_argument('--n', type=int, default=10, help='timed decisions per input type')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--seconds', type=float, default=10.0, help='length of the synthetic audio clip')
    ap.add_argument('--device', help='cuda, mps or cpu (default: picked automatically)')
    ap.add_argument('--json', help='also write the results here')
    ap.add_argument('--dry-run', action='store_true', help='print the device and the plan, load nothing')
    a = ap.parse_args()

    kinds = [k.strip() for k in a.kinds.split(',') if k.strip()]
    bad = [k for k in kinds if k not in QUESTIONS]
    if bad:
        ap.error(f'unknown input types: {bad}')
    models = ['audio', 'decisions'] if a.model == 'both' else [a.model]
    info = device_info(a.device)
    plan = {'models': [REPOS[m] for m in models], 'kinds': kinds, 'n': a.n, 'warmup': a.warmup, 'audio_seconds': a.seconds}
    print(json.dumps({'device': info, 'plan': plan}, indent=1))
    if a.dry_run:
        return

    import torch

    from smaug_decide import SmaugDecide

    results = []
    for m in models:
        t0 = time.time()
        model = SmaugDecide.from_pretrained(REPOS[m], device=info['device'])
        load_s = time.time() - t0
        for kind in kinds:
            question, options, multi = QUESTIONS[kind]
            inp = make_input(kind, a.seconds)
            times = []
            for i in range(a.warmup + a.n):
                t0 = time.time()
                model(inp, question, options, multi=multi, input_type=kind)
                if info['device'] == 'cuda':
                    torch.cuda.synchronize()
                if i >= a.warmup:
                    times.append(1000 * (time.time() - t0))
            times.sort()
            row = {
                'model': REPOS[m],
                'kind': kind,
                'median_ms': round(statistics.median(times), 1),
                'p90_ms': round(times[min(len(times) - 1, int(0.9 * len(times)))], 1),
                'load_s': round(load_s, 1),
                'peak_rss_gb': peak_rss_gb(),
            }
            if info['device'] == 'cuda':
                row['cuda_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            results.append(row)
            print(json.dumps(row))
        del model
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'device': info, 'plan': plan, 'results': results}, f, indent=1)


if __name__ == '__main__':
    main()
