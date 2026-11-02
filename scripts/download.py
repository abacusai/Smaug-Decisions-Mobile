#!/usr/bin/env python3
"""Download Smaug-Decisions models from Hugging Face (a thin wrapper around `hf download`).

    python scripts/download.py --model decisions                       # Transformers weights for Python / the server (~17 GB)
    python scripts/download.py --model audio                         # Smaug-Audio-Decisions-0.7B Transformers weights (~3 GB)
    python scripts/download.py --model audio --variant coreml        # int8 Core ML units for iOS / macOS (~1.4 GB)
    python scripts/download.py --model litert --variant full-vis280  # a LiteRT-LM file for phones (3.45 GB) + manifest.json
    python scripts/download.py --list                                # every model and variant
    python scripts/download.py --model litert --dry-run              # print the command, download nothing

Files go to ./models/<repo name> unless --local-dir is given.
"""

import argparse
import shutil
import subprocess
import sys


TRANSFORMERS_FILES = ['*.json', '*.jinja', 'model*.safetensors', 'head.pt']

# model -> (repo, {variant: (include patterns, description)}, default variant)
MODELS = {
    'decisions': (
        'abacusai/Smaug-Decisions',
        {'transformers': (TRANSFORMERS_FILES, 'Transformers checkpoint (bf16, about 17 GB): Python, server, MLX, GGUF')},
        'transformers',
    ),
    'audio': (
        'abacusai/Smaug-Audio-Decisions-0.7B',
        {
            'transformers': (TRANSFORMERS_FILES, 'Transformers checkpoint (about 3 GB): Python, server, Core ML conversion'),
            'coreml': (['coreml/*'], 'int8 Core ML units, embedding table and readout for iOS / macOS (about 1.4 GB)'),
        },
        'transformers',
    ),
    'litert': (
        'abacusai/Smaug-Decisions-LiteRT',
        {
            'full-vis560': (['smaug-decisions-full-vis560.litertlm', 'manifest.json'], 'text, audio, images (560); 8 GB phones, 3.45 GB'),
            'full-vis280': (['smaug-decisions-full-vis280.litertlm', 'manifest.json'], 'text, audio, images (280); 6 GB phones, 3.45 GB'),
            'noaudio-vis560': (['smaug-decisions-noaudio-vis560.litertlm', 'manifest.json'], 'text and images at 560 tokens (3.15 GB)'),
            'noaudio-vis280': (['smaug-decisions-noaudio-vis280.litertlm', 'manifest.json'], 'text and images at 280 tokens (3.15 GB)'),
            'novision': (['smaug-decisions-novision.litertlm', 'manifest.json'], 'text and audio (3.28 GB)'),
            'textonly': (['smaug-decisions-textonly.litertlm', 'manifest.json'], 'text only (2.97 GB)'),
            'manifest': (['manifest.json'], 'only manifest.json: every file with its size, sha256, inputs and URL'),
        },
        'full-vis560',
    ),
}


def hf_command():
    for exe in ('hf', 'huggingface-cli'):
        if shutil.which(exe):
            return [exe, 'download']
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=sorted(MODELS))
    ap.add_argument('--variant', help='which files to fetch (default: the first variant listed by --list)')
    ap.add_argument('--local-dir', help='target directory (default: models/<repo name>)')
    ap.add_argument('--revision', help='a branch, tag or commit of the model repository')
    ap.add_argument('--list', action='store_true', help='list the models and variants, then exit')
    ap.add_argument('--dry-run', action='store_true', help='print the download command without running it')
    a = ap.parse_args()

    if a.list or not a.model:
        for name, (repo, variants, default) in MODELS.items():
            print(f'{name}  ({repo})')
            for v, (_, desc) in variants.items():
                print(f'  --variant {v:<15} {desc}{"  [default]" if v == default else ""}')
        return 0 if a.list else 2

    repo, variants, default = MODELS[a.model]
    variant = a.variant or default
    if variant not in variants:
        ap.error(f'--model {a.model} has variants: {", ".join(variants)}')
    patterns, _ = variants[variant]
    local_dir = a.local_dir or f'models/{repo.split("/")[1]}'
    base = hf_command()
    cmd = (base or ['hf', 'download']) + [repo, '--include', *patterns, '--local-dir', local_dir]
    if a.revision:
        cmd += ['--revision', a.revision]
    print(' '.join(cmd))
    if a.dry_run:
        return 0
    if base is None:
        print('the Hugging Face CLI is missing: pip install -U huggingface_hub', file=sys.stderr)
        return 1
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
