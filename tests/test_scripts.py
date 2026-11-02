"""The helper scripts start, print their help and do a dry run without models or network access."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SCRIPTS = ['download.py', 'decide.py', 'benchmark.py', 'calibrate.py']


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('script', SCRIPTS)
def test_help(script):
    r = run(f'scripts/{script}', '--help')
    assert r.returncode == 0 and 'usage' in r.stdout


def test_download_dry_run():
    r = run('scripts/download.py', '--model', 'audio', '--variant', 'coreml', '--dry-run')
    assert r.returncode == 0 and 'abacusai/Smaug-Audio-Decisions-0.7B --include coreml/*' in r.stdout
    r = run('scripts/download.py', '--model', 'litert', '--variant', 'full-vis280', '--dry-run')
    assert r.returncode == 0 and 'smaug-decisions-full-vis280.litertlm manifest.json' in r.stdout


def test_decide_dry_run(tmp_path):
    import soundfile as sf

    wav = tmp_path / 'clip.wav'
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    r = run('scripts/decide.py', str(wav), 'Done?', '--model', 'cascade', '--dry-run')
    out = json.loads(r.stdout)
    assert out['input_type'] == 'audio' and set(out['prompts']) == {'Smaug-Audio-Decisions-0.7B', 'Smaug-Decisions'}
    r = run('scripts/decide.py', 'Refund me.', 'Team?', '--options', 'billing', 'shipping', '--model', 'cascade', '--dry-run')
    assert list(json.loads(r.stdout)['prompts']) == ['Smaug-Decisions']


def test_calibrate_from_answers(tmp_path):
    answers = tmp_path / 'answers.jsonl'
    rows = [
        {'fast': {'Yes': 0.95, 'No': 0.05}, 'big': {'Yes': 0.9, 'No': 0.1}, 'label': 'Yes'},
        {'fast': {'Yes': 0.6, 'No': 0.4}, 'big': {'Yes': 0.1, 'No': 0.9}, 'label': 'No'},
    ]
    answers.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    r = run('scripts/calibrate.py', '--answers', str(answers))
    assert r.returncode == 0 and 'Suggested tau' in r.stdout
