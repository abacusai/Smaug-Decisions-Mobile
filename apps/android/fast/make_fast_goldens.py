"""JVM goldens for the fast tier's Kotlin front end (app/src/test/resources/goldens/mel.json): transformers'
Gemma4AudioFeatureExtractor (the audio model's processor) on clips of 0.01-12 s and 30.5 s (cut to the last 30 s), keeping the valid
frames, plus the processor's <|audio|> token count per clip. Also half-precision conversion cases (numpy float16 -> float32) and
the cascade confidence rule on random probability vectors (smaug_decide.cascade.confidence, imported from the package).
usage: make_fast_goldens.py --model <audio model dir> --smaug-pkg <Smaug-Decisions repository root> --out app/src/test/resources/goldens
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import soundfile as sf


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True); ap.add_argument('--smaug-pkg', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.model)
    import smaug_decide as sd
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model)
    fe = proc.feature_extractor
    AUD = proc.tokenizer.convert_tokens_to_ids('<|audio|>')
    rng = np.random.default_rng(3)
    t = lambda s: np.arange(int(s * 16000)) / 16000
    clips = {
        'tiny_0p01s': 0.1 * rng.standard_normal(160),
        'short_0p4s': 0.3 * np.sin(2 * np.pi * 880 * t(0.4)),
        'chirp_1s': 0.5 * np.sin(2 * np.pi * (200 + 3000 * t(1.0)) * t(1.0)),
        'noise_3p7s': 0.2 * rng.standard_normal(int(3.7 * 16000)),
        'mix_12s': 0.3 * np.sin(2 * np.pi * 300 * t(12)) * (np.floor(t(12) * 3) % 2) + 0.05 * rng.standard_normal(12 * 16000),
        'long_30p5s': 0.2 * np.sin(2 * np.pi * 523 * t(30.5)) + 0.02 * rng.standard_normal(int(30.5 * 16000)),
        'loud_clip_2s': np.clip(3 * np.sin(2 * np.pi * 150 * t(2.0)), -1, 1),
    }
    d = os.path.join(a.out, 'mel'); os.makedirs(d, exist_ok=True)
    rows = []
    for name, w in clips.items():
        w = w.astype(np.float32)
        path = os.path.join(d, name + '.wav'); sf.write(path, w, 16000, subtype='FLOAT')
        x = sd.load_audio(path)                                   # last 30 s
        n = len(x); frames = max(0, (n + 160 - 321) // 160 + 1)
        row = {'name': name, 'wav': f'mel/{name}.wav', 'samples_used': n, 'frames': frames}
        if frames > 0:
            f = fe([x], sampling_rate=16000, return_tensors='np')
            feats = f['input_features'][0][:frames]
            m = f.get('input_features_mask', f.get('attention_mask'))
            if m is not None:
                assert int(np.asarray(m[0]).sum()) == frames, (name, int(np.asarray(m[0]).sum()), frames)
            text = sd.render('Q?', ['a', 'b'], [None, None], False, '<|audio|>', None, None, None, 1000)
            ids = proc(text=[text], audio=[x], return_tensors='np')['input_ids'][0].tolist()
            row.update(features_b64=b64(feats), audio_tokens=ids.count(AUD))
        rows.append(row)
        print(name, n, frames, row.get('audio_tokens'))
    halves = np.concatenate([rng.standard_normal(200) * 10, [0.0, -0.0, 1e-7, -3e-6, 6.1e-5, 65504.0, -65504.0, 1e-8, 0.5, 2.0 ** -24]]).astype(np.float16)
    import importlib.util   # the package's cascade.py by path (the name smaug_decide is taken by the release file)
    spec = importlib.util.spec_from_file_location('sd_cascade', os.path.join(a.smaug_pkg, 'smaug_decide', 'cascade.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); confidence = mod.confidence
    conf = []
    for i in range(300):
        k = int(rng.integers(2, 9)); multi = bool(i % 2)
        p = rng.random(k) if multi else rng.dirichlet(np.ones(k) * rng.choice([0.1, 1.0, 5.0]))
        if i % 7 == 0:
            p[0] = 0.9   # exact threshold ties
        conf.append({'p': p.tolist(), 'multi': multi, 'confidence': confidence(dict(enumerate(p.tolist())), multi)})
    json.dump({'clips': rows, 'half_bits': halves.view(np.uint16).tolist(), 'half_values': halves.astype(np.float32).tolist(), 'confidence': conf},
              open(os.path.join(a.out, 'mel.json'), 'w'))
    print('wrote', os.path.join(a.out, 'mel.json'), round(os.path.getsize(os.path.join(a.out, 'mel.json')) / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
