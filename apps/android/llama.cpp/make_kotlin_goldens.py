# Goldens for the Android JVM unit tests (android/app/src/test): the Python reference's prompt rendering, char budget, trim,
# HF image grid + Pillow bicubic resize, readout math, WAV decoding and linear resampling. All inputs are synthetic.
# usage: make_kotlin_goldens.py <out dir = android/app/src/test/resources/goldens> [--smaug <model repo dir with smaug_decide.py>]
# With --smaug, every rendered prompt is also asserted identical to the Transformers reference smaug_decide.render.
import argparse, base64, hashlib, io, json, os, sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llama_decide as L  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--smaug')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
smaug = None
if a.smaug:
    # exec only its constants + trim/render (importing it needs torch + a recent transformers)
    import ast, types
    src = open(os.path.join(a.smaug, 'smaug_decide.py')).read()
    keep = [n for n in ast.parse(src).body if (isinstance(n, ast.FunctionDef) and n.name in ('trim', 'render'))
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper()
                                                  or isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) and e.id.isupper() for e in t.elts)
                                                  for t in n.targets))]
    smaug = types.SimpleNamespace(); ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'smaug_decide.py', 'exec'), ns)
    smaug.render = ns['render']
    # the Kotlin port and llama_decide.py must use the release's own budget constants
    assert (ns['MAX_TOKENS'], ns['IMAGE_TOKENS']) == (L.MAX_TOKENS, L.IMAGE_TOKENS), (ns['MAX_TOKENS'], ns['IMAGE_TOKENS'])
rng = np.random.default_rng(1234)


def budget(media, audio_samples, n_options):
    mt = audio_samples / L.SAMPLE_RATE * L.AUDIO_TOKENS_PER_SECOND + 2 if media == 'audio' else (L.IMAGE_TOKENS + 2 if media == 'image' else 0)
    return int(max(400, (L.MAX_TOKENS - mt - 80 - 10 * n_options) * 3.5))


# ---- render: ~45 cases ----
FOX = 'The quick brown fox jumps over the lazy dog. '
UNI = 'Ünïcödé façade — naïve café, 日本語のテキスト, emoji 😀🎉👍🏽, combining é, RTL שלום, math ∑∞. '
R = []
def case(question, options, media='text', text=None, descriptions=None, multi=False, rules=None, context=None, audio_samples=0, note=''):
    R.append(dict(question=question, options=options, media=media, text=text, descriptions=descriptions, multi=multi, rules=rules,
                  context=context, audio_samples=audio_samples, note=note))

case('Is this message spam?', ['Yes', 'No'], text='hey r u free 2nite?', note='predicate')
case('How upset is the customer?', ['calm', 'annoyed', 'angry'], text='The order never arrived.', note='choice')
case('Which topics?', ['praise', 'bug report', 'feature request', 'design complaint', 'pricing'], text='Love it but export is broken.', multi=True, note='select-all')
case('Which topics?', ['a', 'b'], text='x', multi=True, note='select-all 2 options')
case('What is the user asking about?', ['billing', 'shipping', 'returns', 'account', 'other'], text='Can I change the address?',
     descriptions=['charges, refunds', 'delivery, tracking', None, 'login, password', None], note='descriptions with None')
case('Pick one', ['A', 'B', 'C'], text='t', descriptions=['', 'second', ''], note='empty-string descriptions are skipped')
case('Pick', ['x', 'y'], text='t', descriptions=['only one'], note='descriptions shorter than options: zip truncates options')
case('Is it urgent?', ['Yes', 'No'], text='Server down!', rules='Outages are always urgent.', note='rules')
case('Is it urgent?', ['Yes', 'No'], text='Server down!', context='The user is on-call.', note='context')
case('Is it urgent?', ['Yes', 'No'], text='Server down!', rules='R1. R2.', context='C1.', note='rules + context')
case('Is it urgent?', ['Yes', 'No'], text='Server down!', rules='', context='', note='empty rules/context omitted')
case('Q?', ['Yes', 'No'], text='', note='empty text input omitted')
case('Q?', ['Yes', 'No'], text=None, note='no input at all')
case('Is the text repetitive?', ['Yes', 'No'], text=FOX * 40, note='1800 chars, under the budget: not trimmed')
case('Is the text repetitive?', ['Yes', 'No'], text=FOX * 120, note='long text trimmed')
case('Is the text repetitive?', ['Yes', 'No', 'Maybe'] + [f'opt{i}' for i in range(10)], text=FOX * 120, note='trim budget shrinks with options')
case('Summarize?', ['Yes', 'No'], text=UNI * 80, note='unicode long text trimmed by code points')
case('Summarize?', ['Yes', 'No'], text='😀' * 5000, note='astral-plane only, trimmed')
case('Q?', ['Yes', 'No'], text='x', rules=FOX * 60, note='long rules trimmed at budget // 2')
case('Q?', ['Yes', 'No'], text='x', context=FOX * 120, note='long context trimmed')
case('Q?', ['Yes', 'No'], text=FOX * 120, rules=FOX * 60, context=FOX * 120, note='everything trimmed')
case('Q?', ['Yes', 'No'], text='line one\nline two\n\nline four\ttab', note='newlines and tabs in input')
case('Does it contain <|image|> literally?', ['Yes', 'No'], text='a <|image|> b <turn|> c', note='special-token-looking text is literal')
case('Which language?', ['English', 'French', 'Spanish', 'German'], text='Je voudrais annuler mon abonnement.', note='choice 4')
case('Many options', [f'option {i}' for i in range(20)], text='pick the 7th', note='20 options (max)')
case('Many options', [f'option {i}' for i in range(20)], text=FOX * 120, multi=True, note='20 options select-all long text')
case('Question with “quotes” and émojis 🙂?', ['Ja', 'Nein'], text='Größe', note='unicode question/options')
case('Is anyone speaking?', ['Yes', 'No'], media='audio', audio_samples=16000 * 3, note='audio 3 s')
case('Has the speaker finished?', ['Yes', 'No'], media='audio', audio_samples=73297, note='audio odd length')
case('Which sounds?', ['speech', 'music', 'a pure tone', 'noise', 'silence'], media='audio', audio_samples=16000 * 30, multi=True, note='audio 30 s select-all')
case('Q?', ['Yes', 'No'], media='audio', audio_samples=16000 * 30, rules=FOX * 60, context=FOX * 60, note='audio 30 s, rules/context trimmed')
case('Q?', [f'o{i}' for i in range(20)], media='audio', audio_samples=16000 * 30, context=FOX * 30, note='audio 30 s + 20 options: smallest budget')
case('Q?', ['Yes', 'No'], media='audio', audio_samples=16000 * 10, text='ignored text', note='text ignored when media is given')
case('Which objects?', ['house', 'sun', 'car', 'tree', 'person'], media='image', multi=True, note='image select-all')
case('What kind of image?', ['photograph', 'painting', 'diagram', 'screenshot'], media='image', note='image choice')
case('Is this from space?', ['Yes', 'No'], media='image', rules='Only answer about the image.', context=FOX * 200, note='image with long context')
case('Is this from space?', ['Yes', 'No'], media='image', descriptions=['seen from orbit', None], note='image with descriptions')
for i in range(8):   # random mixes
    n = int(rng.integers(2, 21)); media = ['text', 'audio', 'image'][i % 3]
    words = lambda k: ' '.join(rng.choice(['alpha', 'beta', 'γάμμα', 'δ', '数据', 'x', 'yes', 'no', '🙂', 'zeta'], size=k))
    case(f'random question {i}?', [words(int(rng.integers(1, 4))) for _ in range(n)], media=media,
         text=words(int(rng.integers(1, 2000))), descriptions=[words(2) if rng.random() < 0.5 else None for _ in range(n)],
         multi=bool(rng.random() < 0.5), rules=words(int(rng.integers(0, 600))) or None, context=words(int(rng.integers(0, 1500))) or None,
         audio_samples=int(rng.integers(1000, 16000 * 30)), note=f'random {media}')
for c in R:
    tag = '<|audio|>' if c['media'] == 'audio' else '<|image|>' if c['media'] == 'image' else None
    desc = list(c['descriptions']) if c['descriptions'] else [None] * len(c['options'])
    bud = budget(c['media'], c['audio_samples'], len(c['options']))
    args = (c['question'], c['options'], desc, c['multi'], tag, c['rules'], c['context'], c['text'] if tag is None else None, bud)
    c['budget'] = bud; c['prompt'] = L.render(*args)
    if smaug:
        assert smaug.render(*args) == c['prompt'], c['note']
json.dump(R, open(os.path.join(a.out, 'render.json'), 'w'), ensure_ascii=False, indent=1)

# ---- trim ----
T = []
for s in ['', 'a', 'abcdef', FOX, UNI, '😀🎉👍🏽' * 3, FOX * 10]:
    for n in [0, 1, 2, 3, 4, 5, 9, 10, 33, 100, 200, 1000]:
        if n >= 0:
            T.append(dict(s=s, n=n, out=L.trim(s, n)))
json.dump(T, open(os.path.join(a.out, 'trim.json'), 'w'), ensure_ascii=False, indent=1)

# ---- image grid sizes ----
S = []
dims = [(1, 1), (10, 10), (47, 47), (48, 48), (100, 100), (224, 224), (300, 800), (480, 640), (500, 500), (720, 1280), (1080, 1920),
        (3000, 4000), (4032, 3024), (8000, 6000), (1, 5000), (5000, 1), (20, 4000), (4000, 20), (48, 20000), (20000, 48), (333, 777),
        (1, 47), (47, 1), (100000, 3)] + [tuple(int(v) for v in rng.integers(1, 5000, size=2)) for _ in range(40)]
for h, w in dims:
    th, tw = L.hf_image_size(h, w)
    S.append(dict(h=h, w=w, th=th, tw=tw, tokens=(th // 48) * (tw // 48)))
json.dump(S, open(os.path.join(a.out, 'image_sizes.json'), 'w'), indent=1)

# ---- bicubic resize, bit-exact vs Pillow ----
idir = os.path.join(a.out, 'images'); os.makedirs(idir, exist_ok=True)
def scene(w, h):
    im = Image.new('RGB', (w, h), (135, 206, 235)); d = ImageDraw.Draw(im)
    d.rectangle([0, int(h * .75), w, h], fill=(60, 160, 60)); d.ellipse([int(w * .8), int(h * .05), int(w * .95), int(h * .25)], fill=(255, 220, 0))
    d.polygon([(int(w * .3), int(h * .45)), (int(w * .5), int(h * .25)), (int(w * .7), int(h * .45))], fill=(120, 40, 30))
    d.line([(0, 0), (w, h)], fill=(0, 0, 0), width=1); return im
def noise(w, h): return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), 'RGB')
I = []
srcs = {'scene_640x480': scene(640, 480), 'scene_1920x1080': scene(1920, 1080), 'scene_37x91': scene(37, 91), 'noise_64x48': noise(64, 48),
        'noise_13x7': noise(13, 7), 'noise_800x60': noise(800, 60), 'noise_250x250': noise(250, 250), 'scene_768x816': scene(768, 816),
        'noise_2000x30': noise(2000, 30), 'scene_4000x3000': scene(4000, 3000)}
for name, im in srcs.items():
    im.save(os.path.join(idir, name + '.png'))
    out = L.hf_resize(im)
    I.append(dict(name=name, w=im.width, h=im.height, tw=out.width, th=out.height,
                  sha256=hashlib.sha256(np.asarray(out, dtype=np.uint8).tobytes()).hexdigest()))
    # also a plain resize to a non-grid size, both directions
    for tw, th in [(max(1, im.width // 3), max(1, im.height // 2)), (im.width * 2 + 1, im.height)]:
        o2 = im.resize((tw, th), Image.BICUBIC)
        I.append(dict(name=name, w=im.width, h=im.height, tw=tw, th=th, plain=True,
                      sha256=hashlib.sha256(np.asarray(o2, dtype=np.uint8).tobytes()).hexdigest()))
json.dump(I, open(os.path.join(a.out, 'resize.json'), 'w'), indent=1)

# ---- readout ----
B = 8.010554313659668   # Smaug-Decisions readout.json multi_bias_b
D = []
for i in range(200):
    n = int(rng.integers(2, 21)); z = rng.uniform(-10, 40, size=n).astype(np.float32).astype(np.float64)
    if i % 10 == 0:
        z[:] = z[0]   # ties
    e = np.exp(z - z.max())
    D.append(dict(z=z.tolist(), softmax=(e / e.sum()).tolist(), sigmoid=(1 / (1 + np.exp(-(z - B)))).tolist()))
json.dump(dict(b=B, cases=D), open(os.path.join(a.out, 'readout.json'), 'w'), indent=1)

# ---- audio: WAV decode + resample, vs load_audio ----
adir = os.path.join(a.out, 'audio'); os.makedirs(adir, exist_ok=True)
A = []
def f32b64(x): return base64.b64encode(np.asarray(x, dtype='<f4').tobytes()).decode()
for sr, ch, sub, secs in [(16000, 1, 'PCM_16', 0.5), (44100, 1, 'PCM_16', 0.37), (48000, 2, 'PCM_16', 0.3), (8000, 1, 'PCM_16', 0.61),
                          (22050, 1, 'FLOAT', 0.4), (16000, 1, 'PCM_24', 0.2), (11025, 2, 'PCM_U8', 0.33), (8000, 1, 'PCM_16', 31.0)]:
    n = int(sr * secs); t = np.arange(n) / sr
    w = 0.5 * np.sin(2 * np.pi * 440 * t)[:, None] + 0.05 * rng.standard_normal((n, ch))
    name = f'{sr}_{ch}ch_{sub}_{n}.wav'; sf.write(os.path.join(adir, name), np.clip(w, -1, 1).astype(np.float32), sr, subtype=sub)
    out = L.load_audio(os.path.join(adir, name))
    A.append(dict(file=name, sr=sr, channels=ch, n_out=len(out), sha256=hashlib.sha256(np.asarray(out, dtype='<f4').tobytes()).hexdigest(),
                  head=out[:64].tolist()))
# resample only (float arrays in, float32 out)
RS = []
for sr in [8000, 11025, 22050, 32000, 44100, 48000, 96000, 12345]:
    n = int(rng.integers(50, 3000)); w = rng.uniform(-1, 1, size=n).astype(np.float32)
    out = L.load_audio((w, sr))
    RS.append(dict(sr=sr, x=f32b64(w), y=f32b64(out)))
json.dump(dict(wav=A, resample=RS), open(os.path.join(a.out, 'audio.json'), 'w'), indent=1)
print(f'render {len(R)}, trim {len(T)}, sizes {len(S)}, resize {len(I)}, readout {len(D)}, wav {len(A)}, resample {len(RS)} -> {a.out}'
      + (' (render == smaug_decide.render: OK)' if smaug else ''))
