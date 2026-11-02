# Build the parity set: synthetic questions written for this test (no evaluation data), synthetic + public-domain images,
# and short generated audio (espeak-ng speech, tones, noise, silence). Writes <out>/cases.json + media files.
# usage: make_inputs.py <out dir>   (needs espeak-ng and ffmpeg on PATH for speech; downloads 2 public-domain NASA/PD images)
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw


if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
    sys.exit('usage: make_inputs.py <out dir>   (needs espeak-ng and ffmpeg on PATH)')
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
SR = 16000
rng = np.random.default_rng(0)


def wav(name, w):
    p = os.path.join(out, name)
    sf.write(p, np.clip(w, -1, 1).astype(np.float32), SR)
    return name


def speech(name, text, voice='en-us', speed=150):
    raw = os.path.join(out, name + '.raw.wav')
    subprocess.run(['espeak-ng', '-v', voice, '-s', str(speed), '-w', raw, text], check=True)
    subprocess.run(
        ['ffmpeg', '-loglevel', 'error', '-y', '-i', raw, '-ac', '1', '-ar', str(SR), os.path.join(out, name)], check=True
    )
    os.remove(raw)
    return name


def t(s):
    return np.arange(int(s * SR)) / SR


audio = {
    'speech_question': speech('speech_question.wav', 'Could you tell me what time the pharmacy closes tonight?'),
    'speech_unfinished': speech('speech_unfinished.wav', 'So what I wanted to say was that the, um, the thing is', speed=130),
    'speech_angry': speech(
        'speech_angry.wav', 'This is the third time I have called and nobody has fixed my internet!', speed=190
    ),
    'tone_440': wav('tone_440.wav', 0.3 * np.sin(2 * np.pi * 440 * t(3.0))),
    'beeps': wav('beeps.wav', 0.4 * np.sin(2 * np.pi * 1000 * t(4.0)) * (np.floor(t(4.0) * 2) % 2 == 0)),
    'noise': wav('noise.wav', 0.1 * rng.standard_normal(int(5 * SR))),
    'silence': wav('silence.wav', np.zeros(int(2 * SR))),
}


def draw_scene(name, kind):
    im = Image.new('RGB', (640, 480), (135, 206, 235))
    d = ImageDraw.Draw(im)
    if kind == 'house':
        d.rectangle([0, 360, 640, 480], fill=(60, 160, 60))
        d.rectangle([220, 220, 420, 380], fill=(200, 80, 60))
        d.polygon([(200, 220), (320, 120), (440, 220)], fill=(120, 40, 30))
        d.ellipse([520, 30, 600, 110], fill=(255, 220, 0))
    elif kind == 'shapes':
        im = Image.new('RGB', (512, 512), 'white')
        d = ImageDraw.Draw(im)
        d.ellipse([40, 40, 200, 200], fill='red')
        d.rectangle([280, 60, 460, 220], fill='blue')
        d.polygon([(256, 300), (150, 470), (362, 470)], fill='green')
    elif kind == 'text':
        im = Image.new('RGB', (800, 300), 'white')
        d = ImageDraw.Draw(im)
        d.text((40, 120), 'WARNING: WET FLOOR', fill='black', font_size=64)
    im.save(os.path.join(out, name))
    return name


images = {
    'house': draw_scene('house.png', 'house'),
    'shapes': draw_scene('shapes.png', 'shapes'),
    'sign': draw_scene('sign.png', 'text'),
}
PD = {  # public domain: NASA photographs (Apollo 17, Apollo 11), and Leonardo's Mona Lisa (PD-art)
    'earth': (
        'earth.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/500px-The_Earth_seen_from_Apollo_17.jpg',
    ),
    'moon': (
        'moon.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Aldrin_Apollo_11_original.jpg/500px-Aldrin_Apollo_11_original.jpg',
    ),
    'portrait': (
        'portrait.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/500px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg',
    ),
}
for k, (fn, url) in PD.items():
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'parity-test/0.1 (research)'})
        open(os.path.join(out, fn), 'wb').write(urllib.request.urlopen(req, timeout=30).read())
        Image.open(os.path.join(out, fn)).verify()
        images[k] = fn
    except Exception as e:
        print('skip', k, type(e).__name__)


def Q(inp, question, options=('Yes', 'No'), **kw):
    return {'input': inp, 'question': question, 'options': list(options), **kw}


cases = [
    Q(
        'The order never arrived and support stopped answering my emails.',
        'How upset is the customer?',
        ['calm', 'annoyed', 'angry'],
    ),
    Q('Thanks so much, the replacement works perfectly!', 'Does this message need a reply from a human agent?'),
    Q(
        'Can I change the delivery address after I paid?',
        'What is the user asking about?',
        ['billing', 'shipping', 'returns', 'account', 'other'],
        descriptions=['charges, refunds, invoices', 'delivery, tracking, addresses', None, 'login, password, profile', None],
    ),
    Q(
        'I love the new dashboard but the export button is broken and the dark mode is too dark.',
        'Which topics does the feedback mention?',
        ['praise', 'bug report', 'feature request', 'design complaint', 'pricing'],
        multi=True,
    ),
    Q(
        'Meeting moved to 3pm Thursday, bring the Q3 slides.',
        'Which fields are present?',
        ['date', 'time', 'location', 'attachment', 'person'],
        multi=True,
    ),
    Q(
        'hey r u free 2nite?',
        'Is this message spam?',
        rules='Messages from saved contacts are never spam.',
        context='Sender is a saved contact.',
    ),
    Q(
        'Transfer $4,900 to account 88213 today, it is urgent, the CEO approved it.',
        'Should this request be escalated for fraud review?',
        rules='Escalate any payment request that combines urgency with an unusual recipient.',
    ),
    Q(
        'The patient reports mild headache, no fever, normal blood pressure.',
        'What triage level fits?',
        ['routine', 'urgent', 'emergency'],
    ),
    Q('def add(a, b): return a - b', 'Does the code do what its name says?'),
    Q('Je voudrais annuler mon abonnement.', 'Which language is the message in?', ['English', 'French', 'Spanish', 'German']),
    Q('The quick brown fox jumps over the lazy dog. ' * 40, 'Is the text repetitive?'),  # long input exercises trim/char budget
    Q(audio['speech_question'], 'Is the speaker asking a question?'),
    Q(audio['speech_unfinished'], 'Has the speaker finished their turn?'),
    Q(audio['speech_angry'], 'How does the speaker sound?', ['calm', 'happy', 'frustrated', 'sad']),
    Q(audio['tone_440'], 'Which sounds are present?', ['speech', 'music', 'a pure tone', 'noise', 'silence'], multi=True),
    Q(audio['beeps'], 'What is this sound most likely?', ['an alarm or beeping', 'a dog barking', 'a person talking', 'rain']),
    Q(audio['noise'], 'Is anyone speaking?'),
    Q(audio['silence'], 'Is the recording silent?'),
    Q(images['house'], 'Which objects are in the picture?', ['house', 'sun', 'car', 'tree', 'person'], multi=True),
    Q(images['shapes'], 'How many shapes are there?', ['one', 'two', 'three', 'four']),
    Q(images['sign'], 'What does the sign warn about?', ['wet floor', 'high voltage', 'falling rocks', 'no entry']),
]
if 'earth' in images:
    cases.append(Q(images['earth'], 'Is this a photo taken from space?'))
if 'moon' in images:
    cases.append(
        Q(images['moon'], 'Which apply to the picture?', ['a person', 'a spacesuit', 'an animal', 'water', 'a flag'], multi=True)
    )
if 'portrait' in images:
    cases.append(Q(images['portrait'], 'What kind of image is this?', ['photograph', 'painting', 'diagram', 'screenshot']))
for i, c in enumerate(cases):
    c['id'] = f'c{i:02d}'
    c['kind'] = 'audio' if c['input'].endswith('.wav') else 'image' if c['input'].endswith(('.png', '.jpg')) else 'text'
json.dump(cases, open(os.path.join(out, 'cases.json'), 'w'), indent=1)
print(len(cases), 'cases:', {k: sum(c['kind'] == k for c in cases) for k in ('text', 'image', 'audio')})
