"""Synthetic audio for the fast-tier parity and the cascade tests (generated here: espeak-ng speech, tones, chords, noise;
no benchmark items). Writes <out>/*.wav and <out>/cases.jsonl, one decision per line:
{"id", "input", "question", "options", "descriptions", "multi"}, crossing every clip with the app's audio scenarios plus a few
clip-specific questions. Clip lengths span 0.4 s to 28 s so both the short path and the >512-token trunk band are exercised;
two clips are written at 44.1 kHz stereo / 8 kHz to exercise resampling.
usage: make_clips.py <out dir>
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import SCENARIOS  # noqa: E402

SR = 16000
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)


def t(s, sr=SR):
    return np.arange(int(s * sr)) / sr


def say(text, voice='en-us', speed=150):
    raw = os.path.join(out, '_tmp.wav')
    subprocess.run(['espeak-ng', '-v', voice, '-s', str(speed), '-w', raw, text], check=True, capture_output=True)
    w, sr = sf.read(raw, dtype='float32')
    os.remove(raw)
    n = int(round(len(w) * SR / sr))
    return np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)


def chord(s, freqs, sr=SR):
    x = sum(np.sin(2 * np.pi * f * t(s, sr)) for f in freqs) / len(freqs)
    env = np.minimum(1, t(s, sr) * 8) * np.exp(-0.3 * (t(s, sr) % 1.0))
    return (0.3 * x * env).astype(np.float32)


def melody(s):
    notes = [262, 294, 330, 349, 392, 440, 494, 523]
    seg = int(0.25 * SR)
    return np.concatenate([chord(0.25, [notes[rng.integers(8)], notes[rng.integers(8)] * 1.5]) for _ in range(int(s / 0.25))])


clips = {}
def put(name, w, sr=SR, stereo=False):
    w = np.clip(np.asarray(w, np.float32), -1, 1)
    if stereo:
        w = np.stack([w, 0.8 * w], 1)
    sf.write(os.path.join(out, name), w, sr, subtype='PCM_16')
    clips[name] = round((w.shape[0]) / sr, 2)


q = say('Excuse me, do you know where the nearest train station is?')
put('ask_station.wav', q)
put('done_turn.wav', say('Okay, that is all I needed. Thanks a lot, goodbye.', speed=160))
put('unfinished.wav', say('And then I was going to tell you that the, uh, the reason we', speed=125))
put('complaint.wav', say('I have been waiting for forty minutes and the food is still cold. This is unacceptable!', speed=185))
put('smalltalk.wav', say('Hi! Lovely weather today, isn\'t it? How was your weekend?', voice='en-gb'))
put('two_speakers.wav', np.concatenate([say('Did you finish the report?'), np.zeros(SR // 3, np.float32),
                                        say('Almost, I will send it tonight.', voice='en-gb+f3'), np.zeros(SR // 3, np.float32),
                                        say('Great, thanks.')]))
sp = say('Please turn left at the next corner.')
put('speech_in_noise.wav', sp + 0.06 * rng.standard_normal(len(sp)).astype(np.float32))
put('music_only.wav', melody(6.0))
put('speech_over_music.wav', (lambda s: s + 0.5 * melody(len(s) / SR + 0.25)[:len(s)])(say('Welcome to the show, tonight we have a great lineup.')))
put('siren.wav', 0.4 * np.sin(2 * np.pi * (700 + 300 * np.sin(2 * np.pi * 0.5 * t(5.0))) * t(5.0)))
put('blip_short.wav', 0.3 * np.sin(2 * np.pi * 880 * t(0.4)))
put('quiet_room.wav', 0.003 * rng.standard_normal(int(3 * SR)))
long_talk = np.concatenate([say('Let me explain how the new schedule works.'), np.zeros(SR // 2, np.float32),
                            say('Monday and Wednesday we start at nine, Tuesday and Thursday at ten.'), np.zeros(SR // 2, np.float32),
                            say('Friday is flexible, but please tell your manager in advance.'), np.zeros(SR // 2, np.float32),
                            say('Does that work for everyone?', voice='en-gb')])
put('long_12s.wav', long_talk[: 12 * SR] if len(long_talk) > 12 * SR else np.pad(long_talk, (0, 12 * SR - len(long_talk))))
put('long_28s.wav', np.concatenate([melody(8.0), 0.05 * rng.standard_normal(4 * SR), long_talk, melody(4.0)])[: 28 * SR])
put('resample_44k_stereo.wav', np.interp(np.linspace(0, len(q) - 1, int(len(q) * 44100 / SR)), np.arange(len(q)), q), sr=44100, stereo=True)
put('resample_8k.wav', say('Can you call me back in five minutes?')[::2], sr=8000)

cases = []
for name in clips:
    for s in SCENARIOS:
        if s['kind'] != 'audio':
            continue
        cases.append({'id': f"{name[:-4]}:{s['id']}", 'input': name, 'question': s['question'], 'options': [o for o, _ in s['options']],
                      'descriptions': [d for _, d in s['options']], 'multi': s['multi']})
extra = [('ask_station.wav', 'Is the speaker asking a question?', ['Yes', 'No'], False),
         ('siren.wav', 'What is this sound most likely?', ['an alarm or siren', 'a dog barking', 'a person talking', 'rain'], False),
         ('speech_over_music.wav', 'Which sounds are present?', ['speech', 'music', 'a pure tone', 'noise', 'silence'], True),
         ('complaint.wav', 'How does the speaker sound?', ['calm', 'happy', 'frustrated', 'sad'], False)]
for name, qq, opts, multi in extra:
    cases.append({'id': f'{name[:-4]}:extra', 'input': name, 'question': qq, 'options': opts, 'descriptions': None, 'multi': multi})
with open(os.path.join(out, 'cases.jsonl'), 'w') as f:
    for c in cases:
        f.write(json.dumps(c) + '\n')
print(json.dumps(clips), len(cases), 'cases')
