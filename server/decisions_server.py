"""HTTP server for the decision models: one POST evaluates an input (text, or messages carrying text, one audio clip or one image)
against up to 64 typed questions and returns calibrated answers. No text is generated.

    python server/decisions_server.py --model abacusai/Smaug-Decisions --port 8000
    python server/decisions_server.py --fast abacusai/Smaug-Audio-Decisions-0.7B --threshold 0.9      # the cascade (answers carry "tier")

Question types and answers:
    predicate -> {probability}                                    (Yes / No)
    choice    -> {choice, probabilities: [{value, probability}], confidence}
    score     -> {score, probabilities: [{value, label, probability}], confidence}   (ordered levels, lowest first)
    multi     -> {selected, probabilities: [{value, probability}]}                    (select all that apply)
Confidence: choice (k * max_p - 1) / (k - 1); score 1 - sum_i p_i |i - peak| / mean_i |i - (k - 1) / 2|.

Input: a string, or a list of messages. developer / system text becomes the rules; user text becomes the context (with media)
or the text input (without). Audio arrives as an `input_audio` part ({"data": base64, "format": "wav"}) and keeps its last 30 s;
an image as an `input_image` part with a data URL. Remote URLs are never fetched.

DECISIONS_API_KEY=<key> (or DECISIONS_API_KEY_FILE=<path>) in the environment makes /v1/decisions require
`Authorization: Bearer <key>`. Set it on any server with a public port. /health stays open.
"""

import argparse
import base64
import hmac
import io
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from smaug_decide.cascade import CascadeResult


MAX_QUESTIONS, MAX_OPTIONS, MAX_AUDIO_S, SR = 64, 20, 30.0, 16000
QUESTION_TYPES = ('predicate', 'choice', 'score', 'multi')
HERE = os.path.dirname(os.path.abspath(__file__))


class BadRequest(Exception):
    pass


def b64_payload(url_or_data):
    """bytes from a data: URL or a bare base64 string."""
    s = url_or_data.split(',', 1)[1] if url_or_data.startswith('data:') else url_or_data
    try:
        return base64.b64decode(s, validate=False)
    except ValueError:
        raise BadRequest('invalid base64 payload')


def load_audio(part):
    import numpy as np
    import soundfile as sf

    src = part.get('input_audio') if isinstance(part.get('input_audio'), dict) else part
    raw = src.get('data') or src.get('audio_url') or part.get('audio_url')
    if not raw:
        raise BadRequest('input_audio needs "data" (base64) or "audio_url" (data URL)')
    try:
        w, sr = sf.read(io.BytesIO(b64_payload(raw)), dtype='float32')
    except RuntimeError:
        raise BadRequest('could not decode the audio (send wav, flac or ogg)')
    w = w if w.ndim == 1 else w.mean(1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), SR)
        w = resample_poly(w, SR // g, int(sr) // g).astype(np.float32)
    return w[-int(MAX_AUDIO_S * SR) :]  # the decision is about how the clip ends


def load_image(part):
    from PIL import Image

    url = part.get('image_url') if isinstance(part.get('image_url'), str) else (part.get('image_url') or {}).get('url')
    if not url or not url.startswith('data:'):
        raise BadRequest('input_image needs an image_url data URL (remote URLs are not fetched)')
    try:
        return Image.open(io.BytesIO(b64_payload(url))).convert('RGB')
    except OSError:
        raise BadRequest('could not decode the image')


def parse_input(decision_input):
    """(rules, text, audio, image) from an `input`: a string or a list of messages with content parts."""
    rules, texts, audio, images = [], [], None, []
    messages = [{'role': 'user', 'content': decision_input}] if isinstance(decision_input, str) else decision_input
    if not isinstance(messages, list) or not messages:
        raise BadRequest('"input" must be a string or a non-empty list of messages')
    for m in messages:
        if not isinstance(m, dict):
            raise BadRequest('every message must be an object with "role" and "content"')
        role, content = m.get('role', 'user'), m.get('content')
        parts = [{'type': 'input_text', 'text': content}] if isinstance(content, str) else (content or [])
        for p in parts:
            t = p.get('type') if isinstance(p, dict) else None
            if t in ('input_text', 'text'):
                (rules if role in ('developer', 'system') else texts).append(str(p.get('text', '')))
            elif t == 'input_image':
                images.append(load_image(p))
            elif t == 'input_audio':
                if audio is not None:
                    raise BadRequest('one audio input per request')
                audio = load_audio(p)
            else:
                raise BadRequest(f'unsupported input part type {t!r}')
    if audio is not None and images:
        raise BadRequest('send either audio or an image in one request, not both')
    if len(images) > 1:
        raise BadRequest('one image per request')
    return '\n'.join(rules) or None, '\n'.join(texts) or None, audio, (images[0] if images else None)


def parse_question(q):
    """(name, type, instructions, options, descriptions), or a refusal answer for a question this model cannot answer."""
    if not isinstance(q, dict):
        raise BadRequest('every question must be an object')
    name, t = q.get('name'), q.get('type')
    if not isinstance(name, str) or not name:
        raise BadRequest('every question needs a string "name"')
    if t not in QUESTION_TYPES:
        raise BadRequest(f'question {name!r}: type must be one of {QUESTION_TYPES}')
    instructions = q.get('instructions')
    if not isinstance(instructions, str) or not instructions.strip():
        raise BadRequest(f'question {name!r} needs "instructions"')
    if t == 'predicate':
        opts, desc = ['Yes', 'No'], [None, None]
    elif t in ('choice', 'multi'):
        choices = q.get('choices')
        if not isinstance(choices, list) or not all(isinstance(c, dict) and isinstance(c.get('value'), str) for c in choices):
            raise BadRequest(f'question {name!r}: "choices" must be a list of {{value, description?}}')
        opts, desc = [c['value'] for c in choices], [c.get('description') for c in choices]
    else:
        levels = q.get('levels')
        if not isinstance(levels, list) or not all(isinstance(lv, dict) and isinstance(lv.get('label'), str) for lv in levels):
            raise BadRequest(f'question {name!r}: "levels" must be a list of {{label, description?}}, lowest first')
        opts, desc = [lv['label'] for lv in levels], [lv.get('description') for lv in levels]
    if not 2 <= len(opts) <= MAX_OPTIONS:
        return {'type': 'refusal', 'name': name, 'reason': f'{len(opts)} options; this model answers 2 to {MAX_OPTIONS}'}
    return {'name': name, 'type': t, 'instructions': instructions, 'options': opts, 'descriptions': desc}


def confidence_choice(p):
    k = len(p)
    return (k * max(p) - 1) / (k - 1) if k > 1 else 1.0


def confidence_score(p):
    k = len(p)
    peak = max(range(k), key=lambda i: p[i])
    spread = sum(p[i] * abs(i - peak) for i in range(k))
    norm = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
    return 1.0 - spread / norm if norm > 0 else 1.0


def answer(q, p):
    """Shape one question's probabilities (a list in option order) into its typed answer."""
    name, t, opts = q['name'], q['type'], q['options']
    if t == 'predicate':
        return {'type': 'predicate', 'name': name, 'probability': p[0]}
    if t == 'choice':
        best = max(range(len(p)), key=lambda i: p[i])
        return {
            'type': 'choice',
            'name': name,
            'choice': opts[best],
            'confidence': confidence_choice(p),
            'probabilities': [{'value': o, 'probability': x} for o, x in zip(opts, p)],
        }
    if t == 'score':
        return {
            'type': 'score',
            'name': name,
            'score': sum(i * x for i, x in enumerate(p)),
            'confidence': confidence_score(p),
            'probabilities': [{'value': i, 'label': o, 'probability': x} for i, (o, x) in enumerate(zip(opts, p))],
        }
    return {
        'type': 'multi',
        'name': name,
        'selected': [o for o, x in zip(opts, p) if x > 0.5],
        'probabilities': [{'value': o, 'probability': x} for o, x in zip(opts, p)],
    }


def build_app(model, name, api_key=None):
    """model: a SmaugDecide or a Cascade (or anything with the same __call__). Questions run one after another, one forward pass
    each (two for an escalated cascade question). With a Cascade, every answer also says which tier answered it."""
    api_key = api_key or os.environ.get('DECISIONS_API_KEY')
    if not api_key and os.environ.get('DECISIONS_API_KEY_FILE'):  # a mounted secret file keeps the key out of env and argv
        with open(os.environ['DECISIONS_API_KEY_FILE']) as f:
            api_key = f.read().strip()
    lock = threading.Lock()
    app = FastAPI(title='Smaug-Decisions')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': name}

    @app.get('/v1/models')
    def models():
        return {'data': [{'id': name, 'object': 'model'}]}

    @app.get('/demo')
    def demo():
        """Phone-first demo page: turn gate, room awareness, image checks and text routing."""
        with open(os.path.join(HERE, 'demo.html')) as f:
            return HTMLResponse(f.read())

    @app.get('/playground')
    def playground():
        """Browser page: live turn-taking from the microphone and free-form questions about a recorded or uploaded clip."""
        with open(os.path.join(HERE, 'playground.html')) as f:
            return HTMLResponse(f.read())

    @app.post('/v1/decisions')
    def decisions(body: dict, request: Request):
        t0 = time.time()
        if api_key and not hmac.compare_digest(request.headers.get('authorization', ''), f'Bearer {api_key}'):
            return JSONResponse(
                status_code=401, content={'error': {'message': 'missing or invalid API key', 'type': 'authentication_error'}}
            )
        try:
            questions = body.get('questions')
            if not isinstance(questions, list) or not 1 <= len(questions) <= MAX_QUESTIONS:
                raise BadRequest(f'"questions" must hold 1 to {MAX_QUESTIONS} questions')
            parsed = [parse_question(q) for q in questions]
            names = [q['name'] for q in parsed]
            if len(set(names)) != len(names):
                raise BadRequest('question names must be unique')
            rules, text, audio, image = parse_input(body.get('input'))
        except BadRequest as e:
            return JSONResponse(status_code=400, content={'error': {'message': str(e), 'type': 'invalid_request_error'}})
        media = audio if audio is not None else image
        inp, context = (media, text) if media is not None else (text or '', None)
        # explicit input type: a text input must never be read as a file on the server
        input_type = 'audio' if audio is not None else 'image' if image is not None else 'text'
        answers = []
        with lock:
            for q in parsed:
                if q['type'] == 'refusal':
                    answers.append(q)
                    continue
                probs = model(
                    inp,
                    q['instructions'],
                    q['options'],
                    q['descriptions'],
                    q['type'] == 'multi',
                    rules,
                    context,
                    input_type=input_type,
                )
                tier = None
                if isinstance(probs, CascadeResult):
                    probs, tier = probs.probabilities, probs.tier
                a = answer(q, [probs[o] for o in q['options']])
                if tier:
                    a['tier'] = tier
                answers.append(a)
        return {'model': name, 'answers': answers, 'latency_ms': round(1000 * (time.time() - t0), 1)}

    return app


def main():
    ap = argparse.ArgumentParser(description='Serve a decision model at /v1/decisions.')
    ap.add_argument('--model', default='abacusai/Smaug-Decisions', help='Hugging Face repo id or local checkpoint directory')
    ap.add_argument('--fast', default=None, help='serve the cascade with this fast tier for audio, e.g. abacusai/Smaug-Audio-Decisions-0.7B')
    ap.add_argument('--threshold', type=float, default=0.9, help='cascade: the fast tier answers at or above this confidence')
    ap.add_argument('--device', default=None, help='cuda, mps or cpu (default: picked automatically)')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    import uvicorn

    from smaug_decide import Cascade, SmaugDecide

    if a.fast:
        # load both now, so the first escalated request does not pay for loading the big model
        model = Cascade(
            SmaugDecide.from_pretrained(a.fast, device=a.device),
            SmaugDecide.from_pretrained(a.model, device=a.device),
            threshold=a.threshold,
        )
        name = f'cascade({a.fast} -> {a.model}, tau={a.threshold})'
    else:
        model, name = SmaugDecide.from_pretrained(a.model, device=a.device), a.model
    uvicorn.run(build_app(model, name), host=a.host, port=a.port, workers=1, log_level='info')


if __name__ == '__main__':
    main()
