"""Tests for server/decisions_server.py that need no GPU and no model: a fake model stands in for SmaugDecide and returns fixed
probabilities, so these check request parsing, answer shapes, confidence formulas, refusals, errors and authentication.

    pip install -e ".[dev]"
    pytest tests/test_server.py

For an end-to-end check against a real model, start the server and send the curl request from docs/server.md.
"""

import base64
import io
import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'server'))
fastapi = pytest.importorskip('fastapi')
import decisions_server as ds  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class FakeModel:
    """Records every call; returns probabilities that sum to 1 (one answer) or are independent (select-all)."""

    def __init__(self):
        self.calls = []

    def __call__(self, inp, question, options, descriptions=None, multi=False, rules=None, context=None, input_type=None):
        self.calls.append(
            {
                'inp': inp,
                'question': question,
                'options': options,
                'descriptions': descriptions,
                'multi': multi,
                'rules': rules,
                'context': context,
                'input_type': input_type,
            }
        )
        k = len(options)
        if multi:
            return {o: (0.9 if i == 0 else 0.1) for i, o in enumerate(options)}
        p = [2.0 ** -(i + 1) for i in range(k)]
        p[-1] += 1.0 - sum(p)
        return dict(zip(options, p))


@pytest.fixture
def setup():
    model = FakeModel()
    return model, TestClient(ds.build_app(model, 'fake-model'))


def wav_b64(seconds=1.0, sr=16000):
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), np.float32), sr, format='WAV', subtype='PCM_16')
    return base64.b64encode(buf.getvalue()).decode()


def png_data_url():
    from PIL import Image

    buf = io.BytesIO()
    Image.new('RGB', (8, 8), 'white').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def post(client, body, **kw):
    return client.post('/v1/decisions', json=body, **kw)


def test_health(setup):
    _, client = setup
    assert client.get('/health').json() == {'status': 'ok', 'model': 'fake-model'}
    assert client.get('/v1/models').json()['data'][0]['id'] == 'fake-model'


def test_pages(setup):
    _, client = setup
    for page in ('/demo', '/playground'):
        r = client.get(page)
        assert r.status_code == 200 and '<html' in r.text


def test_all_question_types_on_text(setup):
    model, client = setup
    r = post(
        client,
        {
            'input': [
                {'role': 'developer', 'content': 'You moderate a support inbox.'},
                {'role': 'user', 'content': 'My order arrived broken.'},
            ],
            'questions': [
                {'name': 'refund', 'type': 'predicate', 'instructions': 'Is the customer asking for a refund?'},
                {
                    'name': 'team',
                    'type': 'choice',
                    'instructions': 'Which team should handle this?',
                    'choices': [{'value': 'billing', 'description': 'refunds'}, {'value': 'shipping'}, {'value': 'sales'}],
                },
                {
                    'name': 'mood',
                    'type': 'score',
                    'instructions': 'How upset is the customer?',
                    'levels': [{'label': 'calm'}, {'label': 'annoyed'}, {'label': 'angry'}],
                },
                {
                    'name': 'topics',
                    'type': 'multi',
                    'instructions': 'Which topics come up?',
                    'choices': [{'value': 'damage'}, {'value': 'delay'}],
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    a = {x['name']: x for x in r.json()['answers']}
    assert a['refund'] == {'type': 'predicate', 'name': 'refund', 'probability': 0.5}
    assert a['team']['choice'] == 'billing' and a['team']['confidence'] == pytest.approx((3 * 0.5 - 1) / 2)
    assert sum(p['probability'] for p in a['team']['probabilities']) == pytest.approx(1.0)
    assert a['mood']['score'] == pytest.approx(0 * 0.5 + 1 * 0.25 + 2 * 0.25)
    assert [p['label'] for p in a['mood']['probabilities']] == ['calm', 'annoyed', 'angry']
    assert a['topics']['selected'] == ['damage']
    first = model.calls[0]
    assert first['rules'] == 'You moderate a support inbox.' and first['inp'] == 'My order arrived broken.'
    assert first['input_type'] == 'text' and first['context'] is None
    assert model.calls[1]['descriptions'] == ['refunds', None, None]
    assert model.calls[3]['multi'] is True


def test_text_input_is_never_read_as_a_file(setup):
    model, client = setup
    r = post(
        client, {'input': '/etc/hosts', 'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'Is this a path?'}]}
    )
    assert r.status_code == 200 and model.calls[0]['input_type'] == 'text' and model.calls[0]['inp'] == '/etc/hosts'


def test_audio_with_context(setup):
    model, client = setup
    r = post(
        client,
        {
            'input': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_audio', 'input_audio': {'data': wav_b64(2.0), 'format': 'wav'}},
                        {'type': 'input_text', 'text': 'A caller on a support line.'},
                    ],
                }
            ],
            'questions': [{'name': 'done', 'type': 'predicate', 'instructions': 'Has the speaker finished their turn?'}],
        },
    )
    assert r.status_code == 200, r.text
    call = model.calls[0]
    assert call['input_type'] == 'audio' and call['context'] == 'A caller on a support line.'
    assert isinstance(call['inp'], np.ndarray) and len(call['inp']) == 32000


def test_long_audio_keeps_last_30_seconds(setup):
    model, client = setup
    r = post(
        client,
        {
            'input': [{'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {'data': wav_b64(31.0)}}]}],
            'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'Anyone talking?'}],
        },
    )
    assert r.status_code == 200 and len(model.calls[0]['inp']) == 30 * 16000


def test_image(setup):
    model, client = setup
    r = post(
        client,
        {
            'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': png_data_url()}]}],
            'questions': [{'name': 'person', 'type': 'predicate', 'instructions': 'Does this image show a person?'}],
        },
    )
    assert r.status_code == 200, r.text
    assert model.calls[0]['input_type'] == 'image' and model.calls[0]['inp'].size == (8, 8)


def test_refusal_for_too_many_options(setup):
    model, client = setup
    r = post(
        client,
        {
            'input': 'hello',
            'questions': [
                {'name': 'big', 'type': 'choice', 'instructions': 'Pick one', 'choices': [{'value': str(i)} for i in range(21)]},
                {'name': 'ok', 'type': 'predicate', 'instructions': 'Is this a greeting?'},
            ],
        },
    )
    assert r.status_code == 200
    a = r.json()['answers']
    assert a[0]['type'] == 'refusal' and a[1]['type'] == 'predicate' and len(model.calls) == 1


@pytest.mark.parametrize(
    'body',
    [
        {'input': 'x'},
        {'input': 'x', 'questions': []},
        {'input': 'x', 'questions': [{'name': 'q', 'type': 'essay', 'instructions': 'Write'}]},
        {'input': 'x', 'questions': [{'name': 'q', 'type': 'predicate'}]},
        {'input': 'x', 'questions': [{'name': 'q', 'type': 'choice', 'instructions': 'Pick', 'choices': ['a', 'b']}]},
        {
            'input': 'x',
            'questions': [
                {'name': 'q', 'type': 'predicate', 'instructions': 'a'},
                {'name': 'q', 'type': 'predicate', 'instructions': 'b'},
            ],
        },
        {'input': [], 'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'a'}]},
        {
            'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'https://example.com/a.png'}]}],
            'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'a'}],
        },
        {
            'input': [{'role': 'user', 'content': [{'type': 'input_video'}]}],
            'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'a'}],
        },
    ],
)
def test_bad_requests(setup, body):
    model, client = setup
    r = post(client, body)
    assert r.status_code == 400 and r.json()['error']['type'] == 'invalid_request_error'
    assert model.calls == []


def test_api_key(monkeypatch):
    monkeypatch.setenv('DECISIONS_API_KEY', 'secret-key')
    client = TestClient(ds.build_app(FakeModel(), 'fake-model'))
    body = {'input': 'hi', 'questions': [{'name': 'q', 'type': 'predicate', 'instructions': 'Greeting?'}]}
    assert post(client, body).status_code == 401
    assert post(client, body, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert post(client, body, headers={'Authorization': 'Bearer secret-key'}).status_code == 200
    assert client.get('/health').status_code == 200


def test_confidence_formulas():
    assert ds.confidence_choice([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert ds.confidence_choice([1 / 3] * 3) == pytest.approx(0.0)
    assert ds.confidence_score([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert ds.confidence_score([0.5, 0.0, 0.5]) < 0.5


def test_cascade_answers_carry_the_tier():
    from smaug_decide import Cascade

    class Fixed:
        def __init__(self, p):
            self.p, self.calls = p, 0

        def __call__(self, *args, **kw):
            self.calls += 1
            return self.p

    fast, big = Fixed({'Yes': 0.95, 'No': 0.05}), Fixed({'Yes': 0.3, 'No': 0.7})
    client = TestClient(ds.build_app(Cascade(fast, big, threshold=0.9), 'cascade'))
    q = [{'name': 'done', 'type': 'predicate', 'instructions': 'Has the speaker finished?'}]
    audio = [{'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {'data': wav_b64(), 'format': 'wav'}}]}]
    r = post(client, {'input': audio, 'questions': q}).json()
    assert r['answers'][0]['tier'] == 'fast' and r['answers'][0]['probability'] == pytest.approx(0.95)
    fast.p = {'Yes': 0.6, 'No': 0.4}
    r = post(client, {'input': audio, 'questions': q}).json()
    assert r['answers'][0]['tier'] == 'big' and r['answers'][0]['probability'] == pytest.approx(0.3)
    calls = fast.calls
    r = post(client, {'input': 'Is this done?', 'questions': q}).json()  # text skips the fast tier
    assert r['answers'][0]['tier'] == 'big' and fast.calls == calls
