"""Fast, deterministic tests of the package that need no GPU and no model weights: prompt rendering, input detection, the
readout math, the cascade's routing and confidence rule, threshold calibration and the log-mel front end.

    pytest tests/
"""

import math

import numpy as np
import pytest

from smaug_decide import Cascade, build_prompt, confidence, option_probs, readout_kind, render_letters, render_marker
from smaug_decide.media import SAMPLE_RATE, audio_chunks, classify_input, load_audio
from smaug_decide.readout import combine_windows


def test_render_letters_shape():
    text = render_letters('Which genre?', ['rock', 'jazz'], ['loud', None], False, '<|audio|>', 'Be strict.', None, None, 1000)
    assert text.startswith('<bos><|turn>user\nYou make decisions for apps.')
    assert 'Rules: Be strict.\nInput: <|audio|>\nQuestion: Which genre?\nA) rock — loud\nB) jazz\n' in text
    assert text.endswith('Answer with the letter of the best option.<turn|>\n<|turn>model\n')
    multi = render_letters('Which sounds?', ['a', 'b'], [None, None], True, None, None, None, 'some text', 1000)
    assert 'Question: Which sounds? Select all that apply.' in multi and 'Input: some text' in multi
    assert 'Answer with the letters of all options that apply.' in multi


def test_render_marker_shape():
    text = render_marker('Is it loud?', ['Yes', 'No'], [None, 'quiet'], False, '<|audio|>', None, 'a call', None, 1000)
    assert text == 'task: decide | context: a call\ninput: <|audio|>\nquestion: Is it loud?\n<unused0> Yes\n<unused0> No: quiet\n'
    assert '(select all that apply)' in render_marker('q', ['a', 'b'], [None, None], True, None, None, None, 't', 1000)


def test_long_text_is_trimmed_head_and_tail():
    long = 'x' * 50000 + 'END'
    text, audio, image = build_prompt('letters', long, 'q', ['a', 'b'])
    assert audio is None and image is None and 'END' in text and ' ... ' in text and len(text) < 10000


def test_build_prompt_audio_and_validation():
    clip = np.zeros(SAMPLE_RATE * 2, np.float32)
    text, audio, image = build_prompt('marker', clip, 'q', ['a', 'b'])
    assert 'input: <|audio|>' in text and audio.shape == (SAMPLE_RATE * 2,)
    with pytest.raises(ValueError):
        build_prompt('letters', 'text', 'q', ['only one'])
    with pytest.raises(ValueError):
        build_prompt('letters', 'text', 'q', [str(i) for i in range(21)])
    with pytest.raises(ValueError):
        build_prompt('letters', 'text', 'q', ['a', 'b'], input_type='video')


def test_input_detection(tmp_path):
    assert classify_input('The order never arrived.') == 'text'
    assert classify_input('photo.JPG') == 'image'
    assert classify_input('clip.flac') == 'audio'
    assert classify_input(np.zeros(10)) == 'audio'
    f = tmp_path / 'noext'
    f.write_bytes(b'')
    assert classify_input(str(f)) == 'audio'  # an existing file without an image suffix is read as audio
    text, _, _ = build_prompt('letters', str(f), 'q', ['a', 'b'], input_type='text')
    assert str(f) in text  # input_type='text' never touches the file


def test_load_audio_resamples_and_keeps_the_end():
    w = load_audio((np.ones(8000 * 40, np.float32), 8000))
    assert len(w) == 30 * SAMPLE_RATE
    stereo = np.stack([np.ones(100), -np.ones(100)], 1)
    assert np.allclose(load_audio(stereo), 0.0)


def test_readout_math():
    p = option_probs([2.0, 0.0, -1.0], multi=False)
    assert sum(p) == pytest.approx(1.0) and p[0] > p[1] > p[2]
    capped = option_probs([100.0, 0.0], multi=False, cap=30.0)
    assert capped[0] == pytest.approx(1 / (1 + math.exp(-30 * math.tanh(100 / 30))))
    s = option_probs([1.0, -1.0], multi=True, b=1.0)
    assert s == pytest.approx([0.5, 1 / (1 + math.exp(2.0))])


def test_readout_kind():
    assert readout_kind({'letters': 'ABC', 'final_logit_softcapping': 30.0, 'multi_bias_b': 1.2}) == 'letters'
    assert readout_kind({'marker': '<unused0>', 'marker_id': 6}) == 'marker'


def test_confidence():
    assert confidence({'a': 0.7, 'b': 0.3}) == pytest.approx(0.7)
    assert confidence({'a': 0.95, 'b': 0.2}, multi=True) == pytest.approx(0.8)


class Fixed:
    def __init__(self, probs):
        self.probs, self.calls = probs, 0

    def __call__(self, *args):
        self.calls += 1
        return self.probs


def test_cascade_escalates_only_when_unsure():
    fast, big = Fixed({'Yes': 0.95, 'No': 0.05}), Fixed({'Yes': 0.2, 'No': 0.8})
    r = Cascade(fast, big, threshold=0.9)('clip.wav', 'Done?')
    assert r.tier == 'fast' and big.calls == 0 and not r.routed
    fast.probs = {'Yes': 0.6, 'No': 0.4}
    r = Cascade(fast, big, threshold=0.9)('clip.wav', 'Done?')
    assert r.tier == 'big' and r.probabilities == {'Yes': 0.2, 'No': 0.8} and r.fast_confidence == pytest.approx(0.6)
    fast.probs = {'Yes': 0.9, 'No': 0.1}
    assert Cascade(fast, big, threshold=0.9)('clip.wav', 'Done?').tier == 'fast'  # the threshold itself answers fast


def test_cascade_routes_text_and_images_to_the_big_model():
    fast, big = Fixed({'Yes': 0.99, 'No': 0.01}), Fixed({'Yes': 0.3, 'No': 0.7})
    c = Cascade(fast, big, threshold=0.9)
    for inp, kind in [('The order never arrived.', None), ('photo.jpg', None), ('clip.wav', 'text')]:
        r = c(inp, 'Upset?', input_type=kind)
        assert r.tier == 'big' and r.routed and r.fast_confidence is None
    assert fast.calls == 0
    assert c(np.zeros(SAMPLE_RATE, np.float32), 'Done?').tier == 'fast'  # arrays are audio
    everything = Cascade(fast, big, threshold=0.9, fast_inputs=None)
    assert everything('The order never arrived.', 'Upset?').tier == 'fast' and fast.calls == 2


def test_confidence_rule_table():
    rows = [
        ({'a': 0.5, 'b': 0.5}, False, 0.5),
        ({'a': 0.2, 'b': 0.3, 'c': 0.5}, False, 0.5),
        ({'a': 2.0, 'b': 2.0}, False, 0.5),  # unnormalised input is normalised
        ({'a': 0.99, 'b': 0.02, 'c': 0.5}, True, 0.5),  # select-all: the least certain option decides
        ({'a': 0.99, 'b': 0.01}, True, 0.99),
        ({'a': 0.0, 'b': 1.0}, True, 1.0),
    ]
    for probs, multi, want in rows:
        assert confidence(probs, multi) == pytest.approx(want), (probs, multi)


def test_audio_chunks_align_to_the_end():
    sr = SAMPLE_RATE
    assert [len(c) for c in audio_chunks(np.zeros(20 * sr))] == [20 * sr]
    c = audio_chunks(np.arange(70 * sr, dtype=np.float32))
    assert [len(x) / sr for x in c] == [10.0, 30.0, 30.0] and c[-1][-1] == 70 * sr - 1
    c = audio_chunks(np.arange(62 * sr, dtype=np.float32))  # 2 s head -> replaced by the first full window
    assert [len(x) / sr for x in c] == [30.0, 30.0, 30.0] and c[0][0] == 0 and c[-1][-1] == 62 * sr - 1


def test_combine_windows():
    ps = [{'speech': 0.9, 'music': 0.1}, {'speech': 0.2, 'music': 0.7}]
    assert combine_windows(ps, 'mean') == pytest.approx({'speech': 0.55, 'music': 0.4})
    assert combine_windows(ps, 'any') == {'speech': 0.9, 'music': 0.7}


def test_load_audio_untrimmed():
    assert len(load_audio(np.zeros(70 * SAMPLE_RATE, np.float32), max_seconds=None)) == 70 * SAMPLE_RATE


def test_llama_cpp_prompt_matches_the_package():
    # llamacpp/llama_decide.py keeps a numpy-only copy of the prompt code; the GGUF path is only exact if it renders identically
    import importlib.util
    import os

    from smaug_decide.prompt import char_budget

    path = os.path.join(os.path.dirname(__file__), '..', 'llamacpp', 'llama_decide.py')
    spec = importlib.util.spec_from_file_location('llama_decide', path)
    ld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ld)
    long = 'Ünïcödé 🎧 text. ' * 400
    for args in [
        ('Which genre?', ['rock', 'jazz'], ['loud', None], False, '<|audio|>', 'Be strict.', None, None, 1000),
        ('Which apply?', ['a', 'b', 'c'], [None, None, None], True, None, long, long, long, 700),
        ('Is it a photo?', ['Yes', 'No'], [None, None], False, '<|image|>', None, 'ctx', None, 3000),
    ]:
        assert ld.render(*args) == render_letters(*args)
    for media_tokens, n in [(0, 2), (282, 5), (30 * 25 + 2, 20)]:
        assert int(max(400, (ld.MAX_TOKENS - media_tokens - 80 - 10 * n) * 3.5)) == char_budget('letters', media_tokens, n)
    assert ld.hf_image_size(500, 500) == (768, 768)  # upscaled to the 16 x 16 grid of 48-pixel cells (256 image tokens)
