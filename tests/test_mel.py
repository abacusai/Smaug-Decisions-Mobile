"""The numpy log-mel front end (the spec the Swift and Kotlin ports follow) against the Transformers feature extractor.

The extractor is built from its default settings, which are the ones both published checkpoints ship in processor_config.json,
so no download is needed.
"""

import numpy as np
import pytest

from smaug_decide.mel import MEL_BINS, frame_count, htk_filters, log_mel


def test_frame_count():
    assert [frame_count(n) for n in (0, 160, 320, 321, 481, 16000, 160000)] == [0, 0, 1, 2, 3, 99, 999]
    assert log_mel(np.zeros(100, np.float32)).shape == (0, MEL_BINS)


def test_filters_are_htk_triangles():
    f = htk_filters()
    assert f.shape == (257, 128) and f.min() >= 0.0 and f.max() <= 1.0
    assert (f[:, 1:].sum(0) > 0).all()  # the lowest filter is narrower than one FFT bin and stays empty, as in Transformers
    assert np.all(np.diff(f.argmax(0)) >= 0)  # centres rise with the bin index


@pytest.mark.parametrize('n', [8000, 16000 * 3 + 123, 16000 * 10])
def test_log_mel_matches_transformers(n):
    fe_cls = pytest.importorskip('transformers').Gemma4AudioFeatureExtractor
    audio = (0.1 * np.random.default_rng(n).standard_normal(n)).astype(np.float32)
    ref = fe_cls()([audio], sampling_rate=16000, return_tensors='np')
    feats, mask = ref['input_features'][0], ref['input_features_mask'][0].astype(bool)
    ours = log_mel(audio)
    assert mask.sum() == len(ours) == frame_count(n)
    assert np.abs(feats[mask] - ours).max() < 1e-4
