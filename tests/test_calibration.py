"""Threshold calibration: scoring a cascade at each tau from saved answers, and the suggestion rule."""

import pytest

from smaug_decide.calibration import is_correct, score, suggest, sweep


def rec(fast, big, label, multi=False):
    return {'fast': fast, 'big': big, 'label': label, 'multi': multi, 'fast_ms': 10.0, 'big_ms': 200.0}


RECORDS = [
    rec({'Yes': 0.95, 'No': 0.05}, {'Yes': 0.9, 'No': 0.1}, 'Yes'),  # fast sure and right
    rec({'Yes': 0.60, 'No': 0.40}, {'Yes': 0.1, 'No': 0.9}, 'No'),  # fast unsure and wrong, big right
    rec({'Yes': 0.70, 'No': 0.30}, {'Yes': 0.8, 'No': 0.2}, 'Yes'),  # fast unsure but right
    rec({'a': 0.97, 'b': 0.02}, {'a': 0.9, 'b': 0.6}, ['a'], multi=True),  # select-all: fast sure and right
]


def test_is_correct():
    assert is_correct({'Yes': 0.7, 'No': 0.3}, 'Yes') and not is_correct({'Yes': 0.3, 'No': 0.7}, 'Yes')
    assert is_correct({'a': 0.9, 'b': 0.6, 'c': 0.1}, ['a', 'b'], multi=True)
    assert not is_correct({'a': 0.9, 'b': 0.6, 'c': 0.1}, ['a'], multi=True)
    assert is_correct({'a': 0.9, 'b': 0.1}, 'a', multi=True)


def test_score_table():
    rows = [
        (0.0, 0.75, 0.0, 10.0),  # fast alone: misses the second record
        (0.65, 1.0, 0.25, 60.0),  # escalates only the 0.60 record
        (0.9, 1.0, 0.5, 110.0),  # escalates 0.60 and 0.70
        (0.99, 0.75, 1.0, 210.0),  # escalates everything: big misses the select-all record (b at 0.6)
        (float('inf'), 0.75, 1.0, 200.0),  # big alone: no fast pass
    ]
    for tau, acc, esc, ms in rows:
        r = score(RECORDS, tau)
        assert (r['accuracy'], r['escalation_rate']) == (pytest.approx(acc), pytest.approx(esc)), tau
        assert r['mean_ms'] == pytest.approx(ms), tau


def test_sweep_and_suggest():
    rows = sweep(RECORDS, taus=[0.65, 0.9, 0.99])
    assert [r['tau'] for r in rows] == [0.0, 0.65, 0.9, 0.99, float('inf')]
    assert suggest(rows)['tau'] == 0.65  # lowest escalation at or above Smaug-Decisions alone
    no_ms = [{k: v for k, v in r.items() if not k.endswith('_ms')} for r in RECORDS]
    assert 'mean_ms' not in score(no_ms, 0.9)
    with pytest.raises(ValueError):
        score([], 0.9)
