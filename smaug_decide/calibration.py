"""Choosing the cascade threshold on your own labeled examples: sweep tau and read accuracy, escalation rate and latency.

Each record holds both tiers' answers to one labeled audio question, so every threshold can be scored without running the models
again:

    {"fast": {option: p}, "big": {option: p}, "label": "Yes" or ["speech", "music"], "multi": false,
     "fast_ms": 12.0, "big_ms": 210.0}

A one-answer question is correct when the top option is the label. A select-all question is correct when the options above 0.5
are exactly the labeled set. fast_ms / big_ms are optional; without them the latency columns are omitted.
"""

from .cascade import confidence


DEFAULT_TAUS = [round(0.5 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99


def is_correct(probs, label, multi=False):
    if multi:
        want = {label} if isinstance(label, str) else set(label)
        return {o for o, p in probs.items() if p > 0.5} == want
    return max(probs, key=probs.get) == label


def score(records, tau):
    """Cascade at threshold tau over the records: {tau, accuracy, escalation_rate, mean_ms?}. tau=0 is the fast tier alone and
    tau > 1 is the big model alone."""
    if not records:
        raise ValueError('no records to score')
    correct = escalated = 0
    ms, have_ms = 0.0, all('fast_ms' in r and 'big_ms' in r for r in records)
    for r in records:
        multi = r.get('multi', False)
        esc = confidence(r['fast'], multi) < tau
        escalated += esc
        correct += is_correct(r['big'] if esc else r['fast'], r['label'], multi)
        if have_ms:
            ms += r['fast_ms'] * (tau <= 1.0) + (r['big_ms'] if esc else 0.0)
    out = {'tau': tau, 'accuracy': correct / len(records), 'escalation_rate': escalated / len(records)}
    if have_ms:
        out['mean_ms'] = ms / len(records)
    return out


def sweep(records, taus=DEFAULT_TAUS):
    """Rows for the fast tier alone, every tau, and the big model alone (tau = inf)."""
    return [score(records, 0.0)] + [score(records, t) for t in taus] + [score(records, float('inf'))]


def suggest(rows, tolerance=0.005):
    """The lowest-escalation threshold whose accuracy is within `tolerance` of the big model alone (the last row)."""
    target = rows[-1]['accuracy'] - tolerance
    ok = [r for r in rows[1:-1] if r['accuracy'] >= target]
    if not ok:
        return rows[-1]
    return min(ok, key=lambda r: (r['escalation_rate'], -r['accuracy']))
