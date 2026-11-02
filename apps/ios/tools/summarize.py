"""Summarize an autorun directory (Documents/autorun_*.json pulled from the Simulator) as markdown tables.

    python tools/summarize.py <dir with autorun_*.json>
"""
import json
import os
import sys


def fmt(v):
    return ' / '.join(f'{x:.4f}' for x in v) if v else '—'


def top(opts, p):
    return opts[max(range(len(p)), key=p.__getitem__)] if p else '—'


def main(d):
    s = json.load(open(os.path.join(d, 'autorun_summary.json')))
    print('platform:', s.get('platform'), '| tau', s['tau'], '| policy', json.dumps(s.get('policy')), '| big:', s['big_state'])
    print('tokenizer check:', s.get('tokenizer_check'))
    print('memory end:', {k: round(v) for k, v in s['memory_end'].items()}, '| after release:',
          {k: round(v) for k, v in s.get('memory_after_release', {}).items()})
    print('\n| case | route | fast conf | fast p (sim) | PyTorch | Mac Core ML CPU | max abs dp vs PyTorch | vs Mac CPU | tokens sim/ref | fast ms (audio+trunk) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for c in s['cases']:
        if 'fast_p' not in c:
            continue
        fp, rt, rc = c['fast_p'], c.get('fast_ref_torch') or [], c.get('fast_ref_coreml_cpu') or []
        short = lambda v: fmt(sorted(v, reverse=True)[:2]) if len(v) > 2 else fmt(v)  # noqa: E731
        d1 = max(abs(a - b) for a, b in zip(fp, rt)) if rt else float('nan')
        d2 = max(abs(a - b) for a, b in zip(fp, rc)) if rc else float('nan')
        m = c['fast_ms']
        print(f"| {c['id']} | {c['route']} | {c['fast_confidence']:.4f} | {top(c['options'], fp)} {max(fp):.4f} | {top(c['options'], rt) + ' ' + format(max(rt), '.4f') if rt else '—'} | "
              f"{format(max(rc), '.4f') if rc else '—'} | {d1:.4f} | {d2:.4f} | {c['fast_tokens']}/{c.get('fast_ref_tokens', '—')} | {m['total']:.0f} ({m['audio']:.0f}+{m['trunk']:.0f}) |")
    print('\n| case | route | Smaug-Decisions p (sim) | PyTorch bf16 ref | max abs dp | tokens sim/ref | load s | prefill s | readout s | footprint MB (peak) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for c in s['cases']:
        mem = c['memory']
        if 'big_p' not in c:
            print(f"| {c['id']} | {c['route']} | — | — | — | — | — | — | — | {mem['footprint_mb']:.0f} ({mem['footprint_peak_mb']:.0f}) |")
            continue
        p, r, m = c['big_p'], c['big_ref'], c['big_ms']
        d = f"{max(abs(a - b) for a, b in zip(p, r)):.4f}" if r else 'no ref'
        pp = fmt(p) if len(p) <= 4 else f"{top(c['options'], p)} {max(p):.4f}"
        rr = (fmt(r) if len(r) <= 4 else f"{top(c['options'], r)} {max(r):.4f}") if r else '—'
        print(f"| {c['id']} | {c['route']} | {pp} | {rr} | {d} | {c['big_tokens']}/{c.get('big_ref_tokens', '—')} | {m['load'] / 1000:.1f} | "
              f"{m['prefill'] / 1000:.2f} | {m['readout'] / 1000:.2f} | {mem['footprint_mb']:.0f} ({mem['footprint_peak_mb']:.0f}) |")


if __name__ == '__main__':
    main(sys.argv[1])
