# Print the summary lines of parity_llama.py outputs. usage: summarize.py par_*.json
import json
import sys


if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
    sys.exit('usage: summarize.py par_*.json')
for f in sys.argv[1:]:
    s = json.load(open(f))['summary']
    pk = ' '.join(f'{k}:{v["median_dp"]:.4f}/{v["max_dp"]:.4f}/{v["median_ms"]:.0f}ms' for k, v in s['per_kind'].items())
    print(
        f'{f:<28} n={s["n"]} median|dp|={s["median_dp"]:.4f} max|dp|={s["max_dp"]:.4f} argmax={s["argmax_agree"]:.3f} '
        f'shift={s["mean_z_shift"]:+.3f} | {pk}'
    )
