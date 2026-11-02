# Mac (Apple Silicon)

Three ways to run the models on a Mac:

| Path | Models | Memory | Notes |
|---|---|---|---|
| PyTorch on MPS (the Python package) | Smaug-Decisions, Smaug-Audio-Decisions-0.7B, the cascade | Smaug-Decisions about 16 GB (bfloat16), the audio model about 3 GB | Works as on Linux: `SmaugDecide.from_pretrained` picks MPS automatically. See [python.md](python.md) |
| MLX (`mlx-vlm`) | Smaug-Decisions | 4- or 8-bit weights, a fraction of bfloat16 | Quantised builds for Macs with less memory; this page |
| Core ML | Smaug-Audio-Decisions-0.7B | 670 MB of int8 units and tables | The int8 units the iPhone app runs: **121 ms per decision on an M3 Pro GPU** (measured). See [ios.md](ios.md) and [`scripts/convert/coreml/`](../scripts/convert/coreml) |

llama.cpp also builds on macOS ([llamacpp.md](llamacpp.md)); check it against the PyTorch model with `llamacpp/parity_llama.py`.

## MLX

Run Smaug-Decisions natively with [MLX](https://github.com/ml-explore/mlx) through `mlx-vlm`. The readout is unchanged: the same
prompt, the LM-head logits of the option letters at the last position, softcap, then softmax or `sigmoid(z - b)`.

### Requirements

An Apple Silicon Mac, Python 3.10+, and:

```bash
pip install -e ".[mlx]"        # from the repository root: mlx, mlx-vlm
```

### Steps

```bash
# 1. convert (downloads the checkpoint, quantises, copies readout.json)
scripts/convert/mlx/convert.sh build/Smaug-Decisions-mlx4 4
scripts/convert/mlx/convert.sh build/Smaug-Decisions-mlx8 8                    # optional reference build

# 2. reference probabilities from the PyTorch model, then compare (cases.jsonl: {"id", "input", "question", "options", ...})
python -m smaug_decide --model abacusai/Smaug-Decisions --cases cases.jsonl --out reference.jsonl
python scripts/convert/mlx/parity.py --mlx build/Smaug-Decisions-mlx4 --cases cases.jsonl --reference reference.jsonl

# 3. 4-bit builds only, if you use select-all questions: correct the select-all threshold against the 8-bit build
python scripts/convert/mlx/fix_multi_bias.py --ref build/Smaug-Decisions-mlx8 --quant build/Smaug-Decisions-mlx4 --audio-dir my_clips/
```

| Script | Purpose |
|---|---|
| [`convert.sh`](../scripts/convert/mlx/convert.sh) | `mlx_vlm.convert` with 4- or 8-bit weights (group size 64), plus `readout.json` |
| [`parity.py`](../scripts/convert/mlx/parity.py) | `MLXDecide`, the letter readout on an MLX build, and a comparison against PyTorch reference probabilities |
| [`fix_multi_bias.py`](../scripts/convert/mlx/fix_multi_bias.py) | Label-free correction of the select-all threshold `b` for a quantised build |

`MLXDecide` in `parity.py` is also the shortest path to using the MLX build from your own code:

```python
import sys; sys.path.insert(0, "scripts/convert/mlx")   # from the repository root
from parity import MLXDecide
model = MLXDecide('build/Smaug-Decisions-mlx4')
probs, n_tokens = model('call.wav', 'Has the speaker finished their turn?', ['Yes', 'No'])
```

### Notes

- **Softcap exactly once.** Some MLX model implementations apply the final logit softcap inside the model; `parity.py` detects
  this and skips its own softcap (override with `--cap`).
- **Why the threshold moves.** Quantisation shifts the letter logits by a roughly constant amount. Softmax does not care, but the
  select-all readout `sigmoid(z - b)` does, so a 4-bit build needs `b` corrected by that shift.
- `parity.py` checks audio and text inputs. Measure accuracy and latency on your own cases with it.
