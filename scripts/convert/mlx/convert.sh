#!/bin/bash
# Convert Smaug-Decisions to MLX for Apple Silicon and carry over the readout constants.
# usage: scripts/convert/mlx/convert.sh <out dir> [bits: 4|8] [hf repo or local dir]
#   scripts/convert/mlx/convert.sh build/Smaug-Decisions-mlx4 4
#   scripts/convert/mlx/convert.sh build/Smaug-Decisions-mlx8 8
# Needs: pip install -e ".[mlx]" (mlx, mlx-vlm) on an Apple Silicon Mac.
set -euo pipefail
OUT=$1; BITS=${2:-4}; SRC=${3:-abacusai/Smaug-Decisions}
python -c "import mlx.core as m, mlx_vlm; print('mlx', m.__version__, 'mlx-vlm', mlx_vlm.__version__)"
if [ -d "$SRC" ]; then HF=$SRC; else
  HF=$(python -c "from huggingface_hub import snapshot_download as d; print(d('$SRC', allow_patterns=['*.json', '*.jinja', 'model*.safetensors', 'head.pt']))")
fi
[ -d "$OUT" ] || python -m mlx_vlm.convert --hf-path "$HF" --mlx-path "$OUT" --quantize --q-bits "$BITS" --q-group-size 64
cp "$HF/readout.json" "$OUT/"
du -sh "$OUT"
echo "next: python scripts/convert/mlx/parity.py --mlx $OUT --cases cases.jsonl --reference reference.jsonl"
