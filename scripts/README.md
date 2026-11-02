# Scripts

Run from the repository root. Each takes `--help`; all but the converters also take `--dry-run`, which loads no model.

| Script | What it does |
|---|---|
| [`download.py`](download.py) | Download weights with the Hugging Face CLI: `--model decisions`, `audio` or `litert`, `--variant` (`--list` shows them) |
| [`decide.py`](decide.py) | One decision on a file or a piece of text, with Smaug-Decisions, Smaug-Audio-Decisions-0.7B or the cascade |
| [`benchmark.py`](benchmark.py) | Latency and peak memory on this machine, with synthetic audio, text and image inputs |
| [`calibrate.py`](calibrate.py) | Sweep the cascade threshold on your labeled examples and suggest a value ([guide](../docs/choosing-a-threshold.md)) |
| [`convert/coreml/`](convert/coreml) | Smaug-Audio-Decisions-0.7B to Core ML units for iOS and macOS, with a parity check |
| [`convert/mlx/`](convert/mlx) | Smaug-Decisions to MLX for Apple Silicon, with a parity check ([guide](../docs/mac.md)) |

GGUF conversion for llama.cpp lives in [`llamacpp/`](../llamacpp) ([guide](../docs/llamacpp.md)).
