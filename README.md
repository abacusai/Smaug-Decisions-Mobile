# Smaug-Decisions Mobile

**State-of-the-art audio decisions in a 4B model that also sees and reads, running offline on phones.**

Smaug-Decisions is the only 4B decision model that hears audio, sees images and reads text. Ask a question with fixed options
("Has the speaker finished their turn?", "Which sounds are present?", "Which team should handle this message?") and get a
calibrated probability for every option from one forward pass; nothing is generated, so a decision costs a single prefill and
the probabilities work directly as thresholds.

- **Smaug-Decisions** (4B) scores **60.1** on Earshot, our 6,671-item audio decision benchmark, ahead of Gemini 4 Argon (59.4) and
  Gemini 3.8 Flash (50.5), and is better calibrated (ECE **0.069** vs 0.123 and 0.204).
- **Smaug-Audio-Decisions-0.7B** (744M) answers audio in about 10 ms on an H100 and beats Gemini 3.8 Flash (53.0); its int8 phone weights
  score the same as float.
- **Smaug-Decisions-LiteRT** (3.45 GB) runs Smaug-Decisions on the phone GPU and keeps 59.4 audio skill: text in about 0.6 s, audio in about
  1.6 s on an iPhone 14 Pro Max.

Runs on NVIDIA GPUs, Macs, CPUs, iPhone and Android.

## Models

| Model | Size | Inputs | Audio: Earshot (ECE) | Text: Decision Index | Images: Decision Index | Hugging Face |
|---|---|---|---|---|---|---|
| **Smaug-Decisions** | 4B effective (Gemma 4 E4B fine-tune; 8.6B stored, 3.5B of them embedding lookup tables) | audio up to 30 s, one image, or text; 2-20 options | **60.1** (0.069) | **56.6** | **57.9** | [abacusai/Smaug-Decisions](https://huggingface.co/abacusai/Smaug-Decisions) |
| **Smaug-Audio-Decisions-0.7B** | 744M (EmbeddingGemma 2 fine-tune) | audio | 53.0 (0.086) | &mdash; (audio model) | &mdash; | [abacusai/Smaug-Audio-Decisions-0.7B](https://huggingface.co/abacusai/Smaug-Audio-Decisions-0.7B) |
| **Smaug-Decisions-LiteRT** | Smaug-Decisions for phones (LiteRT-LM, phone GPU): 3.45 GB, or 2.97-3.28 GB without some encoders | as Smaug-Decisions, or a subset (see below); 2,048-token context | 59.4 (0.074) | 53.9 | 55.1\* | [abacusai/Smaug-Decisions-LiteRT](https://huggingface.co/abacusai/Smaug-Decisions-LiteRT) |

Earshot scores are skill, (accuracy - chance) / (1 - chance) x 100, so 0 is guessing and 100 is perfect; ECE is the
calibration error of the top probability. The Decision Index scores are our runs of the public Jev Decision Index kit.
\*10 of the 11 public vision sets; Smaug-Decisions scores 57.2 on the same 10.

**Smaug-Decisions-LiteRT files.** One set of quantized weights, packaged with different encoders. `manifest.json` in the
repository lists each file's size, sha256, inputs and URL; the apps read it to pick a file.

| File | Bytes | Inputs | For |
|---|---|---|---|
| `smaug-decisions-full-vis560.litertlm` | 3,449,280,464 | text, audio, images at 560 image tokens | phones with 8 GB of RAM or more |
| `smaug-decisions-full-vis280.litertlm` | 3,449,280,464 | text, audio, images at 280 image tokens (an image needs about 1 GB less memory) | 6 GB phones |
| `smaug-decisions-noaudio-vis560.litertlm` | 3,147,634,640 | text, images (560) | text and photos, with audio left to Smaug-Audio-Decisions-0.7B |
| `smaug-decisions-noaudio-vis280.litertlm` | 3,147,634,640 | text, images (280) | the same on 6 GB phones |
| `smaug-decisions-novision.litertlm` | 3,275,757,520 | text, audio | audio and text, no photos |
| `smaug-decisions-textonly.litertlm` | 2,974,111,696 | text | text only |

The component files give bit-identical letter logits to the full file on the inputs they support. Smaug-Decisions-LiteRT's scores come
from its quantized weights reproduced in PyTorch, on the same items and harness as full precision.

## The cascade

Smaug-Audio-Decisions-0.7B answers audio questions when it is confident and hands the rest to Smaug-Decisions. Text and images go straight to
Smaug-Decisions, because the audio model is near chance on them.

```
             ┌── audio ──> Smaug-Audio-Decisions-0.7B ── confidence >= tau ──> answer   (tier "fast")
             │                         │
question ────┤                         └── confidence < tau ──> Smaug-Decisions ──> answer   (tier "big", escalated)
             │
             └── text / image ────────────────────────────────> Smaug-Decisions ──> answer   (tier "big", routed)
```

Confidence is the top probability for one-answer questions and, for select-all questions, the least certain option's
max(p, 1 - p). At tau = 0.9 the cascade matches Smaug-Decisions alone on Earshot (60.1) at **130 ms per audio decision instead of
210 ms** on an H100, escalating 60% of the questions. Pick tau on your own data: [choosing a threshold](docs/choosing-a-threshold.md).

## Which device runs what

| Device | Smaug-Audio-Decisions-0.7B | Smaug-Decisions | Cascade |
|---|---|---|---|
| iPhone, 6 GB (iPhone 14 Pro, 15) | Core ML, **0.27-0.37 s** per decision | LiteRT-LM GPU, `smaug-decisions-full-vis280`: text **0.6 s**, audio **1.6 s**, image 7.4 s; peak 1.35 GB with both models | yes |
| iPhone, 8 GB and more (iPhone 15 Pro and later) | Core ML | LiteRT-LM GPU, `smaug-decisions-full-vis560` | yes |
| iPhone, under 6 GB | Core ML | does not fit | 0.7B audio model only; unsure answers are flagged |
| Android, 8 GB and more | ONNX Runtime | LiteRT-LM, `smaug-decisions-full-vis560` | yes |
| Android, 6 GB | ONNX Runtime | LiteRT-LM, `smaug-decisions-full-vis280`, when enough memory is free at that moment | gated |
| Android, 4 GB | ONNX Runtime | does not fit | 0.7B audio model only |
| Mac (Apple Silicon) | PyTorch (MPS), Core ML (121 ms on an M3 Pro GPU) | PyTorch (MPS, about 16 GB), MLX 4/8-bit, llama.cpp | yes (Python) |
| Linux, CPU | PyTorch | PyTorch, llama.cpp (Q4_K_M: 340 ms text, 790 ms audio at 16 threads), LiteRT-LM | yes (Python) |
| NVIDIA GPU server | PyTorch, 9.7 ms on an H100 | PyTorch, 210 ms per audio decision on an H100; llama.cpp CUDA | yes: 130 ms at tau 0.9 |

Phone numbers are wall-clock per decision with the model loaded and its weight cache built. Smaug-Decisions' first load writes the
cache (13-21 s on the iPhone 14 Pro Max; later loads 0.48 s), and the first launch compiles the audio model for the Neural Engine once.
Measured on an iPhone 14 Pro Max, an H100, an M3 Pro and Linux x86; full tables and memory: [docs/devices.md](docs/devices.md).

## Quick start

```bash
pip install -e .                                   # Python 3.10+; installs torch and transformers>=5.20
```

```python
from smaug_decide import Cascade, SmaugDecide

model = SmaugDecide.from_pretrained("abacusai/Smaug-Decisions")      # CUDA, Apple Silicon or CPU, picked automatically
model("call.wav", "Has the speaker finished their turn?")                                  # {'Yes': 0.93, 'No': 0.07}
model("room.wav", "Which sounds are present?", ["speech", "music", "alarm"], multi=True)    # one probability per option
model("receipt.jpg", "Does the total exceed $50?")
model("The order never arrived.", "How upset is the customer?", ["calm", "annoyed", "angry"])

cascade = Cascade("abacusai/Smaug-Audio-Decisions-0.7B", "abacusai/Smaug-Decisions", threshold=0.9)
r = cascade("call.wav", "Has the speaker finished their turn?")
r.probabilities, r.tier                            # 'fast' or 'big'; Smaug-Decisions loads on the first escalation
```

| Platform | Start here |
|---|---|
| Python (CUDA, Apple Silicon, CPU) | [docs/python.md](docs/python.md), [examples/](examples) |
| HTTP server (`/v1/decisions`, single model or cascade) | [docs/server.md](docs/server.md) |
| iPhone (Core ML + LiteRT-LM cascade app) | [docs/ios.md](docs/ios.md) |
| Android (ONNX Runtime + LiteRT-LM cascade app) | [docs/android.md](docs/android.md) |
| Mac (MLX, MPS, Core ML) | [docs/mac.md](docs/mac.md) |
| llama.cpp (GGUF; CPU and CUDA) | [docs/llamacpp.md](docs/llamacpp.md) |
| Measured devices | [docs/devices.md](docs/devices.md) |
| Picking the cascade threshold | [docs/choosing-a-threshold.md](docs/choosing-a-threshold.md) |

Helper scripts, each with `--help` and `--dry-run`:

```bash
python scripts/download.py --model decisions|audio|litert [--variant ...]     # fetch weights (--list shows the variants)
python scripts/decide.py call.wav "Has the speaker finished their turn?" --model cascade
python scripts/benchmark.py --model audio                                    # latency and memory on this machine
python scripts/calibrate.py my_examples.jsonl --out answers.jsonl            # sweep tau on your labeled examples
```

## How it works

The input, optional rules and context, the question and the options are rendered into one prompt
([`smaug_decide/prompt.py`](smaug_decide/prompt.py)), and the model reads it once.

- **Smaug-Decisions (letter readout).** The options are lettered A..T. The decision is the LM-head logits of those letters at the
  answer position, through a tanh softcap of 30; softmax for one-answer questions, `sigmoid(z - b)` per option for select-all
  (`b` is in the checkpoint's `readout.json`).
- **Smaug-Audio-Decisions-0.7B (marker readout).** Every option line starts with a marker token, and a linear head scores the hidden state
  at each marker: softmax over the options, or a sigmoid per option.

Any runtime can reproduce a decision from the prompt code and the readout math in
[`smaug_decide/readout.py`](smaug_decide/readout.py); the phone apps, MLX and llama.cpp paths do exactly that.

## Repository layout

| Path | Contents |
|---|---|
| [`smaug_decide/`](smaug_decide) | Python package: `SmaugDecide`, `Cascade`, prompt rendering, readout math, log-mel front end, threshold calibration, a command line |
| [`server/`](server) | HTTP server with `/v1/decisions`, single model or cascade, and two demo pages |
| [`scripts/`](scripts) | Download, decide, benchmark and calibrate; [`scripts/convert/`](scripts/convert) has the Core ML and MLX converters |
| [`llamacpp/`](llamacpp) | GGUF conversion, a small C core over llama.cpp, `LlamaDecide` in Python and a parity check |
| [`examples/`](examples) | Short runnable scripts: predicate, choice, select-all, image, text with rules, cascade, turn gate |
| [`apps/`](apps) | The iOS and Android cascade apps |
| [`docs/`](docs) | Per-platform guides, device measurements, threshold selection |
| [`tests/`](tests) | Fast tests that need no GPU and no weights: `pip install -e ".[dev]" && pytest` |

## Scope

- **Decisions, not generation:** 2 to 20 fixed options per question (the audio model takes 2 or more).
- **Smaug-Audio-Decisions-0.7B is an audio model;** the cascade sends text and images to Smaug-Decisions.
- **One input per decision:** audio up to 30 s (`long_audio='mean'` or `'any'` covers longer recordings), one image, or text.
  On the phones the audio model hears the last 10 s.
- **Phone prompts up to about 2,000 tokens** (2,048-token context); longer input keeps the head and the tail.
- **Smaug-Decisions on a phone needs 6 GB of RAM;** smaller phones run the audio model.
- **Thresholds:** the probabilities are calibrated; set action thresholds on a few hundred of your own examples.

## License

Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Smaug-Decisions is fine-tuned from Gemma 4 E4B-it and Smaug-Audio-Decisions-0.7B from
EmbeddingGemma 2, both by Google; NOTICE carries their terms.

## Citation

```bibtex
@misc{abacusai2026smaugdecisions,
  title  = {Smaug-Decisions: calibrated typed decisions over audio, images and text},
  author = {Abacus.AI},
  year   = {2026},
  url    = {https://github.com/abacusai/Smaug-Decisions-Mobile}
}
```
