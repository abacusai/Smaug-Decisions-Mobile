# llama.cpp (Smaug-Decisions as GGUF)

Run Smaug-Decisions with [llama.cpp](https://github.com/ggml-org/llama.cpp) and its multimodal library `libmtmd`: **text, image and
audio decisions, on CPU and CUDA**, with nothing generated. The decision is the same as in the PyTorch package: render the prompt
`smaug_decide.render_letters` produces, run one prefill, read the logits of the option letters `A`..`T` at the last position,
then softmax (one answer) or `sigmoid(z - b)` (select-all, `b` from `readout.json`).

Tested at llama.cpp master `f7c01c5` (tag `b11979`). llama.cpp itself needs no patches; the two fixes below live in
[`llamacpp/`](../llamacpp). Convert GGUF builds from the published weights as shown below.

| | Status |
|---|---|
| Text | ✅ CPU and CUDA |
| Image | ✅ CPU and CUDA (with the image pre-resize, see below) |
| Audio | ✅ CPU and CUDA |
| Quantisations checked | bf16, Q8_0, Q5_K_M, Q4_K_M, Q4_K_M with an importance matrix; mmproj f16 and q8_0 |
| Parity with PyTorch | every build checked on 24 inputs; the rendered prompt and the prompt token count match exactly at every quant |

## Two fixes, and why

1. **Drop the tied `lm_head.weight` before conversion.** The Smaug-Decisions checkpoint stores an explicit `lm_head.weight` although
   `tie_word_embeddings` is true (it is bit-identical to the token embeddings). llama.cpp's Gemma 4 converters do not skip it: the
   mmproj conversion fails with `Can not map tensor 'lm_head.weight'`, and the text GGUF gets a duplicate output matrix (+1.3 GB
   at bf16). [`convert.py`](../llamacpp/convert.py) drops the tensor and then runs the stock `convert_hf_to_gguf.py` unchanged.
2. **Resize images to the Transformers grid before encoding.** llama.cpp's Gemma 4 vision preprocessing only rounds an image to
   the nearest patch multiple and never upscales, so a 500x500 image gets about 100 image tokens where the Transformers processor
   gives 256 (100-160 vs 256-270 on our test images, and probability differences up to 0.12 at bf16, 0.31 at Q4_K_M).
   `llama_decide.py` (and the Android app's llama.cpp backend) first resize the image the way the Transformers processor does (scale up or
   down to fill 280 x 9 patches, sides floored to multiples of 48), which gives an exact token-count match.

## Requirements

A llama.cpp checkout built with shared libraries, a C compiler, Python 3.10+ with `numpy`, `pillow` and `soundfile` for
`llama_decide.py`, and the converter's own requirements for the conversion step. The commands below run from the root of this
repository and keep the llama.cpp checkout outside it, in `$LLAMA`.

```bash
LLAMA=$HOME/src/llama.cpp
git clone https://github.com/ggml-org/llama.cpp $LLAMA && git -C $LLAMA checkout f7c01c5
cmake -S $LLAMA -B $LLAMA/build -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release   # add -DGGML_CUDA=ON for NVIDIA
cmake --build $LLAMA/build -j
pip install -r $LLAMA/requirements/requirements-convert_hf_to_gguf.txt    # converter: torch, transformers, gguf-py, ...
pip install numpy pillow soundfile
```

## Convert the published weights

```bash
# 1. the Transformers checkpoint, plus readout.json (it holds the select-all threshold b)
hf download abacusai/Smaug-Decisions --include "*.json" "*.jinja" "model*.safetensors" --local-dir Smaug-Decisions

# 2. the text GGUF at bf16, and the mmproj (vision + audio encoders)
python llamacpp/convert.py $LLAMA Smaug-Decisions --outtype bf16 --outfile smaug-decisions-bf16.gguf
python llamacpp/convert.py $LLAMA Smaug-Decisions --mmproj --outtype f16 --outfile mmproj-smaug-decisions-f16.gguf
python llamacpp/convert.py $LLAMA Smaug-Decisions --mmproj --outtype q8_0 --outfile mmproj-smaug-decisions-q8_0.gguf   # optional, smaller

# 3. quantise
Q=$LLAMA/build/bin
$Q/llama-quantize smaug-decisions-bf16.gguf smaug-decisions-Q8_0.gguf Q8_0
$Q/llama-quantize smaug-decisions-bf16.gguf smaug-decisions-Q5_K_M.gguf Q5_K_M

# 4. Q4_K_M with an importance matrix (the recommended 4-bit build)
python llamacpp/make_calib.py calib.txt 300          # 300 synthetic decision prompts, about 30k tokens
$Q/llama-imatrix -m smaug-decisions-Q8_0.gguf -f calib.txt -o imatrix-smaug-decisions.gguf -c 512
$Q/llama-quantize --imatrix imatrix-smaug-decisions.gguf smaug-decisions-bf16.gguf smaug-decisions-Q4_K_M-imat.gguf Q4_K_M
```

`llamacpp/convert.py` takes the llama.cpp checkout first; every argument after it goes to `convert_hf_to_gguf.py`.

**The importance matrix** is a quantisation calibration step: llama.cpp measures which weights matter most on representative text
and keeps those more precise in the 4-bit build. [`make_calib.py`](../llamacpp/make_calib.py) writes that text: decision prompts in the
model's prompt format, rendered from random synthetic templates (made-up messages, questions and options, each followed by an
answer letter). It contains no evaluation data. On Smaug-Decisions it roughly halves the worst-case probability error of Q4_K_M (tables
below).

## Build the C core

[`decide_core.c`](../llamacpp/decide_core.c) is a small C API over llama.cpp + libmtmd for decision models: load a GGUF (+ mmproj), evaluate
one prompt with an optional image or 16 kHz audio clip, and return the last-position logits of a few token ids. The Python path
calls it through ctypes and the Android app's llama.cpp backend through JNI, so both run the same code.

```bash
bash llamacpp/build_core.sh $LLAMA build llamacpp/libdecide.so      # <checkout> <its build dir> [output]
export LLAMA_DECIDE_LIB=$PWD/llamacpp/libdecide.so                    # default: libdecide.so next to llama_decide.py
```

## Use it

```bash
python llamacpp/llama_decide.py --gguf smaug-decisions-Q4_K_M-imat.gguf --mmproj mmproj-smaug-decisions-f16.gguf \
    --readout Smaug-Decisions/readout.json clip.wav "Is anyone speaking?"
python llamacpp/llama_decide.py --gguf smaug-decisions-Q4_K_M-imat.gguf --readout Smaug-Decisions/readout.json --multi \
    "Meeting moved to 3pm Thursday." "Which fields are present?" date time location
# --ngl -1 puts every layer (and the encoders) on the GPU of a CUDA build; --threads N sets CPU threads
```

```python
import sys; sys.path.insert(0, "llamacpp")        # from the repository root
from llama_decide import LlamaDecide

m = LlamaDecide("smaug-decisions-Q4_K_M-imat.gguf", mmproj="mmproj-smaug-decisions-f16.gguf", readout="Smaug-Decisions/readout.json")
m("call.wav", "Has the speaker finished their turn?")                                 # {'Yes': ..., 'No': ...}
m("photo.jpg", "Which apply?", ["a person", "a dog", "a car"], multi=True)
m("The order never arrived.", "How upset is the customer?", ["calm", "annoyed", "angry"])
m.last_tokens, m.last_ms                                                              # prompt positions, encoders + prefill time
```

`LlamaDecide` takes the same inputs and arguments as `SmaugDecide` (an audio path or 16 kHz array, an image path or PIL image, or
text; `descriptions`, `multi`, `rules`, `context`) and needs only numpy and Pillow, no torch. The mmproj is needed for images and
audio only. `DEC_NO_REPACK=1` lowers CPU memory (see below); `DEC_VERBOSE=1` shows llama.cpp's logs.

## Parity with the PyTorch model

24 inputs written for this check (11 text, 6 images, 7 audio clips): synthetic questions, drawn scenes and three public-domain
pictures, generated speech, tones, noise and silence ([`make_inputs.py`](../llamacpp/make_inputs.py)). The reference is the PyTorch package at
bf16 ([`ref_hf.py`](../llamacpp/ref_hf.py)). |dp| is the largest absolute probability difference over a case's options.

**CUDA (all layers and both encoders on the GPU) vs PyTorch bf16 on the same GPU,** mmproj f16 unless noted:

| GGUF | median \|dp\| | max \|dp\| | text median / max | image median / max | audio median / max | decisions changed |
|---|---|---|---|---|---|---|
| bf16 | 0.0005 | 0.039 | 0.0003 / 0.005 | 0.0003 / 0.015 | 0.0007 / 0.039 | 0 |
| Q8_0 | 0.0005 | 0.052 | 0.0004 / 0.014 | 0.0010 / 0.037 | 0.0014 / 0.052 | 0 |
| Q5_K_M | 0.0039 | 0.040 | 0.012 / 0.040 | 0.0016 / 0.036 | 0.0045 / 0.031 | 0 (one select-all argmax swap between two options at 0.94 / 0.95; thresholded answers agree) |
| **Q4_K_M + imatrix** | **0.0037** | **0.071** | 0.005 / 0.052 | 0.010 / 0.064 | 0.002 / 0.071 | **0** |
| Q4_K_M | 0.0099 | 0.159 | 0.012 / 0.082 | 0.018 / 0.124 | 0.005 / 0.159 | the same select-all near-tie |
| Q4_K_M, mmproj q8_0 | 0.0104 | 0.159 | | 0.012 / 0.108 | 0.009 / 0.159 | 2 select-all argmax swaps (near-ties) |

**CPU (16 threads) vs PyTorch bf16 on CPU:**

| GGUF | median \|dp\| | max \|dp\| |
|---|---|---|
| bf16 | 0.0006 | 0.039 |
| Q8_0 | 0.0010 | 0.075 |
| Q5_K_M | 0.0026 | 0.059 |
| **Q4_K_M + imatrix** | **0.0023** | **0.075** |
| Q4_K_M + imatrix, mmproj q8_0 | 0.0029 | 0.067 |
| Q4_K_M | 0.0089 | 0.143 |

Without the image pre-resize, the image cases reached max |dp| 0.121 at bf16 and 0.306 at Q4_K_M.

**Noise floor.** PyTorch bf16 on CPU and PyTorch bf16 on GPU already differ by median 0.0002, max 0.038, so the bf16 and Q8_0
GGUFs sit at the reference's own noise level. The largest residuals are select-all options near 0.5, where the sigmoid is
steepest.

**Recommendation: Q4_K_M with the importance matrix** for 4-bit (5.3 GB), Q8_0 when memory allows. Plain Q4_K_M is noticeably
worse. We also tried the select-all threshold correction used for the [MLX 4-bit build](mac.md) (shift `b` by the mean
letter-logit shift against bf16, measured on synthetic select-all prompts). Here the shift is not constant (its standard deviation
is larger than its mean), and correcting `b` made max |dp| slightly worse (Q4_K_M 0.143 to 0.176, Q4_K_M + imatrix 0.075 to
0.079). Use the imatrix build with `b` from `readout.json` unchanged.

## Sizes, latency and memory

| File | Size |
|---|---|
| smaug-decisions-bf16.gguf (16.3 GB if the duplicate lm_head were kept) | 14.9 GB |
| smaug-decisions-Q8_0.gguf | 7.97 GB |
| smaug-decisions-Q5_K_M.gguf | 5.72 GB |
| smaug-decisions-Q4_K_M(-imat).gguf (per-layer embedding table 2.2 GB and token embeddings 0.5 GB, both at q6_K) | 5.30 GB |
| mmproj-smaug-decisions-f16.gguf (vision + audio encoders) | 990 MB |
| mmproj-smaug-decisions-q8_0.gguf | 560 MB |

Median time per decision (encoders + prefill, in process, model loaded). Prompts are about 80-110 tokens for text, 165-190 for
audio (3-5 s clips) and 330-355 for images.

| Setup | Text | Audio | Image |
|---|---|---|---|
| PyTorch bf16 (`smaug_decide`), B200 | 75 ms | 151 ms | 128 ms |
| llama.cpp CUDA bf16, B200 | 41 ms | 122 ms | 143 ms |
| llama.cpp CUDA Q8_0 / Q5_K_M / Q4_K_M, B200 | 58-62 ms | 154-157 ms | 186-199 ms |
| llama.cpp CPU Q4_K_M + imatrix, 16 threads (x86 server) | 340 ms | 790 ms | 2.4 s |
| llama.cpp CPU Q4_K_M + imatrix, 4 threads (x86 server) | 0.91 s | 1.96 s | 5.95 s |
| llama.cpp CPU Q8_0, 4 threads (x86 server) | 0.81 s | 1.85 s | 5.84 s |

At these small batches the quantised
CUDA builds are slower than bf16. On CPU the vision encoder dominates image latency (2,520 patches per image).

**Memory** (maximum resident set of one text + image + audio pass, Q4_K_M + imatrix with mmproj q8_0, 4 threads, including about
0.2 GB of Python): **9.2 GB** by default, because llama.cpp's CPU backend keeps repacked copies of the weights next to the mmap'd
file, and **6.3 GB with `DEC_NO_REPACK=1`**, at the same speed on x86 (the repacked kernels may matter more on ARM; not measured).

## Notes

- **Softcap exactly once.** llama.cpp's Gemma 4 graph applies the final logit softcap (30) itself, so the logits the C core
  returns are already softcapped. `llama_decide.py` checks the GGUF for `final_logit_softcapping` and applies the softcap from
  `readout.json` only when the GGUF has none.
- **llama-server cannot serve select-all questions.** Its `n_probs` returns the top-N probabilities over the whole vocabulary:
  letters outside the top N are lost, and select-all needs the absolute letter logits, which a softmax over the vocabulary hides.
  Use the C core for exact decisions.
- **Encoders on the GPU.** With `n_gpu_layers != 0` the vision and audio encoders also run on the GPU backend (`gpu_mm=` to
  override).
- `hf_image_resize=False` turns the image pre-resize off (only useful to compare with llama.cpp's own preprocessing).

## Reproduce the parity check

```bash
cd llamacpp
python make_inputs.py inputs                         # needs espeak-ng and ffmpeg; downloads three public-domain images
python ref_hf.py --cases inputs/cases.json --out ref_bf16.json --device cuda       # PyTorch reference from abacusai/Smaug-Decisions
python parity_llama.py --ref ref_bf16.json --cases inputs/cases.json --gguf ../smaug-decisions-Q4_K_M-imat.gguf \
    --mmproj ../mmproj-smaug-decisions-f16.gguf --ngl -1 --repeat 3 --out par_Q4_K_M-imat.json   # CPU: --ngl 0 --threads N
python summarize.py par_*.json
```

`ref_hf.py` needs the package installed (`pip install -e .` from the repository root). `parity_llama.py` asserts that the rendered
prompt is identical to the reference's for every case, and reports |dp|, argmax agreement, the mean letter-logit shift, prompt
lengths and latency. Generated speech depends on the espeak-ng version, so expect small differences from the numbers above on the
audio inputs.

| Script | Purpose |
|---|---|
| [`convert.py`](../llamacpp/convert.py) | `convert_hf_to_gguf.py` without the duplicate tied `lm_head.weight` |
| [`make_calib.py`](../llamacpp/make_calib.py) | Synthetic calibration prompts for `llama-imatrix` |
| [`decide_core.c`](../llamacpp/decide_core.c), [`decide_core.h`](../llamacpp/decide_core.h), [`build_core.sh`](../llamacpp/build_core.sh) | The C core shared by Python and Android, and its build |
| [`llama_decide.py`](../llamacpp/llama_decide.py) | `LlamaDecide`: decisions from a GGUF through the C core (ctypes), plus a command line |
| [`make_inputs.py`](../llamacpp/make_inputs.py) | Builds the 24-input parity set |
| [`ref_hf.py`](../llamacpp/ref_hf.py) | PyTorch reference letter logits and probabilities for the parity set |
| [`parity_llama.py`](../llamacpp/parity_llama.py), [`summarize.py`](../llamacpp/summarize.py) | GGUF vs reference comparison, and a one-line summary per run |

## Known upstream gaps

Two llama.cpp behaviours are worked around here and could be fixed upstream: the Gemma 4 converters do not skip a tied
`lm_head.weight`, and the Gemma 4 vision preprocessing does not upscale images to the Transformers token grid.
