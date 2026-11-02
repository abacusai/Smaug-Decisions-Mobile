# Devices: what runs where, measured

Every number on this page was measured unless it says **estimated**. Phone latencies are wall-clock per decision with the model
loaded and its weight cache built, unless the row says otherwise.

## Summary

| Device | Smaug-Audio-Decisions-0.7B | Smaug-Decisions | Cascade | Status |
|---|---|---|---|---|
| iPhone 14 Pro Max (6 GB, A16) | Core ML: 0.27-0.37 s | LiteRT-LM GPU, `smaug-decisions-full-vis280`: text 0.6 s, audio 1.6 s, image 7.4 s | full cascade, peak 1.35 GB | measured |
| iPhones with 8 GB or more | Core ML | LiteRT-LM GPU, `smaug-decisions-full-vis560` | full cascade | estimated |
| iPhones under 6 GB | Core ML | does not fit | 0.7B audio model only | estimated |
| Android, 12 GB or more | ONNX Runtime | LiteRT-LM, vision encoder kept after the first image | full cascade | estimated (verified off-device) |
| Android, 8 GB | ONNX Runtime | LiteRT-LM when 2.0 GB (text, audio) or 3.5 GB (image) is free | full cascade | estimated (verified off-device) |
| Android, 6 GB | ONNX Runtime | same gates as 8 GB | usually text and audio, rarely images | expected |
| Android, 4 GB | ONNX Runtime | does not fit | 0.7B audio model only | estimated |
| Mac, Apple Silicon | Core ML: 121 ms (M3 Pro GPU); PyTorch MPS | PyTorch MPS (about 16 GB), MLX, llama.cpp | Python | Core ML measured |
| x86 CPU server | PyTorch | llama.cpp Q4_K_M: 340 ms text, 790 ms audio, 2.4 s image (16 threads); LiteRT-LM `smaug-decisions-full-vis560`: 190 / 441 / 4,927 ms prefill (16 vCPUs) | Python | measured |
| NVIDIA H100 | 9.7 ms | 210 ms per audio decision | 130 ms per audio decision at tau 0.9 | measured |

"Verified off-device" for Android means the app's Kotlin and native code ran on a desktop JVM and the arm64 runtime under
emulation and matched the Python reference; no Android phone has run it yet.

## iPhone 14 Pro Max (measured)

iPhone 14 Pro Max (A16, 6 GB RAM, reported as 5,636 MB), iOS 26.6, Release build of the cascade app, `smaug-decisions-full-vis280.litertlm`
(the file the app picks for 6 GB phones) on LiteRT-LM's GPU backend.

**Memory.** With the `increased-memory-limit` entitlement the app's limit was **4,096 MB** (about 3,054 MB without it on
this class of phone, from developer reports), so the app ran the full cascade. Peak footprint with both models loaded was
**1.35 GB**, and no run was killed for memory.

**Smaug-Audio-Decisions-0.7B (Core ML: trunk on the Neural Engine, audio tower on the GPU)**

| | iPhone 14 Pro Max |
|---|---|
| steady decision | **0.27-0.37 s** (audio tower about 0.23 s, trunk 0.04-0.06 s) |
| model load | 0.8-0.9 s; 23-31 s on the first launch after an install (Neural Engine compile) |
| first decision after an install | 14.5 s (GPU compile of the audio tower) |
| footprint, the audio model alone | 0.39-0.51 GB |
| parity | max \|dp\| 0.0037 vs PyTorch on the app's audio cases; the int8 embedding table moves probabilities by at most 0.0063 vs fp16 (same answers and routes on 40 decisions) |

**Smaug-Decisions (`smaug-decisions-full-vis280`, LiteRT-LM GPU backend)**

| | Smaug-Decisions-LiteRT |
|---|---|
| first load (writes the weight cache) | **13-21 s**, 1.95 GB cache (2.38 GB once an image has run) |
| later loads | **0.48 s** |
| text decision (prefill + readout) | **0.56 + 0.07 s** |
| audio decision | **1.50 + 0.08 s** (the first audio after a load: 3.0 + 1.2 s) |
| image decision (280 image tokens) | **6.1 + 1.3 s** |
| peak app footprint, with the audio model loaded | **1.35 GB** |

- iOS sent memory warnings at a footprint well below the limit. The app releases Smaug-Decisions on critical pressure and reloads it on
  the next escalation in about half a second.
- **Storage:** the file plus its weight cache take about 5.8 GB.
- The CPU backend also works (the app falls back to it if the GPU engine cannot be created) but is 2-4x slower on text and
  audio on this phone.

**Correctness on the phone.** Routing at tau 0.9 was right on every case in every run (confident audio answered by the audio model,
unsure audio escalated, text and images routed to Smaug-Decisions). Smaug-Decisions' tokenizer reproduced the reference token ids 10 of 10
times. The phone GPU gives the same answers as the CPU build of the same weights (text 0.939, image 0.9995, audio 0.976 on the
reference questions) and the same top answers as full-precision Smaug-Decisions on every referenced case.

**Not verified on a phone:** `smaug-decisions-full-vis560` on an 8 GB phone, the in-app download of the real files, microphone
recording and photo picking in the autorun, thermal behaviour over long use.

## iPhone device classes (estimated)

| RAM | Smaug-Decisions build | Smaug-Decisions loads when free memory is at least | An image also needs |
|---|---|---|---|
| 8 GB and 12 GB | 560 image tokens | 1.1 GB | 1.45 GB |
| 6 GB | 280 image tokens | 1.1 GB | 0.55 GB |
| under 6 GB | none: 0.7B audio model only (unsure audio answers are flagged, text and images need Smaug-Decisions) | | |

The thresholds are measured footprints x 1.25. Free memory is `os_proc_available_memory()` with the audio model loaded, checked at launch
and before every Smaug-Decisions load or vision attach. A 6 GB phone fits even without the entitlement: 0.5 GB for the audio model + 1.1 GB + 0.55 GB
= 2.15 GB against about 3 GB.

## Android

The app ran on a desktop JVM (x86, 4 shared vCPUs) with the same Kotlin, JNI and LiteRT-LM code, and the arm64 build of the
runtime ran under qemu with an Android 16 arm64 userland. The Smaug-Decisions checks below compare the app's
path with the Python LiteRT-LM path on the same file.

| Check | Result |
|---|---|
| The audio model ONNX export vs PyTorch (fp32) | median \|dp\| 1.7e-5, max 1.5e-4 over 100 decisions |
| The audio model as shipped (audio int8, trunk 8-bit blocks) | median \|dp\| 0.0076, max 0.063; 80 of 83 top answers agree, the 3 others are near-ties that escalate anyway |
| Smaug-Decisions, app's Kotlin + JNI path vs the Python LiteRT-LM path | max \|dp\| 2.2e-16 on 40 cases, prompts and positions identical |
| Whole cascade vs the Python reference | 0 routing differences on 39 requests |
| arm64 runtime (qemu) vs x86 | max \|dp\| 0.0042 text, 0.0020 audio |

Desktop JVM latency (4 threads; a phone will differ): Smaug-Decisions text 0.80-0.95 s, audio 1.5-1.6 s, image 10-11 s; an escalated
audio decision in the cascade had a median of 2.1 s (15.4 s with the llama.cpp backend on the same machine).

**Memory** (warm weight cache, `/proc/self/smaps_rollup`):

| Configuration | Dirty (anonymous) | Clean file-backed |
|---|---|---|
| Smaug-Decisions, text decisions, no vision encoder | 0.79 GB | 2.38 GB |
| Smaug-Decisions, audio decisions | 0.83 GB | 2.65 GB |
| Smaug-Decisions with the vision encoder attached | about 2.1-2.65 GB | 2.4-2.5 GB |
| Whole cascade after audio or text | 1.90 GB (the audio model 0.65-1.1 GB of it) | 2.67 GB |

Creating the vision encoder alone costs about 1.3 GB, so the app attaches it only for image decisions on phones under 12 GB.

**Packaging.** arm64 APK with the audio model inside (int8 embedding table): 694 MB. As a Play bundle with the audio model in an install-time asset
pack: 582 MB download (base module 36 MB). Smaug-Decisions is downloaded separately (3.45 GB for `smaug-decisions-full-*`) and writes a weight
cache of about 2 GB on its first load.

**Device tiers** (from `totalMem`; the gates come from the measurements above plus headroom and will be tuned on phones):

| Tier | RAM | Policy |
|---|---|---|
| Full | 12 GB+ | cascade; vision encoder kept after the first image |
| Standard | 8 GB | cascade; Smaug-Decisions only when 2.0 GB (text, audio) or 3.5 GB (image) is free at that moment, else the audio model answer is flagged |
| Squeeze | 6 GB | same gates; expected to pass for text and audio on a lightly loaded phone |
| Fast only | 4 GB | 0.7B audio model only |

## Smaug-Decisions-LiteRT builds on a CPU (measured, x86, 16 vCPUs)

Warm weight cache, CPU only. "Dirty" is anonymous memory (what the OS has to keep); "clean" is file-backed pages it can drop and
re-read.

| Build | Size | Dirty text / audio / image | Clean text | Prefill text / audio / image |
|---|---|---|---|---|
| `smaug-decisions-full-vis560` | 3.45 GB | 0.65 / 0.75 / 2.00 GB | 2.02 GB | 190 / 441 / 4,927 ms |
| `smaug-decisions-full-vis280` | 3.45 GB | 0.65 / 0.75 / **1.04** GB | 2.02 GB | 220 / 529 / **1,856** ms |

The engine needs a writable, persistent cache directory: without the weight cache, Smaug-Decisions' dirty memory is about 4.8 GB
instead of 0.8 GB.

## Servers and desktops (measured)

| Setup | Text | Audio | Image |
|---|---|---|---|
| Smaug-Audio-Decisions-0.7B, PyTorch, H100 | | 9.7 ms | |
| Smaug-Decisions, PyTorch bfloat16, H100 | | 210 ms | |
| Cascade at tau 0.9, H100 (mean over Earshot) | | 130 ms | |
| Smaug-Decisions, PyTorch bfloat16, B200 | 75 ms | 151 ms | 128 ms |
| Smaug-Decisions, llama.cpp CUDA bf16, B200 | 41 ms | 122 ms | 143 ms |
| Smaug-Decisions, llama.cpp CPU Q4_K_M + imatrix, 16 threads (x86) | 340 ms | 790 ms | 2.4 s |
| Smaug-Decisions, llama.cpp CPU Q4_K_M + imatrix, 4 threads (x86) | 0.91 s | 1.96 s | 5.95 s |
| Smaug-Audio-Decisions-0.7B, Core ML, Mac M3 Pro GPU | | 121 ms | |

Measure your own machine with `python scripts/benchmark.py`.
