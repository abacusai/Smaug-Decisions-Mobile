# Android

The Android app runs both models on the phone and the cascade between them, with no server.

| Tier | Model | Runtime | Loaded | Shipped |
|---|---|---|---|---|
| fast | Smaug-Audio-Decisions-0.7B | ONNX Runtime (CPU, NEON kernels; no GPU or NPU driver needed) | at launch, memory-mapped from the APK | in the app (audio tower int8, trunk 8-bit, int8 embedding table; about 600 MB) |
| big | Smaug-Decisions | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) over JNI (CPU / XNNPack) | on the first escalation; released in `onTrimMemory` | a Smaug-Decisions-LiteRT file, downloaded by the app (not in the APK) |
| big, fallback build flavor | Smaug-Decisions as GGUF | llama.cpp ([llamacpp.md](llamacpp.md)) | same | not in the APK |

**Verification:** the app's Kotlin and native code ran on a desktop JVM, and the arm64 runtime ran under emulation; both matched
the Python reference (details in [devices.md](devices.md#android)).

Source, build steps and tests: [`apps/android`](../apps/android).

## How the app decides

Routing is the same as on iOS and in the Python `Cascade`: audio goes to the audio model, which answers when its confidence is at least
tau (0.9 by default; an exact port of `smaug_decide.cascade.confidence`) and escalates otherwise. Text and images go straight
to Smaug-Decisions. The log-mel front end is a Kotlin port of [`smaug_decide/mel.py`](../smaug_decide/mel.py), and the prompts are
rendered exactly as `smaug_decide/prompt.py` does, with a budget of about 2,000 tokens for the 2,048-token context.

## Device tiers

| Tier | RAM (`totalMem`) | Smaug-Decisions file | What runs |
|---|---|---|---|
| Full | 12 GB+ | `smaug-decisions-full-vis560` | cascade; vision encoder kept after the first image |
| Standard | 8 GB | `smaug-decisions-full-vis560` | cascade; Smaug-Decisions loads when 1.2 GB (text, audio) or 1.5 GB (image, 2.3 GB if Smaug-Decisions is not loaded yet) is free at that moment, else the audio model's answer is flagged; vision encoder dropped after each image |
| Squeeze | 6 GB | `smaug-decisions-full-vis280` | same gates as Standard |
| Fast only | 4 GB | none | 0.7B audio model only: unsure audio is flagged, text and images need Smaug-Decisions |

The gates come from measured dirty memory (the audio model 0.65-1.1 GB, Smaug-Decisions +0.8 GB for text or audio, +1.3 GB more with the vision
encoder) plus Smaug-Decisions' clean working set of about 2-2.65 GB, which must stay in the page cache for decisions to stay fast. They
will be tuned on phones. The **Models** sheet offers the same choices as iOS (full cascade, Smaug-Decisions for text + images,
Smaug-Decisions for audio + text, 0.7B audio model only) and picks the file from the bundled copy of Smaug-Decisions-LiteRT's `manifest.json`.

## Packaging

- Sideload: one arm64 APK with the audio model inside, 694 MB.
- Google Play: an app bundle with the audio model in an install-time asset pack, 582 MB download (36 MB base module), inside Play's
  limits (500 MB base, 1.5 GB per pack).
- Smaug-Decisions is downloaded separately (3.45 GB for `smaug-decisions-full-*`, 2.97-3.28 GB for the component files) with a resumable,
  sha256-checked background download (or download it on a computer and `adb push` it, see
  [`apps/android`](../apps/android)). LiteRT-LM writes a weight cache of about 2 GB to the app's files directory
  on the first load; keep it, or dirty memory grows from about 0.8 GB to 4.8 GB.
- Every native library is 16 KB page aligned (required on Android 15+ devices with 16 KB pages and by Play).

## Building

You need JDK 21, the Android SDK and NDK, and CMake; the audio model's ONNX files (exported from the published weights with
`apps/android/fast/export_onnx.py` and `quantize_onnx.py`); the LiteRT-LM runtime for `android_arm64`, built from source with a
small patch (raw multimodal prefill and letter logits) by `apps/android/runtime/build_litert.sh`; and a Smaug-Decisions-LiteRT file.
Step-by-step: [`apps/android`](../apps/android#build).
