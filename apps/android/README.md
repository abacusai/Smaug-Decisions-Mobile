# Cascade for Android

A Kotlin (Jetpack Compose) app that runs both models on the phone, with no server:

| Tier | Model | Runtime | Loaded | Shipped |
|---|---|---|---|---|
| fast | Smaug-Audio-Decisions-0.7B (744M, audio) | ONNX Runtime Android, CPU (NEON kernels; no GPU or NPU driver needed) | at launch, memory-mapped from the APK | in the APK (audio tower int8, trunk 8-bit, int8 embedding table; about 600 MB) |
| big | Smaug-Decisions (text, images, audio) | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) through JNI, CPU (XNNPack) | on the first question that needs it; released in `onTrimMemory` | downloaded from [Smaug-Decisions-LiteRT](https://huggingface.co/abacusai/Smaug-Decisions-LiteRT) |
| big, build flavor | Smaug-Decisions as GGUF | llama.cpp (`-PbigBackend=llamacpp`) | same | not in the APK |

Routing is the same as on iOS and in the Python `Cascade`: audio goes to the audio model, whose answer stands when its confidence is at
least tau (0.9 by default; an exact port of `smaug_decide.cascade.confidence`), else the question escalates to Smaug-Decisions. Text
and photos go straight to Smaug-Decisions.

**Verification:** the app's Kotlin, JNI and LiteRT-LM code ran on a desktop JVM (x86 Linux) and matched the Python reference
(Smaug-Decisions max |dp| 2.2e-16 on 40 cases, no routing differences on 39 cascade requests), and the arm64 runtime matched x86
under qemu. Measurements: [docs/devices.md](../../docs/devices.md#android).

## Device tiers and the Models screen

The tier comes from `ActivityManager.MemoryInfo.totalMem` (`DevicePolicy` in `Cascade.kt`):

| Tier | RAM | Smaug-Decisions file | What runs |
|---|---|---|---|
| Full | 12 GB+ | `smaug-decisions-full-vis560` | cascade; the vision encoder stays loaded after the first image |
| Standard | 8 GB | `smaug-decisions-full-vis560` | cascade; Smaug-Decisions only when 1.2 GB (text, audio) or 1.5 GB (image, 2.3 GB if Smaug-Decisions is not loaded yet) is free at that moment, else the audio model's answer is flagged; vision encoder dropped after each image |
| Squeeze | 6 GB | `smaug-decisions-full-vis280` | same gates as Standard |
| Fast only | 4 GB | none | 0.7B audio model only: unsure audio is flagged, text and photos need Smaug-Decisions |

**Models** (the Models sheet) has the same choices as the iOS app: Full cascade, Smaug-Decisions for text + images
(`smaug-decisions-noaudio-*`; audio stays with the audio model), Smaug-Decisions for audio + text (`smaug-decisions-novision`; no photos), and 0.7B audio model only. The app
picks the file from [`app/src/main/assets/models_manifest.json`](app/src/main/assets/models_manifest.json) (a copy of
Smaug-Decisions-LiteRT's `manifest.json`; a `models_manifest.json` in the app's external files directory overrides it), downloads it
with a WorkManager foreground worker (resumable, Wi-Fi-only switch, size and sha256 checked), asks before deleting other
variants, and checks free storage before downloading and before the first load.

To install a file without the in-app download, fetch it on a computer and push it:

```bash
python scripts/download.py --model litert --variant full-vis560       # from the repository root
adb push models/Smaug-Decisions-LiteRT/smaug-decisions-full-vis560.litertlm /sdcard/Android/data/ai.abacus.decide/files/
```

LiteRT-LM writes a weight cache (about 2 GB for the decoder, plus the encoders) to the app's `filesDir/litert_cache` on the first
load. Keep it: without it Smaug-Decisions' dirty memory grows from about 0.8 GB to 4.8 GB.

## Build

JDK 21, Android SDK 36, NDK 29.0.13113456 and CMake 3.31.6 (Android Studio installs them), Python 3.12 with torch,
transformers 5.20+, onnx and onnxruntime 1.23.2 for the audio model export.

```bash
cd apps/android

# 1. The audio model for ONNX Runtime: export, quantise, int8 embedding table (from the published weights)
python ../../scripts/download.py --model audio --local-dir ../../models/Smaug-Audio-Decisions-0.7B
python fast/export_onnx.py --model ../../models/Smaug-Audio-Decisions-0.7B --out onnx_fp32
python fast/quantize_onnx.py --src onnx_fp32 --out-root . --variants int8,q8w        # audio from onnx_int8, trunk from onnx_q8w
mkdir -p apk_assets/fast && cp onnx_int8/audio.onnx onnx_q8w/trunk.onnx onnx_fp32/head.json apk_assets/fast/
python ../ios/tools/quantize_embed.py onnx_fp32/embed_tokens_f16.bin apk_assets/fast    # embed_tokens_int8.bin + _scales.bin

# 2. LiteRT-LM for android_arm64 with the harness patch (Linux x86_64; about 10 min on 24 cores)
runtime/build_litert.sh                                    # -> litert/ (BUILD_X86=1 also builds the desktop library for tests)

# 3. the app
printf 'sdk.dir=%s\nlitertDir=%s\nfastAssets=%s\n' "$ANDROID_HOME" "$PWD/litert" "$PWD/apk_assets" > local.properties
./gradlew test                                             # JVM tests: prompts, image resize, readout, WAV, mel, confidence, tiers, Models, downloader
./gradlew assembleDebug                                    # arm64-v8a APK with the audio model inside (sideload)
ln -s "$PWD/apk_assets" fastpack/src/main/assets && ./gradlew bundleRelease -PfastPack=true   # Play: the audio model in an install-time asset pack
```

The llama.cpp flavor needs `llamaCppDir=<llama.cpp checkout at f7c01c5>` in `local.properties` and `-PbigBackend=llamacpp`.

**Sizes:** the arm64 APK with the audio model inside is 694 MB; as a Play bundle the download is 582 MB (36 MB base module plus an
install-time asset pack), inside Play's limits. Every native library is 16 KB page aligned (Android 15+ devices with 16 KB
pages, and Play's requirement).

## Verify on a computer

```bash
BUILD_X86=1 runtime/build_litert.sh                        # the desktop LiteRT-LM library, if not built yet
./run_litert_host.sh litert <dir with the audio model ONNX files> smaug-decisions-full-vis560.litertlm    # Smaug-Decisions + the whole cascade on the JVM
python fast/parity_onnx.py --model <Smaug-Audio-Decisions-0.7B dir> --onnx <onnx dir> --cases clips/cases.jsonl --out parity.json   # the audio model vs PyTorch
```

`fast/make_clips.py` writes the synthetic clips for the parity check. `run_host_e2e.sh` and `run_cascade_host.sh` do the same
for the llama.cpp flavor.

## On a phone

```bash
./run_device_test.sh smaug-decisions-full-vis560.litertlm     # installs the app and the test APK, pushes the model, runs both device tests
```

It runs `DeviceSelfTest` (Smaug-Decisions: parity with the Python reference, prompt identity, latency, peak memory) and
`DeviceCascadeTest` (the whole cascade with the device's own tier) and pulls both reports. The app's Models sheet also has
developer self-tests. Regenerate the expected values in `assets/selftest_litert` with the Python LiteRT-LM path for the exact
file under test before reading its parity numbers.

## Layout

| Path | What |
|---|---|
| `app/src/main/java/ai/abacus/decide/` | `Cascade.kt` (routing, tiers), `FastTier.kt` + `MelFrontEnd.kt` + `EmbedTable.kt` (the audio model), `LiteRtDecider.kt` + `MediaFiles.kt` (Smaug-Decisions on LiteRT-LM), `ModelSetup.kt` + `ModelManager.kt` (Models choices, manifest, download, storage), `Prompt.kt`, `Readout.kt`, `HfImage.kt`, `Decider.kt` (llama.cpp flavor), `MainActivity.kt` + `Ui.kt` (Compose UI), self-tests |
| `app/src/main/cpp/` | `litert/DecideShim.c` + `litert_jni.c` (the C shim over the LiteRT-LM C API, shared with iOS), `decide_jni.c` (llama.cpp flavor) |
| `app/src/main/assets/` | `models_manifest.json`, the audio model's pre-tokenised scenarios (`fast/scenarios.json`), self-test bundles |
| `app/src/main/res/font/` | Inter, under the SIL Open Font License (`app/src/main/FONTS_LICENSE.txt`) |
| `app/src/test/`, `app/src/androidTest/` | JVM tests with goldens from the Python package; device tests |
| `fast/` | the audio model's export (`export_onnx.py`), quantisation (`quantize_onnx.py`), scenarios (`make_scenarios.py`), test clips, goldens and the cascade self-test bundle, parity with PyTorch |
| `runtime/` | the LiteRT-LM build for Android (`build_litert.sh`) and the harness patch (`apply_patch.py`, `litert_lm_harness.patch`) |
| `llama.cpp/` | the llama.cpp flavor's C core and test-data generators ([README](llama.cpp/README.md)) |
| `fastpack/` | the Play Asset Delivery pack for the audio model |
