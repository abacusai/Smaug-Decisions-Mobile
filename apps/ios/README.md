# Cascade for iOS

A SwiftUI app that runs both models on the iPhone, with no server:

- **Smaug-Audio-Decisions-0.7B** (744M, audio) on Core ML: the trunk on the Neural Engine, the audio tower on the GPU. It ships inside the app
  and loads at launch.
- **Smaug-Decisions** (text, images, audio) on [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)'s GPU backend, from a
  [Smaug-Decisions-LiteRT](https://huggingface.co/abacusai/Smaug-Decisions-LiteRT) file. It is downloaded on demand, memory-mapped, loaded on
  the first question that needs it and released under memory pressure.

Ask about a recording, a photo or a piece of text. Audio goes to the audio model first: when its confidence is at least tau (0.9 by
default, adjustable from 0.50 to 0.99) its answer stands (**fast**), otherwise the question goes to Smaug-Decisions (**escalated**).
Text and photos go straight to Smaug-Decisions (**routed**). Every answer shows the probabilities, which model answered and its
confidence.

## Measured on an iPhone 14 Pro Max

A16, 6 GB RAM, iOS 26, Release build, `smaug-decisions-full-vis280.litertlm`, GPU backend:

| | |
|---|---|
| Smaug-Audio-Decisions-0.7B, one audio decision | **0.27-0.37 s** |
| Smaug-Decisions load | **0.48 s** warm; **13-21 s** the first time (writes a 1.95 GB weight cache, 2.38 GB once an image has run) |
| Smaug-Decisions, text decision | **about 0.6 s** (0.56 s prefill + 0.07 s readout) |
| Smaug-Decisions, audio decision | **about 1.6 s** (the first audio after a load: about 4 s) |
| Smaug-Decisions, image decision (280 image tokens) | **about 7.4 s** |
| Peak app footprint, both models loaded | **1.35 GB** (limit with the increased-memory-limit entitlement: 4,096 MB) |

Routing was correct on every run, and Smaug-Decisions' answers on the phone GPU match the CPU build of the same file. The first launch
after an install also compiles the audio model for the Neural Engine once (23-31 s). Plan for about 5.8 GB of free storage for
Smaug-Decisions: the file plus its weight cache. More numbers: [docs/devices.md](../../docs/devices.md).

## Device modes and the Models screen

The app picks a mode from the phone's RAM and the memory iOS lets it use (`DevicePolicy.swift`):

| Phone | Smaug-Decisions file | Mode |
|---|---|---|
| 8 GB or more (iPhone 15 Pro and later) | `smaug-decisions-full-vis560` (images at 560 tokens) | Full cascade |
| 6 GB (iPhone 14 Pro, 15) | `smaug-decisions-full-vis280` (images at 280 tokens: about 1 GB less memory per image) | Full cascade |
| under 6 GB | none | 0.7B audio model only: unsure audio is flagged "would escalate", text and photos say they need Smaug-Decisions |

Smaug-Decisions loads only when enough memory is free at that moment (1.1 GB, and 0.55-1.45 GB more for an image), checked before
every load.

**Models** (Settings > Models, or the first-run "Get Smaug-Decisions" card) chooses what to install; the app picks the matching file
from the manifest:

| Choice | Smaug-Decisions file | What it answers |
|---|---|---|
| Full cascade (default) | `smaug-decisions-full-vis560` / `-vis280` (3.45 GB) | audio (the audio model; unsure ones escalate), text, photos |
| Smaug-Decisions for text + images | `smaug-decisions-noaudio-vis560` / `-vis280` (3.15 GB) | text and photos; audio stays with the audio model, no escalation |
| Smaug-Decisions for audio + text | `smaug-decisions-novision` (3.28 GB) | audio with escalation, text; no photos |
| 0.7B audio model only | none | audio only; nothing to download |

The screen shows each choice's download and storage cost, downloads in the background (pausable, resumable, sha256-checked),
asks before deleting other variants when you switch, and has the tau slider, Prepare now and Release buttons.

## Getting Smaug-Decisions onto the phone

The app reads [`Cascade/Cascade/Resources/models_manifest.json`](Cascade/Cascade/Resources/models_manifest.json), a copy of
the `manifest.json` in [abacusai/Smaug-Decisions-LiteRT](https://huggingface.co/abacusai/Smaug-Decisions-LiteRT): every variant's file name,
size, sha256, inputs and download URL.

- **In the app:** the first-run screen or Models > Download.
- **From a computer:** download it from the repository root (`python scripts/download.py --model litert --variant full-vis280`)
  and copy the file into the app's Documents: Finder >
  your iPhone > Files > Cascade, or `DEV=<device id> tools/phone.sh put models/Smaug-Decisions-LiteRT/smaug-decisions-full-vis280.litertlm`.
  The app checks the size and sha256 against the manifest before loading.
- **From your own server:** copy a `models_manifest.json` with your URLs (or a `url_overrides` map) into the app's
  Documents; it takes precedence over the bundled one.

## Build

You need a Mac with Apple silicon, Xcode 16 or newer, [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install
xcodegen`) and Python 3 with numpy. From the repository root:

```bash
python scripts/download.py --model audio --variant coreml        # the audio model's Core ML units -> models/Smaug-Audio-Decisions-0.7B/coreml
python scripts/download.py --model litert --variant full-vis280  # optional: a Smaug-Decisions file for the Simulator

cd apps/ios
runtime/build_runtime_ios.sh     # LiteRT-LM with the harness patch -> Cascade/Frameworks (30-60 min cold, ~20 GB bazel cache)
./build.sh setup                 # links the Core ML units and frameworks, writes the int8 embedding table, generates the project
./build.sh test                  # the logic tests (no models needed)
./build.sh sim && ./build.sh install && ./build.sh run      # Simulator
```

`runtime/build_runtime_ios.sh` needs bazelisk and Rust (`brew install bazelisk rustup && rustup-init -y`). It builds LiteRT-LM
at a pinned commit with two small patches ([`runtime/patches/`](runtime/patches)): a multimodal prefill on a raw session and the
raw letter logits after it, which select-all questions need. `--prebuilt` installs Google's stock runtime instead (text,
one-answer questions only).

**On a phone:** open `Cascade/Cascade.xcodeproj` after `./build.sh setup`, set your team and a bundle id, and run. A free
(personal) team refuses the extended-virtual-addressing entitlement; build with
`CODE_SIGN_ENTITLEMENTS=CascadePersonalTeam.entitlements` (or use
`TEAM=<team> XCDEST=<device id> open tools/phone_build.command`, which does that from the Mac's GUI session, where codesign can
reach the keychain). `./build.sh device` compiles the arm64 app without signing.

`build.sh` reads `UNITS` (Core ML units, default `models/Smaug-Audio-Decisions-0.7B/coreml`), `RUNTIME` (frameworks, default
`Cascade/Frameworks`), `SMAUG_DECISIONS_DIR` (Smaug-Decisions files for the Simulator, default `models/Smaug-Decisions-LiteRT`), `SIM` and `BUNDLE_ID`.

## Launch arguments

For testing and measurements (`./build.sh run <args>` in the Simulator, `tools/phone.sh run <name> <args>` on a phone):

| Argument | Effect |
|---|---|
| `--autorun a1,a2,a3,a4,t1,i1` | runs built-in cases and writes `Documents/autorun_<id>.json` and `autorun_summary.json` |
| `--tau x` | sets the threshold |
| `--models full\|audio_text\|text_image\|audio_only` | the Models choice |
| `--backend gpu\|cpu`, `--threads n` | Smaug-Decisions' backend (default GPU on a phone, CPU in the Simulator) and CPU threads |
| `--ram-gb n`, `--mode full\|fast-only`, `--build vis560\|vis280` | pretend to be an n GB phone, or force the mode or the image budget |
| `--clear-cache`, `--drop-cache <s>` | delete LiteRT-LM weight caches (all, or those whose name contains s) |
| `--manifest <path>` | read another manifest |
| `--tokcheck`, `--freeform "a1\|question\|Yes,No"`, `--verify-files` | tokenizer check, a free-form question for the audio model, sha256 of the installed files |
| `--snapshot <name>` | save the app's window to `Documents/snapshot_<name>.png` |

`python tools/summarize.py <dir>` prints tables from a folder of autorun JSONs.

The built-in cases are our own inputs, never benchmark items: a1 (a synthetic melody), a2 (an unfinished turn, synthetic speech),
a3 (two synthetic voices), a4 (a tone sweep), t1 (a fraud text), i1 (a drawn warning sign). The clips come from
[`apps/android/fast/make_clips.py`](../android/fast/make_clips.py). Expected routes at tau 0.9: a1 and a4 fast, a2 and a3
escalated, t1 and i1 routed.

## Layout

| Path | What |
|---|---|
| `Cascade/project.yml` | XcodeGen spec: the app and the hostless logic-test bundle |
| `Cascade/Cascade/Core/` | routing (`Router.swift`, a port of `smaug_decide/cascade.py`), device policy, Models choice and manifest, download, app state |
| `Cascade/Cascade/Fast/` | the audio model on Core ML, the int8 embedding table, the log-mel front end (a port of `smaug_decide/mel.py`), the recorder, free-form tokenisation |
| `Cascade/Cascade/Big/` | Smaug-Decisions on LiteRT-LM: prompt rendering (a port of `smaug_decide/prompt.py`), readout, media preparation, the C shim |
| `Cascade/Cascade/App/` | SwiftUI screens |
| `Cascade/Cascade/Resources/` | `cascade.json` (the audio model's pre-tokenised prompts and the built-in cases), `models_manifest.json`, demo media, synthetic test clips |
| `Cascade/Cascade/Fonts/` | Inter, under the SIL Open Font License (`LICENSE.txt`) |
| `Cascade/CascadeTests/` | logic tests against goldens from the Python package: confidence, routing, prompts, tokenizer, device policy, Models choices, manifest |
| `runtime/` | the LiteRT-LM runtime build and its patches |
| `tools/` | `quantize_embed.py` (the audio model's int8 table), `make_scenarios.py` and `make_cases.py` (pre-tokenised prompts, `cascade.json`), `make_goldens.py` (test goldens), `tokenizer_check.py`, `freeform_ref.py`, `summarize.py`, `phone.sh`, `phone_build.command` |

## Notes

- **Simulator.** Core ML with GPU compute units returned all-zero outputs in the iOS 26.5 Simulator, so the app runs the audio model on
  the CPU there, and Simulator latencies say nothing about a phone.
- **Free-form questions for the audio model** use Smaug-Decisions' tokenizer (both models share the Gemma 4 vocabulary), so they appear once Smaug-Decisions
  is loaded; without Smaug-Decisions, the audio model asks the pre-tokenised questions in `cascade.json`.
- **Measured phone:** the numbers above are from an iPhone 14 Pro Max (6 GB, `-vis280`); 8 GB phones use the 560-token file.
