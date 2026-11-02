# iPhone (iOS)

The iOS app runs both models on the phone and the cascade between them, with no server.

| Tier | Model | Runtime | Loaded | Shipped |
|---|---|---|---|---|
| fast | Smaug-Audio-Decisions-0.7B | Core ML: trunk on the Neural Engine, audio tower on the GPU | at launch | in the app (int8 units and an int8 embedding table; the app is about 600 MB) |
| big | Smaug-Decisions | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), GPU backend (CPU backend available) | on the first escalation; memory-mapped; released under memory pressure | downloaded from Smaug-Decisions-LiteRT (not in the app bundle) |

Measured on an iPhone 14 Pro Max (6 GB) with `smaug-decisions-full-vis280` on the GPU: the audio model answers in **0.27-0.37 s**; Smaug-Decisions loads
in 0.48 s (13-21 s the first time), answers text in about 0.6 s, audio in about 1.6 s and a photo in about 7.4 s; the whole
cascade peaks at 1.35 GB, and routing was correct on every case. Full numbers: [devices.md](devices.md#iphone-14-pro-max-measured).

Source, build steps and launch arguments: [`apps/ios`](../apps/ios).

## How the app decides

| Input | First tier | Rule | Shown as |
|---|---|---|---|
| audio | The audio model | confidence >= tau | fast |
| audio | The audio model, then Smaug-Decisions | confidence < tau | escalated |
| text, photo | Smaug-Decisions | always | routed |
| audio, phone too small for Smaug-Decisions | The audio model | confidence < tau | would escalate (the audio model's answer kept and flagged) |
| text, photo, phone too small | none | | needs Smaug-Decisions |

- The confidence rule is a line-for-line port of `smaug_decide.cascade.confidence` and is tested against the Python values.
  tau defaults to 0.9 and can be set from 0.50 to 0.99 in the app.
- The audio model uses its marker prompt and Smaug-Decisions its letter prompt, both rendered exactly as `smaug_decide/prompt.py` does (checked
  byte for byte in the app's tests). The audio model hears the last 10 s of a clip (the Core ML unit's window), Smaug-Decisions the last 30 s.
- The phone computes the log-mel features itself in Swift, following [`smaug_decide/mel.py`](../smaug_decide/mel.py).
- Prompts are fitted to about 2,000 tokens: the app counts tokens with Smaug-Decisions' tokenizer and shrinks the text budget until
  the prompt fits the 2,048-token context.

## Which Smaug-Decisions file a phone gets

| RAM | File | Smaug-Decisions loads when free memory is at least | An image also needs |
|---|---|---|---|
| 8 GB and 12 GB (iPhone 15 Pro and later) | `smaug-decisions-full-vis560` (images at 560 tokens) | 1.1 GB | 1.45 GB |
| 6 GB (iPhone 14 Pro, 15) | `smaug-decisions-full-vis280` (images at 280 tokens) | 1.1 GB | 0.55 GB |
| under 6 GB | none: 0.7B audio model only | | |

The app requests the `com.apple.developer.kernel.increased-memory-limit` entitlement, which raised the iPhone 14 Pro Max's limit
to 4,096 MB. The vision encoder is attached only when a photo arrives. The **Models** screen narrows what is installed: the full
cascade, Smaug-Decisions for text + images (`smaug-decisions-noaudio-*`; audio stays with the audio model), Smaug-Decisions for audio + text
(`smaug-decisions-novision`; no photos) or 0.7B audio model only. Files, sizes and checksums come from the bundled copy of Smaug-Decisions-LiteRT's
`manifest.json` ([README](../README.md#models)).

## First run and storage

- The app downloads the chosen file in the background (pausable, resumable, sha256-checked). To side-load it instead, download
  it on a computer with `python scripts/download.py --model litert --variant full-vis280` and copy it into the app's Documents
  (Finder > iPhone > Files); see [`apps/ios`](../apps/ios#getting-smaug-decisions-onto-the-phone).
- LiteRT-LM writes a weight cache on Smaug-Decisions' first load (1.95 GB, 13-21 s on the iPhone 14 Pro Max; 2.38 GB once an image
  has run). Later loads take about half a second. The cache lives in `Library/Application Support`, which iOS does not purge,
  and is excluded from backups.
- Plan for about 5.8 GB of free storage: the file plus its cache. The app checks before downloading and before the first load.
- The first launch after an install compiles the audio model for the Neural Engine (23-31 s), once.

## Building

You need a Mac with Apple silicon, Xcode 16 or newer and XcodeGen; the audio model's Core ML units
(`python scripts/download.py --model audio --variant coreml`, or convert them with
[`scripts/convert/coreml/`](../scripts/convert/coreml)); and the LiteRT-LM runtime for iOS, built from source with two small
patches (raw multimodal prefill and the raw letter logits) by `apps/ios/runtime/build_runtime_ios.sh`. Step-by-step:
[`apps/ios`](../apps/ios#build).

## Notes

- **Simulator.** Core ML with GPU compute units returned all-zero outputs in the iOS 26.5 Simulator, so the app uses the CPU
  there. Simulator latencies are not phone latencies.
- **Free-form questions for the audio model** need Smaug-Decisions' tokenizer (the two models share the Gemma 4 vocabulary), so they become
  available once Smaug-Decisions has been loaded; phones that cannot load Smaug-Decisions use pre-tokenised questions.
- **Hard cases stay hard.** When both models are unsure, escalation does not make the answer confident; the probabilities say
  so.
