# Core ML export (fast tier)

Tools to turn the published `abacusai/Smaug-Audio-Decisions-0.7B` weights into the fixed-shape Core ML units the [iOS app](../../../docs/ios.md)
runs, and to check them against the PyTorch model on your own audio. This is plain conversion and int8 weight quantisation of the
published weights; nothing is trained.

**You may not need to convert at all.** The model repository ships ready-made int8 units in `coreml/`:

```bash
python scripts/download.py --model audio --variant coreml        # from the repository root
```

Convert yourself when you want other sequence lengths or audio windows, other quantisation, or to audit the conversion.

## Requirements

macOS 15 or later on Apple Silicon (conversion also runs on Linux, but only macOS can load and test the units), Python 3.10+, and:

```bash
pip install -e ".[coreml]"     # from the repository root: torch, transformers>=5.20, coremltools>=9
```

## Units

| Unit | Inputs -> output | Size (int8) |
|---|---|---|
| `trunk_L512_int8`, `trunk_L1024_int8` | `inputs_embeds [1,L,512]`, additive `mask [1,1,L,L]`, `position_ids [1,L]` -> hidden states `[1,L,768]` | 139 MB each |
| `audio_10s_int8`, `audio_30s_int8` | log-mel `features [1,F,128]`, frame `mask [1,F]` -> audio token embeddings `[1,T,512]` (10 s: F = 999, T = 250) | 295 MB |
| `audio_conv_10s_fp16` + `audio_stack_10s_int8` | the same audio tower split in two so the Neural Engine can compile it (`--split_audio`, see `audio_dense.py`) | 2 MB + 294 MB |
| `embed_tokens_fp16.bin` | the scaled token-embedding table, raw little-endian fp16 `[vocab, 512]`, gathered in Swift | 268 MB |
| `head.json`, `readout.json` | marker head weight and bias; shapes, token ids and the mask constant | small |

The app builds the trunk input itself: token embeddings from the table, with the audio unit's output spliced in at the audio token
positions, then reads the head at each option's `<unused0>` marker.

## Steps

```bash
# 1. convert (minutes per unit; existing units in --out are kept)
python scripts/convert/coreml/convert.py --model abacusai/Smaug-Audio-Decisions-0.7B --out build/coreml --L 512 --audio 10

# 2. check against PyTorch on your own clips (cases.jsonl: {"id", "input": "clip.wav", "question", "options", "multi"?})
python scripts/convert/coreml/parity.py --units build/coreml --cases cases.jsonl --compute CPU_AND_GPU
```

Run these from the repository root. Then point the iOS app's build at `build/coreml` instead of the downloaded units (see
[`apps/ios`](../../../apps/ios)).

| Script | Purpose |
|---|---|
| [`convert.py`](convert.py) | Trunk, audio tower (single or split), embedding table, head and readout |
| [`patches.py`](patches.py) | Traceable mask builders, an fp16-safe RMSNorm and coremltools op fixes, applied on import |
| [`audio_dense.py`](audio_dense.py) | The audio tower's local attention rewritten as dense band attention for the Neural Engine |
| [`check_audio_dense.py`](check_audio_dense.py) | Dense form vs the HF tower on random weights (no checkpoint needed) |
| [`parity.py`](parity.py) | Core ML vs PyTorch float32 on your cases: per-case max probability difference and per-unit latency |

## Measured results

With the published int8 units, 10 s audio window, trunk L512:

| Check | Result |
|---|---|
| Swift log-mel vs transformers' feature extractor | max abs diff 3e-4 (the spec is [`smaug_decide/mel.py`](../../../smaug_decide/mel.py)) |
| Core ML int8 vs PyTorch float32, Mac GPU, 30 clips | probability difference median 0.014, max 0.217 |
| iPhone 14 Pro Max vs Mac Core ML, same clip | max abs diff 0.000 |
| iPhone 14 Pro Max, trunk on the Neural Engine, audio tower on the GPU | 389 ms per decision (0.30-0.37 s in the cascade app) |
| iPhone 14 Pro Max, GPU only | 1,082 ms per decision |
| Mac M3 Pro, GPU | 121 ms per decision |
| App bundle | 670 MB |

The first launch on a phone compiles the trunk for the Neural Engine once, which takes minutes. The single audio unit does not
finish compiling for the Neural Engine, so it runs on the GPU. The split audio units compile for the Neural Engine on a Mac in
about 14 s and run there in about 27 ms, but on the iPhone 14 Pro Max they were no faster than the single unit on the GPU (about
200 ms either way), so the app bundles the single unit by default.

## Status

The scripts here are a cleaned, self-contained version of the ones that produced the published units. The PyTorch side has been
checked against stock transformers 5.20 on small random-weight models: the trunk wrapper (padding mask plus sliding-window band)
and the frozen-mask audio wrapper match the stock forward to within 3e-5, and the dense audio stack matches the HF tower to 2e-6
(`check_audio_dense.py`). The full Core ML conversion of the published weights has not yet been re-run from this repository, so
run `parity.py` on a few of your own clips before you ship units you converted yourself.
