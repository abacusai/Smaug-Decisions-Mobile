# llama.cpp backend (build flavor)

The Android app's fallback backend for Smaug-Decisions (`-PbigBackend=llamacpp`): the C core over llama.cpp + libmtmd that
`app/src/main/cpp/decide_jni.c` links, and the tools that make the app's test data from the same path.

| File | What |
|---|---|
| `decide_core.c`, `decide_core.h` | the C core (the same code as [`llamacpp/`](../../../llamacpp) at the repository root) |
| `llama_decide.py` | `LlamaDecide` with the app's settings (3,072-token budget, 560 image tokens), used by the two generators below |
| `make_kotlin_goldens.py` | goldens for the JVM tests (`app/src/test/resources/goldens`): prompts, trim, image grid and resize, readout, WAV decoding |
| `make_selftest_bundle.py` | the on-device self-test bundle (`app/src/main/assets/selftest`): parity cases with the expected probabilities |
| `decide_replay.c` | a CLI that replays the bundle through the C core, for checking a cross-compiled build without a device |
| `inputs/` | the parity inputs (`cases.json` and its media) that `run_host_e2e.sh` uses by default |

GGUF conversion and the Python path are documented in [docs/llamacpp.md](../../../docs/llamacpp.md); the app expects
`smaug-decisions-Q4_K_M-imat.gguf` and `mmproj-smaug-decisions-q8_0.gguf` in its external files directory.
