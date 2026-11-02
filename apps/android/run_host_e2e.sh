#!/usr/bin/env bash
# Host end-to-end test: builds the app's JNI library (decide_jni.c + llama.cpp/decide_core.c + llama.cpp, static) for this
# desktop, then runs HostE2ETest: the app's own Kotlin Decider -> JNI -> C core -> llama.cpp on the desktop JVM, real GGUF on CPU,
# over the self-test bundle (app/src/main/assets/selftest), compared with the Python path's expected probabilities.
#
# usage: run_host_e2e.sh <llama.cpp checkout> <model.gguf> <mmproj.gguf> [threads=4]
# env:   BUILD_DIR (default build/host-jni), ONLY=c00,c11 (subset of case ids),
#        REGEN_BUNDLE=1 to first regenerate expected.json with Python (llama_decide.py) through the SAME library,
#        HF_REF=<llamacpp/ref_hf.py output json> (optional, adds the Transformers bf16 probabilities when regenerating),
#        CASES=<cases.json> (default llama.cpp/inputs/cases.json; its media must sit next to it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 3 ] || { sed -n '2,11p' "$0"; exit 2; }
LC=$(cd "$1" && pwd); MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); MMPROJ=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
THREADS=${4:-4}
BUILD=${BUILD_DIR:-$HERE/build/host-jni}
NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

cmake -S "$HERE/app/src/main/cpp" -B "$BUILD" -DLLAMA_CPP_DIR="$LC" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$NPROC" --target decide_jni
LIB=$(ls "$BUILD"/libdecide_jni.* | head -1)
echo "host library: $LIB"

if [ "${REGEN_BUNDLE:-0}" = 1 ]; then
    LLAMA_DECIDE_LIB=$LIB python3 "$HERE/llama.cpp/make_selftest_bundle.py" --cases "${CASES:-$HERE/llama.cpp/inputs/cases.json}" \
        --gguf "$MODEL" --mmproj "$MMPROJ" --threads "$THREADS" --out "$HERE/app/src/main/assets/selftest" ${HF_REF:+--hf-ref "$HF_REF"}
fi

cd "$HERE"
./gradlew testDebugUnitTest --tests ai.abacus.decide.HostE2ETest -Pdecide.hostLib="$BUILD" -Pdecide.model="$MODEL" \
    -Pdecide.mmproj="$MMPROJ" -Pdecide.threads="$THREADS" -Pdecide.report="$HERE/build/host-e2e-report.json" ${ONLY:+-Pdecide.only="$ONLY"}
echo "report: $HERE/build/host-e2e-report.json"
