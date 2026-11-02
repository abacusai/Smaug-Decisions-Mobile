#!/usr/bin/env bash
# Cascade end to end on the desktop JVM with the real models: builds the app's JNI library for this desktop (as run_host_e2e.sh),
# then runs CascadeHostTest: FastTier (ONNX Runtime desktop jar) + Cascade + Decider -> JNI -> decide_core -> llama.cpp over
# assets/cascade_selftest (written by fast/make_cascade_bundle.py), with DEC_NO_REPACK=1 like the app.
#
# usage: run_cascade_host.sh <llama.cpp checkout> <fast-tier ONNX dir> <model.gguf> <mmproj.gguf> [threads=4]
# env:   BUILD_DIR (default build/host-jni), TESTS (default: both CascadeHostTest tests)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 4 ] || { sed -n '2,8p' "$0"; exit 2; }
LC=$(cd "$1" && pwd); FAST=$(cd "$2" && pwd); MODEL=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); MMPROJ=$(cd "$(dirname "$4")" && pwd)/$(basename "$4")
THREADS=${5:-4}; BUILD=${BUILD_DIR:-$HERE/build/host-jni}
if [ ! -f "$BUILD/libdecide_jni.so" ]; then
    cmake -S "$HERE/app/src/main/cpp" -B "$BUILD" -DLLAMA_CPP_DIR="$LC" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD" -j"$(getconf _NPROCESSORS_ONLN)" --target decide_jni
fi
cd "$HERE"
./gradlew testDebugUnitTest --tests "${TESTS:-ai.abacus.decide.CascadeHostTest}" -Pabis=x86_64 -Pdecide.hostLib="$BUILD" \
    -Pdecide.fastDir="$FAST" -Pdecide.model="$MODEL" -Pdecide.mmproj="$MMPROJ" -Pdecide.threads="$THREADS" -Pdecide.noRepack=1 \
    -Pdecide.report="$HERE/build/cascade-host-report.json"
echo "report: $HERE/build/cascade-host-report.json"
