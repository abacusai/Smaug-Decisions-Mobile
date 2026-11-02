#!/usr/bin/env bash
# LiteRT-LM backend end to end on the desktop JVM: builds libdecide_litert_jni.so (DecideShim.c + litert_jni.c, linked to the
# patched x86_64 liblitert-lm.so in LITERT_DIR/linux_x86_64), then runs
#   HostE2ETest     LiteRtDecider (Kotlin + JNI + LiteRT-LM) over assets/selftest_litert vs the Python LiteRT path + PyTorch @560
#   CascadeHostTest the whole cascade (ONNX fast tier + LiteRT Smaug-Decisions) over assets/cascade_selftest
# usage: run_litert_host.sh <LITERT_DIR> <fast-tier ONNX dir> <model.litertlm> [threads=4]
# env:   CACHE (XNNPack cache dir; default: the model's dir, where the Python harness also writes it), TESTS, BUILD_DIR
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 3 ] || { sed -n '2,9p' "$0"; exit 2; }
LD=$(cd "$1" && pwd); FAST=$(cd "$2" && pwd); MODEL=$(cd "$(dirname "$3")" && pwd)/$(basename "$3"); THREADS=${4:-4}
BUILD=${BUILD_DIR:-$HERE/build/host-litert}; CACHE=${CACHE:-$(dirname "$MODEL")}
cmake -S "$HERE/app/src/main/cpp" -B "$BUILD" -DDECIDE_BIG_BACKEND=litert -DLITERT_DIR="$LD" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j"$(getconf _NPROCESSORS_ONLN)" --target decide_litert_jni
cd "$HERE"
for T in ${TESTS:-ai.abacus.decide.HostE2ETest ai.abacus.decide.CascadeHostTest}; do
  ./gradlew testDebugUnitTest --tests "$T" -Pdecide.hostLib="$BUILD" -Pdecide.bigBackend=litert -Pdecide.litertModel="$MODEL" \
      -Pdecide.litertCache="$CACHE" -Pdecide.fastDir="$FAST" -Pdecide.threads="$THREADS" -Pdecide.heap=3g \
      -Pdecide.bundle="$HERE/app/src/main/assets/selftest_litert" -Pdecide.report="$HERE/build/litert-$(echo "$T" | sed 's/.*\.//').json"
done
