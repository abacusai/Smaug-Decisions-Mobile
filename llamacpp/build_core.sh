#!/bin/bash
# Build libdecide.so against a llama.cpp checkout + build dir (shared libs). usage: build_core.sh <llama.cpp dir> <build dir name> [out]
set -euo pipefail
LC=$1; B=$2; OUT=${3:-$(dirname "$0")/libdecide.so}
HERE=$(cd "$(dirname "$0")" && pwd)
gcc -O2 -fPIC -shared -o "$OUT" "$HERE/decide_core.c" -I"$LC/include" -I"$LC/ggml/include" -I"$LC/tools/mtmd" \
    -L"$LC/$B/bin" -lllama -lmtmd -lggml -lggml-base -Wl,-rpath,"$LC/$B/bin"
echo "built $OUT"
