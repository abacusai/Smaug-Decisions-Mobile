#!/bin/bash
# Builds the LiteRT-LM C API (liblitert-lm.so) with the harness patch (multimodal raw-session prefill + raw letter logits;
# apply_patch.py, the same change as litert_lm_harness.patch) for android_arm64 (the app) and, with BUILD_X86=1, linux_x86_64
# (the desktop JVM tests). Base: google-ai-edge/LiteRT-LM 2c1c9a67 (== litert-lm 0.19.0), Android NDK r28b.
#
#   runtime/build_litert.sh                     # -> litert/{include/c, android_arm64, jniLibs/arm64-v8a}
#   BUILD_X86=1 runtime/build_litert.sh         # also litert/linux_x86_64
#
# Linux x86_64 host with clang, lld, git, git-lfs, unzip, curl and python3. Env: WORK (checkout + NDK, default runtime/.build),
# OUT (default ../litert next to app/), JOBS (bazel --jobs). A cold build takes about 10 minutes on 24 cores.
set -euxo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HERE/.build}; OUT=${OUT:-$HERE/../litert}; JOBS=${JOBS:-$(nproc)}
COMMIT=2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4
mkdir -p "$WORK/bin" "$OUT"; cd "$WORK"
[ -x bin/bazelisk ] || curl -sSL -o bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64
chmod +x bin/bazelisk
command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ] || { echo "install Rust first: https://rustup.rs"; exit 1; }
[ -d android-ndk-r28b ] || (curl -sSL -o ndk.zip https://dl.google.com/android/repository/android-ndk-r28b-linux.zip && unzip -q ndk.zip && rm ndk.zip)
[ -d LiteRT-LM ] || git clone -q https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM && git checkout -q "$COMMIT"
git lfs install --local && git lfs pull --include="prebuilt/linux_x86_64/*,prebuilt/android_arm64/*"
python3 "$HERE/apply_patch.py" .
export PATH=$WORK/bin:$HOME/.cargo/bin:$PATH
mkdir -p "$OUT/include/c" && cp c/*.h "$OUT/include/c/"
if [ "${BUILD_X86:-0}" = 1 ]; then
  bazelisk build --config=linux_x86_64 --jobs="$JOBS" --deleted_packages=swift --java_runtime_version=remotejdk_17 //c:litert-lm
  mkdir -p "$OUT/linux_x86_64" && cp -L bazel-bin/c/liblitert-lm.so prebuilt/linux_x86_64/*.so "$OUT/linux_x86_64/"
fi
export ANDROID_NDK_HOME=$WORK/android-ndk-r28b
# 16 KB ELF alignment (Android 15+ devices with 16 KB pages, and Google Play's requirement)
bazelisk build --config=android_arm64 --jobs="$JOBS" --deleted_packages=swift --java_runtime_version=remotejdk_17 \
  --linkopt=-Wl,-z,max-page-size=16384 --linkopt=-Wl,-z,common-page-size=16384 //c:litert-lm
mkdir -p "$OUT/android_arm64" && cp -L bazel-bin/c/liblitert-lm.so prebuilt/android_arm64/*.so "$OUT/android_arm64/"
# the library has no SONAME; give it one, or bionic cannot match the LITERT_LM_C_API_1.0 version needs of the JNI library
python3 -m pip -q install --user patchelf && "$HOME/.local/bin/patchelf" --set-soname liblitert-lm.so "$OUT/android_arm64/liblitert-lm.so"
mkdir -p "$OUT/jniLibs/arm64-v8a" && cp "$OUT"/android_arm64/*.so "$OUT/jniLibs/arm64-v8a/"
file "$OUT"/*/liblitert-lm.so
echo "LiteRT-LM runtime in $OUT (set litertDir=$OUT in local.properties)"
