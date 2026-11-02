#!/usr/bin/env bash
# Builds the LiteRT-LM C API as CLiteRTLM.xcframework (iOS device arm64 + iOS Simulator arm64) WITH our harness patch
# (multimodal raw-session prefill + absolute letter logits) and puts it, with GemmaModelConstraintProvider.xcframework, in
# Cascade/Frameworks, where ../build.sh setup and the Xcode project expect them.
#
#   runtime/build_runtime_ios.sh              # patched source build (default; ~30-60 min cold, ~20 GB of bazel cache)
#   runtime/build_runtime_ios.sh --prebuilt   # fallback: Google's stock prebuilt v0.18.1 xcframework (text only, one-answer)
#
# macOS only. Needs a full Xcode (not just the Command Line Tools), git, bazelisk and Rust (cargo):
#   brew install bazelisk rustup && rustup-init -y
# Env: WORK=<dir> (LiteRT-LM checkout, default ./.build/LiteRT-LM), JOBS=<n> (bazel --jobs), BAZEL_FLAGS="<extra build flags>",
#      BAZEL_STARTUP_FLAGS="--output_user_root=<dir>" (keep bazel's cache somewhere specific), SLICES=sim (simulator slice only:
#      half the build time and disk), RUSTUP_IOS_TARGETS=1 (also `rustup target add` the iOS Rust targets).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMMIT=2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4   # google-ai-edge/LiteRT-LM main 2026-10-28 == v0.19.0 (PyPI litert-lm-api 0.19.0)
REPO=https://github.com/google-ai-edge/LiteRT-LM.git
WORK=${WORK:-$HERE/.build/LiteRT-LM}
OUT=$HERE/../Cascade/Frameworks
FW=$OUT/CLiteRTLM.xcframework
# The binary the stock Package.swift at $COMMIT pins (its checksum is from that Package.swift; verified to match the download).
PREBUILT_URL=https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.18.1/CLiteRTLM.xcframework.zip
PREBUILT_SHA256=5c51905e38c77ccc611dc2f8c2af6cf5144621046f1dde6518aab83f78488937
# libGemmaModelConstraintProvider.dylib: a prebuilt dylib (Git LFS in the repo) that the source-built CLiteRTLM links via
# @rpath; it is packaged as its own xcframework so Xcode embeds and signs the right slice. sha256 per slice at $COMMIT:
# shellcheck disable=SC2034  # read indirectly as ${!sha}
CONSTRAINT_SHA256_ios_arm64=ac29411811aece37119f1546d7cfa5b5aa411a3c6c64db2c65bfa633f8120762
# shellcheck disable=SC2034
CONSTRAINT_SHA256_ios_sim_arm64=a51a1a8348e924de4af18bd2948f1cced3be284dc8556a28340745ddf8baeb80
HARNESS_SYMBOLS=(_litert_lm_experimental_session_run_prefill_multimodal _litert_lm_experimental_get_last_scoring_logits)

die() { echo "error: $*" >&2; exit 1; }
need() { for t in "$@"; do command -v "$t" >/dev/null || die "missing '$t' (see the header of this script)"; done; }

# Packages the two prebuilt libGemmaModelConstraintProvider.dylib slices as Frameworks/GemmaModelConstraintProvider.xcframework
# (the app project embeds it in both modes; the stock prebuilt CLiteRTLM simply does not load it).
make_constraint_xcframework() {
  local dir slice dst sha
  dir=$(mktemp -d)
  for slice in ios_arm64 ios_sim_arm64; do
    mkdir -p "$dir/$slice"
    dst=$dir/$slice/libGemmaModelConstraintProvider.dylib
    local src=$WORK/prebuilt/$slice/libGemmaModelConstraintProvider.dylib
    if [[ -f $src ]] && ! head -c 40 "$src" | grep -q '^version https://git-lfs'; then
      cp "$src" "$dst"
    else
      curl -fsSL --retry 3 -o "$dst" \
        "https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/$COMMIT/prebuilt/$slice/libGemmaModelConstraintProvider.dylib"
    fi
    sha=CONSTRAINT_SHA256_$slice
    [[ $(shasum -a 256 "$dst" | cut -d' ' -f1) == "${!sha}" ]] || die "sha256 mismatch for $slice libGemmaModelConstraintProvider.dylib"
  done
  rm -rf "$OUT/GemmaModelConstraintProvider.xcframework" && mkdir -p "$OUT"
  xcodebuild -create-xcframework -library "$dir/ios_arm64/libGemmaModelConstraintProvider.dylib" \
    -library "$dir/ios_sim_arm64/libGemmaModelConstraintProvider.dylib" \
    -output "$OUT/GemmaModelConstraintProvider.xcframework" >/dev/null
  rm -rf "$dir"
  echo "packaged $OUT/GemmaModelConstraintProvider.xcframework"
}

install_prebuilt() {
  need curl unzip shasum
  local zip
  zip="$(mktemp -d)/CLiteRTLM.xcframework.zip"
  echo "Downloading the stock prebuilt CLiteRTLM v0.18.1 ..."
  curl -fL --retry 3 -o "$zip" "$PREBUILT_URL"
  echo "$PREBUILT_SHA256  $zip" | shasum -a 256 -c -
  rm -rf "$FW" && mkdir -p "$OUT"
  unzip -q "$zip" -d "$OUT" && rm -f "$zip"
  xattr -dr com.apple.quarantine "$FW" 2>/dev/null || true
  make_constraint_xcframework
  echo
  echo "Installed the STOCK runtime (pre-1.0 C API, no harness patch): $FW"
  echo "The app runs text questions, one-answer only (one prefill per option; softcap cannot be applied, so the top"
  echo "answer matches PyTorch but the probabilities come out sharper). Images, audio and select-all need the patched build."
}

checkout() {
  if [[ ! -d $WORK/.git ]]; then
    echo "Fetching LiteRT-LM @ ${COMMIT:0:7} into $WORK ..."
    mkdir -p "$WORK"
    git -C "$WORK" init -q
    git -C "$WORK" remote add origin "$REPO"
  fi
  if [[ $(git -C "$WORK" rev-parse HEAD 2>/dev/null) != "$COMMIT" ]]; then
    git -C "$WORK" fetch -q --depth 1 origin "$COMMIT"
    git -C "$WORK" checkout -q --detach FETCH_HEAD
  fi
  # The iOS slices link the prebuilt constraint-provider dylibs, which are Git LFS objects (pointer files in a plain
  # checkout). Fetch them over HTTPS from GitHub's LFS media host and check each against its pointer's sha256; this needs
  # no git-lfs and works when git's URL is rewritten to ssh (where the LFS batch API fails).
  local f oid tmp
  for f in "$WORK"/prebuilt/{ios_arm64,ios_sim_arm64,macos_arm64}/*.dylib; do
    head -c 40 "$f" | grep -q '^version https://git-lfs' || continue
    oid=$(sed -n 's/^oid sha256://p' "$f")
    tmp=$f.download
    curl -fsSL --retry 3 -o "$tmp" "https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/$COMMIT/${f#"$WORK"/}"
    [[ $(shasum -a 256 "$tmp" | cut -d' ' -f1) == "$oid" ]] || die "sha256 mismatch for ${f#"$WORK"/}"
    mv "$tmp" "$f" && echo "fetched LFS object ${f#"$WORK"/}"
  done
}

# Applies a patch once; a re-run sees it already applied and skips it.
apply_once() {
  local patch=$1
  if git -C "$WORK" apply --check "$patch" 2>/dev/null; then
    git -C "$WORK" apply "$patch" && echo "applied $(basename "$patch")"
  elif git -C "$WORK" apply -R --check "$patch" 2>/dev/null; then
    echo "already applied: $(basename "$patch")"
  else
    die "$(basename "$patch") does not apply to $WORK (is it at ${COMMIT:0:7}?)"
  fi
}

build_patched() {
  [[ $(xcode-select -p) == *Xcode*.app/* ]] || die "select a full Xcode: sudo xcode-select -s /Applications/Xcode.app"
  need git curl shasum bazelisk cargo unzip nm otool
  # Note: llguidance (Rust) is in the dep graph; only if rules_rust uses the host toolchain does it need the iOS std libs.
  [[ ${RUSTUP_IOS_TARGETS:-} == 1 ]] && rustup target add aarch64-apple-ios aarch64-apple-ios-sim
  checkout
  # 1) the harness patch exactly as built and verified on Linux (c/conversation.cc, scoring_cpu_util, //c:litert-lm deps)
  apply_once "$HERE/patches/litert_lm_harness.patch"
  # 2) iOS additions: the same deps on //c:engine (+ //c:engine_cpu), which //swift:CLiteRTLM links, and
  #    public declarations in c/experimental.h (+ the LITERT_LM_HARNESS_PATCH macro the app's C shim keys on)
  apply_once "$HERE/patches/ios_engine_exports.patch"

  cd "$WORK"
  git checkout -q swift/BUILD
  if [[ ${SLICES:-} == sim ]]; then   # drop the device slice from //swift:CLiteRTLM for a faster Simulator-only build
    perl -0pi -e 's/"device": \[\s*"arm64",\s*\],//' swift/BUILD
    ! grep -q '"device"' swift/BUILD || die "could not drop the device slice from swift/BUILD"
  fi
  # //swift:CLiteRTLM is the repo's own apple_xcframework (swift/BUILD): ios = {device: [arm64], simulator: [arm64]},
  # minimum iOS 15. rules_apple transitions to both slices itself. The config the official build uses is not
  # in the repo; --config=ios_arm64 (.bazelrc: ios copts, c++20, cpu/platform) is tried first, then plain --config=ios.
  local flags=(-c opt --java_runtime_version=remotejdk_17)
  [[ -n ${JOBS:-} ]] && flags+=("--jobs=$JOBS")
  # shellcheck disable=SC2206  # BAZEL_FLAGS is intentionally word-split
  [[ -n ${BAZEL_FLAGS:-} ]] && flags+=(${BAZEL_FLAGS})
  local startup=()
  # shellcheck disable=SC2206  # BAZEL_STARTUP_FLAGS is intentionally word-split
  [[ -n ${BAZEL_STARTUP_FLAGS:-} ]] && startup=(${BAZEL_STARTUP_FLAGS})
  local cfg built=""
  for cfg in ios_arm64 ios; do
    echo "bazelisk ${startup[*]:-} build ${flags[*]} --config=$cfg //swift:CLiteRTLM"
    if bazelisk ${startup[@]+"${startup[@]}"} build "${flags[@]}" "--config=$cfg" //swift:CLiteRTLM; then built=$cfg; break; fi
    echo "build with --config=$cfg failed; trying the next config" >&2
  done
  [[ -n $built ]] || die "patched build failed. Fallback: $0 --prebuilt  (text-only, one-answer)"

  local artifact
  artifact=$(bazelisk ${startup[@]+"${startup[@]}"} cquery "${flags[@]}" "--config=$built" --output=files //swift:CLiteRTLM 2>/dev/null | grep -m1 'xcframework')
  [[ -n $artifact ]] || die "no xcframework output found for //swift:CLiteRTLM"
  rm -rf "$FW" && mkdir -p "$OUT"
  if [[ $artifact == *.zip ]]; then
    unzip -q -o "$artifact" -d "$OUT"
  else
    cp -R "$artifact" "$FW"
  fi
  [[ -d $FW ]] || die "expected $FW after unpacking $artifact"
  chmod -R u+w "$FW"
  make_constraint_xcframework
  verify_patched
}

verify_patched() {
  local slice bin sym ok=1
  grep -q LITERT_LM_HARNESS_PATCH "$FW"/*/CLiteRTLM.framework/Headers/experimental.h || { echo "!! experimental.h lacks the harness declarations"; ok=0; }
  for slice in "$FW"/*/CLiteRTLM.framework; do
    bin=$slice/CLiteRTLM
    [[ -f $bin ]] || { bin=$(find "$slice" -maxdepth 1 -type f -perm -u+x | head -1); }
    echo "-- $(basename "$(dirname "$slice")"): $(lipo -archs "$bin" 2>/dev/null || echo '?') $(file -b "$bin" | cut -c1-60)"
    for sym in "${HARNESS_SYMBOLS[@]}"; do
      # (no `nm | grep -q`: with pipefail, grep -q closing the pipe early makes nm fail with SIGPIPE)
      if [[ $(nm -gU "$bin" 2>/dev/null | grep -c " $sym\$") -ge 1 ]]; then echo "   exports $sym"; else echo "!! $sym NOT exported"; ok=0; fi
    done
    if [[ $(otool -L "$bin" | grep -c libGemmaModelConstraintProvider) -ge 1 ]]; then
      echo "   links @rpath/libGemmaModelConstraintProvider.dylib (embedded via GemmaModelConstraintProvider.xcframework)"
    fi
  done
  echo
  if [[ $ok == 1 ]]; then echo "OK: patched CLiteRTLM.xcframework at $FW"; else die "built, but the checks above failed: $FW"; fi
}

[[ $(uname -s) == Darwin ]] || die "run this on macOS (it builds Apple frameworks with Xcode)"
case ${1:-} in
  --prebuilt) install_prebuilt ;;
  "") build_patched ;;
  *) die "usage: $0 [--prebuilt]" ;;
esac
echo "Next: ../build.sh setup"
