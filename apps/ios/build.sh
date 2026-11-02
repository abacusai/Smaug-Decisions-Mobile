#!/bin/bash
# Build, test and run the Cascade app on a Mac (Xcode 16+, xcodegen). Everything big is APFS-cloned (cp -c), not copied.
# usage: ./build.sh setup | test | sim | install | run [launch args...] | device
#   setup    link the Smaug-Audio-Decisions-0.7B Core ML units + LiteRT-LM xcframeworks into the project and generate Cascade.xcodeproj
#   test     hostless logic tests (CascadeTests) in the Simulator
#   sim      build the app for the Simulator
#   install  install it on the booted Simulator and clone the Smaug-Decisions files into the app's Documents
#   run      launch it (e.g. ./build.sh run --autorun a1,a2,a3,a4,t1,i1 --tokcheck)
#   device   compile the device (arm64) app without signing; needs the device slice of CLiteRTLM.xcframework
# environment (defaults: the repository's models/ folder, as written by scripts/download.py):
#   UNITS    Smaug-Audio-Decisions-0.7B Core ML units (trunk_L512_int8.mlpackage, audio_10s_int8.mlpackage, embed_tokens_fp16.bin, head.json);
#            python scripts/download.py --model audio --variant coreml. setup bundles the embedding table as int8 + per-row scales
#            (embed_tokens_int8.bin, embed_tokens_int8_scales.bin), made from embed_tokens_fp16.bin by tools/quantize_embed.py
#            into $UNITS the first time (PYTHON: a python3 with numpy)
#   RUNTIME  dir with CLiteRTLM.xcframework + GemmaModelConstraintProvider.xcframework (runtime/build_runtime_ios.sh writes
#            them to Cascade/Frameworks, the default)
#   SMAUG_DECISIONS_DIR  dir with Smaug-Decisions-LiteRT files (the files Resources/models_manifest.json names, e.g. smaug-decisions-full-vis560.litertlm
#            for >= 8 GB phones, smaug-decisions-full-vis280.litertlm for 6 GB phones); install clones every .litertlm it finds.
#            python scripts/download.py --model litert --variant full-vis280 (Simulator only: on a phone the app downloads it)
#   SIM      Simulator name (iPhone 17 Pro); BUNDLE_ID (ai.abacus.Cascade)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); P=$HERE/Cascade; R=$P/Cascade/Resources
REPO=$(cd "$HERE/../.." && pwd)
UNITS=${UNITS:-$REPO/models/Smaug-Audio-Decisions-0.7B/coreml}
RUNTIME=${RUNTIME:-$P/Frameworks}
SMAUG_DECISIONS_DIR=${SMAUG_DECISIONS_DIR:-$REPO/models/Smaug-Decisions-LiteRT}
SIM=${SIM:-iPhone 17 Pro}
export BUNDLE_ID=${BUNDLE_ID:-ai.abacus.Cascade} DEVELOPMENT_TEAM=${DEVELOPMENT_TEAM:-}
export PATH=/opt/homebrew/bin:$PATH
MODE=${1:-sim}; shift || true
cd "$P"
case "$MODE" in
  setup)
    if [ ! -e "$UNITS/embed_tokens_int8.bin" ]; then
      [ -e "$UNITS/embed_tokens_fp16.bin" ] || { echo "missing $UNITS/embed_tokens_fp16.bin"; exit 1; }
      ${PYTHON:-python3} "$HERE/tools/quantize_embed.py" "$UNITS/embed_tokens_fp16.bin" "$UNITS"
    fi
    rm -f "$R/embed_tokens_fp16.bin"   # the app ships the int8 table; an fp16 copy in Resources would be bundled too
    for f in trunk_L512_int8.mlpackage audio_10s_int8.mlpackage embed_tokens_int8.bin embed_tokens_int8_scales.bin head.json; do
      [ -e "$UNITS/$f" ] || { echo "missing $UNITS/$f"; exit 1; }
      [ -e "$R/$f" ] || cp -cR "$UNITS/$f" "$R/"
    done
    mkdir -p Frameworks
    for f in CLiteRTLM.xcframework GemmaModelConstraintProvider.xcframework; do [ -e "Frameworks/$f" ] || cp -cR "$RUNTIME/$f" Frameworks/; done
    xcodegen generate --quiet && echo "generated $P/Cascade.xcodeproj ($(du -sh "$R" | cut -f1) resources)" ;;
  test)
    xcodebuild test -project Cascade.xcodeproj -scheme CascadeTests -destination "platform=iOS Simulator,name=$SIM" \
      -derivedDataPath build/test 2>&1 | grep -E "Test Case.*(passed|failed)|error:|Executed|TEST (SUCCEEDED|FAILED)" ;;
  sim)
    xcodebuild build -project Cascade.xcodeproj -scheme Cascade -configuration Debug -destination "platform=iOS Simulator,name=$SIM" \
      -derivedDataPath build/sim 2>&1 | grep -E "error:|warning:|BUILD (SUCCEEDED|FAILED)" | grep -v -e "appintents" -e "quoted-include-in-framework-header" | head -60 ;;
  install)
    xcrun simctl boot "$SIM" 2>/dev/null || true
    APP=build/sim/Build/Products/Debug-iphonesimulator/Cascade.app
    xcrun simctl install "$SIM" "$APP"
    DATA=$(xcrun simctl get_app_container "$SIM" "$BUNDLE_ID" data)
    for f in "$SMAUG_DECISIONS_DIR"/*.litertlm; do [ -e "$f" ] && { [ -e "$DATA/Documents/$(basename "$f")" ] || cp -c "$f" "$DATA/Documents/"; }; done
    # PARITY=1: the fp16 table in Documents, for --fast-parity (int8 vs fp16 on the same compute units)
    [ "${PARITY:-0}" = 1 ] && cp -c "$UNITS/embed_tokens_fp16.bin" "$DATA/Documents/" 2>/dev/null
    echo "installed $(du -sh "$APP" | cut -f1) app; Documents: $(ls "$DATA/Documents" | tr '\n' ' ')" ;;
  run)
    xcrun simctl terminate "$SIM" "$BUNDLE_ID" 2>/dev/null || true
    xcrun simctl launch "$SIM" "$BUNDLE_ID" "$@" ;;
  device)
    xcodebuild build -project Cascade.xcodeproj -scheme Cascade -configuration Release -destination 'generic/platform=iOS' \
      CODE_SIGNING_ALLOWED=NO -derivedDataPath build/dev 2>&1 | grep -E "error:|BUILD (SUCCEEDED|FAILED)" | head -40 ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
