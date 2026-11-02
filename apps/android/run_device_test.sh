#!/usr/bin/env bash
# One-command phone test: builds the APK for the attached device's ABI (+ the instrumentation APK), installs both, pushes the
# GGUF + mmproj into the app's external files dir (skipped when already there with the same size), runs DeviceSelfTest (Smaug-Decisions:
# replays assets/selftest: max |dp| vs Python, prompt identity, latency, peak memory) and DeviceCascadeTest (the whole cascade on
# assets/cascade_selftest with the device's own tier) and pulls both reports.
#
# usage: run_device_test.sh <model.litertlm> [options]            (LiteRT-LM build, the default)
#        run_device_test.sh <model.gguf> <mmproj.gguf> [options]  (llama.cpp fallback build, -PbigBackend=llamacpp)
#   -s SERIAL     adb device serial (else $ANDROID_SERIAL / the only device)
#   --threads N   llama.cpp threads (default: min(4, cores))
#   --repeat N    run each case N times, latency = median (default 1)
#   --only IDS    comma-separated case ids (e.g. c00,c11,c18)
#   --max-dp X    fail threshold for max |dp| vs Python (default 0.1; arm64 and x86 Q4 kernels differ by a few 1e-2)
#   --no-build    reuse the APKs already in app/build/outputs
#   --out DIR     where to put the report (default build/device-test/<timestamp>)
# Needs: adb, a JDK 17+ and the Android SDK/NDK for the build (see README), a 12 GB+ RAM arm64 phone (8 GB is marginal).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { sed -n '2,16p' "$0"; exit 2; }
case $1 in
    *.litertlm) MODEL=$1; MMPROJ=""; LITERT=1; shift 1 ;;
    *) MODEL=$1; MMPROJ=$2; LITERT=0; shift 2 ;;
esac
ADB=${ADB:-adb}; THREADS=""; REPEAT=1; ONLY=""; MAXDP=0.1; BUILD=1; OUT=""
while [ $# -gt 0 ]; do
    case $1 in
        -s) ADB="$ADB -s $2"; shift 2 ;;
        --threads) THREADS=$2; shift 2 ;;
        --repeat) REPEAT=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        --max-dp) MAXDP=$2; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        --out) OUT=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done
for f in "$MODEL" $MMPROJ; do [ -f "$f" ] || { echo "missing $f"; exit 2; }; done
PKG=ai.abacus.decide
OUT=${OUT:-$HERE/build/device-test/$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
if [ $LITERT = 1 ]; then BUNDLE=$HERE/app/src/main/assets/selftest_litert/expected.json; BACKEND=litert
else BUNDLE=$HERE/app/src/main/assets/selftest/expected.json; BACKEND=llamacpp; fi
name_of() { sed -n "s/^ *\"$1\": \"\([^\"]*\)\".*/\1/p" "$BUNDLE" | head -1; }
GGUF_NAME=$(name_of gguf); MMPROJ_NAME=$(name_of mmproj)
[ "$(basename "$MODEL")" = "$GGUF_NAME" ] || echo "note: pushing $(basename "$MODEL") as $GGUF_NAME (the name the bundle's expected values were made with)"

$ADB get-state >/dev/null
ABI=$($ADB shell getprop ro.product.cpu.abi | tr -d '\r')
MEM_KB=$($ADB shell cat /proc/meminfo | tr -d '\r' | awk '/MemTotal/ {print $2}')
echo "device: $($ADB shell getprop ro.product.model | tr -d '\r'), $ABI, $((MEM_KB / 1024)) MB RAM"
[ "$MEM_KB" -ge 4500000 ] || echo "note: under 4.5 GB RAM: the app runs the fast tier only (DevicePolicy FAST_ONLY)"
case $ABI in arm64-v8a|x86_64) ;; *) echo "unsupported ABI $ABI (built: arm64-v8a, x86_64)"; exit 1 ;; esac

if [ $BUILD = 1 ]; then (cd "$HERE" && ./gradlew assembleDebug assembleDebugAndroidTest -Pabis="$ABI" -PbigBackend=$BACKEND); fi
APK=$HERE/app/build/outputs/apk/debug/app-$ABI-debug.apk
TAPK=$HERE/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
$ADB install -r -t "$APK"
$ADB install -r -t "$TAPK"

# let the app create its external files dir, then push the models there
$ADB shell am start -W -n $PKG/.MainActivity >/dev/null
DEV_DIR=/sdcard/Android/data/$PKG/files
$ADB shell mkdir -p $DEV_DIR
push() {  # local, remote name
    local size; size=$(wc -c < "$1" | tr -d ' ')
    local rsize; rsize=$($ADB shell stat -c %s "$DEV_DIR/$2" 2>/dev/null | tr -d '\r' || true)
    if [ "$size" = "$rsize" ]; then echo "$2 already on device"; else $ADB push "$1" "$DEV_DIR/$2"; fi
}
push "$MODEL" "$GGUF_NAME"
[ -n "$MMPROJ" ] && push "$MMPROJ" "$MMPROJ_NAME"
$ADB shell am force-stop $PKG
$ADB shell rm -f $DEV_DIR/selftest-report.json $DEV_DIR/cascade-report.json
$ADB logcat -c || true

ARGS="-e class $PKG.DeviceSelfTest,$PKG.DeviceCascadeTest -e repeat $REPEAT -e max_dp $MAXDP"
[ -n "$THREADS" ] && ARGS="$ARGS -e threads $THREADS"
[ -n "$ONLY" ] && ARGS="$ARGS -e only $ONLY"
echo "running the self-test (several minutes on a phone CPU)..."
set +e
$ADB shell am instrument -w -r $ARGS $PKG.test/androidx.test.runner.AndroidJUnitRunner | tr -d '\r' | tee "$OUT/instrument.txt"
set -e
$ADB logcat -d -s decide-selftest:I decide-cascade:I decide:I | tr -d '\r' > "$OUT/logcat.txt" || true
$ADB pull $DEV_DIR/selftest-report.json "$OUT/" >/dev/null 2>&1 || echo "no Smaug-Decisions report on the device (crash or OOM? see $OUT/logcat.txt)"
$ADB pull $DEV_DIR/cascade-report.json "$OUT/" >/dev/null 2>&1 || echo "no cascade report on the device (see $OUT/logcat.txt)"
grep -h "decide-selftest" "$OUT/logcat.txt" | sed 's/.*decide-selftest[^:]*: //' | tail -n 40 || true
echo "report: $OUT/selftest-report.json"
grep -q "^OK (2 tests)" "$OUT/instrument.txt"
