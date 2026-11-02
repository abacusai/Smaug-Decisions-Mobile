#!/bin/bash
# Mac: drive the Cascade app on a connected iPhone with devicectl.
#   tools/phone.sh run <name> <launch args...>   launch, wait for Documents/autorun_summary.json, pull Documents/*.json to work/phone/<name>
#   tools/phone.sh ls                            list the app's Documents
#   tools/phone.sh put <local file>              copy a file into the app's Documents
# env: DEV (device id from `xcrun devicectl list devices`, required), APP (bundle id, default ai.abacus.Cascade), W (output dir)
set -uo pipefail
DEV=${DEV:?set DEV to the iPhone id from xcrun devicectl list devices}; APP=${APP:-ai.abacus.Cascade}
W=${W:-$(cd "$(dirname "$0")/.." && pwd)/build/phone}
dc() { xcrun devicectl "$@" 2>&1; }
case "$1" in
  ls) dc device info files --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --subdirectory Documents | tail -n +1 ;;
  put) dc device copy to --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source "$2" --destination "Documents/$(basename "$2")" | tail -3 ;;
  rm) dc device copy to --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source /dev/null --destination "Documents/$2" >/dev/null ;;
  run)
    name=$2; shift 2
    mkdir -p "$W/$name"; rm -f "$W/$name"/*
    # clear the previous run's results so we can wait for this run's summary
    for f in autorun_summary.json state.json; do : > /tmp/empty.json; dc device copy to --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source /tmp/empty.json --destination "Documents/$f" >/dev/null; done
    t0=$(date +%s)
    dc device process launch --device "$DEV" --terminate-existing "$APP" "$@" | grep -i -E "error|launched" | head -3
    for _ in $(seq 1 ${WAIT_LOOPS:-150}); do
      sleep 8
      dc device copy from --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source Documents/autorun_summary.json --destination "$W/$name/autorun_summary.json" >/dev/null
      [ -s "$W/$name/autorun_summary.json" ] && break
    done
    echo "waited $(( $(date +%s) - t0 )) s"
    case " $* " in *" --snapshot "*) sleep 8
      for f in $(dc device info files --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --subdirectory Documents | grep -o -E "snapshot_[A-Za-z0-9_]+\.png" | sort -u); do
        dc device copy from --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source "Documents/$f" --destination "$W/$name/$f" >/dev/null; done ;; esac
    for f in $(dc device info files --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --subdirectory Documents | grep -o -E "(autorun_[a-z0-9]+|state)\.json" | sort -u); do
      dc device copy from --device "$DEV" --domain-type appDataContainer --domain-identifier "$APP" --source "Documents/$f" --destination "$W/$name/$f" >/dev/null
    done
    ls "$W/$name" ;;
esac
