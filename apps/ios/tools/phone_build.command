#!/bin/bash
# Build + sign the Cascade app for a connected iPhone. Run it from the Mac's GUI session (double-click, or `open` it): codesign
# needs the login keychain, which an ssh session cannot unlock (errSecInternalComponent).
#   TEAM=<your team id> XCDEST=<xcodebuild device id> BUNDLE=<a bundle id your team may sign> open tools/phone_build.command
# (open passes the environment of a GUI session; export the variables in ~/.zprofile or edit the defaults below)
# A free (personal) team rejects extended-virtual-addressing, so CascadePersonalTeam.entitlements keeps only
# increased-memory-limit.
cd "$(dirname "$0")/../Cascade" || exit 1
LOG=${LOG:-$PWD/build/phone_build.log}; mkdir -p "$(dirname "$LOG")"
TEAM=${TEAM:?set TEAM to your Apple development team id}; XCDEST=${XCDEST:?set XCDEST to the iPhone id from xcodebuild -showdestinations}
BUNDLE=${BUNDLE:-ai.abacus.Cascade}
{
  date
  xcodebuild build -project Cascade.xcodeproj -scheme Cascade -configuration Release -destination "id=$XCDEST" \
    -allowProvisioningUpdates DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE" \
    CODE_SIGN_ENTITLEMENTS=CascadePersonalTeam.entitlements -derivedDataPath build/phone 2>&1 | grep -E "rror:|BUILD (SUCCEEDED|FAILED)|Signing Identity|Provisioning Profile"
  echo "EXIT ${PIPESTATUS[0]}"
} > "$LOG" 2>&1
