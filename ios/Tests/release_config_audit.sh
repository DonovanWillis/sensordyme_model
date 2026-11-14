#!/usr/bin/env bash
# Prove, on the compiled product, that a Release build of ORYN EDGE carries no
# debug affordance (M8.5 launch-readiness audit).
#
#   bash ios/Tests/release_config_audit.sh [path/to/SensorDyme.app]
#
# Without an argument it builds Release for the iOS Simulator itself (ad-hoc
# signed, DerivedData in a temporary directory it deletes) and audits that.
#
# What is checked, and why it is provable rather than asserted:
#
#  1. No "-oryn" launch flag survives in the binary. Every debug launch hook
#     the app has (DebugLaunch.swift, BLESimulatedTransport.launchEndpoints)
#     spells its flag as a "-oryn…" string literal inside `#if DEBUG`. In a
#     Release build those literals must not exist, so the compiled binary is
#     the proof: if `strings` finds any "-oryn", DEBUG code leaked into Release.
#  2. BLESimulatedTransport does not exist in the product. The whole file is
#     `#if DEBUG`; its class name appears in the binary (type metadata) iff it
#     was compiled. CoreBluetoothTransport must be present — it is the only
#     transport a Release build can construct (BLETransports.make()).
#  3. MockAPI exists but is reachable only through AppState.backend when
#     isDemo is set, and the only Release path that sets it is the labelled
#     "Explore the demo device" button (AuthView). That is a product feature,
#     not a debug hook; this script checks the structural half (no flag strings
#     that could set it from a launch argument).
#  4. The app's Info.plist declares exactly the privacy strings setup needs
#     (Bluetooth, local network + both Bonjour service types) and no background
#     modes, and the UI-test bundle is not embedded in the app.
set -euo pipefail
cd "$(dirname "$0")/../.."

APP="${1:-}"
CLEANUP=""
if [ -z "$APP" ]; then
  free_kb=$(df -k / | awk 'NR==2 {print $4}')
  [ "$free_kb" -gt $((3072 * 1024)) ] || { echo "need >3 GB free to build"; exit 3; }
  DD="$(mktemp -d "${TMPDIR:-/tmp}/sd-release-audit.XXXXXX")"
  CLEANUP="$DD"
  trap '[ -n "$CLEANUP" ] && rm -rf "$CLEANUP" ios/SensorDyme.xcodeproj ios/Info.plist' EXIT
  (cd ios && xcodegen generate >/dev/null)
  xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme -configuration Release \
    -destination 'generic/platform=iOS Simulator' -derivedDataPath "$DD" build \
    CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= > "$DD/build.log" 2>&1 \
    || { tail -40 "$DD/build.log"; exit 1; }
  APP="$DD/Build/Products/Release-iphonesimulator/SensorDyme.app"
fi

BIN="$APP/SensorDyme"
[ -f "$BIN" ] || { echo "no app binary at $BIN"; exit 2; }

# Everything executable in the bundle. An Xcode Debug build puts the app's
# actual code in SensorDyme.debug.dylib and ships a stub launcher, so auditing
# only $BIN would wave a Debug build through; a Release build has no dylibs.
BINARIES=("$BIN")
while IFS= read -r dylib; do BINARIES+=("$dylib"); done \
  < <(find "$APP" -name '*.dylib' -type f 2>/dev/null)

status=0
fail() { echo "FAIL: $1"; status=1; }
pass() { echo "  ok: $1"; }

echo "== release config audit: $APP"
echo "   binaries: ${BINARIES[*]#$APP/}"

# 1. Debug launch flags: all of them begin "-oryn". (Dumped to a file first:
#    under pipefail, `strings | grep -q` turns a hit into SIGPIPE for strings,
#    and the check would call a dirty binary clean.)
STRS="$(mktemp)"
strings "${BINARIES[@]}" > "$STRS" || true
if grep -q -- '-oryn' "$STRS"; then
  grep -- '-oryn' "$STRS" | sed 's/^/    found: /'
  fail "debug launch flags present in the binary (DEBUG code compiled into Release)"
else
  pass "no -oryn launch flag in the binary"
fi

# 2. Transports: the simulated one absent, the real one present. Class names
#    live in mangled Swift symbols, so `nm` is the probe, not `strings`.
#    (Symbols land in a file first: `nm | grep -q` under pipefail reports nm's
#    SIGPIPE when grep stops early, and a hit then counted as a miss.)
SYMS="$(mktemp)"
nm "${BINARIES[@]}" 2>/dev/null > "$SYMS" || true
if grep -q 'BLESimulatedTransport' "$SYMS"; then
  fail "BLESimulatedTransport was compiled into this build"
else
  pass "BLESimulatedTransport absent"
fi
if grep -q 'CoreBluetoothTransport' "$SYMS"; then
  pass "CoreBluetoothTransport present (the only transport)"
else
  fail "CoreBluetoothTransport missing — wrong binary?"
fi
rm -f "$SYMS"

# 3. The simulated-BLE opt-in flag specifically (also covered by check 1).
if grep -q 'orynSimulatedBLE' "$STRS"; then
  fail "simulated-BLE launch flag present"
else
  pass "no simulated-BLE flag"
fi
rm -f "$STRS"

# 4. Info.plist inventory.
PLIST="$APP/Info.plist"
plval() { /usr/libexec/PlistBuddy -c "Print :$1" "$PLIST" 2>/dev/null; }
for key in NSBluetoothAlwaysUsageDescription NSLocalNetworkUsageDescription; do
  if [ -n "$(plval $key)" ]; then pass "$key declared"; else fail "$key missing"; fi
done
bonjour="$(plval NSBonjourServices | tr -d ' \n')"
case "$bonjour" in
  *_sensordyme._tcp*_sensordyme-api._tcp*|*_sensordyme-api._tcp*_sensordyme._tcp*)
    pass "NSBonjourServices lists both service types" ;;
  *) fail "NSBonjourServices incomplete: $bonjour" ;;
esac
if [ -n "$(plval UIBackgroundModes)" ]; then
  fail "unexpected UIBackgroundModes declared"
else
  pass "no background modes"
fi

# 5. Nothing test-only rides along in the app bundle.
if [ -d "$APP/PlugIns" ]; then
  fail "PlugIns embedded in the app: $(ls "$APP/PlugIns")"
else
  pass "no embedded test bundle"
fi

[ $status -eq 0 ] && echo "== release config audit: PASS" || echo "== release config audit: FAIL"
exit $status
