#!/usr/bin/env bash
# The real iPhone app (iOS Simulator) against the real backend and the virtual device.
#
#   scripts/sim/ios_e2e.sh [OUT_DIR] [SIMULATOR_UDID]
#
# Starts cloud/app.py (sim.backend, throwaway data) and a virtual SensorDyme
# (sim.virtual_device, scenario "attentive"), signs in with a real /v1/login and
# claims the device with its factory claim code, builds the app (Debug, ad-hoc
# "sign to run locally" so the Keychain works), and drives it with DEBUG-only
# launch arguments (ios/Sources/DebugLaunch.swift), saving a screenshot per step:
#   ready -> stopped (POST /protection) -> Ask Jupiter dwell prompt -> unsupported
#   prompt refused -> device offline (status unknown) -> device restarted (ready).
# Everything it creates (build products, generated project, data dirs, app
# install) is removed at the end. Simulated device; no camera footage.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-$(mktemp -d)/ios-e2e}"; SIM="${2:-$(xcrun simctl list devices available | awk -F'[()]' '/iPhone/ {print $2; exit}')}"
PY="${SIM_PY:-python3}"; WORK="$(mktemp -d)"; mkdir -p "$OUT"
API=http://127.0.0.1:8787; APP=com.sensordyme.app
cleanup() {
  pkill -f "sim.virtual_device --backend $API" 2>/dev/null || true
  pkill -f "sim.backend --port 8787" 2>/dev/null || true
  xcrun simctl uninstall "$SIM" "$APP" 2>/dev/null || true
  xcrun simctl shutdown "$SIM" 2>/dev/null || true
  rm -rf "$WORK" ios/SensorDyme.xcodeproj ios/Info.plist
}
trap cleanup EXIT
shot() { sleep "${2:-12}"; xcrun simctl io "$SIM" screenshot "$OUT/$1.png" >/dev/null; echo "  screenshot $1"; }
launch() { xcrun simctl launch --terminate-running-process "$SIM" "$APP" -orynNoPrompt "$@" >/dev/null; }
api() { curl -sf "$@" -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json'; }
device() { PYTHONDONTWRITEBYTECODE=1 "$PY" -m sim.virtual_device --backend "$API" --factory "$WORK/backend/factory.json" \
             --state-dir "$WORK/device" --scenario attentive --no-dashboard >> "$WORK/device.log" 2>&1 & }

echo "== backend + virtual device"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m sim.backend --port 8787 --data-dir "$WORK/backend" > "$WORK/backend.log" 2>&1 &
for _ in $(seq 60); do curl -sf "$API/v1/health" >/dev/null && break; sleep 0.5; done
device
CLAIM=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['devices'][0]['claim_code'])" "$WORK/backend/factory.json")
TOKEN=$(curl -sf -X POST "$API/v1/login" -H 'Content-Type: application/json' \
          -d '{"email":"demo@sensordyme.local","password":"sensordyme-demo"}' | "$PY" -c "import sys,json;print(json.load(sys.stdin)['token'])")
api -X POST "$API/v1/devices/claim" -d "{\"claim_code\":\"$CLAIM\"}" >/dev/null
sleep 20

echo "== build and install (Debug, ad-hoc signed for the Simulator)"
(cd ios && xcodegen generate >/dev/null && xcodebuild -scheme SensorDyme -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' -derivedDataPath "$WORK/dd" \
  CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= build | grep -E "BUILD (SUCCEEDED|FAILED)")
xcrun simctl boot "$SIM" 2>/dev/null || true; xcrun simctl bootstatus "$SIM" -b >/dev/null
xcrun simctl install "$SIM" "$WORK/dd/Build/Products/Debug-iphonesimulator/SensorDyme.app"

echo "== journeys"
launch -orynSession "$TOKEN" demo@sensordyme.local -orynOpenDevice; shot 01_ready 15
api -X POST "$API/v1/devices/1/protection" -d '{"action":"stop","confirm":true}' >/dev/null; sleep 6
launch -orynOpenDevice; shot 02_stopped
api -X POST "$API/v1/devices/1/protection" -d '{"action":"resume"}' >/dev/null; sleep 4
launch -orynOpenDevice -orynOpenJupiter -orynPrompt "Alert me if someone stays inside the marked area for longer than 10 seconds"; shot 03_dwell_prompt 16
launch -orynOpenDevice -orynOpenJupiter -orynPrompt "Alert me when a stranger I don't recognize walks in"; shot 04_unsupported_prompt 16
pkill -f "sim.virtual_device --backend $API"; sleep 38
launch -orynOpenDevice; shot 05_offline
device; sleep 25
launch -orynOpenDevice; shot 06_recovered
cp "$WORK"/*.log "$OUT/" 2>/dev/null || true
echo "screenshots in $OUT"
