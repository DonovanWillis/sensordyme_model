#!/usr/bin/env bash
# XCUITest journeys: the real iPhone app (iOS Simulator) against the real backend
# and the virtual device, with faults injected by scripts/sim/uitest_control.py.
#
#   scripts/sim/ios_uitest.sh [OUT_DIR] [SIMULATOR_UDID] [-only-testing:...]
#
# OUT_DIR gets summary.md (pass/fail per journey), shots/ (a PNG per journey
# step and the accessibility audit reports), xcodebuild.log, the stack's logs
# and the result bundle. Build products go in OUT_DIR/dd and are deleted at the
# end (KEEP_DD=1 keeps them; REUSE_BUILD=1 then skips the build on the next run).
# Each xcodebuild starts only with more than 1.8 GB free. Diagnostics collection
# is off: XCTest's own accessibility tracing otherwise wrote ~600 MB into the
# result bundle and filled the disk. Simulated device; no camera, no network
# beyond 127.0.0.1.
#
# Ad-hoc signed ("sign to run locally"): an unsigned Simulator build has no
# Keychain, and the app keeps its session there.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-$(mktemp -d)/ios-uitest}"
SIM="${2:-E72547BB-CBFE-4CFF-93A4-39948F3F8705}"
ONLY=("${@:3}")
PY="${SIM_PY:-$HOME/.venvs/sensordyme-m01/bin/python}"
mkdir -p "$OUT/shots"
OUT="$(cd "$OUT" && pwd)"

need_space() {
  local free_kb; free_kb=$(df -k / | awk 'NR==2 {print $4}')
  if [ "$free_kb" -lt 1887437 ]; then
    echo "only $((free_kb / 1024)) MB free; $1 needs more than 1.8 GB. Stopping."; exit 3
  fi
}
need_space "this run"
for port in 8787 8790 8797; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then echo "port $port is in use"; exit 3; fi
done

cleanup() {
  curl -sf -X POST http://127.0.0.1:8790/shutdown >/dev/null 2>&1 || true
  [ -n "${CTL:-}" ] && wait "$CTL" 2>/dev/null || true
  xcrun simctl shutdown "$SIM" 2>/dev/null || true
  [ "${KEEP_DD:-0}" = 1 ] || rm -rf "$OUT/dd"
  rm -rf ios/SensorDyme.xcodeproj ios/Info.plist
}
trap cleanup EXIT

echo "== backend + virtual device + fault proxy"
# Fresh data every run: a backend kept from an earlier run already has the device
# claimed, and the discovery journey (claiming it through Add device) then skips.
rm -rf "$OUT/stack"
"$PY" scripts/sim/uitest_control.py --work "$OUT/stack" --python "$PY" > "$OUT/control.out" 2>&1 &
CTL=$!
for _ in $(seq 120); do curl -sf http://127.0.0.1:8790/state >/dev/null && break; sleep 0.5; done
curl -sf http://127.0.0.1:8790/state >/dev/null || { cat "$OUT/control.out"; exit 1; }

echo "== build for testing (Debug, ad-hoc signed)"
(cd ios && xcodegen generate >/dev/null)
xcrun simctl boot "$SIM" 2>/dev/null || true
xcrun simctl bootstatus "$SIM" -b >/dev/null
COMMON=(-project ios/SensorDyme.xcodeproj -scheme SensorDyme -destination "id=$SIM" -derivedDataPath "$OUT/dd"
        CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM=)
if [ "${REUSE_BUILD:-0}" = 1 ] && ls "$OUT"/dd/Build/Products/*.xctestrun >/dev/null 2>&1; then
  echo "   (reusing the build in $OUT/dd)"
else
  need_space "the build"
  xcodebuild "${COMMON[@]}" build-for-testing > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
fi

echo "== journeys"
rm -rf "$OUT/result.xcresult"
need_space "the test run"
set +e
TEST_RUNNER_ORYN_UITEST_SHOTS="$OUT/shots" TEST_RUNNER_ORYN_UITEST_ARGS="${ORYN_UITEST_ARGS:-}" \
  xcodebuild "${COMMON[@]}" test-without-building -collect-test-diagnostics never -resultBundlePath "$OUT/result.xcresult" ${ONLY[@]+"${ONLY[@]}"} \
  > "$OUT/xcodebuild.log" 2>&1
rc=$?
set -e
cp "$OUT"/stack/*.log "$OUT/" 2>/dev/null || true

python3 - "$OUT" <<'PY'
import re, sys
out = sys.argv[1]
log = open(f"{out}/xcodebuild.log").read()
rows = re.findall(r"Test [Cc]ase '-\[\S+ (test\w+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)", log)
fails = {}
for name, where, msg in re.findall(r"error: -\[\S+ (test\w+)\] : (?:(\S+:\d+): )?(.*)", log):
    fails.setdefault(name, msg.strip()[:160])
for m in re.finditer(r"/([\w]+\.swift):(\d+): error: -\[\S+ (test\w+)\] : (.*)", log):
    fails.setdefault(m.group(3), f"{m.group(1)}:{m.group(2)} {m.group(4).strip()[:160]}")
lines = ["| Journey | Result | Seconds | First failure |", "|---|---|---|---|"]
for name, result, secs in rows:
    lines.append(f"| {name} | {result.upper()} | {float(secs):.0f} | {fails.get(name, '') if result == 'failed' else ''} |")
passed = sum(r == "passed" for _, r, _ in rows)
summary = "\n".join(lines) + f"\n\n{passed}/{len(rows)} passed\n"
open(f"{out}/summary.md", "w").write(summary)
print(summary)
PY
echo "screenshots and audit reports in $OUT/shots; full log $OUT/xcodebuild.log"
exit $rc
