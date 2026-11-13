#!/usr/bin/env bash
# XCUITest journeys: the real iPhone app (iOS Simulator) against the real backend
# and the virtual device, with faults injected by scripts/sim/uitest_control.py.
#
#   scripts/sim/ios_uitest.sh [OUT_DIR] [SIMULATOR_UDID] [-only-testing:...]
#
# The whole suite (every journey and both accessibility audits) runs in ONE
# xcodebuild invocation, one test at a time, with no retries
# (-retry-tests-on-failure and -test-iterations are never passed: a flaky test
# must fail, not pass on its second try). Every test starts from its own fresh
# backend, account and device (Stack.fresh, POST /fresh), so the order does not
# matter: ORDER=random runs it in XCTest's random order (TestExecutionOrdering
# in the .xctestrun); the order the tests actually ran in is the order of
# summary.md's table.
#
# OUT_DIR gets summary.md (pass/fail/skip per test, in execution order, and any
# test that never reported), shots/ (a PNG per journey step and the
# accessibility audit reports), xcodebuild.log, the stack's logs and the result
# bundle. The exit status is non-zero unless every test that ran passed and
# none is missing (a skip is not a pass).
#
# Build products go in OUT_DIR/dd and are deleted at the end (KEEP_DD=1 keeps
# them; REUSE_BUILD=1 then skips the build on the next run of the same commit).
# A build starts only with more than MIN_BUILD_MB (3072) free, a test run with
# more than 1.8 GB. Diagnostics collection is off: XCTest's own accessibility
# tracing otherwise wrote ~600 MB into the result bundle and filled the disk.
# Simulated device; no camera, no network beyond 127.0.0.1.
#
# Simulator state: the app and its test runner are uninstalled and the
# Simulator's keychain reset before the run, so a run never inherits a session,
# settings or a paired camera from an earlier one. Within the run, each test's
# launches start signed out (-orynSignOut) and sign in to that test's own new
# account (AppState.claimLocalState clears the previous account's phone state).
#
# Ad-hoc signed ("sign to run locally"): an unsigned Simulator build has no
# Keychain, and the app keeps its session there.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-$(mktemp -d)/ios-uitest}"
SIM="${2:-E72547BB-CBFE-4CFF-93A4-39948F3F8705}"
ONLY=("${@:3}")
PY="${SIM_PY:-$HOME/.venvs/sensordyme-m01/bin/python}"
ORDER="${ORDER:-alphabetical}"
MIN_BUILD_MB="${MIN_BUILD_MB:-3072}"
case "$ORDER" in alphabetical|random) ;; *) echo "ORDER must be alphabetical or random"; exit 2 ;; esac
mkdir -p "$OUT/shots"
OUT="$(cd "$OUT" && pwd)"

need_space() {  # need_space MB WHAT
  local free_kb; free_kb=$(df -k / | awk 'NR==2 {print $4}')
  if [ "$free_kb" -lt $(($1 * 1024)) ]; then
    echo "only $((free_kb / 1024)) MB free; $2 needs more than $1 MB. Stopping."; exit 3
  fi
}
need_space 1843 "this run"
for port in 8787 8790 8797; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then echo "port $port is in use"; exit 3; fi
done

cleanup() {
  curl -sf -X POST http://127.0.0.1:8790/shutdown >/dev/null 2>&1 || true
  [ -n "${CTL:-}" ] && wait "$CTL" 2>/dev/null || true
  # M8.3 security: the throwaway stack (backend DB, .jwt_secret, factory.json
  # with device keys and claim codes, device config) never lands in the
  # evidence folder; only its logs are kept.
  if [ -n "${STACK:-}" ] && [ -d "$STACK" ]; then
    find "$STACK" -name '*.log' -exec cp {} "$OUT/" \; 2>/dev/null || true
    rm -rf "$STACK"
  fi
  xcrun simctl shutdown "$SIM" 2>/dev/null || true
  [ "${KEEP_DD:-0}" = 1 ] || rm -rf "$OUT/dd"
  rm -rf ios/SensorDyme.xcodeproj ios/Info.plist
}
trap cleanup EXIT

echo "== backend + virtual device + fault proxy"
# Each test asks for its own fresh backend (POST /fresh); this one only has to
# be there for the controller to answer.
STACK="$(mktemp -d "${TMPDIR:-/tmp}/sd-uitest-stack.XXXXXX")"
"$PY" scripts/sim/uitest_control.py --work "$STACK" --python "$PY" > "$OUT/control.out" 2>&1 &
CTL=$!
for _ in $(seq 120); do curl -sf http://127.0.0.1:8790/state >/dev/null && break; sleep 0.5; done
curl -sf http://127.0.0.1:8790/state >/dev/null || { cat "$OUT/control.out"; exit 1; }

echo "== build for testing (Debug, ad-hoc signed)"
(cd ios && xcodegen generate >/dev/null)
xcrun simctl boot "$SIM" 2>/dev/null || true
xcrun simctl bootstatus "$SIM" -b >/dev/null
DEST=(-destination "id=$SIM")
COMMON=(-project ios/SensorDyme.xcodeproj -scheme SensorDyme "${DEST[@]}" -derivedDataPath "$OUT/dd"
        CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM=)
if [ "${REUSE_BUILD:-0}" = 1 ] && ls "$OUT"/dd/Build/Products/*.xctestrun >/dev/null 2>&1; then
  echo "   (reusing the build in $OUT/dd)"
else
  need_space "$MIN_BUILD_MB" "the build"
  xcodebuild "${COMMON[@]}" build-for-testing > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
fi

echo "== Simulator: no app, no keychain from an earlier run"
xcrun simctl uninstall "$SIM" com.sensordyme.app 2>/dev/null || true
xcrun simctl uninstall "$SIM" com.sensordyme.app.uitests.xctrunner 2>/dev/null || true
xcrun simctl keychain "$SIM" reset

# One test at a time, in the order asked for. Run from a copy of the build's
# .xctestrun with the ordering and parallelization written into it, so what
# ran is what the file says (and is kept in OUT for the record).
SRC_RUN=$(ls "$OUT"/dd/Build/Products/*.xctestrun | grep -v "/oryn-[a-z]*\.xctestrun$" | head -1)
RUNFILE="$OUT/dd/Build/Products/oryn-$ORDER.xctestrun"
cp "$SRC_RUN" "$RUNFILE"
"$PY" - "$RUNFILE" "$ORDER" <<'PY'
import plistlib, sys
path, order = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    plan = plistlib.load(f)
# Format 2 lists targets under TestConfigurations; format 1 (what a scheme
# without a test plan builds) has one top-level key per target.
if "TestConfigurations" in plan:
    targets = [t for c in plan["TestConfigurations"] for t in c.get("TestTargets", [])]
else:
    targets = [v for k, v in plan.items() if k != "__xctestrun_metadata__" and isinstance(v, dict)]
if not targets:
    sys.exit(f"no test targets found in {path}")
for target in targets:
    target["ParallelizationEnabled"] = False
    if order == "random":
        target["TestExecutionOrdering"] = "random"
    else:
        target.pop("TestExecutionOrdering", None)
with open(path, "wb") as f:
    plistlib.dump(plan, f)
PY
cp "$RUNFILE" "$OUT/run.xctestrun"

echo "== tests (order: $ORDER)"
rm -rf "$OUT/result.xcresult"
need_space 1843 "the test run"
set +e
TEST_RUNNER_ORYN_UITEST_SHOTS="$OUT/shots" TEST_RUNNER_ORYN_UITEST_ARGS="${ORYN_UITEST_ARGS:-}" \
TEST_RUNNER_ORYN_UITEST_REQUIRE_STACK=1 \
  xcodebuild test-without-building -xctestrun "$RUNFILE" "${DEST[@]}" -parallel-testing-enabled NO \
  -collect-test-diagnostics never -resultBundlePath "$OUT/result.xcresult" ${ONLY[@]+"${ONLY[@]}"} \
  > "$OUT/xcodebuild.log" 2>&1
rc=$?
set -e
find "$STACK" -name '*.log' -exec cp {} "$OUT/" \; 2>/dev/null || true

expected_only=""
[ ${#ONLY[@]} -gt 0 ] && expected_only="${ONLY[*]}"
sumrc=0
python3 - "$OUT" "$ORDER" "$(git rev-parse --short HEAD)" "$expected_only" <<'PY' || sumrc=$?
import glob, re, sys
out, order, commit, only = sys.argv[1:5]
log = open(f"{out}/xcodebuild.log").read()
rows = re.findall(r"Test [Cc]ase '-\[\S+ (test\w+)\]' (passed|failed|skipped) \(([0-9.]+) seconds\)", log)
fails = {}
for m in re.finditer(r"/([\w]+\.swift):(\d+): error: -\[\S+ (test\w+)\] : (.*)", log):
    fails.setdefault(m.group(3), f"{m.group(1)}:{m.group(2)} {m.group(4).strip()[:160]}")
for name, msg in re.findall(r"error: -\[\S+ (test\w+)\] : (.*)", log):
    fails.setdefault(name, msg.strip()[:160])
skips = dict(re.findall(r"-\[\S+ (test\w+)\] : Test skipped - (.*)", log))
# Every test the sources define (or the -only-testing ones) must have reported.
# (class, test) for every test the sources define; with -only-testing, the ones
# it names (a class names all of its tests).
defined = set()
for path in glob.glob("ios/UITests/*.swift"):
    for block in re.split(r"(?=\bclass \w+\s*:\s*XCTestCase)", open(path).read()):
        cls = re.match(r"class (\w+)", block)
        if cls:
            defined |= {(cls.group(1), t) for t in re.findall(r"func (test\w+)\(\)", block)}
def selected(cls, test):
    for item in only.split():
        parts = item.split(":", 1)[-1].split("/")  # -only-testing:Target/Class[/test]
        if parts[1:2] == [cls] and (len(parts) < 3 or parts[2] == test):
            return True
    return False
expected = {t for c, t in defined if not only or selected(c, t)}
ran = [name for name, _, _ in rows]
missing = sorted(expected - set(ran))
lines = [f"Commit {commit}; order {order}; one xcodebuild invocation; no retries.", "",
         "| # | Test | Result | Seconds | First failure / skip reason |", "|---|---|---|---|---|"]
for i, (name, result, secs) in enumerate(rows, 1):
    why = fails.get(name, "") if result == "failed" else skips.get(name, "") if result == "skipped" else ""
    lines.append(f"| {i} | {name} | {result.upper()} | {float(secs):.0f} | {why} |")
for name in missing:
    lines.append(f"| - | {name} | MISSING (never reported) | | |")
passed = sum(r == "passed" for _, r, _ in rows)
failed = sum(r == "failed" for _, r, _ in rows)
skipped = sum(r == "skipped" for _, r, _ in rows)
total = sum(float(s) for _, _, s in rows)
summary = ("\n".join(lines) + f"\n\n{passed}/{len(expected)} passed, {failed} failed, {skipped} skipped, "
           f"{len(missing)} missing; {total / 60:.1f} min of tests\n")
open(f"{out}/summary.md", "w").write(summary)
print(summary)
sys.exit(0 if passed == len(expected) and not failed and not skipped and not missing else 1)
PY
echo "screenshots and audit reports in $OUT/shots; full log $OUT/xcodebuild.log"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$sumrc"
