#!/usr/bin/env bash
# Rehearse every demo mode, on the device, in one command.
#
# This exists because Phase 5 found a mode (occupancy) that could not start at
# all, and found it only by cycling modes with service restarts -- not by any
# test. The unit tests all passed. `make demo-check` proves the CURRENT mode is
# healthy; this proves EVERY mode still starts, still reaches the NPU, and still
# reports truthfully after a restart.
#
# `systemctl reset-failed` before each restart is deliberate and is NOT hiding a
# failure: the unit sets StartLimitBurst=5 with StartLimitAction=reboot, so a
# rehearsal that restarts more than five times would reboot the Pi mid-test and
# measure systemd instead of SensorDyme. Real crash loops still trip the guard,
# because they restart every 3s inside the 120s window.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODES=${MODES:-"driver_attention baby_monitoring fall_detection occupancy"}
PORT=$(.venv/bin/python -c "import yaml;print((yaml.safe_load(open('config.yaml')).get('dashboard') or {}).get('port',8899))")
# Carry the dashboard token when one is configured. Without it every health poll
# below 401s and the rehearsal reports every mode as failed on a device that is
# working correctly -- which would make the security feature look like a bug.
TOKEN=$(.venv/bin/python -c "import yaml;print((yaml.safe_load(open('config.yaml')).get('dashboard') or {}).get('token') or '')")
AUTH=(); [ -n "$TOKEN" ] && AUTH=(-H "X-Device-Token: $TOKEN")
ORIGINAL=$(grep -m1 '^module:' config.yaml | awk '{print $2}')
FAILURES=0
# One private file per run, emptied before every mode. A fixed /tmp path that
# was only ever overwritten on SUCCESS let a mode whose dashboard never answered
# be scored from the PREVIOUS mode's healthy reply -- and print OK.
HEALTH=$(mktemp "${TMPDIR:-/tmp}/sensordyme-cycle.XXXXXX") || exit 1

restore() {
  rm -f "$HEALTH"
  sed -i "s/^module: .*/module: $ORIGINAL/" config.yaml
  sudo systemctl reset-failed sensordyme 2>/dev/null
  sudo systemctl restart sensordyme
  echo "  restored module: $ORIGINAL"
}
trap restore EXIT

printf '%-20s %8s %8s %10s %8s %7s  %s\n' MODE BOOT_S ACTIVE MONITORING NPU FPS RESULT
for mode in $MODES; do
  sed -i "s/^module: .*/module: $mode/" config.yaml
  sudo systemctl reset-failed sensordyme 2>/dev/null
  : > "$HEALTH"
  t0=$(date +%s.%N)
  sudo systemctl restart sensordyme
  boot="timeout"
  for _ in $(seq 1 90); do
    if curl -sf "${AUTH[@]}" --max-time 3 "localhost:$PORT/api/health" -o "$HEALTH" 2>/dev/null; then
      boot=$(echo "$(date +%s.%N) - $t0" | bc); break
    fi
    sleep 1
  done
  sleep 12   # let inference actually run before judging it
  # The judged sample is THIS read. If it fails, nothing is left to score, so
  # the mode fails rather than inheriting the boot-time reply.
  : > "$HEALTH"
  curl -sf "${AUTH[@]}" --max-time 5 "localhost:$PORT/api/health" -o "$HEALTH" 2>/dev/null || : > "$HEALTH"
  if ! .venv/bin/python - "$mode" "$boot" "$(systemctl is-active sensordyme)" "$HEALTH" <<'PY'
import json, sys
mode, boot, active, path = sys.argv[1:5]
try:
    dev = json.load(open(path)); dev = dev.get('device', dev)
except Exception:
    dev = {}
cm = dev.get('camera', {}).get('metrics', {}); am = dev.get('accelerator', {}).get('metrics', {})
mon = dev.get('monitoring_active')
try: b = f"{float(boot):.1f}"
except Exception: b = boot
# fall_detection consumes no YOLO (MediaPipe Pose only), so zero NPU inferences
# is CORRECT there and must not be scored as a failure.
needs_npu = mode != 'fall_detection'
# The device must be running the mode under test. An AI-custom override (see
# demo_preflight.py's _override_state) keeps one module running whatever
# config.yaml says, and every "mode" then scored the same healthy module.
running = str(dev.get('inference', {}).get('detail') or '')
ran_mode = f"({mode})" in running
ok = (active == 'active' and mon is True and ran_mode
      and dev.get('camera', {}).get('state') == 'ok'
      and (not needs_npu or (am.get('inferences') or 0) > 0)
      and (cm.get('evidence_losing_stalls_1s') or 0) == 0)
result = 'OK' if ok else ('FAILED' if ran_mode or not dev else f"FAILED (running: {running!r})")
print(f"{mode:<20} {b:>8} {active:>8} {str(mon):>10} {str(am.get('inferences')):>8} "
      f"{str(cm.get('fps')):>7}  {result}")
raise SystemExit(0 if ok else 1)
PY
  then FAILURES=$((FAILURES+1)); fi
done

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "ALL MODES OK"
else
  echo "$FAILURES MODE(S) FAILED"
fi
exit "$FAILURES"
