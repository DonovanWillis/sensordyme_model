#!/usr/bin/env bash
# Thermal soak logger for the Pi 5 + Hailo-8L, READ-ONLY.
#
# Runs ALONGSIDE the normal service (that is the point: thermals under the real
# workload). It starts, stops and changes nothing; it only reads vcgencmd, sysfs
# and the service's own /api/health, and appends rows to a CSV outside the
# installed tree.
#
#   SENSORDYME_TOKEN=... bash thermal_soak.sh [--minutes 30] [--interval 5] \
#       [--out ~/hwval/soak_$(date +%Y%m%dT%H%M%S).csv]
#
# Columns: utc, elapsed_s, soc_temp_c, throttled (raw hex), arm_hz, core_v,
# fan_rpm, load1, hailo_temp_c, processed_fps, camera_fps, processed_p95_ms,
# hailo_inferences, monitoring_state
#
# processed_fps = processed_frames in the health window / measured time between
# polls. The
# window is drained per caller ("dashboard"), so do not run the app, a browser
# on /api/health, or stage_timing.py health at the same time.
#
# Hailo die temperature: OFF by default. Whether the installed HailoRT's CLI can
# report it is not established, and opening a second device handle while the
# service holds the chip is untested and could disturb it. If an operator has
# verified a safe read-only command for this HailoRT version, pass it in
# HAILO_TEMP_CMD (it must print a single number in degrees C).
set -u

MINUTES=30
INTERVAL=5
OUT="${HOME}/hwval/soak_$(date -u +%Y%m%dT%H%M%SZ).csv"
URL="http://127.0.0.1:8899/api/health"
INSTALL_DIR="/home/pi/sensordyme"

while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

out_dir="$(dirname "$OUT")"
mkdir -p "$out_dir"
case "$(cd "$out_dir" && pwd -P)/" in
  "$(cd "$INSTALL_DIR" 2>/dev/null && pwd -P)/"*)
    echo "REFUSED: --out is inside $INSTALL_DIR; write somewhere else" >&2; exit 2 ;;
esac

if ! systemctl is-active --quiet sensordyme.service; then
  echo "NOTE: sensordyme.service is not active: this soak measures an IDLE device," >&2
  echo "      not the production workload. Health columns will be empty." >&2
fi
if [ -z "${SENSORDYME_TOKEN:-}" ]; then
  echo "NOTE: SENSORDYME_TOKEN is not set; /api/health may answer 401 (fps columns empty)." >&2
fi

vc() { command -v vcgencmd >/dev/null 2>&1 && vcgencmd "$@" 2>/dev/null; }

soc_temp() { vc measure_temp | sed -n "s/^temp=\([0-9.]*\).*/\1/p"; }
throttled() { vc get_throttled | sed -n 's/^throttled=//p'; }
arm_hz() { vc measure_clock arm | sed -n 's/^frequency([0-9]*)=//p'; }
core_v() { vc measure_volts core | sed -n 's/^volt=\([0-9.]*\)V/\1/p'; }
fan_rpm() { cat /sys/devices/platform/cooling_fan/hwmon/hwmon*/fan1_input 2>/dev/null | head -1; }
load1() { cut -d' ' -f1 /proc/loadavg; }
hailo_temp() { [ -n "${HAILO_TEMP_CMD:-}" ] && timeout 5 bash -c "$HAILO_TEMP_CMD" 2>/dev/null | head -1; }

health() {
  local hdr=()
  [ -n "${SENSORDYME_TOKEN:-}" ] && hdr=(-H "X-Device-Token: ${SENSORDYME_TOKEN}")
  curl -s -m 4 "${hdr[@]}" "$URL" 2>/dev/null | python3 -c '
import json, sys
interval = float(sys.argv[1])
try:
    d = json.load(sys.stdin)["device"]
except Exception:
    print(",,,,"); sys.exit(0)
m = lambda c: (d.get(c) or {}).get("metrics") or {}
frames = m("inference").get("processed_frames")
fps = "" if frames is None else f"{frames / interval:.2f}"
vals = [fps, m("camera").get("fps"), m("inference").get("processed_p95_ms"),
        m("accelerator").get("inferences"), (d.get("monitoring") or {}).get("state")]
print(",".join("" if v is None else str(v) for v in vals))
' "$1"
}

echo "utc,elapsed_s,soc_temp_c,throttled,arm_hz,core_v,fan_rpm,load1,hailo_temp_c,processed_fps,camera_fps,processed_p95_ms,hailo_inferences,monitoring_state" > "$OUT"
echo "soak: ${MINUTES} min every ${INTERVAL}s -> $OUT"
health 1 >/dev/null  # first poll only establishes the health window baseline
prev_poll=$(date +%s.%N)
start=$(date +%s)
end=$((start + MINUTES * 60))
while [ "$(date +%s)" -lt "$end" ]; do
  sleep "$INTERVAL"
  now=$(date +%s)
  sys_cols="$(soc_temp),$(throttled),$(arm_hz),$(core_v),$(fan_rpm),$(load1),$(hailo_temp)"
  poll=$(date +%s.%N)
  window=$(python3 -c "print(max(0.001, $poll - $prev_poll))")
  prev_poll=$poll
  row="$(date -u +%Y-%m-%dT%H:%M:%SZ),$((now - start)),${sys_cols},$(health "$window")"
  echo "$row" >> "$OUT"
  echo "$row"
done

echo
echo "done: $OUT"
python3 - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
def col(k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (TypeError, ValueError):
            pass
    return out
for k in ("soc_temp_c", "processed_fps", "arm_hz"):
    v = col(k)
    if v:
        print(f"{k:>14}: first {v[0]:.1f}  last {v[-1]:.1f}  min {min(v):.1f}  max {max(v):.1f}")
flags = {r["throttled"] for r in rows if r["throttled"]}
print(f"     throttled: {sorted(flags) or ['(not read)']}  (0x0 = never; bit 2 = throttling now, bit 18 = has throttled)")
PY
