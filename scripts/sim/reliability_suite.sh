#!/usr/bin/env bash
# The virtual-device reliability + stress suites and the suites they touch (M6.5).
#   PY=/path/to/python bash scripts/sim/reliability_suite.sh [OUT_DIR]
# On the fleet: ./sdrun submit -t oryn --repo <repo> --ref <commit> --no-gpu --cpus 2 --mem 1G \
#   --timeout 1800 -- bash scripts/sim/reliability_suite.sh
# (PY defaults to the fleet's repo venv; OUT_DIR to $SDJOB_OUT or ./out.)
#
# One pytest process per file: the driver suites load real MediaPipe graphs, and in one
# process they and the sim suites together exceeded a 1 GiB job cap (M6.5 run 4419a2).
# Peak RSS per file is reported where GNU time exists (Linux).
set -uo pipefail
PY="${PY:-${SDJOB_DATA:-}/venvs/sensordyme-a779b4d/bin/python}"
OUT="${1:-${SDJOB_OUT:-out}}"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
FILES=(
  tests/test_sim_reliability.py tests/test_sim_stress.py
  tests/test_sim_e2e.py tests/test_sim_units.py tests/test_sim_seams.py
  tests/test_detections.py tests/test_monitoring_mode.py tests/test_object_class_mapping.py
  tests/test_phase1_driver_fixes.py tests/test_driver_attention_module.py tests/test_driver_review_fixes.py
  tests/test_rule_delivery.py tests/test_m01_role_floors.py tests/test_calibration.py
  tests/test_phase1_reboot_bounding.py
)
TIME=()
if [ -x /usr/bin/time ] && /usr/bin/time -v true >/dev/null 2>&1; then
  TIME=(/usr/bin/time -v -o "$OUT/time.txt")
fi
rc=0
t0=$(date +%s)
for f in "${FILES[@]}"; do
  name=$(basename "$f" .py)
  t1=$(date +%s)
  rm -f "$OUT/time.txt"
  ${TIME[@]+"${TIME[@]}"} "$PY" -m pytest -p no:cacheprovider -q -s -rs --timeout=600 \
    --junitxml="$OUT/junit-$name.xml" "$f" > "$OUT/pytest-$name.txt" 2>&1
  frc=$?
  [ "$frc" -ne 0 ] && rc=$frc
  rss=""
  [ -f "$OUT/time.txt" ] && rss=$(awk -F': ' '/Maximum resident/ {printf "%d MiB", $2/1024}' "$OUT/time.txt")
  echo "$name rc=$frc $(( $(date +%s) - t1 ))s ${rss} | $(tail -1 "$OUT/pytest-$name.txt")"
  grep -hE "^(FAILED|ERROR)|^(backlog|retry storm|race|threads|restarts|20 min|backoff):" \
    "$OUT/pytest-$name.txt" | head -12
done
echo "all rc=$rc in $(( $(date +%s) - t0 ))s"
exit "$rc"
