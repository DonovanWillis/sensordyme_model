#!/usr/bin/env bash
# The virtual-device reliability + stress suites and the suites they touch (M6.5).
#   PY=/path/to/python bash scripts/sim/reliability_suite.sh [OUT_DIR]
# On the fleet: ./sdrun submit -t oryn --repo <repo> --ref <commit> --no-gpu --cpus 2 --mem 1G \
#   --timeout 1800 -- bash scripts/sim/reliability_suite.sh
# (PY defaults to the fleet's repo venv; OUT_DIR to $SDJOB_OUT or ./out.)
set -uo pipefail
PY="${PY:-${SDJOB_DATA:-}/venvs/sensordyme-a779b4d/bin/python}"
OUT="${1:-${SDJOB_OUT:-out}}"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
t0=$(date +%s)
"$PY" -m pytest -p no:cacheprovider -q -s -rs --timeout=600 --junitxml="$OUT/junit.xml" \
  tests/test_sim_reliability.py tests/test_sim_stress.py \
  tests/test_sim_e2e.py tests/test_sim_units.py tests/test_sim_seams.py \
  tests/test_detections.py tests/test_monitoring_mode.py tests/test_object_class_mapping.py \
  tests/test_phase1_driver_fixes.py tests/test_driver_attention_module.py tests/test_driver_review_fixes.py \
  tests/test_rule_delivery.py tests/test_m01_role_floors.py tests/test_calibration.py \
  tests/test_phase1_reboot_bounding.py > "$OUT/pytest.txt" 2>&1
rc=$?
echo "pytest rc=$rc in $(( $(date +%s) - t0 ))s"
grep -E "^(FAILED|ERROR)|^(backlog|retry storm|race|threads|restarts|20 min|backoff):" "$OUT/pytest.txt" | head -30
tail -1 "$OUT/pytest.txt"
exit "$rc"
