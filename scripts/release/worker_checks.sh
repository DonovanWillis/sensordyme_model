#!/usr/bin/env bash
# The parts of `make verify-release` that need Linux, plus the tests around the changed
# release surfaces, for a worker job (oryn). The job source is an archive, not a git
# checkout, so the manifest build/gates (git objects) stay on the controller.
#   sdrun submit -t oryn ... --no-gpu -- bash scripts/release/worker_checks.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 2
PY="${PYTHON:-$SDJOB_DATA/venvs/sensordyme-a779b4d/bin/python}"
rc=0
bash scripts/release/verify_systemd_units.sh || rc=1
"$PY" scripts/release/check_device_deps.py || rc=1
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider -q -rs --disable-warnings --timeout=300 \
  tests/test_device_dependencies.py tests/test_least_privilege_unit.py tests/test_release_hardening.py \
  tests/test_m6_security_release_manifest.py tests/test_ble_advertising.py tests/test_ble_payload.py \
  tests/test_ble_scan_cache.py tests/test_m5_hw_validation.py tests/test_m6_autonomy_boot.py \
  tests/test_m6_security_device.py tests/test_m6_security_preflight.py tests/test_network_review_fixes.py \
  tests/test_phase1_reboot_bounding.py tests/test_phase4_no_agpl_runtime.py tests/test_provisioning.py \
  tests/test_runtime_review_fixes.py tests/test_m02_recovery.py 2>&1 | tail -15 || rc=1
exit $rc
