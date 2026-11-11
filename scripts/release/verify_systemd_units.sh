#!/usr/bin/env bash
# Validate the SensorDyme systemd units with systemd's own parser, off-device.
#
#   bash scripts/release/verify_systemd_units.sh
#
# Renders each unit into a scratch tree (so ExecStart paths exist and verify judges
# the unit, not this host), then runs `systemd-analyze verify` on: the stock unit,
# the least-privilege unit, and the least-privilege unit with its optional drop-in;
# and prints the offline `systemd-analyze security` exposure score of both.
# Needs a Linux host with systemd (oryn's job distro has it); on macOS it says so
# and exits 0, because the static checks in tests/test_least_privilege_unit.py
# still run there. Exit 1 if any verify reports a problem in a SensorDyme unit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v systemd-analyze >/dev/null 2>&1; then
  echo "systemd-analyze not found: systemd validation SKIPPED on this host (static tests still apply)"
  exit 0
fi
systemd-analyze --version | head -1
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
APP="$T/opt"; UNITS="$T/units"
mkdir -p "$APP/.venv/bin" "$APP/scripts" "$UNITS/sensordyme.service.d"
ln -s "$(command -v python3)" "$APP/.venv/bin/python"
: > "$APP/scripts/preflight.py"; : > "$APP/main.py"
cp "$HERE/deploy/systemd/sensordyme-recover.service" "$UNITS/"

render() {  # render SRC DEST: fixed layout and template tokens -> the scratch tree
  python3 - "$1" "$2" "$APP" "$(id -un)" <<'PY'
import sys
src, dest, app, user = sys.argv[1:]
text = open(src).read().replace("/opt/sensordyme", app)
text = text.replace("__SENSORDYME_DIR__", app).replace("__SENSORDYME_USER__", user)
open(dest, "w").write(text)
PY
}

status=0
check() {  # check LABEL UNIT
  echo "== verify: $1"
  out="$(SYSTEMD_UNIT_PATH="$UNITS:" systemd-analyze verify --man=no "$2" 2>&1)"; rc=$?
  # The recover unit's ExecStart (/usr/local/lib/sensordyme/...) exists only on an
  # installed device; that one finding is expected here and is not a unit defect.
  problems="$(printf '%s\n' "$out" | grep -v '^$' | grep -v 'sensordyme-recover' | grep -v '/usr/local/lib/sensordyme' || true)"
  printf '%s\n' "$out" | sed 's/^/   /'
  if [ -n "$problems" ]; then echo "   FAIL ($rc)"; status=1; else echo "   OK (rc=$rc)"; fi
}

render "$HERE/sensordyme.service" "$UNITS/stock.service"
check "stock sensordyme.service" "$UNITS/stock.service"
render "$HERE/deploy/systemd/sensordyme.service.hardened" "$UNITS/sensordyme.service"
check "sensordyme.service.hardened" "$UNITS/sensordyme.service"
cp "$HERE/deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf" "$UNITS/sensordyme.service.d/"
check "hardened + btmgmt-advertising.conf" "$UNITS/sensordyme.service"
rm "$UNITS/sensordyme.service.d/btmgmt-advertising.conf"

echo "== exposure (systemd-analyze security --offline; lower is better)"
for unit in stock.service sensordyme.service; do
  score="$(SYSTEMD_UNIT_PATH="$UNITS:" systemd-analyze security --offline=true "$UNITS/$unit" 2>/dev/null | tail -1)"
  echo "   $unit: ${score:-n/a}"
done
exit $status
