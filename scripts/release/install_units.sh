#!/usr/bin/env bash
# Install (or render) the SensorDyme systemd units for THIS checkout.
#
# Why this exists: `make deploy` rsyncs code but never touches
# /etc/systemd/system, so a release that changes sensordyme.service (M6 does:
# bounded start-limit reboots) would otherwise land with the OLD unit still in
# force. setup_pi.sh installs the same files on a first install.
#
#   bash scripts/release/install_units.sh --render-only OUT_DIR [--user NAME]
#       render the units into OUT_DIR and change nothing (no sudo; works on a Mac)
#   sudo bash scripts/release/install_units.sh --user NAME
#       install them, then `systemctl daemon-reload`. Does NOT restart the
#       service: the operator does that, deliberately (docs/MONDAY_RUNBOOK.md).
#
# Installs:
#   /etc/systemd/system/sensordyme.service          (rendered: user + directory)
#   /etc/systemd/system/sensordyme-recover.service  (bounded start-limit reboot)
#   /usr/local/lib/sensordyme/start_limit_guard.py  (root-owned, 0755)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$HERE/../.." && pwd)"
APP_DIR="$SRC_DIR"
SERVICE_USER="${SENSORDYME_USER:-${SUDO_USER:-$(id -un)}}"
RENDER_ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --render-only) RENDER_ONLY="${2:?--render-only needs a directory}"; shift 2 ;;
    --user) SERVICE_USER="${2:?--user needs a name}"; shift 2 ;;
    --app-dir) APP_DIR="${2:?--app-dir needs a directory}"; shift 2 ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$SERVICE_USER" = "root" ]; then
  echo "ERROR: refusing to run the SensorDyme service as root (pass --user NAME)." >&2
  exit 1
fi
case "$SERVICE_USER$APP_DIR" in
  *'#'*) echo "ERROR: '#' in the user or directory would break the unit rendering." >&2; exit 1 ;;
esac

render() {
  sed "s#__SENSORDYME_DIR__#${APP_DIR}#g; s#__SENSORDYME_USER__#${SERVICE_USER}#g" "$1"
}

if [ -n "$RENDER_ONLY" ]; then
  mkdir -p "$RENDER_ONLY"
  render "$SRC_DIR/sensordyme.service" > "$RENDER_ONLY/sensordyme.service"
  cp "$SRC_DIR/deploy/systemd/sensordyme-recover.service" "$RENDER_ONLY/sensordyme-recover.service"
  cp "$SRC_DIR/scripts/start_limit_guard.py" "$RENDER_ONLY/start_limit_guard.py"
  echo "rendered for user '$SERVICE_USER', directory '$APP_DIR' -> $RENDER_ONLY"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: installing units needs root: sudo bash $0 --user $SERVICE_USER" >&2
  exit 1
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "ERROR: service user '$SERVICE_USER' does not exist." >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
render "$SRC_DIR/sensordyme.service" > "$TMP/sensordyme.service"
if command -v systemd-analyze >/dev/null 2>&1; then
  # Advisory: verify complains about units it cannot see yet; print, do not block.
  systemd-analyze verify "$TMP/sensordyme.service" 2>&1 | sed 's/^/  verify: /' || true
fi

install -D -m 0755 -o root -g root "$SRC_DIR/scripts/start_limit_guard.py" \
  /usr/local/lib/sensordyme/start_limit_guard.py
install -m 0644 -o root -g root "$SRC_DIR/deploy/systemd/sensordyme-recover.service" \
  /etc/systemd/system/sensordyme-recover.service
install -m 0644 -o root -g root "$TMP/sensordyme.service" /etc/systemd/system/sensordyme.service
systemctl daemon-reload
echo "installed units for user '$SERVICE_USER', directory '$APP_DIR' (service NOT restarted)"
sha256sum /etc/systemd/system/sensordyme.service /etc/systemd/system/sensordyme-recover.service \
  /usr/local/lib/sensordyme/start_limit_guard.py
