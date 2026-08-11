#!/bin/bash
# Installs the SensorDyme cloud backend as a macOS LaunchAgent, so it survives
# reboots, crashes, and closed terminal windows instead of only running while
# someone remembers to leave `uvicorn` running in a window.
#
# This is the fix for a device and app that both look healthy (Wi-Fi joined,
# DHCP fine, device dashboard reachable, mDNS advertising) but the app still
# says "Can't reach backend" and the device's own health reports
# cloud_uplink degraded: the backend on this Mac simply is not running right
# now. See deploy/launchd/com.sensordyme.backend.plist.template for why.
#
# Safe to re-run: it stops any existing instance of this agent before
# reinstalling, so re-running after a `git pull` picks up code changes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_LABEL="com.sensordyme.backend"
PLIST_DEST="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

if [ ! -x "$REPO_DIR/.venv/bin/uvicorn" ]; then
    echo "error: $REPO_DIR/.venv/bin/uvicorn not found." >&2
    echo "Set up the venv first: python3 -m venv .venv && .venv/bin/pip install -r cloud/requirements.txt" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"
sed "s#__REPO_DIR__#${REPO_DIR}#g" \
    "$REPO_DIR/deploy/launchd/com.sensordyme.backend.plist.template" \
    > "$PLIST_DEST"

UID_NUM="$(id -u)"
DOMAIN="gui/${UID_NUM}"

# bootout is expected to fail (exit nonzero) the first time this ever runs,
# when nothing is loaded yet -- that is not an error.
launchctl bootout "$DOMAIN/$PLIST_LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
launchctl enable "$DOMAIN/$PLIST_LABEL"

sleep 2
if curl -sS -m 3 -o /dev/null -w '' "http://localhost:8787/v1/health" 2>/dev/null; then
    echo "SensorDyme backend installed and running (launchd: ${PLIST_LABEL})."
    echo "It will now start automatically at login and restart itself if it crashes."
    echo "Logs: ${REPO_DIR}/cloud/backend.log"
else
    echo "Installed, but it did not answer on :8787 within a few seconds." >&2
    echo "Check ${REPO_DIR}/cloud/backend.log for the reason." >&2
    exit 1
fi

echo
echo "Useful commands:"
echo "  Check status:  launchctl print ${DOMAIN}/${PLIST_LABEL}"
echo "  Stop for now:  launchctl bootout ${DOMAIN}/${PLIST_LABEL}"
echo "  Diagnose:      .venv/bin/python scripts/diagnose_backend_mac.py"
