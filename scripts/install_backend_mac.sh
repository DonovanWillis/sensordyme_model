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

# Refuse to install if something OTHER than our own agent already holds the port.
# Installing over it produces an agent that can never start, reported only as a
# throttled respawn in the log -- and meanwhile the thing that IS on :8787
# answers health checks, so everything looks fine while this agent is dead.
PORT=8787
if HOLDER="$(lsof -iTCP:${PORT} -sTCP:LISTEN -n -P -t 2>/dev/null)"; then
    if [ -n "$HOLDER" ]; then
        for pid in $HOLDER; do
            if ! ps -p "$pid" -o command= 2>/dev/null | grep -q "cloud.app:app"; then
                echo "error: port ${PORT} is held by PID ${pid}, which is not the SensorDyme backend:" >&2
                ps -p "$pid" -o command= >&2
                echo "Stop it first, or the agent installed here will never be able to bind." >&2
                exit 1
            fi
        done
    fi
fi

mkdir -p "$HOME/Library/LaunchAgents"

# Keep a copy of whatever is already installed. This script is the documented way
# to repair the agent, and the thing most likely to need repairing is a plist
# somebody edited by hand -- so the edit should be recoverable rather than simply
# gone.
if [ -f "$PLIST_DEST" ]; then
    BACKUP="${PLIST_DEST}.replaced-$(date +%Y%m%d-%H%M%S)"
    cp "$PLIST_DEST" "$BACKUP"
    echo "Kept a copy of the agent that was already installed: ${BACKUP}"
fi

sed "s#__REPO_DIR__#${REPO_DIR}#g" \
    "$REPO_DIR/deploy/launchd/com.sensordyme.backend.plist.template" \
    > "$PLIST_DEST"

# Verify what was just written actually starts the backend, before loading it.
# The failure this catches really happened: the installed plist had been edited to
# run `main.py` -- the camera/detector entrypoint -- so launchd dutifully kept a
# detector alive and called it "the backend", while the API server the phone needs
# was not running at all. Nothing else in the chain notices, because a stale
# uvicorn from before the edit was still answering.
if ! /usr/libexec/PlistBuddy -c 'Print :ProgramArguments' "$PLIST_DEST" >/dev/null 2>&1; then
    echo "error: the plist just written is not readable. Is the template valid?" >&2
    exit 1
fi
if ! grep -q 'cloud.app:app' "$PLIST_DEST"; then
    echo "error: the plist just written does not start cloud.app:app." >&2
    echo "Check deploy/launchd/com.sensordyme.backend.plist.template" >&2
    exit 1
fi

UID_NUM="$(id -u)"
DOMAIN="gui/${UID_NUM}"

# bootout is expected to fail (exit nonzero) the first time this ever runs,
# when nothing is loaded yet -- that is not an error.
launchctl bootout "$DOMAIN/$PLIST_LABEL" 2>/dev/null || true

# Wait for the unload to actually settle before loading the replacement.
# bootout returns before launchd has finished retiring the job, and bootstrapping
# into that window fails with the thoroughly unhelpful "Bootstrap failed: 5:
# Input/output error" -- which looks like a broken plist and is not. Observed
# happening on a re-run over an already-loaded agent.
for _ in $(seq 1 30); do
    launchctl print "$DOMAIN/$PLIST_LABEL" >/dev/null 2>&1 || break
    sleep 0.2
done

if ! launchctl bootstrap "$DOMAIN" "$PLIST_DEST" 2>/dev/null; then
    # One retry: the wait above covers the common case, but the window is a race
    # and a single retry costs nothing compared to failing with error 5.
    sleep 1
    launchctl bootstrap "$DOMAIN" "$PLIST_DEST"
fi
launchctl enable "$DOMAIN/$PLIST_LABEL"

# Poll rather than sleeping a fixed two seconds: uvicorn plus the mDNS
# registration (which probes the network for name conflicts) can take longer than
# that on a cold start, and reporting a failure that was only slowness sends
# people to debug a backend that came up fine a second later.
answered=0
for _ in $(seq 1 20); do
    # -f: an error status is not "answered". Without it any process on the port
    # that speaks HTTP at all -- including one returning 404 or 500 -- passed.
    if curl -fsS -m 2 -o /dev/null "http://localhost:${PORT}/v1/health" 2>/dev/null; then
        answered=1
        break
    fi
    sleep 1
done

if [ "$answered" -eq 1 ]; then
    echo "SensorDyme backend installed and running (launchd: ${PLIST_LABEL})."
    echo "It will now start automatically at login and restart itself if it crashes."
    echo "Logs: ${REPO_DIR}/cloud/backend.log"
else
    echo "Installed, but it did not answer on :${PORT} within 20 seconds." >&2
    echo "Last lines of ${REPO_DIR}/cloud/backend.log:" >&2
    tail -n 20 "${REPO_DIR}/cloud/backend.log" >&2 2>/dev/null || true
    exit 1
fi

echo
echo "Useful commands:"
echo "  Check status:  launchctl print ${DOMAIN}/${PLIST_LABEL}"
echo "  Restart now:   launchctl kickstart -k ${DOMAIN}/${PLIST_LABEL}"
echo "  Stop for now:  launchctl bootout ${DOMAIN}/${PLIST_LABEL}"
echo "  Diagnose:      .venv/bin/python scripts/diagnose_backend_mac.py"
