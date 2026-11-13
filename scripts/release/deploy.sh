#!/usr/bin/env bash
# Staged, verified, rollback-safe deploy of a git ref to a device (M8.5, M84-6).
# `make deploy` runs this after `deploy-preflight`; docs/DEPLOYMENT.md is the procedure.
#
#   DEVICE=user@host DIR=sensordyme REF=HEAD PY=.venv/bin/python bash scripts/release/deploy.sh
#
#   1. bundle    tarball + manifest from the git objects of REF (never the working
#                tree), per-file sha256; refused if a release gate FAILs.
#   2. upload    both to $DIR.incoming/ on the device (nothing live changes).
#   3. stage     the guard FROM THE SAME COMMIT, streamed over SSH, verifies every
#                file against the manifest whose sha256 is on its command line, then
#                builds the release's own venv from the hashed lock (wheels only) and
#                checks the MediaPipe model files. Any failure: nothing live changed.
#   4. activate  on the device, in its own session (a dropped SSH connection cannot
#                interrupt it): carry config.yaml over, migrate it in the new release,
#                switch the service path atomically, install the units, restart, and
#                confirm -- or switch back to the previous release (no network).
#   5. wait      for that activation's outcome: exit 0 confirmed, 1 rolled back,
#                3 no answer in DEPLOY_WAIT s (the device still finishes on its own).
#
# DIR is the service path relative to the login directory (default `sensordyme`) or
# absolute; it must already be a staged layout (`make deploy-adopt` once).
# Optional: FIND_LINKS=<dir on the device> (offline wheels, still hash-checked),
# DEPLOY_TIMEOUT / DEPLOY_SETTLE (health), DEPLOY_WAIT, TESTS (recorded in the manifest).
set -euo pipefail

: "${DEVICE:?DEVICE=user@host is required}"
DIR="${DIR:-sensordyme}"
REF="${REF:-HEAD}"
PY="${PY:-.venv/bin/python}"
TIMEOUT="${DEPLOY_TIMEOUT:-180}"
SETTLE="${DEPLOY_SETTLE:-30}"
WAIT="${DEPLOY_WAIT:-900}"
case "$DIR" in
  *[!A-Za-z0-9._/-]*|'') echo "DIR='$DIR': use a plain path relative to the login directory, or absolute (no ~)" >&2; exit 2 ;;
esac
case "$DEVICE" in
  *[!A-Za-z0-9._@:-]*) echo "DEVICE='$DEVICE' is not a plain user@host" >&2; exit 2 ;;
esac
case "${FIND_LINKS:-}" in
  *[!A-Za-z0-9._/-]*) echo "FIND_LINKS must be a plain path" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "==> 1/5 bundle $REF (git objects only)"
"$PY" "$ROOT/scripts/release/make_release_manifest.py" bundle --repo "$ROOT" --ref "$REF" --out-dir "$OUT" \
  ${TESTS:+--test-ref "$TESTS"}
ID="$(cat "$OUT/RELEASE_ID")"
COMMIT="$(cat "$OUT/COMMIT")"
TGZ="sensordyme-$ID.tar.gz"
MAN="sensordyme-$ID.manifest.json"
MSHA="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$OUT/$MAN")"
ATTEMPT="$(date +%Y%m%dT%H%M%S)-$$"
GUARD="$DIR.releases/$ID/scripts/release/release_guard.py"
guard_from_commit() { git -C "$ROOT" show "$COMMIT:scripts/release/release_guard.py"; }

echo "==> 2/5 upload release $ID to $DEVICE:$DIR.incoming/"
ssh "$DEVICE" "mkdir -p '$DIR.incoming'"
scp -q "$OUT/$TGZ" "$OUT/$MAN" "$DEVICE:$DIR.incoming/"

echo "==> 3/5 stage and verify on the device (manifest sha256 $MSHA)"
guard_from_commit | ssh "$DEVICE" "python3 - stage --root '$DIR' --bundle '$DIR.incoming/$TGZ' \
  --manifest '$DIR.incoming/$MAN' --manifest-sha256 $MSHA ${FIND_LINKS:+--find-links '$FIND_LINKS'}"

echo "==> 4/5 activate (detached on the device; attempt $ATTEMPT)"
ssh "$DEVICE" "python3 '$GUARD' activate --root '$DIR' --id $ID --user \"\$(id -un)\" \
  --timeout $TIMEOUT --settle $SETTLE --attempt $ATTEMPT --detach"

echo "==> 5/5 wait for the outcome (up to ${WAIT}s)"
guard_from_commit | ssh "$DEVICE" "python3 - status --root '$DIR' --wait $ID --attempt $ATTEMPT --timeout $WAIT"
