#!/usr/bin/env bash
# ONE repeatable, offline release verification. Light enough for the Mac controller
# (seconds; no downloads, no device, no GPU); also runs on a worker.
#
#   make verify-release [REF=<git ref>]        # or: bash scripts/release/verify.sh [REF]
#
# Steps (each prints PASS/FAIL/SKIP; exit 1 if any FAIL):
#   1. device dependencies   scripts/release/check_device_deps.py (pins vs imports; no dev/test packages)
#   2. release manifest      build for REF from git objects into a temp dir (never the working tree)
#   3. release gates         make_release_manifest.py check: model pins, decoder legacy, /v1 protocol
#                            compatibility, device lock, config schema, status tiers
#   4. systemd units         scripts/release/verify_systemd_units.sh (SKIP where systemd is absent)
#   5. focused tests         the release/packaging tests only (not the full suite)
# Nothing here changes the repo, the .hef, or any device.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${1:-${REF:-HEAD}}"
cd "$ROOT" || exit 2

PY="${PYTHON:-}"
for candidate in "$PY" "$ROOT/.venv/bin/python" "$HOME/.venvs/sensordyme-m01/bin/python" python3; do
  [ -n "$candidate" ] || continue
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import yaml" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
"$PY" -c "import yaml" >/dev/null 2>&1 || { echo "need a python with PyYAML (set PYTHON=...)"; exit 2; }

OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
declare -a RESULTS=()
step() {  # step NAME COMMAND...
  local name="$1"; shift
  echo "== $name"
  "$@" > "$OUT/step.log" 2>&1; local rc=$?
  sed 's/^/   /' "$OUT/step.log" | tail -25
  if [ $rc -eq 0 ]; then
    if grep -q "SKIPPED" "$OUT/step.log"; then RESULTS+=("SKIP  $name"); else RESULTS+=("PASS  $name"); fi
  else
    RESULTS+=("FAIL  $name (exit $rc)")
  fi
}

COMMIT="$(git rev-parse --verify "$REF^{commit}" 2>/dev/null)" || { echo "unknown ref: $REF"; exit 2; }
echo "release verification of $REF = $COMMIT (python: $PY)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || \
  echo "NOTE: the working tree has uncommitted changes; steps 2-3 describe the COMMIT, not them"

step "1 device dependencies" "$PY" scripts/release/check_device_deps.py
step "2 release manifest" "$PY" scripts/release/make_release_manifest.py build --ref "$COMMIT" \
     --out-json "$OUT/manifest.json" --out-md "$OUT/manifest.md"
if [ -s "$OUT/manifest.json" ]; then
  step "3 release gates" "$PY" scripts/release/make_release_manifest.py check --manifest "$OUT/manifest.json"
else
  RESULTS+=("FAIL  3 release gates (no manifest)")
fi
step "4 systemd units" bash scripts/release/verify_systemd_units.sh
if "$PY" -c "import pytest" >/dev/null 2>&1; then
  step "5 focused tests" env PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -p no:cacheprovider -q --disable-warnings -o addopts="" \
       tests/test_device_dependencies.py tests/test_least_privilege_unit.py \
       tests/test_release_hardening.py tests/test_m6_security_release_manifest.py
else
  RESULTS+=("SKIP  5 focused tests (no pytest in $PY)")
fi

echo
echo "== summary for $COMMIT"
printf '   %s\n' "${RESULTS[@]}"
if [ -n "${KEEP_MANIFEST:-}" ] && [ -s "$OUT/manifest.json" ]; then
  mkdir -p "$KEEP_MANIFEST" && cp "$OUT/manifest.json" "$OUT/manifest.md" "$KEEP_MANIFEST/" && \
    echo "   manifest kept in $KEEP_MANIFEST"
fi
if printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'; then echo "RELEASE VERIFICATION: FAIL"; exit 1; fi
echo "RELEASE VERIFICATION: PASS"
