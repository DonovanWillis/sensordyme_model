#!/usr/bin/env bash
# SensorDyme Python development environment: Linux, WSL and macOS.
#
# Creates (or refreshes) a virtualenv OUTSIDE the checkout by default, installs
# the repository's pinned requirements into it, and runs a smoke test (the
# backend + device files in scripts/dev/backend_tests.txt). Where this fits in
# the three-machine workflow: docs/DEVELOPER_SETUP.md.
#
#   scripts/dev/bootstrap_python.sh                     # ~/.venvs/sensordyme, backend profile
#   scripts/dev/bootstrap_python.sh --profile full      # what CI's pytest job installs
#   scripts/dev/bootstrap_python.sh --smoke-only        # re-run the smoke test only
#   scripts/dev/bootstrap_python.sh -- tests/test_rules.py -k fall   # your own pytest args
#
# Profiles (both resolve the repo's own pin files; nothing is pinned here):
#   backend  requirements-device.txt + cloud/requirements.txt + requirements-dev.txt
#            Everything the backend, the device runtime and the test suite import,
#            WITHOUT torch/ultralytics (multi-GB on Linux; only the mac_max profile,
#            benchmarks and training use them).
#   full     requirements.txt + cloud/requirements.txt + requirements-dev.txt
#            Same as the CI test job (.github/workflows/ci.yml). Pulls torch.
#
# Uses uv when it is on PATH (or at ~/.local/bin/uv), else `python -m venv` + pip.
# Python 3.11 or 3.12 only: mediapipe 0.10.14 has no 3.13 wheel.
# It never installs system packages and never writes inside the checkout unless
# you point --venv there yourself.
set -euo pipefail

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --venv DIR      virtualenv location (default: $SENSORDYME_VENV or ~/.venvs/sensordyme)
  --profile P     backend (default) or full
  --python SPEC   interpreter: a version for uv (3.12, 3.11) or a path/command
                  (default: $SENSORDYME_PYTHON or 3.12; without uv, falls back to
                  python3.12, python3.11)
  --no-smoke      install only
  --smoke-only    skip the install; run the smoke test in an existing venv
  --report DIR    also write install.log, freeze.txt, smoke.txt and junit.xml to DIR
  -h, --help      this text
  -- ARGS         pytest arguments that replace the default smoke-test file list
EOF
}

die() { echo "bootstrap_python: $*" >&2; exit 2; }
say() { echo ">>> $*"; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${SENSORDYME_VENV:-$HOME/.venvs/sensordyme}
PROFILE=backend
PYTHON_SPEC=${SENSORDYME_PYTHON:-3.12}
INSTALL=1
SMOKE=1
REPORT=""
SMOKE_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) [ $# -ge 2 ] || die "--venv needs a directory"; VENV=$2; shift 2 ;;
    --profile) [ $# -ge 2 ] || die "--profile needs backend|full"; PROFILE=$2; shift 2 ;;
    --python) [ $# -ge 2 ] || die "--python needs a version or path"; PYTHON_SPEC=$2; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --smoke-only) INSTALL=0; shift ;;
    --report) [ $# -ge 2 ] || die "--report needs a directory"; REPORT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; SMOKE_ARGS=("$@"); break ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

case "$(uname -s)" in
  Linux|Darwin) ;;
  MINGW*|MSYS*|CYGWIN*)
    die "this is the Linux/WSL/macOS script. On native Windows follow docs/DEVELOPER_SETUP.md (\"Windows-native\"), or run it inside WSL." ;;
  *) die "unsupported OS: $(uname -s)" ;;
esac

case "$PROFILE" in
  backend) REQS=(-r requirements-device.txt -r cloud/requirements.txt -r requirements-dev.txt) ;;
  full)    REQS=(-r requirements.txt -r cloud/requirements.txt -r requirements-dev.txt) ;;
  *) die "unknown profile: $PROFILE (backend|full)" ;;
esac

case "$VENV" in /*) ;; *) VENV="$PWD/$VENV" ;; esac
case "$VENV/" in
  "$REPO"/*) say "note: $VENV is inside the checkout; only .venv/ is git-ignored." ;;
esac
if [ -n "$REPORT" ]; then
  case "$REPORT" in /*) ;; *) REPORT="$PWD/$REPORT" ;; esac
  mkdir -p "$REPORT"
fi

UV=${UV:-}
if [ -z "$UV" ]; then
  if command -v uv >/dev/null 2>&1; then UV=$(command -v uv)
  elif [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"
  fi
fi

VPY="$VENV/bin/python"
cd "$REPO"

# Output goes to the terminal and, with --report, to a file as well.
logto() { if [ -n "$REPORT" ]; then tee "$REPORT/$1"; else cat; fi; }

if [ "$INSTALL" = 1 ]; then
  if [ ! -x "$VPY" ]; then
    mkdir -p "$(dirname "$VENV")"
    if [ -n "$UV" ]; then
      say "creating $VENV with $("$UV" --version) (python $PYTHON_SPEC)"
      "$UV" venv --python "$PYTHON_SPEC" "$VENV"
    else
      PY=""
      for c in "$PYTHON_SPEC" "python$PYTHON_SPEC" python3.12 python3.11; do
        if command -v "$c" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
      done
      [ -n "$PY" ] || die "no Python 3.11/3.12 found and no uv. Install uv (https://docs.astral.sh/uv/) or pass --python /path/to/python3.12"
      say "creating $VENV with $("$PY" --version 2>&1) at $PY (no uv: venv + pip)"
      "$PY" -m venv "$VENV" || die "venv creation failed (Debian/Ubuntu: the python3.12-venv package, or use uv)"
    fi
  else
    say "reusing $VENV"
  fi
fi

[ -x "$VPY" ] || die "no interpreter at $VPY (run without --smoke-only first)"
"$VPY" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' \
  || die "$VPY is $("$VPY" --version 2>&1); SensorDyme needs 3.11 or 3.12 (mediapipe). Delete $VENV and pass --python 3.12"

if [ "$INSTALL" = 1 ]; then
  say "installing the '$PROFILE' profile: ${REQS[*]}"
  t0=$(date +%s)
  if [ -n "$UV" ]; then
    "$UV" pip install --python "$VPY" "${REQS[@]}" 2>&1 | logto install.log
  else
    "$VPY" -m pip install --disable-pip-version-check "${REQS[@]}" 2>&1 | logto install.log
  fi
  say "install finished in $(( $(date +%s) - t0 ))s"

  # Provenance: which profile, interpreter, commit and pin files built this venv.
  sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
  {
    echo "profile: $PROFILE"
    echo "python: $("$VPY" --version 2>&1)"
    if [ -n "$UV" ]; then echo "installer: $("$UV" --version)"; else echo "installer: pip"; fi
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo 'unknown (not a git checkout)')"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha requirements.txt requirements-device.txt cloud/requirements.txt requirements-dev.txt
  } > "$VENV/sensordyme-bootstrap.txt"
  if [ -n "$REPORT" ]; then
    cp "$VENV/sensordyme-bootstrap.txt" "$REPORT/bootstrap.txt"
    if [ -n "$UV" ]; then "$UV" pip freeze --python "$VPY" > "$REPORT/freeze.txt"
    else "$VPY" -m pip freeze --disable-pip-version-check > "$REPORT/freeze.txt"; fi
  fi
fi

rc=0
if [ "$SMOKE" = 1 ]; then
  TESTS=()
  if [ ${#SMOKE_ARGS[@]} -gt 0 ]; then
    TESTS=("${SMOKE_ARGS[@]}")
  else
    while IFS= read -r line || [ -n "$line" ]; do
      line=${line%%#*}
      line=$(printf '%s' "$line" | tr -d '[:space:]')
      [ -n "$line" ] && TESTS+=("$line")
    done < scripts/dev/backend_tests.txt
  fi
  [ ${#TESTS[@]} -gt 0 ] || die "no tests to run (scripts/dev/backend_tests.txt is empty)"
  JUNIT=()
  [ -n "$REPORT" ] && JUNIT=("--junitxml=$REPORT/junit.xml")
  say "smoke test: pytest ${TESTS[*]}"
  set +e
  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen SDL_AUDIODRIVER=dummy \
    "$VPY" -m pytest -p no:cacheprovider -q -rs ${JUNIT[@]+"${JUNIT[@]}"} "${TESTS[@]}" 2>&1 | logto smoke.txt
  rc=${PIPESTATUS[0]}
  set -e
  say "smoke test exit code: $rc"
fi

cat <<EOF

Venv: $VENV
  source "$VENV/bin/activate"
  make test VENV="$VENV/bin"            # the full suite
  make test-backend VENV="$VENV/bin"    # the quick backend + device subset
EOF
exit "$rc"
