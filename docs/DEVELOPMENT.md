# Development

How to set up, run, test, lint and type-check SensorDyme on a Mac.

For verifying a **deployment** (Pi boot, health, recovery, networking) see
[TESTING.md](TESTING.md). This document is about the development loop.

---

## 1. Setup

Requires Python 3.11 or 3.12 — **not 3.13**. `mediapipe==0.10.14` publishes no
3.13 wheel, so the install fails outright there.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then the developer toolchain (ruff, pyright, pip-audit, pre-commit, mutmut,
coverage). This needs [uv](https://docs.astral.sh/uv/):

```bash
make setup-dev
```

> The project venv has **no `pip` module**. `python -m pip install ...` will
> not work inside it. Use `uv pip install --python .venv/bin/python <pkg>`,
> which is what `make setup-dev` and CI both do.

Optionally, install the git hooks:

```bash
make hooks      # writes .git/hooks/pre-commit
```

`make` with no arguments lists every target.

---

## 2. Run it

```bash
make dev              # default profile
make dev-debug        # with the debug overlay
make dev-max          # mac_max profile (heavier models, VLM judge)
```

The three comparison implementations run the same framework with someone
else's algorithm, on the same camera and the same `data/events.csv`, so their
accuracy can be measured head to head against this repo's own pipeline:

```bash
make dev-openpilot      # comma.ai openpilot v0.9.7, model + policy verbatim
make dev-skinnemoen     # erlendskinnemoen's MediaPipe FaceMesh DMS, verbatim
make dev-dashsentinel   # DashSentinel @3874e05b, vendored under third_party/
```

Those three modules and `third_party/` are **vendored deliberately**. Their
constants, naming and structure are the thing being measured, so both ruff and
pyright are narrowed on them on purpose (see `ruff.toml` per-file-ignores and
`pyrightconfig.json` `ignore`). Do not reformat them.

---

## 3. Tests

```bash
make test        # full suite, ~20s
make test-cov    # + coverage over core/ modules/ services/ cloud/ jupiter/ hardware/
```

As of 2026-08-29: **1233 tests, all passing, in ~23 seconds.**

> Every count in this document was measured on 2026-08-29 while several people
> were actively adding tests and fixing modules, so the exact numbers move.
> They are here to convey the *shape* of the thing — a suite in the low
> thousands running in under half a minute, a lint report in the hundreds not
> the thousands — not as a figure to assert against. Re-measure with
> `make test`, `make lint` and `make typecheck`.

`pytest.ini` sets a **60-second per-test timeout**. The whole suite runs in 20
seconds, so a test that reaches 60 is not slow, it is stuck — a thread that
never joins, a socket read with no timeout, a watchdog waiting on a device that
is not there. Those shapes are everywhere in `services/supervisor.py`,
`core/uplink.py` and `services/ble_gatt.py`, and without the timeout a single
hang burns the entire CI budget and reports nothing.

Collection scope is pytest's default. Bare `pytest` picks up `tests/` **plus 3
tests from `third_party/DashSentinel/tests`** — the vendored repo's own suite,
kept because it is the check that the vendored copy still behaves like the
upstream it is supposed to reproduce.

`--cov` deliberately excludes `scripts/` and `benchmarks/`. Those are operator
tools run by hand; including them inflates the coverage number without saying
anything about whether a fall still gets detected.

### Mutation testing

```bash
make mutants     # SLOW: minutes to hours. Not a CI gate.
```

Passing tests are not the same as tests that would notice a regression. A prior
manual audit mutated 18 safety constants — fall-detection stillness windows,
drowsiness PERCLOS cutoffs, the numbers where being wrong means missing a real
event — and the suite killed only **2 of 18 (11%)**. `make mutants` makes that
measurement repeatable instead of a one-off. Treat a low kill rate on a safety
threshold as a missing test, not as a mutmut problem.

---

## 4. Accuracy benchmarks

Three different things, easily confused:

```bash
make bench              # latency / thermal. Unattended.
make accuracy           # timed HUMAN scenario protocol. Needs a person.
make accuracy-dataset   # offline accuracy vs the State Farm dataset. Unattended.
make accuracy-gate      # same, scored against the committed baseline. CI-safe.
```

`make accuracy` is **not** unattended. It drives
`benchmarks/accuracy_protocol.py`, which walks a person through timed scenarios
and scores what the running detector reported. `make dev-max` must already be
running **and calibrated** in another terminal before you start it, or the
timings mean nothing. This is the one the README documents.

`make accuracy-dataset` and `make accuracy-gate` drive
`benchmarks/accuracy_state_farm.py` against `data/state_farm` (~6GB,
gitignored, prepared by `scripts/prepare_state_farm.py`). No human needed.
`accuracy-gate` compares against `benchmarks/results/baseline_phase1.json`. It
has three distinct exit codes, and the distinction is what makes it safe to
automate:

| exit | meaning | CI behaviour |
|---|---|---|
| `0` | clean run, nothing regressed | pass |
| `1` | a metric regressed past `--tolerance` (default 0.03) | **fail** |
| `2` | could not run — dataset missing, or `--compare` baseline missing | skip, green |

Code `2` is deliberately not `1`. An earlier version used
`raise SystemExit("message")`, and Python exits `1` for any non-integer
argument — so "I could not run" and "accuracy regressed" were the same code, and
any CI workaround broad enough to tolerate a missing dataset would also have
swallowed every real regression.

> Read the script's own CAVEAT before quoting any of its numbers: State Farm's
> camera is passenger-side looking across the cabin, SensorDyme's mount is
> dash-centre driver-facing. These are domain-shifted **lower bounds**, useful
> for regression detection, not for certification.

### What is and is not versioned under `benchmarks/results/`

`.gitignore` keeps `benchmarks/results/*` out of git — timestamped runs from all
three harnesses are machine- and moment-specific — with one deliberate
exception, `!benchmarks/results/baseline_*.json`. The baseline is the regression
reference `--compare` scores against, so it has to be versioned or the gate has
nothing to compare to.

---

## 5. Lint

```bash
make lint        # ruff check .
make lint-fix    # apply ruff's SAFE fixes only, then show what is left
```

`ruff.toml` **pins the rule set**, and that is the point of it. Before it
existed there was no ruff config anywhere in this repo, so ruff walked up the
directory tree looking for one and inherited whatever it found — running
`ruff check` from different working directories produced 321, 523, 648 and 687
findings for the same code. Lint output that depends on where you are standing
is not a signal.

Current state (2026-08-29, ruff 0.16.4):

| | findings |
|---|---|
| ruff defaults, no config (`ruff check --isolated`) | 715 |
| **this repo's pinned rule set** | **327** |

The rule set is chosen for what actually breaks an unattended device: pyflakes
(`F`), bugbear (`B`), blind-except (`BLE`), bandit (`S`), and the pylint error
and warning sets. Rules that would have dominated the report without finding
bugs — 2546 missing annotations, 2000 docstring-style, 466 magic-value
comparisons, 197 `print` calls — are listed in `ruff.toml` with their measured
counts and the reason each was left out, so the decision is reviewable.

Two real bugs it found on the first run:

- `core/network.py:224` — `threading` used but never imported (`F821`), which
  would have raised `NameError` on the path that reached it. Since fixed.
- `scripts/jupiter_eval.py:108` — a closure capturing a loop variable (`B023`),
  so every callback sees the last iteration's value. **Still open.**

### Formatting is not a gate

```bash
make fmt-check   # informational; never fails, and `make check` skips it
```

`ruff format` would rewrite **144 of the 209** source files in one commit. That
rewrites `git blame` across the whole project and reformats the vendored
comparison modules, whose exact formatting is part of what makes them a
faithful copy. Taking that commit is a human decision; `line-length = 100` in
`ruff.toml` describes the existing style (p95 = 88, p99 = 103) and is the
target for new code.

---

## 6. Type checking

```bash
make typecheck   # pyright
```

`pyrightconfig.json` does three things that matter:

1. **Points pyright at `.venv`.** Without it pyright used the system Python, so
   numpy, cv2, mediapipe, pytest and fastapi all resolved to nothing.
2. **Teaches it the flat layout** (`extraPaths: [".", "scripts"]`), because
   `core`, `modules` and `services` are imported by name from the repo root.
3. **Stubs the Pi-only modules** in `typings/` — `picamera2`, `libcamera`,
   `gpiozero`, `hailo_platform`, `dbus`, `gi`. These genuinely do not exist on
   a Mac (they are apt packages on Pi OS), every import of them is already
   inside a `try/except ImportError` with a fallback, and unstubbed they
   produced 14 errors nobody on a Mac can fix.

| | errors |
|---|---|
| no config | 596 |
| **with `pyrightconfig.json` + `typings/`** | **8** |

Those 8 are all specific and real: `core/detectors.py` returning a plain tuple
where a `Box` is declared, an `Optional` called as a function in
`services/ble_gatt.py`, and possibly-unbound locals in `services/ble_gatt.py`
and three test files. (It was 15 when the config first landed; the missing
`threading` import in `core/network.py` and the possibly-unbound `batch` in
`core/uplink.py` have since been fixed — which is the config doing its job.)

Mode is `basic`, not `strict`: large parts of this codebase are unannotated and
`strict` would report thousands of missing-annotation errors. The noisy
Optional/attribute rules are **warnings** rather than errors — still printed
(484 of them), but they do not fail the gate, so "the gate is red" means
"something here will actually raise". `cloud/` is the exception: it holds real
accounts and device keys, so those rules are promoted back to errors there.

> **Version skew:** Homebrew ships pyright 1.1.413; the newest release on PyPI
> is 1.1.411, which is what `requirements-dev.txt` and CI pin. They disagree
> slightly (1.1.413 reports a few more). To reproduce CI exactly:
> `uvx pyright@1.1.411`.

---

## 7. Dependency audit

```bash
make audit
```

Resolves `requirements.txt`, `cloud/requirements.txt` and
`requirements-dev.txt`, transitive dependencies included, against the PyPI
advisory database. It must run on Python 3.12 — mediapipe 0.10.14 has no 3.13
wheel and resolution fails before anything is audited.

Known state as of 2026-08-29 — **16 advisories in 4 packages**:

| package | version | count | how it gets here | fixed in |
|---|---|---|---|---|
| starlette | 0.46.2 | 9 | transitive, via `fastapi==0.115.*` | 0.47.2 → 1.3.1 |
| zeroconf | 0.132.2 | 5 | `cloud/requirements.txt` pins `0.132.*` | 0.149.x |
| protobuf | 4.25.9 | 1 | transitive, via mediapipe | 5.29.6 / 6.33.5 |
| pytest | 8.3.3 | 1 | test-only | 9.0.3 |

The `zeroconf` pin is the notable one: `0.132.*` locks the series that contains
all five advisories, and the fixes are only in 0.149.x. Fixing any of these
means editing production pins, which is a deliberate change with a test run
behind it — not something CI should force.

---

## 8. CI gates

`make check` runs `lint`, `typecheck` and `test` — the same three blocking
gates as CI, in the same order, so a red CI is reproducible locally without
pushing.

`.github/workflows/ci.yml` defines four jobs on push to `main` and on every PR:

| job | blocking | notes |
|---|---|---|
| `lint` (ruff) | yes | no dependencies needed; finishes in under a minute |
| `typecheck` (pyright) | yes | installs `.venv`, which `pyrightconfig.json` requires |
| `test` (pytest + coverage) | yes | uploads `coverage.xml` and `junit.xml` |
| `accuracy-gate` | yes, **when it runs at all** | skips green on hosted runners |
| `audit` (pip-audit) | **no** | reports into the job summary |

`accuracy-gate` **will skip on every hosted run.** `data/state_farm` is ~6GB and
gitignored, so a GitHub-hosted runner does not have it; the job detects that and
exits green with a note in the summary. It becomes a live gate only on a
self-hosted runner with the dataset on disk.

The job uses **two** mechanisms, and they cover different things:

- The **pre-flight dataset check** is a cost optimisation. Hosted runners never
  have `data/state_farm`, so without it every push would apt-get, build a venv
  and download ~800MB of torch just to discover the dataset is absent.
- The **exit-code branch** is the correctness mechanism, and it catches what the
  pre-flight cannot see. The pre-flight only tests for `val_manifest.csv`, so if
  `baseline_phase1.json` were deleted the check would pass, the harness would
  exit `2`, and the job would go red on a "cannot run" condition. Branching on
  the code prevents that false red.

Neither is redundant; don't remove one as duplication of the other.

`audit` is non-blocking on purpose. It queries an external advisory database
that changes without this repo changing; a new CVE published against starlette
on a Tuesday must not turn every open PR red. There are already 16 findings —
making it blocking on day one means disabling it on day two.

> ### The CI workflow has never run
>
> This repository has **no git remote** (`git remote -v` is empty); it exists
> only on one laptop. No runner has ever executed a line of `ci.yml`. Every
> command in it was verified by hand on macOS; the Ubuntu runner, the caching
> and the YAML wiring are unproven. **Expect the first run to fail** — the most
> likely causes are the Linux dependency install (`ultralytics` pulls torch,
> ~800MB) and tests that need a display, camera or audio device. Do not treat a
> green badge as meaningful until it has actually run once.
>
> `.coderabbit.yaml` is in the same position: it configures PR review, and
> reviews run on pull requests, so it does nothing until work lands via PRs on
> a remote.

### Pre-commit hooks

`.pre-commit-config.yaml` runs ruff (with autofix), a private-key detector, a
2MB added-file guard, merge-conflict and case-collision checks, and YAML/JSON/
TOML validation. It does **not** run pyright or pytest: a hook slower than a
few seconds gets bypassed, and a hook that is routinely bypassed is worse than
no hook.

The secrets and large-file hooks are the ones that earn their place here.
`cloud/*.db` — real account emails, password hashes, salts, device keys, claim
codes — was committed and had to be purged from history on 2026-08-06. And
`data/state_farm` is 5.9GB sitting next to a repo whose entire history is 26MB.

---

## 9. Deploying

```bash
make deploy      # rsync → install deps → migrate config → restart the service
```

Read [TESTING.md](TESTING.md) for what to verify after a deploy.

Two things about this target are worth knowing:

- **`config.yaml` is never deployed.** It is device state — this unit's
  `device_key`, its cloud address, its camera orientation, its `.hef` path.
  Overwriting it silently unpairs the device from the backend, and the symptom
  shows up much later as "the app stopped seeing it". Config changes ship as
  explicit migrations (`scripts/migrate_config.py`).
- **`rsync` runs without `--delete`.** A file deleted from the repo stays on
  the device forever. That is the current behaviour and it is intentional
  insurance against a mistyped `--exclude` wiping `data/`, but it means a
  removed module can linger and still be named by a stale config.

---

## Reference

| file | what it pins |
|---|---|
| `requirements.txt` | production deps; installed on **both** Mac and Pi |
| `cloud/requirements.txt` | the FastAPI backend's deps |
| `requirements-dev.txt` | the dev toolchain; **never** installed on the Pi |
| `ruff.toml` | the lint rule set, with the reason for every inclusion and exclusion |
| `pyrightconfig.json` | type-check config (JSONC — `//` comments are legal) |
| `typings/` | Any-typed stubs for Pi-only modules |
| `pytest.ini` | the 60s per-test timeout and `--strict-markers` |
| `.pre-commit-config.yaml` | git hooks |
| `.github/workflows/ci.yml` | CI (unverified — see above) |
