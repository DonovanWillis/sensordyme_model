.PHONY: help dev dev-debug dev-max dev-openpilot dev-skinnemoen dev-dashsentinel \
        bench accuracy accuracy-dataset accuracy-gate report deploy \
        test test-cov test-ios lint lint-fix fmt-check typecheck audit check mutants \
        setup-dev hooks clean

VENV := .venv/bin

# `make` with no target prints the menu rather than silently running the first
# rule. The good commands are only useful if they are discoverable.
.DEFAULT_GOAL := help

help: ## Show this help
	@echo "SensorDyme — make targets"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Full developer guide: docs/DEVELOPMENT.md"

# =============================================================================
# Run the detector
# =============================================================================

dev: ## Run the detector (default profile)
	$(VENV)/python main.py

dev-debug: ## Run the detector with debug overlay
	$(VENV)/python main.py --debug

dev-max: ## Run the detector at the mac_max profile
	$(VENV)/python main.py --profile mac_max --debug

dev-openpilot: ## Comparison: comma.ai openpilot driver monitoring, verbatim (model + policy)
	$(VENV)/python main.py --module driver_attention_openpilot --debug

dev-skinnemoen: ## Comparison: erlendskinnemoen's MediaPipe FaceMesh DMS, verbatim
	$(VENV)/python main.py --module driver_attention_skinnemoen --debug

dev-dashsentinel: ## Comparison: DashSentinel (vendored under third_party/)
	$(VENV)/python main.py --module driver_attention_dashsentinel --debug

# =============================================================================
# Benchmarks and reports
# =============================================================================

bench: ## Run the latency/thermal benchmark suite
	$(VENV)/python benchmarks/run_benchmarks.py

# NAMING — read before renaming any of these three.
#
# There are two different things called "accuracy" in this project and they are
# not interchangeable:
#
#   accuracy          HUMAN protocol. A person performs a timed scenario script
#                     in front of a running detector. Documented in README.md
#                     (line ~188) as `make accuracy`, so the name is load-bearing
#                     outside this file and is left alone.
#   accuracy-dataset  OFFLINE harness. Scores the detector against the State Farm
#                     dataset. Unattended, reproducible, no human needed.
#
# The request that added accuracy-dataset asked for it to be called `accuracy`.
# That name was already taken by the human protocol above and is referenced in
# README.md, which is outside this task's ownership — repointing it would have
# left the README describing a command that does something else entirely. If the
# rename is genuinely wanted, it is a one-line swap here PLUS an edit to
# README.md:188 and docs/DEVELOPMENT.md.

accuracy: ## HUMAN timed scenario protocol (needs `make dev-max` running + calibrated)
	$(VENV)/python benchmarks/accuracy_protocol.py run

accuracy-dataset: ## Offline accuracy run against the State Farm dataset (no human needed)
	@echo '>>> Needs data/state_farm. It is ~6GB and gitignored; if it is missing,'
	@echo '>>> prepare it with: .venv/bin/python scripts/prepare_state_farm.py'
	@echo
	$(VENV)/python benchmarks/accuracy_state_farm.py --per-class 100

# Exits NONZERO when a metric regresses past --tolerance (default 0.03), so this
# is the CI-usable form. baseline_phase1.json is the committed regression
# reference — it is the one file under benchmarks/results/ that is NOT gitignored
# (see .gitignore), because a gate with no baseline has nothing to compare to.
accuracy-gate: ## Score against the committed baseline; nonzero on regression
	$(VENV)/python benchmarks/accuracy_state_farm.py --per-class 100 \
		--compare benchmarks/results/baseline_phase1.json

report: ## Generate the weekly PDF report from data/events.csv
	$(VENV)/python -c "from core.reports import generate_weekly_report; print(generate_weekly_report('data/events.csv', 'data/reports'))"

# =============================================================================
# Quality gates
#
# These are the four things CI runs (.github/workflows/ci.yml). `make check`
# runs the same set locally, in the same order, so a red CI is reproducible on
# your machine without pushing.
#
# ruff and pyright are NOT in the project venv (see requirements-dev.txt). The
# targets below prefer whatever is on PATH — a Homebrew ruff/pyright works fine
# — and fall back to `uvx`, which fetches the pinned version on demand. That
# keeps `make lint` working on a fresh clone with nothing installed but uv,
# while `make setup-dev` gives you the pinned versions for real.
# =============================================================================

RUFF    ?= $(shell command -v ruff    2>/dev/null || echo "uvx ruff@0.16.4")
PYRIGHT ?= $(shell command -v pyright 2>/dev/null || echo "uvx pyright@1.1.411")

test: ## Run the full test suite (~20s)
	$(VENV)/python -m pytest -q

test-cov: ## Run the tests with a coverage report over the decision-making code
	$(VENV)/python -m pytest -q \
		--cov=core --cov=modules --cov=services --cov=cloud --cov=jupiter --cov=hardware \
		--cov-report=term-missing --cov-report=xml
	@echo
	@echo "Scope note: coverage is measured over the code that MAKES DECISIONS."
	@echo "scripts/ and benchmarks/ are operator tools run by hand and are left out"
	@echo "on purpose — including them inflates the number without telling you"
	@echo "anything about whether a fall still gets detected."

lint: ## Lint with ruff (rule set pinned in ruff.toml)
	$(RUFF) check .

lint-fix: ## Apply ruff's SAFE autofixes, then show what is left
	@echo '>>> Applying SAFE fixes only. Review the diff before committing: git diff'
	@echo '>>> --unsafe-fixes is deliberately not used; it can change behaviour.'
	$(RUFF) check . --fix

fmt-check: ## Show what `ruff format` WOULD change (informational — not a gate)
	@echo '>>> INFORMATIONAL ONLY. This is not a CI gate and "make check" skips it.'
	@echo '>>> "ruff format" currently rewrites 144 of the 209 source files. Taking'
	@echo '>>> that commit rewrites git blame across the whole project and'
	@echo '>>> reformats the vendored comparison modules, whose exact formatting is'
	@echo '>>> part of what makes them a faithful copy. That is a human decision,'
	@echo '>>> not a hook. See the [format] section of ruff.toml.'
	@echo
	-$(RUFF) format --check --diff .

typecheck: ## Type-check with pyright (config in pyrightconfig.json)
	$(PYRIGHT)

audit: ## Audit the pinned dependencies for known CVEs
	@echo ">>> Auditing the REQUIREMENTS FILES, not the local .venv: those pins are"
	@echo ">>> what actually ships to the Pi and to the cloud host, and they give the"
	@echo ">>> same answer on every machine. As of 2026-08-29 this reports 16"
	@echo ">>> advisories, concentrated in starlette (via fastapi) and zeroconf."
	@echo ">>> requirements-device.txt is listed EXPLICITLY, not left to the"
	@echo ">>> '-r requirements-device.txt' include inside requirements.txt: if"
	@echo ">>> pip-audit ever stops following includes, the pins that actually"
	@echo ">>> ship would be the ones silently going unaudited."
	uvx --python 3.12 pip-audit@2.10.1 --progress-spinner=off \
		-r requirements-device.txt -r requirements.txt \
		-r cloud/requirements.txt -r requirements-dev.txt

test-ios: ## Run the Swift logic tests (ios/Tests). Needs a Mac with Swift.
	@if command -v swift >/dev/null 2>&1; then \
		bash ios/Tests/run.sh; \
	else \
		echo "swift not found — skipping the iOS logic tests (not a Mac?)"; \
	fi

check: lint typecheck test test-ios ## Run every CI gate locally (lint + typecheck + tests)
	@echo
	@echo 'All gates passed. "make audit" is deliberately NOT part of this: it'
	@echo 'queries the PyPI advisory database, so it can start failing on a day'
	@echo 'when nothing in this repo changed. CI runs it as a separate,'
	@echo 'non-blocking job for the same reason.'

mutants: ## Mutation-test the safety thresholds (SLOW: minutes to hours)
	@echo ">>> A prior manual audit mutated 18 safety constants and the suite killed"
	@echo ">>> only 2 of them (11%). Those are the fall-detection stillness windows"
	@echo ">>> and drowsiness cutoffs — the numbers where being wrong means missing"
	@echo ">>> a real event while every test stays green. This target makes that"
	@echo ">>> measurement repeatable. Run it deliberately; it is not a CI gate."
	@echo
	$(VENV)/python -m mutmut run --paths-to-mutate modules/,core/
	$(VENV)/python -m mutmut results

# =============================================================================
# Environment
# =============================================================================

setup-dev: ## Install the pinned dev toolchain into .venv (needs uv)
	@command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }
	uv pip install --python $(VENV)/python -r requirements-dev.txt

hooks: ## Install the git pre-commit hooks
	@test -x $(VENV)/pre-commit || { echo "pre-commit not installed. Run: make setup-dev"; exit 1; }
	$(VENV)/pre-commit install

clean: ## Remove caches and coverage artifacts (never touches data/ or .venv)
	rm -rf .ruff_cache .pytest_cache htmlcov coverage.xml .coverage .coverage.*
	@# -path .venv -prune comes FIRST so find never descends into the venv:
	@# it is faster, and it means this can never delete a dependency's cache.
	find . \( -path ./.venv -o -path ./data \) -prune -o \
		-name '__pycache__' -type d -print0 | xargs -0 rm -rf

# =============================================================================
# Deploy
# =============================================================================

# config.yaml is DEVICE STATE, not code: it holds this unit's device_key, its
# cloud address, its camera orientation and its .hef path. Copying the repo's
# copy over it silently unpairs the device from the backend and un-rotates the
# camera, and the failure shows up much later as "the app stopped seeing it".
# Config changes ship as explicit migrations (scripts/migrate_config.py), never
# as a side effect of deploying code.
#
# CHANGED 2026-08-29: the dependency install below is new. Before it, deploy
# rsync'd code and restarted the service but never installed anything — so the
# first deploy that added a line to requirements.txt bricked the device with an
# ImportError at ExecStartPre and the only symptom was "it stopped coming back
# after a restart". `.venv/bin/pip install -r requirements-device.txt` is exactly
# what setup_pi.sh does, inside the same venv the systemd unit runs from
# (sensordyme.service: ExecStart=.../.venv/bin/python), and it is a no-op when
# nothing changed.
#
# CHANGED (Phase 4): requirements.txt -> requirements-device.txt. requirements.txt
# pins ultralytics (AGPL-3.0), and this line is how packages land on hardware we
# convey — so deploying the developer set would put the AGPL package back on the
# device on the next update even with setup_pi.sh fixed. See docs/LICENSING.md.
#
# NOT changed: the host (pi@sensordyme.local) and the rsync flags. rsync still
# runs WITHOUT --delete, so a file deleted from the repo stays on the device
# forever — including a module that main.py no longer imports but that a stale
# config.yaml still names. Adding --delete would also be the fastest way to
# destroy data/ on a device if an exclude were ever mistyped, so it is flagged
# here rather than changed unilaterally.
#
# `--exclude='*.pt'` is a LICENCE control, not a size one. A .pt checkpoint is
# gitignored but still sits in the working tree after any dev run (ultralytics
# auto-downloads yolo11n.pt on first use), and rsync copies the working tree, not
# the index — so a plain `make deploy` from a developer's Mac was putting an
# AGPL-3.0 model weight onto the device even with the pip install fixed. The
# weights carry the same licence as the code that produced them; see
# docs/LICENSING.md §"Model weights". Deliberately NOT excluding *.onnx:
# dmonitoring_model.onnx is openpilot's driver-monitoring model, which the device
# genuinely runs and which is not AGPL.
#
# `--exclude='*.hef'` closes the SAME hole for the format that actually matters.
# A .hef is the only thing the Pi can execute, models/ is gitignored, and
# docs/ACCELERATOR.md used to tell readers to download yolov11n.hef /
# yolov8s_h8l.hef into it -- Ultralytics-derived, so AGPL weights in a different
# wrapper. Excluding .pt while shipping .hef would have been a licence control
# aimed at the one format the device cannot run.
#
# `*.db` / `*.sqlite3` / `*.csv` are a DATA control, the third kind. `--exclude='data'`
# only covers the data/ DIRECTORY, so a dry run of this rsync was carrying
# cloud/sensordyme.db -- the backend's own database, 639KB -- plus e.csv,
# events.csv, t.csv and telemetry.csv from the repo root onto every device.
# Shipping a developer's event history and a backend DB to a customer is a
# privacy problem before it is a tidiness one, and nothing about it is visible
# in `make deploy`'s output.
#
# It also makes deployment DETERMINISTIC rather than merely legal: the device's
# model now comes only from setup_pi.sh, which pins a URL and verifies sha256 and
# refuses to install a mismatch. A model must never arrive on a device because it
# happened to be in someone's working tree.
demo-check: ## run the demo preflight on the device (DEMO READY / NOT READY)
	ssh pi@sensordyme.local 'cd ~/sensordyme && .venv/bin/python scripts/demo_preflight.py'

demo-cycle: ## rehearse every demo mode on the device, with restarts
	ssh pi@sensordyme.local 'cd ~/sensordyme && bash scripts/demo_cycle.sh'

deploy: ## rsync to pi@sensordyme.local, install deps, migrate config, restart
	rsync -av --exclude='.venv' --exclude='data' --exclude='.git' --exclude='__pycache__' \
		--exclude='config.yaml' --exclude='*.log' --exclude='.secrets' --exclude='*.pt' \
		--exclude='*.hef' --exclude='*.db' --exclude='*.sqlite3' --exclude='*.csv' \
		--exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='.claude*' \
		./ pi@sensordyme.local:~/sensordyme/
	ssh pi@sensordyme.local 'cd ~/sensordyme && .venv/bin/pip install -q -r requirements-device.txt'
	ssh pi@sensordyme.local 'cd ~/sensordyme && python3 scripts/migrate_config.py config.yaml'
	ssh pi@sensordyme.local 'cd ~/sensordyme && sudo systemctl restart sensordyme.service'
