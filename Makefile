.PHONY: help dev dev-debug dev-max dev-openpilot dev-skinnemoen dev-dashsentinel \
        bench accuracy accuracy-dataset accuracy-gate report deploy deploy-preflight \
        deploy-adopt deploy-status deploy-rollback demo-check demo-cycle \
        release-manifest release-verify verify-release \
        test test-cov test-ios lint lint-fix fmt-check typecheck audit check mutants \
        setup-dev hooks clean sim sim-backend sim-device sim-scenarios sim-e2e acceptance acceptance-full

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
# Virtual device: the real runtime + real backend on this Mac, scripted
# perception, clearly SIMULATED (docs/VIRTUAL_DEVICE.md). Point the iOS
# Simulator at http://localhost:8787 (its default) and claim with the printed
# code. Override the interpreter with SIM_PY=..., e.g.
#   make sim SIM_PY=~/.venvs/sensordyme-m01/bin/python SCENARIO=drowsy
# =============================================================================

SIM_PY     ?= $(VENV)/python
SCENARIO   ?= tour
SIM_BACKEND ?= http://127.0.0.1:8787

sim: ## Virtual device + local backend in one go (SCENARIO=tour; --list-scenarios via sim-scenarios)
	$(SIM_PY) -m sim.virtual_device --with-backend --backend $(SIM_BACKEND) --scenario $(SCENARIO)

sim-backend: ## The real backend locally on :8787, throwaway data, demo account, factory.json
	$(SIM_PY) -m sim.backend --port $(lastword $(subst :, ,$(SIM_BACKEND)))

sim-device: ## A virtual device against a running backend (FACTORY=path/to/factory.json)
	$(SIM_PY) -m sim.virtual_device --backend $(SIM_BACKEND) --scenario $(SCENARIO) \
		$(if $(FACTORY),--factory $(FACTORY),)

sim-scenarios: ## List the virtual device's scenarios
	$(SIM_PY) -m sim.virtual_device --list-scenarios

sim-e2e: ## Deterministic end-to-end tests: virtual device x real backend (seconds, no sleeps)
	PYTHONDONTWRITEBYTECODE=1 $(SIM_PY) -m pytest -p no:cacheprovider -q \
		tests/test_sim_e2e.py tests/test_sim_units.py tests/test_sim_seams.py

ACCEPTANCE_OUT ?= data/acceptance

acceptance: ## M8.5: every software suite + 30-journey matrix + short soaks + BLE harness; evidence in $(ACCEPTANCE_OUT)/acceptance.{json,md}
	PYTHONDONTWRITEBYTECODE=1 $(SIM_PY) scripts/sim/acceptance.py --out $(ACCEPTANCE_OUT) $(ACCEPTANCE_ARGS)

# The full acceptance additionally runs every XCUITest journey (the real iPhone
# app) via scripts/sim/ios_uitest.sh. It REQUIRES the iOS Simulator and
# xcodebuild, takes ~25 extra minutes, and must not run while anything else owns
# the Simulator (ports 8787/8790/8797 must be free). docs/ACCEPTANCE.md.
acceptance-full: ## M8.5: acceptance + the XCUITest journeys (needs the iOS Simulator; ~25 min extra)
	PYTHONDONTWRITEBYTECODE=1 $(SIM_PY) scripts/sim/acceptance.py --out $(ACCEPTANCE_OUT) --with-ios $(ACCEPTANCE_ARGS)

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

# Exits NONZERO when a metric regresses past --tolerance (default 0.03). Do NOT
# branch on `make accuracy-gate`'s exit code:
# make reports any failing recipe as 2, so the script's 1 (regression) and 2
# (cannot run) arrive identical. CI calls the script directly for that reason.
# baseline_phase1.json is the committed regression
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
	@echo ">>> same answer on every machine. As of 2026-10-14 this reports 2"
	@echo ">>> advisories: protobuf (via mediapipe) and pytest (test-only)."
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
# Deploy (M8.5: staged releases; docs/DEPLOYMENT.md)
# =============================================================================

# The device and its service path. DIR is relative to the login directory (or
# absolute) and is the path the systemd unit names; on a staged device it is a
# symlink into $(DIR).releases/. Before M8.5 both were hard-wired to
# pi@sensordyme.local:~/sensordyme; Bookworm images have no default `pi` user.
DEVICE ?= pi@sensordyme.local
DIR ?= sensordyme
# The guard every device-side step runs, streamed from the ref being deployed.
GUARD_AT_REF = git show $(REF):scripts/release/release_guard.py

# config.yaml is DEVICE STATE, not code: it holds this unit's device_key, its
# cloud address, its camera orientation and its .hef path. A deploy never ships
# one (the manifest excludes it and the stage refuses a bundle that contains it);
# the running release's copy is carried into the new release and migrated there
# (scripts/migrate_config.py), so a rollback runs on the pre-migration copy.
#
# History, kept because each line was a real failure:
# * 2026-08-29: deploys installed no dependencies, so a new requirement bricked
#   the device with an ImportError at ExecStartPre.
# * Phase 4: requirements-device.txt, not requirements.txt (ultralytics is
#   AGPL-3.0; docs/LICENSING.md).
# * The old rsync copied the WORKING TREE: AGPL *.pt/*.hef weights, the backend's
#   cloud/sensordyme.db, developer CSVs and cloud/.jwt_secret all reached devices
#   until each got an --exclude. M8.5 removes that whole class: a release is the
#   git objects of one commit, filtered by the same excludes as the manifest
#   (scripts/hw_validation/integrity_manifest.py), never a working tree.
# * M8.4 (M84-4/5/6): the in-place rsync was not atomic, verified nothing on the
#   device, had no --delete, and installed unhashed packages; its `||` rollback ran
#   over the same SSH session that had just failed. M8.5 replaces it:
#     stage     every file verified against the manifest (sha256 passed over SSH)
#               before it is written, in releases/<commit12>/ beside the live one;
#               its own venv from `pip install --require-hashes --only-binary=:all:
#               -r requirements-device.lock`; MediaPipe model files checked;
#     activate  detached ON the device: carry + migrate config.yaml, one rename(2)
#               of the service symlink, units, restart, confirm -- or switch back
#               (the previous release keeps its venv and config: no network).
# The model (.hef) still comes only from setup_pi.sh (URL + sha256 pin) and lives
# in the shared state dir; a model never arrives with a release.
demo-check: ## run the demo preflight on the device (DEMO READY / NOT READY)
	ssh $(DEVICE) 'cd $(DIR) && .venv/bin/python scripts/demo_preflight.py'

demo-cycle: ## rehearse every demo mode on the device, with restarts
	ssh $(DEVICE) 'cd $(DIR) && bash scripts/demo_cycle.sh'

# The device's config.yaml is never overwritten (above), so check IT -- not the
# repo's -- against the code about to land: schema, role floor, what the role
# needs, and whether the .hef on the device is the build its config pins. A
# device's role is never changed by a deploy; pass ROLE= to state which role
# this deployment is for (e.g. `make deploy ROLE=driver_safety`).
deploy-preflight: ## check the device's own config against this code (ROLE=driver_safety to require it)
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	scp -q $(DEVICE):$(DIR)/config.yaml "$$tmp/config.yaml" && \
	hef=$$($(VENV)/python scripts/deploy_preflight.py --config "$$tmp/config.yaml" --print-hef-path) && \
	if [ -n "$$hef" ]; then \
		set -- --hef-sha256 "$$(ssh $(DEVICE) "cd $(DIR) && sha256sum '$$hef' 2>/dev/null | cut -d' ' -f1")"; \
	else set --; fi && \
	$(VENV)/python scripts/deploy_preflight.py --config "$$tmp/config.yaml" $(if $(ROLE),--expect-role $(ROLE)) "$$@"

deploy: deploy-preflight ## stage REF= (default HEAD) on DEVICE=/DIR=, verify, switch atomically, confirm or roll back
	DEVICE='$(DEVICE)' DIR='$(DIR)' REF='$(REF)' PY='$(VENV)/python' TESTS='$(TESTS)' \
		FIND_LINKS='$(FIND_LINKS)' bash scripts/release/deploy.sh

deploy-adopt: ## ONE TIME per device: stop, turn DIR into the staged layout, start (docs/DEPLOYMENT.md)
	ssh $(DEVICE) 'sudo systemctl stop sensordyme.service'
	$(GUARD_AT_REF) | ssh $(DEVICE) 'python3 - adopt --root $(DIR)' || \
		{ ssh $(DEVICE) 'sudo systemctl start sensordyme.service'; echo "adopt refused; service restarted unchanged"; exit 1; }
	ssh $(DEVICE) 'sudo systemctl start sensordyme.service && sleep 30 && systemctl is-active sensordyme.service'

deploy-status: ## the device's releases and the last deploy's outcome
	$(GUARD_AT_REF) | ssh $(DEVICE) 'python3 - status --root $(DIR)'

deploy-rollback: ## switch DEVICE back to the release before the last confirmed deploy (TO=<id> for another)
	$(GUARD_AT_REF) | ssh $(DEVICE) 'python3 - switch --root $(DIR) --to $(or $(TO),previous) --user "$$(id -un)"'

# =============================================================================
# Release candidates (M6). A manifest pins exactly what a ref puts on a device;
# `release-verify` compares a device snapshot (scripts/hw_validation/
# integrity_manifest.py snapshot, run read-only ON the Pi) against it.
# See docs/RELEASE_MANIFEST.md and docs/MONDAY_RUNBOOK.md.
# =============================================================================
REF ?= HEAD
# Outside the checkout on purpose: a manifest left in the working tree is noise in
# every `git status` (deploys ship git objects only, so it can no longer leak).
RELEASE_DIR ?= $(HOME)/sensordyme-release

release-manifest: ## write $(RELEASE_DIR)/manifest.{json,md} for REF= (default HEAD); TESTS="..." records a test run
	@mkdir -p $(RELEASE_DIR)
	$(VENV)/python scripts/release/make_release_manifest.py build --ref $(REF) \
		--out-json $(RELEASE_DIR)/manifest.json --out-md $(RELEASE_DIR)/manifest.md \
		$(if $(TESTS),--test-ref "$(TESTS)")

release-verify: ## compare SNAPSHOT= (device snapshot JSON) against $(RELEASE_DIR)/manifest.json
	@test -n "$(SNAPSHOT)" || { echo "usage: make release-verify SNAPSHOT=path/to/snapshot.json"; exit 2; }
	$(VENV)/python scripts/release/make_release_manifest.py verify \
		--manifest $(RELEASE_DIR)/manifest.json --snapshot $(SNAPSHOT) --out $(RELEASE_DIR)/verify.json

# M6.5: the one offline release check (device deps, manifest + release gates, systemd
# units, focused tests). Seconds on the Mac; changes nothing. See docs/RELEASE_MANIFEST.md.
verify-release: ## offline release verification of REF= (default HEAD): deps, gates, units, focused tests
	bash scripts/release/verify.sh $(REF)
