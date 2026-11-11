# Mission 6.5 checkpoint log

Branch `feat/m6.5-overnight-hardening`, from `integration/m6-rc` at `f20d1cb`. Nothing pushed or
merged. One entry per checkpoint: commit, feature or defect, tests, resource use, outstanding limitation.

## A. Release candidate verified, branch created

- **Commit:** `f20d1cb` (RC; suite 2,676 passed / 5 skipped / 0 failed in oryn job `20261111-045454-2b68b1`).
- **Resources:** Mac 4.6 GB free at start; oryn idle, GPU 1.3 GB used by the owner's desktop.
- **Limitation:** none new.

## E. One authorized long training run (EXP-08 full, stock labels)

- **`536d786`: early stopping, best checkpoint, wall-clock budget, resume.**
  - `research/pilot/early_stop_patch.py` makes exact-match, idempotent edits to the pinned DAMO-YOLO trainer:
    - per-epoch val metrics go to `sd_epochs.jsonl`;
    - `best_ckpt.pth` is saved on val AP (COCO AP50:95, 3 classes);
    - a patience stop;
    - a deadline stop at an epoch boundary;
    - on resume, the best AP and patience count are restored from the log.
  - `exp08.sh` gains `WARMUP_EPOCHS`, `NO_AUG_EPOCHS`, `EARLY_STOP_PATIENCE`, `TRAIN_BUDGET_S` and `RESUME=1`. Resume uses the newest epoch checkpoint, loaded with `weights_only=True`. Scoring uses the best checkpoint.
- **Tests:**
  - `tests/test_early_stop_patch.py`: 6 tests on a stub trainer (patience, no-stop default, resume continuity, missing metric, deadline, refusal of an unexpected tree).
  - The patch was dry-run on the pinned tree, with `py_compile`.
- **Smoke on oryn (pilot scope, 2 epochs):**
  - Job `20261111-054305-6e409e`, with a budget of 60 s: stopped by the deadline after epoch 1, then saved the best checkpoint, wrote the log and scored. Run time 314 s, peak host memory 7.5 GB, peak GPU memory 4.9 GB, 63 °C.
  - Job `20261111-054846-802b51`, `RESUME=1`: "Resume Training from Epoch: 1", then trained epoch 2. Result: phone 0.656, person 0.615, hand 0.446 AP50. That is in line with the 2-epoch stock pilots (0.651 / 0.631 / 0.466 mean).
- **Pre-launch checks:**
  - source `536d786`;
  - dataset manifest `b878e8429d221f3f` (train 4,883 / val 1,585; test never exported);
  - base checkpoint sha256 `a5ee6382…` OK;
  - GPU 1.3 GB used, 11%, 39 °C, the owner's game not running;
  - oryn `$SDJOB_DATA` about 935 GB free; no permission blocker.
- **Launched: job `20261111-055506-564ff3` at 2026-11-11 05:55 EST.**
  - Settings: stock labels (weight 1), seed 1234, max 60 epochs, warmup 2, no-aug 5, patience 10 evaluations, training budget 20,700 s, job timeout 21,600 s, `--vram-mb 4500`, `--temp-c 80`, 6 CPUs, 8 GB.
  - No automatic restart.
- **Results** (job `20261111-055506-564ff3`, 05:55–09:01 EST, 11,198 s):
  - 60/60 epochs; no early stop (patience peaked at 9/10).
  - Best epoch 60, `best_ckpt.pth` sha256 `baad3842…f37b`.
  - Val AP50: hand 0.706, phone 0.798, person 0.690.
  - Meets 3 of the 4 pre-registered criteria; misses person preservation by 0.003.
  - Peak resources: card memory 4.9 GB, GPU temperature 64 °C, host memory 5.7 GB.
  - Details in `docs/research/EXP08_FULL_RESULTS.md`; registry binding `hand_box_detection.exp08_full60` is `experiment_only` (`eadc7e2`).
- **Limitations:**
  - Early stopping can fire before the final 5 no-aug/min-LR epochs, so those epochs never run.
  - The resumed EMA restarts from the saved EMA weights.
  - While training holds 8 GB of oryn's 9.2 GB job memory, other oryn jobs are limited to 1 GB. The torch export check OOM-killed at 1 GB (job `20261111-060011-c8d211`) and is deferred until training ends.

## F. Evaluation tooling (`7339ac8`)

- `research/pilot/compare_checkpoints.py` collates existing `detector_bench` reports, val only, refusing mixed manifests. It reports per-class AP50, AP50:95 and CI, precision at recall ≥ 0.70, P/R and false boxes on verified negatives at 0.2/0.3/0.5, and ECE (all scores and score ≥ 0.3). Tested by `tests/test_compare_checkpoints.py` (3 tests).
- `research/pilot/export_compare.py` exports a checkpoint the way the Hailo zoo graph was exported (RepConv fused, raw head), diffs the op set and outputs against the zoo ONNX, and times both in onnxruntime. It is a structural check, not a Hailo compile.

- **F results** (`eadc7e2`). Three-way val comparison, the export check and latency are in `docs/research/EXP08_FULL_RESULTS.md`.
  - Export/latency job `20261111-090242-3ab280`; head-subgraph op comparison job `20261111-090525-b99173`.
  - The head subgraph matches the zoo graph behind the `.hef` (77 Conv, identical kernels). Not compiled; INT8 unvalidated.

## B. Security (merged from agent worktrees)

- **4A/4C** (merge of `worktree-agent-a3f943…`).
  - **Signed commands:** `sd-auth-v1` (`core/command_auth.py`), HKDF + HMAC-SHA256 from the standard library. Requests are bound to device, command, body, nonce and the request they answer; replay is refused; keys rotate and revoke via `cloud/manage.py`. Config `cloud.command_auth` off|permissive|required, default permissive.
  - **Credential storage:** `core/secret_store.py` (0600, atomic, checksummed, redacted) plus `scripts/migrate_device_key.py`, which defaults to a dry run.
  - **Tests:** oryn `20261111-061442-e529e8`, 618 passed / 1 skipped.
  - **Limitations:** the backend still stores keys in plaintext; the nonce cache is per process; permissive mode cannot stop a signature-stripping attacker on the network.
- **4B** (merge of `m6.5/ble-pop`).
  - Proof of possession: setup code on a QR or label, PBKDF2 600k + HKDF + HMAC, AES-GCM credentials, explicit state machine, lockout. Default `legacy`; `deployment: production` refuses legacy.
  - **Tests:** Mac 109 passed; the Swift vector tests pass. Its oryn job hit a queue timeout behind training, so its dependent tests run in the combined suite.
  - **Defect found:** Swift JSON key order differed from Python; fixed with sorted keys on both sides.
  - **Limitations:** no BLE hardware; offline code guessing from a recorded session; no app setup screen yet.
- **4D/4E/release** (merge of `worktree-agent-a6e6732…`).
  - **Least privilege:** `deploy/systemd/sensordyme.service.hardened`, prepared and not installed. `systemd-analyze` exposure 9.0 → 1.7 (oryn `20261111-060654-3b7a25`). Pi 5 BLE advertising needs a `CAP_NET_ADMIN` drop-in.
  - **Dependencies:** `pytest` and `anthropic` are out of `requirements-device.txt`; `cryptography==50.0.2` added; `requirements-device.lock` hashed (37 packages).
  - **Release tooling:** `make verify-release`, gates, `release_status.yaml`.
  - **Open:** mediapipe pulls `opencv-contrib-python` next to the pinned `opencv-python`.

## C/G. iOS UI automation and usability (merge of `worktree-agent-a5188a2…`)

- **XCUITest:** 20 journeys against the real backend and virtual device; 19 pass.
- **Journey 10 fails.** The inline Ask Jupiter bar's field never takes keyboard focus.
  - `b8fcff5` fixes a real stale-sheet hazard, but the rerun on 2026-11-11 still failed on focus.
  - This is an open defect.
- **Fixed in the app:**
  - wrong instructions for turning rules off;
  - preset reasons at 2.4:1 contrast;
  - 3.9:1 grey text;
  - small tap targets;
  - banners VoiceOver could not activate;
  - Dynamic Type, Reduce Motion and spoken state labels.
- **Post-fix accessibility audit** (first run after the fixes): 40 new issues, unfixed and recorded in `~/sensordyme-audit/2026-11-10/m65-evidence/ios-audit-post-fix/`. They are mainly mono labels that ignore Dynamic Type, clipping, and 8 contrast failures.
- **Resource:** the Mac disk hit ENOSPC once. Free space swung between 0.1 and 4.5 GB from activity outside this mission (system caches, downloads), which bounded how much UI testing could run.

## D. Deep reliability (merge of `worktree-agent-aaf31b3…`)

- **What was added:** `sim/invariants.py` (an invariant monitor on every transition), 21 multi-stage scenarios, and 12 bounded stress tests.
- **Every invariant held.** Measured on oryn:
  - queue held at 500 under 3,000 offline alerts;
  - retry storm limited to 16 attempts in 2 s;
  - 800 concurrent alerts, none doubled;
  - threads, file handles and memory objects stable over 12 restarts and 20 minutes with faults.
- **Real defects found and fixed:**
  1. Person boxes never reached custom rules.
  2. A non-object `calibration.json` failed the driver module on every boot.
- **Pre-existing failure:** the oryn `test_m01_role_floors` OOM also happens on the base commit `536d786`. Fixed in `73ba319` (the test now stubs `require_backend`).

## Release hardening (integrator)

- `release_status.yaml`: the six M6.5 entries filled with evidence and tiers; the status test now requires them filled.
- `make verify-release` PASS at `bc4ceba`; systemd step skipped on macOS.
- `docs/MONDAY_RUNBOOK.md`: an M6.5 notes block for K and L.
