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
- **Results:** pending.
- **Limitations:**
  - Early stopping can fire before the final 5 no-aug/min-LR epochs, so those epochs never run.
  - The resumed EMA restarts from the saved EMA weights.
  - While training holds 8 GB of oryn's 9.2 GB job memory, other oryn jobs are limited to 1 GB. The torch export check OOM-killed at 1 GB (job `20261111-060011-c8d211`) and is deferred until training ends.

## F. Evaluation tooling (`7339ac8`)

- `research/pilot/compare_checkpoints.py` collates existing `detector_bench` reports, val only, refusing mixed manifests. It reports per-class AP50, AP50:95 and CI, precision at recall ≥ 0.70, P/R and false boxes on verified negatives at 0.2/0.3/0.5, and ECE (all scores and score ≥ 0.3). Tested by `tests/test_compare_checkpoints.py` (3 tests).
- `research/pilot/export_compare.py` exports a checkpoint the way the Hailo zoo graph was exported (RepConv fused, raw head), diffs the op set and outputs against the zoo ONNX, and times both in onnxruntime. It is a structural check, not a Hailo compile.
