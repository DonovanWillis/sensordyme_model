# Integration manifest — integration/m6-rc

Local only (not pushed). Base: `integration/all` = `origin/integration/all` = `50e9441` (collaborator's ORYN EDGE merge).

| Order | Branch | Tip | How | Commits |
|---|---|---|---|---|
| ff | `feat/m0.1-alert-episodes-safety-floor` | `67df21f` | fast-forward | 6 |
| ff | `feat/m0.2-reliable-monitoring` | `973aba3` | fast-forward | 5 |
| ff | `feat/app-status-contract` | `895e783` | fast-forward | 3 |
| merge | `feat/m4-detector-baseline` | `4fc6fae` | `6aa3a8a` (no conflicts) | 30 (shared ancestry counted) |
| merge | `feat/m5-decoder-candidate` | `6aea91e` | `52264d0` (no conflicts) | 13 (shared ancestry counted) |
| merge | `feat/m5-model-registry` | `e68075f` | `b7be39c` (no conflicts) | 14 (shared ancestry counted) |
| merge | `feat/m5-hw-validation` | `3814933` | `a32a84a` (no conflicts) | 13 (shared ancestry counted) |
| merge | `chore/redact-audit-identifiers` | `afee07a` | `b538feb` (no conflicts) | 1 (shared ancestry counted) |

`feat/m4-detector-baseline` contains `feat/m4-eval-harness` (`7bd60f1`).

## Every incorporated commit

```
5e91397 M0.1: episode-aware alert policy with bounded repetition and escalation
4eaaa8a M0.1: deployment roles with role-specific safety floors at config load
949e2bd M0.1: additive custom rules, fail-closed rule validation, preset floor, explicit stop
03ed88e M0.1: backend role enforcement and explicit stop; acceptance and fuzz tests
a41bcde M0.1: dashboard-channel role test; type-check and lint cleanups
67df21f M0.1: restore CRLF line endings in modules/base.py and tests/test_debounce.py
39d11bf M0.2: detector retry with bounded backoff, fresh accelerator probe, stall-proof evidence clock, watchdog hold release, no reboot loop on bad config or module
f28318a M0.2: authoritative monitoring state, role-aware capability health, actuator readiness, impaired alert
9282ead M0.2: deployment preflight against the device's own config; make deploy runs it first
e6ccdb9 M0.2: document the monitoring state model and the hardware checks it still needs
973aba3 M0.2: deploy preflight CLI no longer crashes under python -OO
7bd60f1 M4-B: real-video evaluation harness — manifests, annotations, deterministic replay, event/frame metrics
3581f22 M4-C: Open Images V7 subset pipeline — licence-filtered selection, provenance, dedup, author-grouped splits
2bf19e3 App status contract: typed device status and machine-readable refusal codes
286bff6 M4-C: detector benchmark — production-faithful DAMO-YOLO decode variants, YOLOX, Open Images scoring
4569246 M4: draft in-cabin pilot recording protocol (stationary vehicle, 3-5 consenting adults; not approved)
26d4c35 iOS: show the device's real monitoring state, not a green "All clear"
895e783 Status contract: a live device without health is unknown, not stale
bf52575 M4-C: do not score a class on images verified-positive but never boxed (annotation gaps)
81ee668 M4-C: load torch's CUDA libraries before creating the ORT CUDA session (as the fleet benchmark does)
9afe4e0 M4-C: fix GpuMonitor shadowing Thread._stop; record GPU state before each run (shared worker)
6636b70 M4-C: COCO export of train/val splits (person, phone, hand); reports unverified-class label-noise exposure
47d719a M4-C: benchmark progress lines, per-phase timing, partial report after every run
3b68cf6 M4-C: DAMO-YOLO-S head nodes from its own zoo config
7f6d2b9 M4: EXP-08 fine-tune spec and the bounded 2-epoch pilot script (resource envelope fixed before running)
6a4c2df M5: EXP-08a pilot — validation predictions with DAMO's own pipeline, Open Images scoring incl. hand, training telemetry
998d3b4 M5: selectable DAMO-YOLO reference decoder candidate (legacy stays the default)
52b825d M5: EXP-08a venv layers sdgpu's site-packages (sdgpu is itself a venv)
cfb7f8f M5: bench runs the product decoder (legacy/reference), preprocessing and NMS variants; multilabel FP analysis; hand-gate interaction study
7bb01a6 M5: EXP-08a launches via python -m torch.distributed.run (torchrun lives in sdgpu's bin)
afee07a docs: redact deployment identifiers from the public audit report
6d8df09 M5: in-cabin readiness — enforced label vocabulary v1, operator checklist, consent workflow, transfer procedure (nothing recorded)
e4317af M5: EXP-08a installs setuptools (thop imports distutils, removed in Python 3.12)
9ea7835 M5: typed model registry — cards, evaluation ladder, preset/role chains, re-evaluation impact
5b71129 M5: document the model registry, the evaluation ladder and the backend integration plan
7383be7 M5: EXP-08a checkpoint from the authors' Drive mirror, sha256-pinned and verified as a safe-loadable zip
e68075f M5: registry — remove State Farm as evidence (competition-use-only data cannot support a product status)
a894ec5 M5: hardware validation scripts for the Pi 5 + Hailo-8L + Camera Module 3 Wide
3814933 M5: document the reversible hardware validation procedure (not executed)
4fc6fae M5: bench saves float32 device-decoder outputs as JSON floats
6aea91e M5: document the decoder candidate — defects, measured effect, multilabel FP analysis, threshold trade-offs, Hailo compatibility gaps
6aa3a8a Integrate feat/m4-detector-baseline into the M6 release candidate
52264d0 Integrate feat/m5-decoder-candidate into the M6 release candidate
b7be39c Integrate feat/m5-model-registry into the M6 release candidate
a32a84a Integrate feat/m5-hw-validation into the M6 release candidate
b538feb Integrate chore/redact-audit-identifiers into the M6 release candidate
```

## Verification

- Full suite on oryn at the merge head: **2,330 passed, 5 skipped, 0 failed** (job 20261111-032119-cc88f3); ruff clean.
- `git diff 50e9441 HEAD -- ios`: only app-status-contract additions and the redaction fixture; the collaborator's files are otherwise unchanged.
- Every source branch tip is an ancestor of this branch (no dropped commits).
