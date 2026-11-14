# Mission 8.1 checkpoint log: pre-hardware correctness, reliability and release validation

Branch `feat/m8.1-release-validation`, local only, created from M8 HEAD `618880c`. Each entry records what was verified and how; nothing here ran on the Pi.

## P7. Repository state (2026-11-12, verified after `git fetch --all --prune`)

- `618880c` (M8) is on no remote. `upstream/integration/all` is still `11c524c`, PR #2 (our M7) merged. `origin/integration/all` is still `50e9441`. `origin/feat/m7-perception-engine` is `8519dec`.
- No new collaborator commits, and no open PRs in either repository.
- The active `gh` account and git credentials are both `monishallada`. Both repositories are public, so anything pushed earlier stays public history.
- Outgoing M8 range `11c524c..618880c` (43 commits): gitleaks found no leaks. No key, database, `.env` or setup-code files. The only private address is `192.168.1.20`, a test asserting that the simulated peripheral refuses non-loopback addresses. No tailnet hostnames or emails.
- Nothing pushed, merged or force-pushed in M8.1.

## P4. Compute fleet (2026-11-12 ~04:09 local)

- **oryn:** Tailscale shows it offline, last seen 6 h ago. `sdrun check -t oryn` exits 69 (SSH connect timed out). Not reachable from this Mac; no remote recovery attempted (see the recovery checklist in the final report).
- **quantcompute:** reachable, round trip 0.54 s.
  - Environment `sddml`; 8 CPUs; 16 GB job cap (49.6 GB available); 4 GB VRAM job cap; 2 slots; no active jobs.
  - Disk is not reported by `sdrun check`.
- **P5 dependency:** `research/m8/oryn_m8.sh` needs four prediction files that exist only in oryn's `$SDJOB_DATA`:
  - `bench_preds/m5.t.{reference,legacy}@native_val.jsonl`
  - `m7_detector/preds_full60_nms05.jsonl`
  - `exp08_full60/preds_val.jsonl`
  
  The EXP-08 checkpoint is on oryn too. The Mac has only summary JSONs (`~/sensordyme-fleet/runs/20261111-15*`). P5 is blocked until oryn returns; no substitute was fabricated.

## P6. Pre-hardware release checks (on `618880c` unless stated)

| # | Check | Result | How |
|---|---|---|---|
| 1 | Release file hashes reproducible | PASS | `make release-manifest VENV=… REF=618880c` twice: identical `manifest.json` (554 files, aggregate `7b574a50b0e52a4d…`). `git archive \| gzip -n` twice: identical sha256 `834468f0…6412` |
| 2 | Model artifact pins | PASS | `e69136d3…960e` identical in `setup_pi.sh` `MODEL_SHA256`, `config.yaml` `accelerator.yolo_hef_sha256` and `core/model_registry.yaml`; release gate "model pins consistent: True" |
| 3 | Device config compatibility | PASS, with an expected refusal | `deploy_preflight.py --expect-role driver_safety`: the release's own config and the `11c524c` config (after `migrate_config.py`) are both OK to deploy. The `50e9441` config (no `role`) is **refused** by design: add `role: driver_safety` by hand. Added to runbook F0 and the operator checklist |
| 4 | Legacy decoder default | PASS | `config.yaml` `accelerator.yolo_decoder: legacy`; release gate "decoder is legacy" |
| 5 | Production `.hef` unchanged | PASS | no `.hef` or model file in the diff `11c524c..618880c`; pin lines unchanged (only comments and the M8 setup_pi check) |
| 6 | Runtime dependencies match the lock | PASS | `requirements-device.{txt,lock}` and `requirements.txt` unchanged since `11c524c`; `check_device_deps.py`: device dependencies OK; release gate "device lock agrees with the device pins, every package hashed" |
| 7 | Security migration gated | PASS | `core/command_auth.py` `DEFAULT_MODE = "permissive"`, not overridden in `config.yaml`; `network.provisioning.auth: legacy`; `deployment` unset; new `manage.py` commands opt-in; no key rotation anywhere |
| 8 | Role and mandatory safeguards | PASS | `role: driver_safety`; 193 passed across rollback-guard, role-floor and M0.1 tests, `test_m6_autonomy_boot`, release-hardening and the driver capability tests |
| 9 | Rollback reproducible | PASS (software) | `tests/test_m8_lifecycle_release.py`; runbook O1–O3 unchanged; reproducible tarball (row 1). Real systemd is hardware-pending (runbook R6) |
| 10 | No false hardware claims | PASS | no `device_verified`, `production_permitted` or `validated_on_device` held in `core/model_registry.yaml`; release-status gate "no hardware-validation claim"; no such phrase in `release_status.yaml` |

**Defects found and fixed in the runbook:**
- `make release-manifest`, `deploy-preflight` and `release-verify` call `$(VENV)/python`, which defaults to `.venv/bin`. This checkout has no `.venv`, so A2 failed with exit 127 in the dry run. The runbook now exports `VENV=$HOME/.venvs/sensordyme-m01/bin` and passes it to each call.
- `make deploy-preflight` is hard-wired to `pi@sensordyme.local:~/sensordyme`. F0 now has a manual block for a device whose user, host or directory differs.

**New:** `docs/MONDAY_OPERATOR_CHECKLIST.md`.

## P3. Pyright (merged `bc7fabd`)

- Errors went from 42 to **0**; warnings from 978 to 981. The three new warnings are `reportMissingModuleSource` for the new `typings/onnx` stub, which is by design.
- No rule was disabled, the type-checking mode is unchanged, and no file-level ignores were added. Two existing `# type: ignore` comments were removed by fixing what they hid.
- Categories, every resolution and the genuine bugs are listed in `docs/TYPECHECK_STATUS.md`. One genuine bug: `uitest_control` could send "Bearer None" after a failed login.
- Full suite after the merge: 3,224 passed, 5 skipped, 0 failed (`m81-evidence/full_pytest_bc7fabd.txt`).

## P4 update: oryn back (2026-11-12 ~15:25)

- Tailscale shows oryn online, but `sdrun` refused it with exit 76, a host-identity mismatch: the hostname is now `ORYN`, while the fleet expected its previous Windows machine name.
- The pinned ed25519 host key still matched under `StrictHostKeyChecking=yes`, so it is the same machine, renamed.
- The user confirmed the rename. `expect_hostname` in `~/sensordyme-fleet/config/fleet.json` was changed to `ORYN`; the backup is `fleet.json.bak-20261112`.
- `sdrun check -t oryn`: RTX 4060 idle at 36 °C, 7.5 GB VRAM free, 9.6 GB RAM available, no active jobs.

## P5. CV (oryn job `20261112-152527-afe36f`, CPU only; `docs/research/M8_1_CV_FOLLOWUP.md`)

- **Person regression, paired, significant:** recall at precision 0.80 is 0.627 against 0.661 (Δ 0.033, CI 0.018–0.052).
- **Large people:** the gap is only 0.019, under M8's 0.02 bar for Arch B, so Arch B is not justified.
- **Phone:** recall at precision 0.80 is 0.789 against 0.594, not significant (p = 0.064 that EXP-08 is better; 128 phones).
- **Predicted-hand gate:** EXP-08's own predicted hands gate held phones as well as human labels. False-phone images go from 33.3% to 2.4%, and in-hand recall stays at 0.810.
- **Verdict:** no training before Monday. EXP-09 is justified, but needs approval.

## P1. Health freshness (merged `92d036d`, wording follow-up `07bcb40`)

- **Device:** each sample carries `sd-health-v1` stamps: `boot_id`, `generation`, `revision` and monotonic generation time. The uplink adds `age_s` at send time.
- **Backend:** judges freshness by the stamp (new, reboot, revised, repeat, out of order, retired boot, invalid, unverified), never by the device wall clock. `stale` = `link_stale` OR `health_stale`, so stale health falls to `unknown`.
- **Compatibility:** an older device without a stamp is reported as "unverified" and behaves as before.
- **App:** shows the new `monitoringStale` presence state.
- **Tests:** 9 end-to-end scenarios (`tests/test_m81_health_freshness_sim.py`) plus 26 unit tests.
- **Full suite at `07bcb40`:** 3,259 passed, 5 skipped, 0 failed. Pyright 0 errors. `make verify-release` PASS.
- Evidence, including a control trace that reproduces the old "protecting" on a sample 122 s old: `m81-evidence/freshness/`.

## P2. Deterministic iOS suite (merged `7eca2d9`; integrator fixes `13054e4`, `676801a`)

- **Cause of the skip:** test classes run alphabetically, so the audit class claimed the single shared device before journey 03 ran.
- **Fix:** a controller `POST /fresh` gives every test its own backend, database, JWT secret, rate limiters, account and device. Faults are reset each time, parallel testing is off, and the runner exits non-zero on any failure, skip or missing test.
- **Journey 28** (frozen health): the app shows Stale, "Status unknown", no "All clear" or "Protecting", and recovers after the thaw.
- **Integrator fixes found through the suite:**
  - The running mode's row kept its green check on frozen health. It now shows `activeUnconfirmed`, "Selected, status unknown", which journey 28 asserts.
  - `SDButtonStyle` read `isEnabled` where the environment is not installed, so disabled buttons could draw at full opacity.
- **Result at `676801a`: 28/28 in ONE invocation** (alphabetical, no retries, 30.5 min). The stream also ran 28/28 twice at `72aa192`, once alphabetical and once in random order.

## Final verification at `676801a`

- Full Python suite: 3,259 passed, 5 skipped, 0 failed.
- `ruff check` clean.
- Pyright: 0 errors, 984 warnings.
- `make verify-release` PASS.
- Swift logic tests: 1,116 checks, 0 failures.

## Remaining software defects (known, not fixed in M8.1)

1. **A stopped device's capability rows read "Not working — face landmarks stale"**, because components go stale while nothing runs. The overall state correctly says Stopped. The fix needs a "paused" capability state in the device→app contract and the app. Using "starting" would show misleading start-up wording.
2. **`make deploy` is hard-wired to `pi@sensordyme.local:~/sensordyme`.** The runbook works around it with a manual preflight; parameterising the target is a follow-up.
3. **Repo-wide `ruff check .` also lints `.claude/worktrees/*`.** Use `--exclude .claude`. Not a code defect.
