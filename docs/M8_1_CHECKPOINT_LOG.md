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
