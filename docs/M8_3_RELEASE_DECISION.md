# M8.3 release decision: which candidate goes to the Monday hardware session

**Status: recommendation only.** Nothing has been pushed, merged into a shared branch or deployed. Every step below that publishes code needs the owner's approval.

## Candidates (verified 2026-11-13)

| | Minimal: `feat/m8.2-paused-capabilities` @ `0e9343b` | Integrated: `feat/m8.3-hardware-truth` (HEAD recorded at approval) |
|---|---|---|
| Contents | M0.1–M8.1 + paused capabilities; pre-redesign app | All of the minimal candidate + Liquid Precision redesign (`66cf6b6`) + M8.3 (security, truth, events, hardware, stream, iOS product) |
| Python suite | 3,263 passed / 5 skipped / 0 failed | 3,463 passed / 5 skipped / 0 failed (at `1d3ae7e`). Later commits change no Python except `scripts/sim/ios_uitest.sh`, which was validated by a journey run |
| Pyright / ruff | 0 / clean | 0 / clean |
| `make verify-release` | PASS | PASS (`1d3ae7e`) |
| Swift logic tests | 1,128 checks | 1,501 checks (`1d3ae7e`), plus new iOS-product suites (33 suites) |
| XCUITest, one invocation | **Never run on this commit.** Its app changes were only exercised inside the design branch | **28/28** at `ec71c04`, whose tree is byte-identical to the merge `f75d557`. Later: runner fix (journey 05 re-run, passed) and docs only |
| Hardware | none | none |

## What the minimal candidate lacks for a first hardware session

| Gap in the minimal candidate | Why it matters on Monday | Fixed in M8.3 by |
|---|---|---|
| A malformed Hailo result is silently wrong: an NCHW box head gives wrong boxes, a missing output gives "no detections", and NaN boxes crash the driver module | The first contact with the real `.hef` is exactly where layout surprises appear. Silent zero detections would look like "nobody in view" | HARDWARE `ee9f6e6` (named errors; load-time layout checks) |
| The app labels a frozen picture LIVE for up to 15 s, and a new viewer gets the cached last frame however old | The live feed is how the operator judges the camera on Monday | STREAM (per-frame seq/age, evidence-based liveness) |
| One sample can say "failed" beside "protecting"; the backend passes contradictions through; heat never degrades | Status is the session's main instrument | TRUTH (single evaluation, `_make_coherent`, thermal condition) |
| A stop has no outcome and is applied whenever the device next reconnects | K-section stop/resume checks | TRUTH (pending/confirmed/mismatch/timed_out; late stops withdrawn) |
| A re-reported episode becomes a second stored alert and push; event times trust the Pi clock (no RTC); no evidence that the buzzer actually played | G2/L3 alert checks and the app timeline | EVENTS (episode uniqueness, `occurred_at`, alert delivery evidence) |
| Secrets in logs (ntfy topic, PSK on a stalled rollback); no `no-store` | Monday logs are collected as evidence | SECURITY |

## Risks of the integrated candidate

- **Size:** 166 commits since `11c524c` (247 files, +37k/−3.5k lines). All of it is simulator- or test-verified only.
- **Behaviour changes the operator must know:**
  - A stop that cannot reach the device within 30 s is withdrawn, so the device keeps monitoring and the app says so.
  - While stopped, the dashboard streams unannotated camera video to the token holder (camera fallback). It is on by default, and only code can turn it off.
  - Sustained heat (≥ 80 °C, with the existing hysteresis constant) reports Degraded.
  - The app looks different from the M8 build.
- **Untested on hardware:**
  - URLSession's per-part multipart behaviour on iOS (seen on macOS);
  - CoreBluetooth permission states;
  - Liquid Glass and haptics.

## Recommendation: the integrated candidate

1. **The device and backend code is the payload that matters on Monday.** The integrated candidate's device and backend changes fix silent-failure and false-status defects that the hardware session would otherwise hit or misread.
2. **The integrated candidate is the one with complete evidence.** It has a single 28/28 run on its exact tree; the minimal candidate's app has never had a full run of its own.
3. **Backporting the M8.3 device and backend fixes onto the old app is not tested.** That combination does not exist; building it would mean a new, untested composition.

**Fall-back:** if the owner prefers to keep the app unchanged for Monday, run the session from the integrated candidate's **device and backend** code, and install the M8.1 app (`15bf35b`).
- The status contract is additive, so the M8.1 app tolerates the new fields.
- The M8.1 app does not show the new liveness, protection outcomes or alert-delivery evidence.

## Merge and rollback plan (each step needs approval; nothing is force-pushed)

1. **Freeze:** record `C=$(git rev-parse feat/m8.3-hardware-truth)`. Run `make verify-release REF=$C` (expect PASS), then gitleaks over `11c524c..$C` (expect no leaks; findings so far are known test vectors only).
2. **Publish for review:** `git push -u origin feat/m8.3-hardware-truth` (fork). Open a PR to `DonovanWillis/sensordyme_model:integration/all`.
   - `integration/all` is `11c524c`, an ancestor of `$C`, so the PR merges without conflicts.
   - Note for squash vs merge: commit `78238bc` mentions a worker's old machine name (redacted in `15bf35b`). A squash merge keeps it out of `integration/all` history; a merge commit keeps it.
3. **Hardware session:** run from `$C` per `docs/MONDAY_HARDWARE_SESSION.md` and `docs/MONDAY_RUNBOOK.md`, before or after the PR merge.
4. **Rollback:**
   - Device: runbook §O (O1 previous tree, O2 E1 backup, O3 SD image).
   - App: reinstall a build from `15bf35b` or `618880c`.
   - Repository: if the PR was merged and must be undone, `git revert -m 1 <merge>` on `integration/all`, never a reset or force-push. `feat/m8.3-hardware-truth` stays as the record.
5. **What to merge if the session finds a blocker:** fix on a branch from `$C`, re-run `make verify-release` and the one-invocation XCUITest suite, then repeat step 2.
