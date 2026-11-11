# Autonomy review — Mission 6

Can the device go from power-on to protecting someone, and stay there, with no
human and no network? Reviewed from the code on `m6/autonomy-security-runbook`
(based on `integration/m6-rc`, `c01e43b`). Software only: nothing ran on a Pi.
The physical checks for every link are in `docs/MONDAY_RUNBOOK.md` §K and in
`docs/HARDWARE_DEPENDENCY_MATRIX.md`.

## The chain

| # | Link | What the code does | Verdict |
|---|---|---|---|
| 1 | Power-on → systemd | `WantedBy=multi-user.target`, `After=multi-user.target bluetooth.service`, `Wants=bluetooth.service`. There is no `network-online` dependency: the device must work with no Wi-Fi. `After=multi-user.target` plus `WantedBy=` is not an ordering cycle, because systemd skips the implicit target→unit `After=` when the unit is already ordered after the target. | OK (asserted: `test_the_service_never_waits_for_the_network`) |
| 2 | Preflight | `ExecStartPre=-…/preflight.py` is advisory (leading `-`), so a full card or failing check cannot stop or loop the unit. | OK (existing `test_f7_*`) |
| 3 | Config | Unreadable, malformed, non-mapping or schema-invalid config exits **78**; `RestartPreventExitStatus=78` stops the unit. | OK (M0.2 tests); the guard also never reboots on it (`test_main_never_reboots_for_a_config_refusal`) |
| 4 | Module and model | A module that cannot be built becomes `UnavailableModule` and the device boots FAILED but reachable. The `.hef` is verified against `accelerator.yolo_hef_sha256` at load; a mismatch means DEGRADED, never a crash. | OK (M0.2) |
| 5 | Camera | 5 attempts, each with a 10 s verify read and 2 s backoff; after that the device boots without a camera while capture keeps retrying. | OK (`test_camera_open_retries.py`) |
| 6 | READY / watchdog | Warm-up inference runs before `READY=1`. systemd arms `WatchdogSec=45` only after READY. `TimeoutStartSec=180` exceeds the 60 s worst-case camera bring-up with 120 s left for model load. | OK (`test_the_start_timeout_covers_the_worst_case_camera_bring_up`); the real READY time is a Monday measurement |
| 7 | Monitoring | The inference loop and health monitor start before READY. Critical components (camera, inference) withhold the watchdog when down, with bounded in-process restarts and then the persisted, rate-limited last resort. | OK (M0.2) |
| 8 | Crash at startup (import error after a bad deploy, corrupt `presets.yaml`, broken venv) | **Defect, fixed (A2).** `StartLimitAction=reboot` reset its counter on every boot, giving an unbounded reboot loop (audit 05 F12/F13, row 18). | Fixed |
| 9 | First boot without Wi-Fi | BLE provisioning opens as soon as the device is offline (`grace_s: 0`). Monitoring, the local buzzer and `/api/health` on the device do not depend on the network. | OK in code; BLE is a Monday test |
| 10 | Backend link after network loss | **Defect, fixed (A1).** The uplink retried every 2 s for the whole outage, with no backoff and no jitter. Events queue (bounded, newest kept). | Fixed |
| 11 | Health before network | `/api/health` is served locally from the first second. Health reaches the backend only once the uplink connects (the backend shows "unknown" until then, as the status contract says). | OK |
| 12 | Commands after reconnection | Desired-state convergence: preset, rules and zones are re-sent until the device reports them. **Defect, fixed (A1/S3):** a repeated stop/resume id was applied twice. | Fixed |
| 13 | Persistent identity | `SD-XXXXXXXX` = sha256(SoC serial), stable across reboot, reinstall and reflash. Backend pairing = `cloud.device_key` in `config.yaml`; app live-feed pairing = `dashboard.token` in `config.yaml`; saved mode, rules, calibration and the reboot budget live in `data/`; Wi-Fi in NetworkManager. **These are what a reinstall must preserve** (runbook §E). | OK, with one residual risk below |
| 14 | Release changes reaching the device | **Defect, fixed (A3).** `make deploy` never re-installed `/etc/systemd/system/sensordyme.service`, so a release that changed the unit ran under the old one. | Fixed |
| 15 | Developer profile on a device | **Defect, fixed (A4/S6).** `profile: mac_max` on a Pi cannot start (no torch/MPS), which gave a crash loop and also enabled frame upload. The deploy preflight now refuses it. | Fixed |

## Fixes (each with the test that fails on the old code)

* **A1 — bounded reconnection** (`core/uplink.py`). After N consecutive failed syncs the wait is `ceiling = min(cap, poll × 2^N)`, then a random value in `[ceiling/2, ceiling]`. The cap is 30 s, configurable as `cloud.max_backoff_s`. The first success returns to the poll interval, and queued events are delivered exactly once.
  Tests: `tests/test_m6_autonomy_uplink.py` (6).
* **A2 — bounded start-limit reboot** (`sensordyme.service` `StartLimitAction=none` plus `OnFailure=sensordyme-recover.service`; `deploy/systemd/sensordyme-recover.service`; `scripts/start_limit_guard.py`, installed root-owned in `/usr/local/lib/sensordyme/`). It reboots only on `Result=start-limit-hit`, at most 3 times in 6 h, and only after durably recording the reboot. A clock that went backwards counts against the budget. Past the budget the unit stays failed and the device stays up for SSH.
  Tests: `tests/test_m6_autonomy_boot.py` (decision logic, persistence, unrecordable reboot refused, dry-run, unit wiring, root-owned script).
* **A3 — units installed on deploy** (`scripts/release/install_units.sh`, called by `make deploy` before the restart; `setup_pi.sh` installs the guard and the recovery unit).
  Tests: `test_install_units_renders_without_placeholders`, `test_install_units_refuses_root`, `test_setup_installs_the_guard_before_the_service`, `test_make_deploy_installs_the_units_before_restarting`.
* **A4 — device profile enforced at deploy** (`scripts/deploy_preflight.py`).
  Test: `tests/test_m6_security_preflight.py::test_the_developer_profile_is_refused_on_a_device`.

## Residual risks (documented, not changed)

1. **When the reboot budget is spent, the device is offline to the app.** The unit is failed, so there is no BLE, no `/api/health` and no buzzer. That is deliberate (a reboot loop is worse, and SSH works), but a person must notice. Next step: a minimal "failed" beacon (for example a buzzer pattern from the guard) or a hardware watchdog policy.
2. **Identity fallback.** If a board ever lacks a `/proc/cpuinfo` Serial, the id is derived from the wlan0, then eth0, MAC: stable, but interface-dependent. Not changed, because changing the derivation would change existing ids and break pairing. Monday check: runbook §C2.
3. **The reboot policy depends on systemd 252 semantics** for `OnFailure=` plus `Result=start-limit-hit`. This is unit-tested against a fake `systemctl` only; the real trigger is runbook §K5.
4. **Clock.** The Pi 5 RTC keeps time only with a battery fitted. Without NTP (offline in a car), event timestamps can be wrong until sync. Events dedupe by `event_key`, not time, so delivery is unaffected; the reboot-budget window errs conservative.
5. **Deploy needs the network** for `pip install`. An offline deploy keeps the previous venv (runbook §F2), which the manifest verify then flags if the pins changed.
6. **The BLE provisioning window** is open whenever the device is offline. That is an autonomy feature and a security risk at the same time (see `docs/SECURITY_REVIEW_M6.md` §3). The window policy (`services/provisioning.py`) is outside this mission.
7. **The in-process last resort's reboot** uses `sudo -n systemctl reboot` through the sudoers rule. It is bounded by `data/.reboot_state.json` (`health.reboot_max_total`, default 5, spaced at least `reboot_min_interval_s` apart, with the count reset after 24 h without one). That is a separate budget from A2's. Worst case, the two together allow 5 in-process reboots per campaign plus 3 start-limit reboots per 6 h.
