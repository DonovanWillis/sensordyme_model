"""SensorDyme software acceptance (M8.4, extended for M8.5 "Launch Ready"):
every meaningful digital-twin suite, one command, one evidence summary, and a
matrix of the owner's 30 required journeys (docs/ACCEPTANCE.md).

    make acceptance                       # = python scripts/sim/acceptance.py --out data/acceptance
    make acceptance-full                  # = the same + --with-ios (needs the iOS Simulator)
    python scripts/sim/acceptance.py --out DIR [--quick] [--only NAME ...] [--list] [--with-ios]

Runs, each in its own process (the driver suites load real MediaPipe graphs;
one process for everything once exceeded a 1 GiB fleet job cap):

  * the deterministic virtual-device suites (SimClock: journeys, failure
    injection, reliability, storage integrity, truth invariants, events,
    lifecycle, onboarding, streaming, security end to end);
  * a short sim-clock soak (scripts/sim/twin_soak.py simclock): device-vs-backend
    event integrity and the InvariantMonitor over a lossy link;
  * the live /stream measurement, short (scripts/sim/stream_soak.py), judged only
    on completing and delivering frames -- no timing thresholds;
  * the Swift onboarding harness against the simulated BLE peripheral
    (python -m sim.ble_harness --all), where swiftc exists.

With --with-ios it also runs the full XCUITest suite (scripts/sim/ios_uitest.sh:
the real iPhone app in the iOS Simulator against the real backend and the
virtual device) and folds its per-test results into the summary. Off by
default: it needs xcodebuild, the Simulator and ~25 minutes, and must not run
while anything else owns the Simulator. The report always says whether it ran.

Writes DIR/acceptance.json and DIR/acceptance.md: per suite pass/fail/skip
counts, duration, exit code; a JOURNEY MATRIX (the owner's 30 journeys, each
mapped to the suites and tests that prove it, with coverage notes); the commit
(and whether the tree was dirty); the host platform; and `simulated: true` --
none of this is a hardware result.
Exit status is nonzero if any suite failed. Skips are reported, never hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Suite:
    name: str
    what: str
    tests: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    #: Skip (reported) unless this is true on the host.
    needs: Optional[str] = None
    quick: bool = True


SUITES: tuple[Suite, ...] = (
    # ---- M8.4 (this stream)
    Suite("twin_journeys", "full user journeys through the app's API, invariants attached",
          ("tests/test_m84_twin_journeys.py", "tests/test_m84_twin_fidelity.py")),
    Suite("twin_faults", "lossy/duplicate/reordered link, malformed NPU tensors, ack loss, backend restart, "
                         "read-only data dir", ("tests/test_m84_twin_faults.py",)),
    Suite("reliability_recovery", "repeated inference failure, fault storms, no duplicate actuation",
          ("tests/test_m84_reliability_recovery.py",)),
    Suite("reliability_storage", "full card on a real volume, writers, spool migration, retention",
          ("tests/test_m84_reliability_storage.py",)),
    # ---- the virtual device and its truth model (M6.5 - M8.3)
    Suite("sim_e2e", "virtual device x real backend, end to end", ("tests/test_sim_e2e.py",)),
    Suite("sim_reliability", "multi-stage reliability scenarios", ("tests/test_sim_reliability.py",)),
    Suite("sim_stress", "backlog, retry storm, races, restarts", ("tests/test_sim_stress.py",)),
    Suite("sim_units_seams", "sim building blocks and production seams",
          ("tests/test_sim_units.py", "tests/test_sim_seams.py")),
    Suite("truth_failure_matrix", "M8.3 failure matrix rows", ("tests/test_m83_truth_failure_matrix.py",)),
    Suite("truth_invariants_ack", "truth invariants and protection acknowledgements",
          ("tests/test_m83_truth_invariants.py", "tests/test_m83_truth_protection_ack.py")),
    Suite("health_freshness", "stale / reordered / frozen health (device-driven and backend-level)",
          ("tests/test_m81_health_freshness_sim.py", "tests/test_m81_health_freshness.py")),
    Suite("events", "event pipeline: delivery evidence, dedupe, clocks",
          ("tests/test_m83_events_sim.py", "tests/test_m83_events_backend.py", "tests/test_m83_events_delivery.py")),
    Suite("lifecycle", "boot, LKG config, spool, power cuts",
          ("tests/test_m8_lifecycle_e2e.py", "tests/test_m8_lifecycle_boot.py", "tests/test_m8_lifecycle_uplink.py")),
    Suite("nodes_and_modes", "shared perception, driver activation, nodes",
          ("tests/test_m7_shared_perception_sim.py", "tests/test_m8_driver_activation_sim.py",
           "tests/test_m8_nodes_e2e.py")),
    Suite("security_e2e", "credentials, command replay across restarts", ("tests/test_m8_security_e2e.py",)),
    Suite("stream", "/stream framing, liveness, viewers", ("tests/test_m83_stream_device.py",
                                                            "tests/test_m83_stream_runtime.py")),
    Suite("onboarding", "BLE provisioning peripheral + proof of possession (+ Swift app code where swiftc exists)",
          ("tests/test_m8_onboarding_peripheral.py", "tests/test_m8_onboarding_setup_code.py",
           "tests/test_m8_onboarding_crosslang.py", "tests/test_m8_onboarding_e2e.py",
           "tests/test_ble_pop.py")),
    # ---- M8.5 (journeys the audit found uncovered by the M8.4 command)
    Suite("backend_accounts", "accounts, tokens and the notification chain on the real backend "
                              "(journeys 2, 14)",
          ("tests/test_cloud_backend.py", "tests/test_phase1_cloud_auth.py",
           "tests/test_notification_gate.py", "tests/test_phase1_cloud_notify.py")),
    Suite("security_commands", "sd-auth-v1 on both sides: forged/replayed/revoked commands and keys "
                               "(journeys 27, 28)",
          ("tests/test_command_auth.py", "tests/test_command_auth_backend.py",
           "tests/test_command_auth_uplink.py", "tests/test_m8_security_backend.py",
           "tests/test_m8_security_device.py")),
    Suite("privsec_boundaries", "account deletion, camera privacy while stopped, paused capabilities "
                                "(journeys 17, 18, 27)",
          ("tests/test_m84_privsec_account.py", "tests/test_m84_privsec_camera_device.py",
           "tests/test_m84_privsec_camera_backend.py", "tests/test_m82_paused_capabilities.py")),
    Suite("camera_units", "the real camera open/retry path and CaptureService on fakes (journey 8)",
          ("tests/test_camera_open_retries.py", "tests/test_capture_service.py")),
    Suite("m85_phone_behaviour", "phone handling vs off-road attention through the real module; "
                                 "the shared phone-evidence pieces",
          ("tests/test_m85_phone_scenarios.py", "tests/test_m85_phone_evidence.py")),
    Suite("deploy", "hashed-lock installs, verified model artifacts, staged release with rollback "
                    "(journeys 29-30; deploy.sh e2e against a fake device)",
          ("tests/test_m85_deploy_lock.py", "tests/test_m85_deploy_staged.py",
           "tests/test_m85_deploy_e2e.py", "tests/test_m85_deploy_artifacts.py")),
    Suite("m85_journeys", "M8.5 journey gap-closers: phone-use alert e2e, device-to-push chain, "
                          "stop closes the stream gate, config restore over LKG, deleted account, "
                          "signed-unknown command, the real factory step (journeys 7, 11, 14, 18, 26-28)",
          ("tests/test_m85_journeys_app.py", "tests/test_m85_journeys_security.py")),
    # ---- scripts
    Suite("soak_simclock_short", "0.35 h of device time on the SimClock through a lossy link",
          command=("scripts/sim/twin_soak.py", "simclock", "--hours", "0.35", "--out", "{out}/soak_simclock")),
    Suite("stream_soak_short", "live /stream on the virtual device (measurement only, no thresholds)",
          command=("scripts/sim/stream_soak.py", "--out", "{out}/stream", "--seconds", "6",
                   "--soak-minutes", "0.5", "--only", "attentive"), quick=False),
    Suite("ble_harness", "the app's onboarding code vs the simulated peripheral, every scenario",
          command=("-m", "sim.ble_harness", "--all", "--evidence-dir", "{out}/ble"), needs="swiftc", quick=False),
)


@dataclass
class Journey:
    """One of the owner's 30 required journeys: which suites of THIS command prove
    it (its result is theirs), the load-bearing tests inside them, the XCUITest
    journeys that prove the app-presentation half, and what is NOT covered."""

    num: int
    name: str
    suites: tuple[str, ...]
    tests: str
    ios: tuple[str, ...] = ()
    note: str = ""


#: The audit behind every row is docs/ACCEPTANCE.md (M8.5). "J" below is
#: tests/test_m84_twin_journeys.py::test_the_whole_product_journey_from_first_boot_to_sign_out.
JOURNEYS: tuple[Journey, ...] = (
    Journey(1, "fresh app installation", ("lifecycle", "sim_e2e"),
            "test_m8_lifecycle_e2e.py::test_first_boot_with_no_state_comes_up_protecting; "
            "test_sim_e2e.py::test_first_boot_unclaimed_then_claimed_reaches_ready",
            ("test01_Launch", "test02_SignIn"),
            "device/backend first boot from no state; the app install + first launch half is XCUITest only"),
    Journey(2, "account sign-in", ("backend_accounts", "twin_journeys", "onboarding"),
            "test_cloud_backend.py::test_full_customer_journey (signup/login/claim); "
            "test_phase1_cloud_auth.py (token hygiene); J (signup token works, logout -> 401); "
            "test_m8_onboarding_e2e.py (the Swift app code's real /v1/login)",
            ("test02_SignIn",),
            "no automated UI signup flow anywhere (the XCUITest controller creates accounts)"),
    Journey(3, "device discovery", ("twin_journeys", "onboarding", "sim_units_seams"),
            "J (unclaimed device on no account); test_m8_onboarding_peripheral.py (real GATT table advertised, "
            "closed window refuses); test_sim_units.py::test_the_device_local_api_says_simulated",
            ("test03_Discovery",),
            "BLE scan = Swift app code vs the simulated peripheral (swiftc only); mDNS end to end is live-only"),
    Journey(4, "secure simulated pairing", ("onboarding", "ble_harness"),
            "test_m8_onboarding_peripheral.py::test_first_time_setup_proves_the_code_then_joins_and_closes_the_window; "
            "test_m8_onboarding_crosslang.py (the app's PoP flow vs the real device code: wrong code, lockout, "
            "tampering, expiry); test_ble_pop.py (vectors, setup-code file, replay)",
            (),
            "protocol level on real app + device code; the real radio (CoreBluetooth/BlueZ) is a Sunday gate"),
    Journey(5, "Wi-Fi provisioning simulation", ("onboarding", "ble_harness"),
            "test_m8_onboarding_peripheral.py (join recorded, long writes reassembled); "
            "test_m8_onboarding_e2e.py::test_first_time_pairing_then_reprovisioning_the_same_unit "
            "(after the simulated join the unit reports online to a live backend)",
            (),
            "the Wi-Fi radio is simulated; the real nmcli layer is unit-tested (test_wifi_*); physical join = Sunday"),
    Journey(6, "device claiming", ("twin_journeys", "sim_e2e", "security_commands", "backend_accounts"),
            "J (wrong code 404, claim, SD-SIM row, simulated: true); test_m8_security_backend.py "
            "(claim-guessing limited, already-claimed 409); "
            "test_m84_twin_journeys.py::test_another_account_can_neither_see_claim_nor_command_this_device",
            ("test03_Discovery",), ""),
    Journey(7, "device registration", ("m85_journeys", "sim_e2e", "security_commands"),
            "test_m85_journeys_security.py::test_manage_create_device_provisions_a_unit_the_backend_claims_and_"
            "authenticates (the real factory command); test_sim_e2e.py::test_restart_keeps_identity_pairing_mode_"
            "and_rules; test_m8_security_backend.py (key revocation/rotation)",
            (),
            "the virtual device's SD-SIM identity stands in for core/identity (SoC serial, unit-tested); "
            "registration = first authenticated ingest, there is no separate endpoint"),
    Journey(8, "camera initialization simulation", ("camera_units", "sim_e2e", "lifecycle"),
            "test_camera_open_retries.py (real _open_camera retry/release); test_capture_service.py (real "
            "CaptureService reopen/restart); test_m8_lifecycle_e2e.py::test_camera_absent_at_boot_fails_"
            "truthfully_then_recovers; test_sim_e2e.py camera freeze/stall",
            ("test04_Ready",),
            "the virtual device injects SimCamera past the real open path (fidelity map); a real sensor is a "
            "hardware gate"),
    Journey(9, "monitoring engine readiness", ("twin_journeys", "sim_e2e", "nodes_and_modes"),
            "J (ready/protecting confirmed, state_model.derive); test_sim_e2e.py::test_first_boot_unclaimed_then_"
            "claimed_reaches_ready (ready is earned: starting -> ready); test_m8_driver_capabilities.py is in "
            "`make test` (calibration first)",
            ("test04_Ready", "test14_MonitoringStatus"), ""),
    Journey(10, "Distracted Driving activation", ("nodes_and_modes", "twin_journeys", "sim_e2e"),
            "test_m8_driver_activation_sim.py::test_distracted_driving_activation_reaches_the_device_and_alerts_"
            "reach_the_app (module swap confirmed only by the device's report); J (variant change); "
            "test_sim_e2e.py (role-forbidden preset refused)",
            ("test08_PresetActivate", "test23_DriverSafetyCard"), ""),
    Journey(11, "supported distraction event", ("nodes_and_modes", "sim_e2e", "m85_journeys", "m85_phone_behaviour"),
            "test_m8_driver_activation_sim.py (look-away -> one loud distraction row); test_sim_e2e.py scripted "
            "scenes (phone/distracted/absent types); test_m85_journeys_app.py::test_phone_use_reaches_the_app_"
            "loud_with_its_output_evidence; test_m85_phone_scenarios.py (handling vs mounted/passenger, module "
            "level, at shipped thresholds)",
            ("test18_AlertHistory",),
            "M8.5 handling-cue decisions are module-level pins (synthetic evidence, unvalidated heuristics)"),
    Journey(12, "alert generation", ("twin_journeys", "events", "m85_journeys"),
            "J (loud, decision alerted, output simulated, sim_buzzer played); test_m83_events_sim.py (output "
            "evidence, failing buzzer reported failed, late output report updates the row)",
            ("test18_AlertHistory",),
            "output evidence is the simulated buzzer; an audible GPIO buzzer is a hardware gate"),
    Journey(13, "event persistence", ("events", "twin_journeys", "twin_faults", "nodes_and_modes"),
            "J (paging by before_id, explain, 404); test_m8_driver_activation_sim.py::test_event_history_is_"
            "ordered_complete_limited_and_survives_a_restart; test_m83_events_backend.py (replayed batch, backend "
            "restart); twin_faults (one row through ack loss and restarts)",
            ("test18_AlertHistory",), ""),
    Journey(14, "app notification handling", ("m85_journeys", "backend_accounts", "events"),
            "test_m85_journeys_app.py::test_a_loud_alert_from_the_device_is_pushed_once_and_labelled_simulated "
            "(device -> ingest -> gate -> send_push); test_phase1_cloud_notify.py (real HTTP to a push-server "
            "stand-in); test_notification_gate.py (cooldown, severities); test_m83_events_backend.py (one "
            "notification per alert, '(simulated)' on the lock screen)",
            (),
            "push = ntfy topic (no APNs); the app itself polls and posts LOCAL notifications "
            "(ios/Tests/NotificationPolicyTests.swift, pure logic) -- delivery into the app is not end to end"),
    Journey(15, "custom Jupiter policy creation", ("nodes_and_modes", "twin_journeys", "sim_e2e"),
            "test_m8_nodes_e2e.py::test_e2e_a_supported_custom_prompt_runs_on_the_shared_tracker_and_survives_a_"
            "restart; J (draft -> activate -> active only once the device runs it -> fires); test_sim_e2e.py "
            "(stale draft 409, unsupported refused honestly)",
            ("test10_JupiterPrompt", "test11_PolicyPreview", "test13_RuleEnableDisable"),
            "deterministic parser path; no LLM interpreter in the twin (refusal is interpreter_unavailable)"),
    Journey(16, "compatible concurrent monitoring", ("nodes_and_modes",),
            "test_m8_nodes_e2e.py::test_e2e_simultaneous_compatible_nodes_share_one_detector_run_per_frame and "
            "::test_e2e_driver_safety_keeps_its_module_while_companions_and_rules_join; "
            "test_m7_shared_perception_sim.py (one detector run shared by module + rule + companion)",
            ("test24_CompanionNode",), ""),
    Journey(17, "device stop", ("twin_journeys", "truth_invariants_ack", "privsec_boundaries", "sim_e2e"),
            "J (confirm required; outcome confirmed only by the echo; capabilities drop); "
            "test_m83_truth_protection_ack.py (pending/timeout/mismatch/supersede); "
            "test_m82_paused_capabilities.py (paused, not failing)",
            ("test05_Stop",), ""),
    Journey(18, "camera privacy enforcement", ("m85_journeys", "privsec_boundaries", "stream"),
            "test_m85_journeys_app.py::test_a_backend_stop_closes_the_remote_stream_gate_and_resume_reopens_it; "
            "test_m84_privsec_camera_device.py (a remote viewer gets no frame and the explicit reason, not even "
            "a pre-stop picture); test_m84_privsec_camera_backend.py (owner-only, signed, never inherited); "
            "test_m83_stream_runtime.py (nothing analysed while stopped)",
            (),
            "by design the camera keeps capturing while stopped (local maintenance view); what is enforced is "
            "no analysis and no remote frames"),
    Journey(19, "device resume", ("twin_journeys", "truth_invariants_ack", "sim_e2e"),
            "J (ready/protecting, command id echoed); test_m83_truth_protection_ack.py (a resume that times out "
            "still applies when the device returns); test_sim_e2e.py (alerts fire again after resume)",
            ("test06_Resume",), ""),
    Journey(20, "monitoring recovery", ("reliability_recovery", "twin_faults", "sim_reliability"),
            "test_m84_reliability_recovery.py (module failing every frame is failed, never protecting, then "
            "recovers; seeded fault storms; a reconnect never re-sounds or re-stores an alert); twin_faults "
            "(malformed accelerator tensors degrade, never crash)",
            ("test17_Recovery", "test25_PerceptionDegraded"),
            "duplicate-actuation is counted across a reconnect; a camera reconnect deliberately re-sounds "
            "locally (one stored row)"),
    Journey(21, "backend outage", ("sim_e2e", "lifecycle", "twin_faults", "truth_failure_matrix"),
            "test_sim_e2e.py::test_backend_outage_goes_stale_device_keeps_monitoring_and_reconnects; "
            "test_m8_lifecycle_e2e.py (local alerts continue, queue bounded, flushed once); twin_faults "
            "(backend restart with a pending stop converges once)",
            ("test15_Offline",),
            "modelled as an unreachable network or an instant backend restart; a backend held down for hours "
            "is not simulated"),
    Journey(22, "offline event queue", ("lifecycle", "events", "reliability_storage", "sim_stress"),
            "test_m8_lifecycle_e2e.py (uplink_spool.jsonl on disk, restored over a power cut); "
            "test_m83_events_sim.py (power cut mid-outage: delivered once with its evidence); "
            "test_m84_reliability_storage.py (legacy/torn spools, capacity boundary, full card); "
            "test_sim_stress.py (3000-event backlog stays bounded, drains once)",
            (),
            "bounded by design: beyond capacity the oldest are dropped and counted"),
    Journey(23, "reconnection", ("twin_journeys", "twin_faults", "events", "reliability_recovery"),
            "test_m84_twin_journeys.py::test_an_alert_raised_while_disconnected_reaches_history_once_with_its_"
            "true_time; twin_faults (lossy/duplicating/reordering links deliver every event and command once); "
            "reliability_recovery (no re-sound, no second row)",
            ("test15_Offline", "test16_Stale"), ""),
    Journey(24, "stale health detection", ("health_freshness", "twin_journeys", "truth_failure_matrix"),
            "test_m84_twin_journeys.py::test_invalid_requests_and_a_frozen_device_are_told_apart_from_a_healthy_"
            "one; test_m81_health_freshness_sim.py (frozen snapshot on a live link goes unknown; reordered and "
            "old-boot reports ignored; clock jumps change nothing); test_m81_health_freshness.py (backend rules)",
            ("test16_Stale", "test26_StaleTelemetry", "test28_FrozenHealthSnapshot"), ""),
    Journey(25, "device restart", ("twin_journeys", "sim_e2e", "lifecycle", "truth_failure_matrix", "sim_stress"),
            "J (reboot: same identity, mode, rule); test_m8_lifecycle_e2e.py (app restart AND power cut, checked "
            "before any backend contact); truth_failure_matrix F3/F4 (offline while down, new boot protecting); "
            "test_sim_stress.py (12 cycles leak nothing)",
            ("test17_Recovery",),
            "a stop is deliberately not persisted: a restarted device comes back protecting"),
    Journey(26, "configuration restoration", ("m85_journeys", "lifecycle", "sim_e2e"),
            "test_m85_journeys_app.py::test_preset_zones_and_rules_all_return_after_a_reboot_onto_the_last_known_"
            "good_config (everything configured, config.yaml damaged, all three back + fault reported); "
            "test_m8_lifecycle_boot.py (every CONFIG_DAMAGE kind -> LKG; broken LKG never used); "
            "test_sim_e2e.py::test_restart_keeps_identity_pairing_mode_and_rules",
            (),
            "damaged SAVED STATE deliberately falls back to the role default and reports the fault "
            "(test_m8_lifecycle_boot.py) -- that is the design, not a gap"),
    Journey(27, "account revocation", ("m85_journeys", "privsec_boundaries", "security_commands", "security_e2e"),
            "test_m85_journeys_security.py::test_a_deleted_accounts_token_can_neither_see_nor_command_and_the_"
            "device_keeps_protecting; test_m84_privsec_account.py (deletion, sessions end, released device "
            "carries nothing over); test_m8_security_backend.py (revoked device/key cannot ingest or receive "
            "commands); test_m8_security_e2e.py (revoked key refused end to end, nothing stored)",
            (), ""),
    Journey(28, "invalid command rejection", ("m85_journeys", "security_commands", "security_e2e",
                                              "sim_reliability", "twin_journeys"),
            "test_m8_security_e2e.py::test_command_replay_is_refused_across_device_and_backend_restarts; "
            "test_command_auth_uplink.py::test_an_invalid_signed_command_is_never_applied (unknown key, wrong "
            "device, stale, tampered, malformed); test_m85_journeys_security.py (a VALIDLY SIGNED unknown type "
            "is obeyed by nothing); test_sim_reliability.py (malformed commands change nothing on the real "
            "runtime); J (unknown action 422)",
            ("test19_InvalidResponses", "test27_UnreadableReplies"), ""),
    Journey(29, "failed update rollback", ("deploy",),
            "test_m85_deploy_e2e.py (a crash-looping release is rolled back to the previous good one; a "
            "tampered bundle is refused at stage); test_m85_deploy_staged.py (switch-back without network, "
            "failed unit install / migration roll back, 7 tamper kinds refused)",
            (),
            "health = systemd state + config check against fakes; the legacy in-place rollback "
            "(test_m8_lifecycle_release.py, test_m84_privsec_release.py) runs in `make test`; a real Pi/systemd "
            "run is a hardware gate"),
    Journey(30, "successful staged update", ("deploy",),
            "test_m85_deploy_e2e.py (make deploy's script: bundle -> verify every file -> hashed venv -> atomic "
            "switch -> CONFIRMED; re-deploying the running release is a re-verify no-op); "
            "test_m85_deploy_staged.py (one os.replace of a ready symlink, config carried at 0600, prune); "
            "test_m85_deploy_lock.py (every install path hashed, wheels only); test_m85_deploy_artifacts.py "
            "(model files verified where loaded, mismatch fails closed)",
            (),
            "ssh/systemd/pip are recorders on this Mac; the real staged deploy to the Pi is a hardware gate"),
)


def _journey_rows(results: list[dict], ios: Optional[dict]) -> list[dict]:
    """The 30-journey matrix, judged from THIS run's suite results."""
    by_name = {r["name"]: r["result"] for r in results}
    ios_tests = (ios or {}).get("tests") or {}
    rows = []
    for j in JOURNEYS:
        states = [by_name.get(s) for s in j.suites]
        ran = [s for s in states if s is not None]
        if any(s == "failed" for s in ran):
            result = "failed"
        elif not ran or len(ran) < len(states):
            result = "incomplete (suite not run)"
        elif all(s == "skipped" for s in ran):
            result = "skipped"
        else:
            result = "passed" + (" (some skips)" if "skipped" in ran else "")
        if not j.ios:
            ios_result = "-"
        elif ios is None:
            ios_result = "not run (--with-ios)"
        else:
            per = {t: ios_tests.get(t, "missing") for t in j.ios}
            ios_result = ("passed" if all(v == "passed" for v in per.values())
                          else "; ".join(f"{t}: {v}" for t, v in per.items() if v != "passed"))
        rows.append({"journey": j.num, "name": j.name, "result": result, "suites": list(j.suites),
                     "tests": j.tests, "ios": list(j.ios), "ios_result": ios_result, "note": j.note})
    return rows


def _commit() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or os.environ.get("SDJOB_COMMIT", "unknown"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _junit(path: Path) -> dict:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "errors": 0}
    failures: list[str] = []
    skips: list[str] = []
    try:
        root = ET.parse(path).getroot()  # noqa: S314 - our own junit output
    except (OSError, ET.ParseError):
        return {**counts, "failures": ["no junit report"], "skips": []}
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None:
            counts["failed"] += 1
            failures.append(name)
        elif case.find("error") is not None:
            counts["errors"] += 1
            failures.append(name)
        elif (skipped := case.find("skipped")) is not None:
            if "xfail" in (skipped.get("type", "") + skipped.get("message", "")).lower():
                counts["xfailed"] += 1
            else:
                counts["skipped"] += 1
                skips.append(f"{name}: {skipped.get('message', '')[:160]}")
        else:
            counts["passed"] += 1
    return {**counts, "failures": failures, "skips": skips}


def run_suite(suite: Suite, out: Path, py: str) -> dict:
    started = time.monotonic()
    log = out / "logs" / f"{suite.name}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "SD_M84_EVIDENCE_DIR": str(out / "journeys")}
    if suite.needs == "swiftc" and shutil.which("swiftc") is None:
        return {"name": suite.name, "what": suite.what, "result": "skipped", "reason": "needs swiftc (macOS)",
                "duration_s": 0.0}
    if suite.tests:
        totals = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "errors": 0}
        failures: list[str] = []
        skips: list[str] = []
        codes = []
        for test in suite.tests:
            junit = out / "junit" / f"{Path(test).stem}.xml"
            junit.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as handle:
                # --timeout: pytest.ini's 60 s per-test watchdog is a hang detector, and under a
                # loaded host the 20-minute sim scenarios exceed it without anything being wrong.
                code = subprocess.run([py, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rs", "--timeout=1200",
                                       f"--junitxml={junit}", test], cwd=REPO, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, check=False).returncode
            codes.append(code)
            parsed = _junit(junit)
            for key in totals:
                totals[key] += parsed[key]
            failures += parsed["failures"]
            skips += parsed["skips"]
        # pytest exit 5 = nothing collected (e.g. a whole module skipped by importorskip)
        bad = [c for c in codes if c not in (0, 5)]
        result = "failed" if bad or totals["failed"] or totals["errors"] else (
            "passed" if totals["passed"] else "skipped")
        return {"name": suite.name, "what": suite.what, "result": result, **totals, "exit_codes": codes,
                "failures": failures[:30], "skips": skips[:30], "duration_s": round(time.monotonic() - started, 1),
                "log": str(log.relative_to(out))}
    argv = [py, *(a.format(out=out) for a in suite.command)]
    with open(log, "w") as handle:
        code = subprocess.run(argv, cwd=REPO, env=env, stdout=handle,
                              stderr=subprocess.STDOUT, check=False).returncode
    record = {"name": suite.name, "what": suite.what, "result": "passed" if code == 0 else "failed",
              "exit_code": code, "duration_s": round(time.monotonic() - started, 1), "log": str(log.relative_to(out))}
    record.update(_script_facts(suite, out))
    if record.get("problems"):
        record["result"] = "failed"
    return record


def _script_facts(suite: Suite, out: Path) -> dict:
    """The few numbers a script run is judged on (and reported with)."""
    if suite.name == "soak_simclock_short":
        try:
            data = json.loads((out / "soak_simclock" / "summary_simclock.json").read_text())
        except (OSError, ValueError):
            return {"problems": ["no soak summary"]}
        events = data.get("events", {})
        problems = []
        if events.get("lost") or events.get("duplicate_keys") or events.get("duplicate_type_timestamp"):
            problems.append(f"event integrity: {events}")
        if (data.get("invariants") or {}).get("violations"):
            problems.append(f"invariant violations: {data.get('violations')}")
        return {"facts": {"sim_seconds": data.get("sim_seconds_completed"), "events": events,
                          "invariants": data.get("invariants"), "link": data.get("link")}, "problems": problems}
    if suite.name == "stream_soak_short":
        try:
            data = json.loads((out / "stream" / "stream_measurements.json").read_text())
        except (OSError, ValueError):
            return {"problems": ["no stream measurements"]}
        text = json.dumps(data)
        return {"facts": {"bytes": len(text)}, "problems": [] if '"frames"' in text else ["no frames measured"]}
    return {}


def run_ios(out: Path) -> dict:
    """The full XCUITest suite (scripts/sim/ios_uitest.sh): the real iPhone app in the
    iOS Simulator against the real backend and the virtual device. Only with --with-ios:
    it needs xcodebuild, the Simulator and ~25 minutes, and must not run while another
    stream owns the Simulator. The per-test results come from the run's summary.md."""
    started = time.monotonic()
    ios_out = out / "ios-uitest"
    log = out / "logs" / "ios_uitest.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as handle:
        code = subprocess.run(["bash", "scripts/sim/ios_uitest.sh", str(ios_out)], cwd=REPO,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                              stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    tests: dict[str, str] = {}
    missing: list[str] = []
    summary_md = ios_out / "summary.md"
    try:
        for line in summary_md.read_text().splitlines():
            row = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(row) >= 3 and row[1].startswith("test"):
                if row[2].startswith("MISSING"):
                    missing.append(row[1])
                else:
                    tests[row[1]] = row[2].lower()
    except OSError:
        pass
    counts = {"passed": sum(r == "passed" for r in tests.values()),
              "failed": sum(r == "failed" for r in tests.values()),
              "skipped": sum(r == "skipped" for r in tests.values())}
    record = {"name": "ios_uitest", "what": "XCUITest journeys: the real iPhone app (iOS Simulator) "
                                            "vs the real backend and virtual device",
              "result": "passed" if code == 0 else "failed", "exit_code": code, **counts,
              "failures": sorted(n for n, r in tests.items() if r == "failed") + [f"{n} (never reported)"
                                                                                  for n in missing],
              "skips": sorted(n for n, r in tests.items() if r == "skipped"),
              "tests": tests, "missing": missing,
              "duration_s": round(time.monotonic() - started, 1), "log": str(log.relative_to(out))}
    if not tests and not missing:
        record["result"] = "failed"
        record["failures"] = [f"no summary.md parsed from {summary_md}"]
    return record


def _fidelity() -> dict:
    """The twin's fidelity map (sim/fidelity.py): what was real, what was a stand-in."""
    sys.path.insert(0, str(REPO))
    from sim import fidelity

    return {"by_kind": fidelity.summary(),
            "stand_ins": {name: entry["fakes"] for name, entry in fidelity.FIDELITY.items()
                          if entry["code"] != "real"}}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(REPO / "data" / "acceptance"))  # data/ is git-ignored
    parser.add_argument("--quick", action="store_true", help="skip the live-time scripts (stream soak, BLE harness)")
    parser.add_argument("--only", nargs="*", default=None, help="suite names to run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--with-ios", action="store_true",
                        help="also run the full XCUITest suite (scripts/sim/ios_uitest.sh). Needs xcodebuild and "
                             "the iOS Simulator, takes ~25 min, and must not run while anything else owns the "
                             "Simulator. `make acceptance-full`.")
    args = parser.parse_args(argv)
    if args.list:
        for suite in SUITES:
            print(f"{suite.name:24s} {suite.what}")
        return 0
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen = [s for s in SUITES if (args.only is None or s.name in args.only) and (s.quick or not args.quick)]
    started = time.time()
    results = []
    for suite in chosen:
        record = run_suite(suite, out, args.python)
        results.append(record)
        counts = " ".join(f"{k}={record[k]}" for k in ("passed", "failed", "skipped", "xfailed") if k in record)
        print(f"{record['result'].upper():7s} {suite.name:24s} {record['duration_s']:7.1f}s {counts}", flush=True)
    ios: Optional[dict] = None
    if args.with_ios:
        ios = run_ios(out)
        results.append(ios)
        print(f"{ios['result'].upper():7s} {'ios_uitest':24s} {ios['duration_s']:7.1f}s "
              f"passed={ios['passed']} failed={ios['failed']} skipped={ios['skipped']}", flush=True)
    journeys = _journey_rows(results, ios)
    summary = {
        "simulated": True,
        "label": "Digital twin (virtual device: scripted camera and models, real runtime/uplink/backend). "
                 "NOT a hardware result.",
        **_commit(), "platform": platform.platform(), "python": platform.python_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "duration_s": round(time.time() - started, 1),
        "ok": all(r["result"] != "failed" for r in results),
        "ios": {"ran": bool(args.with_ios),
                **({"result": ios["result"]} if ios else
                   {"note": "XCUITest journeys not run; `make acceptance-full` runs them (iOS Simulator)"})},
        "fidelity": _fidelity(),
        "totals": {k: sum(int(r.get(k, 0) or 0) for r in results) for k in ("passed", "failed", "skipped", "xfailed")},
        "suites": results,
        "journeys": journeys,
    }
    (out / "acceptance.json").write_text(json.dumps(summary, indent=1))
    lines = [f"# SensorDyme acceptance (M8.5) -- {'PASS' if summary['ok'] else 'FAIL'} (SIMULATED)", "",
             (f"commit `{summary['commit']}`{' (dirty tree)' if summary['dirty'] else ''}, {summary['platform']}, "
              f"{summary['duration_s']} s. {summary['label']}"), "",
             ("XCUITest journeys (the real iPhone app in the Simulator): "
              + (f"ran, {ios['result']}" if ios else "NOT run -- `make acceptance-full` (needs the iOS Simulator)")),
             "",
             "| suite | result | passed | failed | skipped | xfailed | s | what |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['result']} | {r.get('passed', '')} | {r.get('failed', '')} | "
                     f"{r.get('skipped', '')} | {r.get('xfailed', '')} | {r['duration_s']} | {r['what']} |")
    for r in results:
        for item in r.get("failures", []) + r.get("problems", []):
            lines.append(f"- FAIL {r['name']}: {item}")
        for item in r.get("skips", []):
            lines.append(f"- skip {r['name']}: {item}")
    lines += ["", "## Journey matrix (the owner's 30 required journeys, all simulated)", "",
              "Each journey's result is the worst result of the suites that prove it in this run; the iOS",
              "column is the XCUITest half (app presentation). Coverage notes name what the simulation",
              "excludes -- those are the Sunday hardware gates (docs/ACCEPTANCE.md).", "",
              "| # | journey | result | iOS (XCUITest) | suites | tests | coverage notes |",
              "|---|---|---|---|---|---|---|"]
    for j in journeys:
        lines.append(f"| {j['journey']} | {j['name']} | {j['result']} | {j['ios_result']} | "
                     f"{', '.join(j['suites'])} | {j['tests']} | {j['note']} |")
    (out / "acceptance.md").write_text("\n".join(lines) + "\n")
    t = summary["totals"]
    print(f"{'PASS' if summary['ok'] else 'FAIL'}: {t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped, "
          f"{t['xfailed']} xfailed in {summary['duration_s']} s -> {out / 'acceptance.md'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
