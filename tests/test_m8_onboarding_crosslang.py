"""The app's Swift onboarding code against the device's Python state machine, message by message.

Each scenario builds nothing new: sim/ble_harness.py compiles the app's own
SetupFlow, Onboarding, BLEProvisioner, BLEPoPFlow and BLEProofOfPossession
(plus the DEBUG simulated transport) with swiftc, starts a fresh
sim/ble_peripheral.py (services/provisioning.py + services/ble_pop.py), and
runs one journey. The Swift side checks what the app did; these tests check
what the DEVICE saw and did, from its own transcript. macOS with swiftc only.
NOT a Bluetooth validation: no radio carries these messages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from sim import ble_harness

pytestmark = pytest.mark.skipif(ble_harness.swiftc() is None, reason="needs swiftc on macOS")


@pytest.fixture(scope="module")
def binary(tmp_path_factory) -> Path:
    return ble_harness.build(tmp_path_factory.mktemp("harness-build"))


def _auth(run) -> list[str]:
    """What crossed the auth characteristic, as the device recorded it."""
    out = []
    for record in run.transcript:
        value = record.get("value")
        if record.get("char") == "auth" and isinstance(value, dict):
            side = "app" if record["dir"] == "app->device" else "dev"
            out.append(f"{side}:{value.get('op')}" + (f"/{value['error']}" if value.get("error") else ""))
    return out


def _statuses(run) -> list[dict]:
    return [r["value"].get("provision") for r in run.transcript
            if r.get("char") == "status" and isinstance(r.get("value"), dict) and r["value"].get("provision")]


def _run(binary, tmp_path, scenario):
    run = ble_harness.run_scenario(binary, scenario, tmp_path / scenario)
    failed = [name for name, ok in run.checks().items() if not ok]
    errors = [e for e in run.evidence if e.get("kind") == "error"]
    assert run.ok, f"{scenario}: failed checks {failed}, errors {errors}"
    assert run.checks(), "the app side checked something"
    return run


def test_success_the_device_verifies_the_proof_decrypts_and_joins(binary, tmp_path):
    run = _run(binary, tmp_path, "success")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:proven", "app:creds", "dev:accepted"]
    outcomes = [p["state"] for p in _statuses(run)]
    assert outcomes[:1] == ["connecting"] and "ok" in outcomes
    assert not any(r.get("char") == "wifi_creds" for r in run.transcript), "nothing in clear on 5d1e0003"


def test_wrong_code_is_refused_and_the_right_one_then_proves_on_the_same_link(binary, tmp_path):
    run = _run(binary, tmp_path, "wrong-code")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:error/wrong_code",
                          "app:hello", "dev:challenge", "app:proof", "dev:proven"]
    assert sum(1 for r in run.transcript if r.get("kind") == "connect") == 1, "one Bluetooth connection throughout"


def test_lockout_the_device_reports_its_wait_and_the_app_honours_it(binary, tmp_path):
    run = _run(binary, tmp_path, "lockout")
    auth = _auth(run)
    assert auth.count("dev:error/wrong_code") == 3 and "dev:error/locked" in auth and auth[-1] == "dev:proven"
    locked = [r["value"] for r in run.transcript if isinstance(r.get("value"), dict) and r["value"].get("error") in
              ("wrong_code", "locked") and "retry_after_s" in r["value"]]
    assert locked and all(1 <= v["retry_after_s"] <= 2 for v in locked), "the device's own wait (base 2 s)"
    assert "app:creds" not in auth


def test_a_label_for_another_device_sends_nothing(binary, tmp_path):
    run = _run(binary, tmp_path, "wrong-label")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:proven"], "only the own label was used"


def test_a_challenge_naming_another_device_ends_the_exchange(binary, tmp_path):
    run = _run(binary, tmp_path, "impostor-id")
    assert _auth(run) == ["app:hello", "dev:challenge"], "no proof was offered to the impostor"


def test_a_tampered_device_proof_gets_no_credentials(binary, tmp_path):
    run = _run(binary, tmp_path, "tamper-proven")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:proven"]
    assert not _statuses(run), "no join was ever attempted"


def test_tampered_credentials_are_refused_by_the_device(binary, tmp_path):
    run = _run(binary, tmp_path, "tamper-creds")
    assert _auth(run)[-2:] == ["app:creds", "dev:error/malformed"]
    assert not _statuses(run), "the device joined nothing"


def test_a_late_proof_is_refused_as_expired_and_a_prompt_retry_works(binary, tmp_path):
    run = _run(binary, tmp_path, "timeout")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:error/unexpected",
                          "app:hello", "dev:challenge", "app:proof", "dev:proven"]


def test_credentials_after_the_proof_expired_renew_the_handshake_once(binary, tmp_path):
    run = _run(binary, tmp_path, "proof-expired")
    assert _auth(run) == ["app:hello", "dev:challenge", "app:proof", "dev:proven", "app:creds", "dev:error/unexpected",
                          "app:hello", "dev:challenge", "app:proof", "dev:proven", "app:creds", "dev:accepted"]
    assert [p["state"] for p in _statuses(run)][-1] == "ok"


def test_a_refused_passphrase_is_reported_and_a_second_attempt_joins(binary, tmp_path):
    run = _run(binary, tmp_path, "wrong-password")
    outcomes = [(p["state"], p["attempt"]) for p in _statuses(run)]
    assert ("error", 1) in outcomes and ("ok", 2) in outcomes


def test_no_secret_reaches_either_record(binary, tmp_path):
    run = _run(binary, tmp_path, "success")
    code = (tmp_path / "success" / "device" / "ble_setup_code").read_text().strip()
    for path in (tmp_path / "success").glob("*.jsonl"):
        text = path.read_text()
        for secret in (code, code.replace("-", ""), ble_harness.LAB_PSK):
            assert secret not in text, f"{secret[:2]}… in {path.name}"
    tags = [r["value"].get("tag") for r in run.transcript if isinstance(r.get("value"), dict) and "tag" in r["value"]]
    assert tags and all(t == "<32 bytes>" for t in tags)
    assert json.dumps(run.evidence).count("SDSETUP") == 0
