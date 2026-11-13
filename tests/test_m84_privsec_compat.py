"""M8.4 safe updates: old and new app / backend / device combinations around Decision C.

Each row of COMPAT is one pairing of versions and what it must do -- operate,
or fail with an explicit, named reason; never silently claim something false.
The same table is in docs/SECURITY_REVIEW_M6.md §7 (compatibility matrix).
App-side rows (an M8.4 app against an old backend) are pinned by
ios/Tests/StreamLivenessTests.swift (`CameraModePresentation.resolve(nil)` is
`unknown`, never "off").
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from test_m84_privsec_camera_backend import (  # noqa: F401
    _camera_block,
    _ingest,
    _set,
    _status,
    backend,
)

from core import command_auth as a
from core.camera_privacy import parse_command
from core.dashboard import _heartbeat_part

COMPAT = [
    # (id, pairing, required behaviour)
    ("old-device/new-backend", "M8.3 device (no monitoring.camera) + M8.4 backend",
     "POST camera-mode -> 409 firmware_unsupported; status.camera_mode.supported=false; never sent"),
    ("future-device/new-backend", "device reporting camera schema 2 + M8.4 backend",
     "treated as unsupported (fail closed): no schema-1 command is sent to it"),
    ("new-device/old-backend", "M8.4 device + backend without camera-mode",
     "no set_camera_mode ever arrives: camera-only stays OFF, camera off while stopped (the safe default)"),
    ("new-backend/new-device-extra-fields", "set_camera_mode with fields this device does not know",
     "unknown fields ignored, known ones strictly typed"),
    ("m83-app/new-device", "M8.3 app (asks for heartbeats, does not know camera_off_while_stopped)",
     "heartbeat parts keep every M8.3 field; the app shows NO NEW FRAMES (no frame leaks)"),
    ("pre-m83-client/new-device", "pre-M8.3 app or browser <img> (no heartbeats)",
     "no non-JPEG part is ever sent; while stopped it receives nothing"),
    ("auth-contract", "sd-auth envelope of an unknown version",
     "refused as unsupported_version, never executed"),
]


@pytest.mark.parametrize("row", COMPAT, ids=[r[0] for r in COMPAT])
def test_every_row_is_documented(row):
    from pathlib import Path

    doc = (Path(__file__).resolve().parent.parent / "docs" / "SECURITY_REVIEW_M6.md").read_text()
    assert row[0] in doc, f"compatibility row {row[0]} is not in docs/SECURITY_REVIEW_M6.md §7"


def test_old_device_new_backend(backend):  # noqa: F811
    b = backend
    _ingest(b, None)
    assert _set(b, "alice", b.ids["alice"], True).json()["code"] == "firmware_unsupported"
    assert _status(b)["camera_mode"]["supported"] is False
    assert "set_camera_mode" not in (_ingest(b, None)["command"] or {})


def test_future_device_new_backend(backend):  # noqa: F811
    b = backend
    future = {**_camera_block(True), "schema": 2}
    _ingest(b, future)
    assert _status(b)["camera_mode"]["supported"] is False
    assert "set_camera_mode" not in (_ingest(b, future)["command"] or {})


def test_new_device_old_backend_defaults_off(tmp_path):
    from core.camera_privacy import CameraPrivacy

    privacy = CameraPrivacy.load()  # nothing ever delivered
    assert privacy.camera_only is False and privacy.remote_stream(stopped=True) == "off_while_stopped"


def test_unknown_command_fields_are_ignored_known_ones_strict():
    assert parse_command({"camera_only": False, "id": "a", "schedule": "22:00"}) == {"camera_only": False, "id": "a"}
    assert parse_command({"camera_only": "false", "id": "a"}) is None


def test_m83_app_heartbeat_fields_are_unchanged():
    part = _heartbeat_part("a1b2c3d4", "camera_off_while_stopped", None, 12).decode()
    for field in ("--frame", "Content-Type: text/plain", "Content-Length: 0", "X-Stream-Heartbeat: 1",
                  "X-Stream-Id: a1b2c3d4", "X-Frame-Seq: 12"):
        assert field in part
    assert part.endswith("\r\n\r\n\r\n")


def test_auth_contract_version_mismatch_is_refused():
    key = a.DeviceKey("dk_" + "77" * 24)
    command = {"set_camera_mode": {"camera_only": True, "id": "x"}}
    envelope = a.sign_command(key, command, device="SD-0A1B2C3D", request_nonce="ab" * 16)
    envelope["v"] = "sd-auth-v2"
    with pytest.raises(a.AuthError) as refused:
        a.verify_command(a.Keyring(key), command, envelope, devices={"SD-0A1B2C3D"}, request_nonce="ab" * 16)
    assert refused.value.reason == "unsupported_version"
