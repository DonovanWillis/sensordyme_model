"""An unusable setup-code file keeps Bluetooth shut and says why; it never crashes provisioning.

Security review finding (M8): load_setup_code raised SetupCodeError only for
permission and format problems. An unreadable file (mode 0000, owned by the
service user) escaped as PermissionError and non-ASCII content as
UnicodeDecodeError; ProvisioningService._open_ble_window catches only
SetupCodeError/ImportError, so the provisioning thread died instead of keeping
the window shut and retrying.
"""

from __future__ import annotations

import os
import threading
from typing import ClassVar

import pytest

pytest.importorskip("cryptography")

from services import ble_pop as pop

CODE = "ABCD-EFGH-JKMN"
DEVICE = "SD-1A2B3C4D"


def _write(path, data: bytes, mode: int = 0o600) -> str:
    path.write_bytes(data)
    path.chmod(mode)
    return str(path)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-0000 file")
def test_an_unreadable_code_file_is_a_setup_code_error(tmp_path):
    path = _write(tmp_path / "code", f"{CODE}\n".encode(), mode=0o000)
    try:
        with pytest.raises(pop.SetupCodeError, match="cannot be read"):
            pop.load_setup_code(path)
    finally:
        (tmp_path / "code").chmod(0o600)


def test_non_ascii_content_is_a_setup_code_error_that_does_not_echo_it(tmp_path):
    path = _write(tmp_path / "code", "ÄBCD-EFGH-JKMN\n".encode())
    with pytest.raises(pop.SetupCodeError, match="not ASCII") as caught:
        pop.load_setup_code(path)
    assert "BCD" not in str(caught.value), "the file's content never appears in the error"


def test_a_directory_in_place_of_the_file_is_refused(tmp_path):
    (tmp_path / "code").mkdir()
    with pytest.raises(pop.SetupCodeError):
        pop.load_setup_code(str(tmp_path / "code"))


def test_a_good_file_still_loads(tmp_path):
    assert pop.load_setup_code(_write(tmp_path / "code", f"{CODE}\n".encode())) == "ABCDEFGHJKMN"


# ---------------------------------------------------------------------- the service fails closed
class _Wifi:
    def is_online(self):
        return False

    def current_ssid(self):
        return None

    def ip_address(self):
        return None

    def watch(self, on_change, stop):
        return None

    def scan(self):
        return []


class _BLE:
    instances: ClassVar[list] = []

    def __init__(self, device_name, device_info, on_credentials, scan_networks=None, on_command=None, on_auth=None):
        _BLE.instances.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def set_status(self, status):
        pass

    def set_scans_paused(self, paused):
        pass


@pytest.mark.parametrize("problem", ["unreadable", "non_ascii"])
def test_the_service_keeps_bluetooth_shut_reports_why_and_recovers(tmp_path, problem):
    if problem == "unreadable" and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a mode-0000 file")
    from services.provisioning import ProvisioningService

    path = tmp_path / "code"
    if problem == "unreadable":
        _write(path, f"{CODE}\n".encode(), mode=0o000)
    else:
        _write(path, "ÄBCD-EFGH-JKMN\n".encode())
    _BLE.instances = []
    service = ProvisioningService(_Wifi(), {"auth": "pop", "setup_code_path": str(path)},
                                  ble_server_factory=_BLE, identity=(DEVICE, "SensorDyme-3C4D"))
    service._open_ble_window()  # raised PermissionError / UnicodeDecodeError before the fix
    assert _BLE.instances == [], "Bluetooth stays shut"
    service.state = "provisioning"
    service._thread = threading.current_thread()  # a live run loop, for status()
    healthy, detail = service.status()
    assert not healthy and "Bluetooth setup could not start" in detail and "setup code unusable" in detail
    assert "BCD" not in detail

    # Fixed on disk: the next retry opens the window (fail closed, not fail dead).
    path.chmod(0o600)
    path.write_text(f"{CODE}\n")
    service._ble_failed_at = None
    service._open_ble_window()
    assert len(_BLE.instances) == 1
