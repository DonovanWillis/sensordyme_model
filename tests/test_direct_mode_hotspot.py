"""G11: the legacy setup hotspot (scripts/provisioning_hotspot.sh) is never open and never leaks.

The script is sourced under bash with `nmcli` replaced by a shell function that
records its arguments; nothing reaches NetworkManager.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from core import direct_mode as dm

REPO = Path(__file__).resolve().parent.parent
HOTSPOT = REPO / "scripts" / "provisioning_hotspot.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _start(tmp_path, psk_file: Path) -> tuple[list[str], str]:
    calls = tmp_path / "nmcli.log"
    script = (f'nmcli() {{ printf "%s\\n" "$*" >> "{calls}"; echo "Hint: the password is SECRET"; }}; '
              f'export SENSORDYME_HOTSPOT_PSK_FILE="{psk_file}"; source "{HOTSPOT}"; start_hotspot SensorDyme-ABCD')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=10, check=False)  # noqa: S603, S607
    assert out.returncode == 0, out.stderr
    return calls.read_text().splitlines(), out.stdout + out.stderr


def test_the_hotspot_uses_the_label_passphrase_when_there_is_one(tmp_path):
    psk_file = tmp_path / "direct_mode_psk"
    psk = dm.generate_psk()
    dm.write_psk(str(psk_file), psk)
    calls, output = _start(tmp_path, psk_file)
    assert calls == [f"device wifi hotspot ifname wlan0 ssid SensorDyme-ABCD band bg password {psk}"]
    assert psk not in output and "SECRET" not in output, "nmcli's output never reaches the journal"


@pytest.mark.parametrize("content", [None, "", "password\n", "ABCD-EFGH-JKMN-PQRS\n", "abcd-efgh-jkmn\n"])
def test_without_a_usable_label_passphrase_nmcli_generates_one_and_nothing_is_printed(tmp_path, content):
    psk_file = tmp_path / "direct_mode_psk"
    if content is not None:
        psk_file.write_text(content)
    calls, output = _start(tmp_path, psk_file)
    assert calls == ["device wifi hotspot ifname wlan0 ssid SensorDyme-ABCD band bg"]
    assert "SECRET" not in output


def test_a_symlinked_passphrase_file_is_not_followed(tmp_path):
    real = tmp_path / "elsewhere"
    dm.write_psk(str(real), dm.generate_psk())
    link = tmp_path / "direct_mode_psk"
    link.symlink_to(real)
    calls, _ = _start(tmp_path, link)
    assert "password" not in calls[0]


def test_the_script_no_longer_claims_the_hotspot_is_open():
    text = HOTSPOT.read_text()
    assert "no password" not in text
    assert "never open" in text
    assert "NOT a portable operating mode" in text
    assert "docs/DIRECT_MODE.md" in text


def test_setup_pi_still_installs_the_hotspot_only_for_method_hotspot():
    """The legacy path is kept; `ble` stays the shipped default."""
    setup = (REPO / "setup_pi.sh").read_text()
    assert "ExecStart=$(pwd)/scripts/provisioning_hotspot.sh" in setup
    assert "method: ble" in (REPO / "config.yaml").read_text()
