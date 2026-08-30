"""Opt-in TLS for the device -> cloud connection.

`verify` is what actually gets handed to `requests.post` on every sync. Off
(plain http://, the default) this whole file is inert; the three cases below
are what `ca_cert_path` / `tls_insecure_skip_verify` resolve to, and that the
resolved value is the one that actually reaches the HTTP call, not just
computed and ignored.
"""

import types

from core.uplink import CloudUplink

BASE_CONFIG = {"base_url": "https://backend.local:8787", "device_key": "dk_x"}


def test_default_config_verifies_normally():
    """No TLS config at all -- the overwhelmingly common case, and the one
    that must need zero changes: a real, CA-signed certificate (Let's
    Encrypt, Caddy) verifies fine against the system trust store already."""
    uplink = CloudUplink(BASE_CONFIG)
    assert uplink._verify is True


def test_ca_cert_path_pins_trust_to_that_one_cert():
    uplink = CloudUplink({**BASE_CONFIG, "ca_cert_path": "/etc/sensordyme/dev-cert.pem"})
    assert uplink._verify == "/etc/sensordyme/dev-cert.pem"


def test_insecure_skip_verify_disables_verification_and_warns_once(capsys):
    uplink = CloudUplink({**BASE_CONFIG, "tls_insecure_skip_verify": True})
    assert uplink._verify is False
    out = capsys.readouterr().out
    assert "tls_insecure_skip_verify is ON" in out
    assert "development only" in out.lower()


def test_a_cert_path_wins_over_insecure_skip_verify_when_both_are_set():
    """Pinning to a specific cert is strictly safer than disabling
    verification entirely -- if someone sets both, the safer one applies."""
    uplink = CloudUplink({
        **BASE_CONFIG, "ca_cert_path": "/etc/sensordyme/dev-cert.pem",
        "tls_insecure_skip_verify": True,
    })
    assert uplink._verify == "/etc/sensordyme/dev-cert.pem"


def test_no_tls_config_never_prints_the_insecure_warning(capsys):
    CloudUplink(BASE_CONFIG)
    assert "tls_insecure_skip_verify" not in capsys.readouterr().out


class _RecordingRequests:
    """Captures the kwargs `_run` actually calls `.post()` with, then stops
    the loop -- the same one-iteration pattern test_uplink_backlog.py uses."""

    def __init__(self, stop):
        self.stop = stop
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append(kwargs)
        self.stop.set()
        response = types.SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=dict,
        )
        return response


def _drain_once(uplink):
    uplink._run()
    assert uplink._requests.calls, "the worker body never ran"


def test_the_resolved_verify_value_actually_reaches_the_http_call():
    uplink = CloudUplink({**BASE_CONFIG, "ca_cert_path": "/etc/sensordyme/dev-cert.pem"})
    uplink._requests = _RecordingRequests(stop=uplink._stop)
    _drain_once(uplink)
    assert uplink._requests.calls[0]["verify"] == "/etc/sensordyme/dev-cert.pem"
