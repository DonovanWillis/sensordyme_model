"""The backend announcing itself on the LAN, so the app never needs an address.

The failures this guards against are all silent ones. mDNS gives no error when a
service type is wrong, iOS gives no error when a service type is undeclared, and
an advertised-but-wrong port fails only on the phone, at sign-in, with a message
that blames the network. So the checks here are mostly about agreement between
files that cannot see each other.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest
import yaml

from cloud import mdns

REPO = Path(__file__).resolve().parents[1]


def service_name(service_type: str) -> str:
    """The `sensordyme-api` part of `_sensordyme-api._tcp.local.`"""
    return service_type.split(".")[0].lstrip("_")


class TestServiceType:
    def test_service_name_fits_the_dns_sd_limit(self):
        """RFC 6763 caps a service name at 15 bytes.

        The first name tried here was `_sensordyme-cloud._tcp`, whose name is 16
        bytes. zeroconf rejects that outright, so the server started, announced
        nothing, and only a warning line said why. Anything longer than this fails
        the same way, so the limit is asserted rather than remembered.
        """
        assert len(service_name(mdns.SERVICE_TYPE).encode()) <= 15

    def test_service_type_is_well_formed(self):
        assert mdns.SERVICE_TYPE.startswith("_")
        assert mdns.SERVICE_TYPE.endswith("._tcp.local.")

    def test_ios_declares_the_service_it_browses_for(self):
        """iOS returns nothing for a Bonjour type the app has not declared.

        Not an error — nothing. So a rename here that misses the app's declared
        services looks exactly like "there is no server on this network", on every
        phone, with no clue pointing back at this file.

        Asserted against project.yml rather than Info.plist because project.yml is
        the tracked source of truth: ios/Info.plist is gitignored and generated
        from it by xcodegen, so it may not exist at all in a fresh checkout.
        """
        spec = yaml.safe_load((REPO / "ios" / "project.yml").read_text())
        declared = spec["targets"]["SensorDyme"]["info"]["properties"]["NSBonjourServices"]
        browsed = mdns.SERVICE_TYPE.removesuffix(".local.")
        assert browsed in declared, (
            f"cloud/mdns.py advertises {browsed} but ios/project.yml declares {declared}; "
            "the app would silently find nothing"
        )

    def test_generated_info_plist_has_not_drifted(self):
        """Guard the working copy against a hand-edit that xcodegen will revert.

        Info.plist is generated, so editing it directly works right up until the
        next `xcodegen generate` silently puts it back — and discovery breaks
        again, long after the change that caused it. Skipped when the artifact has
        not been generated yet, which is the normal state of a fresh clone.
        """
        plist = REPO / "ios" / "Info.plist"
        if not plist.exists():
            pytest.skip("ios/Info.plist is a generated artifact; run xcodegen first")
        spec = yaml.safe_load((REPO / "ios" / "project.yml").read_text())
        declared = spec["targets"]["SensorDyme"]["info"]["properties"]["NSBonjourServices"]
        generated = plistlib.loads(plist.read_bytes())["NSBonjourServices"]
        assert sorted(declared) == sorted(generated)


class TestAdvertisedPort:
    def test_defaults_to_the_documented_port(self, monkeypatch):
        monkeypatch.delenv("SENSORDYME_PORT", raising=False)
        assert mdns.advertised_port() == mdns.DEFAULT_PORT

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("SENSORDYME_PORT", "9000")
        assert mdns.advertised_port() == 9000

    def test_empty_value_is_not_an_error(self, monkeypatch):
        """`SENSORDYME_PORT=` from an unset shell variable must not crash boot."""
        monkeypatch.setenv("SENSORDYME_PORT", "")
        assert mdns.advertised_port() == mdns.DEFAULT_PORT

    def test_nonsense_falls_back_rather_than_raising(self, monkeypatch):
        monkeypatch.setenv("SENSORDYME_PORT", "eight-thousand")
        assert mdns.advertised_port() == mdns.DEFAULT_PORT


class TestEnabled:
    def test_on_by_default(self, monkeypatch):
        monkeypatch.delenv("SENSORDYME_MDNS", raising=False)
        assert mdns.enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " off "])
    def test_off_switches(self, monkeypatch, value):
        monkeypatch.setenv("SENSORDYME_MDNS", value)
        assert mdns.enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
    def test_anything_else_leaves_it_on(self, monkeypatch, value):
        monkeypatch.setenv("SENSORDYME_MDNS", value)
        assert mdns.enabled() is True


class TestLanIP:
    def test_never_returns_loopback(self):
        """Advertising 127.0.0.1 tells every phone to connect to itself.

        Resolving the machine's own hostname is the obvious way to get an address
        and commonly answers loopback on a Mac, which is why lan_ip() does not do
        that. The result is either a real address or None — never a useless one.
        """
        address = mdns.lan_ip()
        assert address is None or not address.startswith("127.")


class TestAdvertiser:
    def test_stop_is_safe_before_start(self):
        """Shutdown runs in a `finally`, including when startup failed."""
        mdns.Advertiser(8787).stop()  # must not raise

    def test_stop_is_idempotent(self):
        advertiser = mdns.Advertiser(8787)
        advertiser.stop()
        advertiser.stop()

    def test_missing_zeroconf_degrades_instead_of_crashing(self, monkeypatch):
        """A backend that refuses to boot because it could not advertise itself
        would be strictly worse than one that boots and asks for an address."""
        import builtins

        real_import = builtins.__import__

        def no_zeroconf(name, *args, **kwargs):
            if name == "zeroconf":
                raise ImportError("no zeroconf")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_zeroconf)
        assert mdns.Advertiser(8787).start() is False

    def test_no_lan_address_degrades_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr(mdns, "lan_ip", lambda: None)
        assert mdns.Advertiser(8787).start() is False

    def test_default_name_has_no_dots(self, monkeypatch):
        """A dot in the instance name splits the mDNS label and produces a name
        that is not the one anybody meant to publish."""
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "some-mac.local")
        assert "." not in mdns.Advertiser(8787).name
