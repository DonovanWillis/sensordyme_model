"""The backend announcing itself on the LAN, so the app never needs an address.

The failures this guards against are all silent ones. mDNS gives no error when a
service type is wrong, iOS gives no error when a service type is undeclared, and
an advertised-but-wrong port fails only on the phone, at sign-in, with a message
that blames the network. So the checks here are mostly about agreement between
files that cannot see each other.
"""

from __future__ import annotations

import plistlib
import socket
import threading as _threading
from pathlib import Path

import pytest
import yaml
import zeroconf as _zeroconf

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


class TestSwiftLiteralsAgreeWithThePublisher:
    """The Swift constants that actually drive NWBrowser must match what the
    backend and the device publish.

    This is the gap that let a real regression through. The existing agreement
    test checks `cloud/mdns.py` against `ios/project.yml`'s NSBonjourServices —
    a *declaration* list containing BOTH service types. But the app browses for
    whatever string is written in the Swift source, and nothing read that. So
    when `DeviceDiscovery` was switched from `_sensordyme._tcp` (the device) to
    `_sensordyme-api._tcp` (the backend), every test still passed: both types are
    declared, so the plist check was satisfied either way. The app listed the Mac
    backend as if it were a camera and the real camera never appeared.

    Two different services, and confusing them is silent on both sides:
      * `_sensordyme._tcp`     — the DEVICE dashboard, published by avahi on the
                                 Pi (scripts/publish_mdns.py). Browsed by
                                 `DeviceDiscovery`.
      * `_sensordyme-api._tcp` — the BACKEND, published by cloud/mdns.py.
                                 Browsed by `BackendLocator`.

    Reading Swift from a Python test is not elegant. It is the only thing that
    covers this at all: `ios/` is excluded from ruff and pyright, there is no
    XCTest target, and nothing automated ran the Swift suite when this was added.
    A regex against a `static let` is crude but it fails loudly on the one change
    that matters, which is the whole job.
    """

    IOS_SOURCES = REPO / "ios" / "Sources"
    #: The device's own service. Not derived from cloud/mdns.py — that publishes
    #: the backend — so it is stated once, here, and asserted against both ends.
    DEVICE_SERVICE_TYPE = "_sensordyme._tcp"

    def _static_let(self, filename: str, name: str) -> str:
        import re

        source = (self.IOS_SOURCES / filename).read_text()
        match = re.search(rf'static let {name}\s*=\s*"([^"]+)"', source)
        assert match, f"{filename} no longer declares `static let {name}` as a literal"
        return match.group(1)

    def test_backend_locator_browses_what_the_backend_publishes(self):
        browsed = self._static_let("BackendLocator.swift", "serviceType")
        published = mdns.SERVICE_TYPE.removesuffix(".local.")
        assert browsed == published, (
            f"BackendLocator browses {browsed!r} but cloud/mdns.py publishes "
            f"{published!r}; the app would find no backend and say nothing about why"
        )

    def test_device_discovery_browses_the_device_not_the_backend(self):
        browsed = self._static_let("Discovery.swift", "serviceType")
        assert browsed == self.DEVICE_SERVICE_TYPE, (
            f"DeviceDiscovery browses {browsed!r}, not {self.DEVICE_SERVICE_TYPE!r}"
        )

    def test_the_two_browsers_do_not_look_for_the_same_service(self):
        """The regression itself: one class pointed at the other's service."""
        device = self._static_let("Discovery.swift", "serviceType")
        backend = self._static_let("BackendLocator.swift", "serviceType")
        assert device != backend, (
            "DeviceDiscovery and BackendLocator browse the same Bonjour type, so "
            "one of them is looking for the wrong thing: the device dashboard and "
            "the cloud backend are separate services"
        )

    def test_both_browsed_types_are_declared_to_ios(self):
        """A type the app has not declared returns nothing, with no error."""
        spec = yaml.safe_load((REPO / "ios" / "project.yml").read_text())
        declared = spec["targets"]["SensorDyme"]["info"]["properties"]["NSBonjourServices"]
        for filename in ("BackendLocator.swift", "Discovery.swift"):
            browsed = self._static_let(filename, "serviceType")
            assert browsed in declared, (
                f"{filename} browses {browsed!r}, which ios/project.yml does not "
                f"declare in NSBonjourServices ({declared}); iOS returns nothing"
            )

    def test_the_health_identity_check_matches_what_health_returns(self):
        """The app refuses a server whose /v1/health `service` is not this string,
        so a change to the payload silently makes every backend unrecognisable."""
        expected = self._static_let("BackendLocator.swift", "expectedService")
        fastapi = pytest.importorskip("fastapi")
        del fastapi
        from fastapi.testclient import TestClient

        from cloud.app import app

        payload = TestClient(app).get("/v1/health").json()
        assert payload["service"] == expected, (
            f"/v1/health reports service={payload['service']!r} but the app only "
            f"accepts {expected!r}, so it would reject the real backend"
        )


class TestBuildTimeBackendURL:
    """The build must not ship pointed at somebody's LAN address.

    This is the regression that took the app off the air, and it is worth a test
    because nothing else catches it: `SENSORDYME_API_BASE_URL` was set to
    `http://11.22.7.64:8787`, a build-time URL outranks mDNS discovery, and the
    network that address named was one the Mac had already left. So every request
    went to an address that could not answer, discovery was never consulted even
    though it had already found the healthy Mac, and the address could not be
    changed without editing this file and rebuilding.

    An empty value is the correct default: the app reads it as "not configured"
    and finds the Mac on the network, which is the behaviour that survives DHCP.
    A real deployed HTTPS URL is also fine — it is a name, valid from anywhere.
    What is never fine is a bare IP or a `.local` name, both of which are true on
    exactly one network.
    """

    def _configured(self) -> str:
        spec = yaml.safe_load((REPO / "ios" / "project.yml").read_text())
        return str(spec["targets"]["SensorDyme"]["settings"]["base"]["SENSORDYME_API_BASE_URL"])

    def test_is_not_a_lan_address(self):
        import re

        value = self._configured().strip()
        if not value:
            return  # empty is the intended default
        host = re.sub(r"^[a-zA-Z]+://", "", value).split("/")[0].split(":")[0]
        is_ip_literal = re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) is not None
        assert not is_ip_literal, (
            f"ios/project.yml ships SENSORDYME_API_BASE_URL={value!r}. A build-time URL "
            f"outranks mDNS discovery, so a LAN IP here strands every build on one "
            f"network and cannot be changed without a rebuild. Leave it empty for "
            f"development, or set a real deployed hostname."
        )
        assert not host.endswith(".local"), (
            f"ios/project.yml ships SENSORDYME_API_BASE_URL={value!r}; a .local name is "
            f"only meaningful on one network and outranks discovery."
        )

    def test_a_non_empty_value_is_a_complete_url(self):
        """An unparseable value still becomes the app's top-priority candidate."""
        value = self._configured().strip()
        if not value:
            return
        assert value.startswith(("http://", "https://")), (
            f"SENSORDYME_API_BASE_URL={value!r} has no scheme, so it is not a usable base URL"
        )


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
        advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
        try:
            assert advertiser.start() is False
        finally:
            advertiser.stop()

    def test_default_name_has_no_dots(self, monkeypatch):
        """A dot in the instance name splits the mDNS label and produces a name
        that is not the one anybody meant to publish."""
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "some-mac.local")
        assert "." not in mdns.Advertiser(8787).name


class _FakeZeroconf:
    """Records what went on the wire, so a test can assert the record is true
    without needing a second machine to browse from."""

    def __init__(self) -> None:
        self.registered: list = []
        self.updated: list = []
        self.unregistered: list = []
        self.closed = False
        self.update_error: Exception | None = None
        self.register_error: Exception | None = None

    def register_service(self, info) -> None:
        if self.register_error is not None:
            raise self.register_error
        self.registered.append(info)

    def update_service(self, info) -> None:
        if self.update_error is not None:
            raise self.update_error
        self.updated.append(info)

    def unregister_service(self, info) -> None:
        self.unregistered.append(info)

    def close(self) -> None:
        self.closed = True


class _Built(list):
    """Every Zeroconf the Advertiser constructed, newest last, plus a way to make
    the NEXT construction fail -- which is what a rebind under a flapping
    interface actually looks like."""

    arm: dict


def _zeroconf_failing(built: _Built, *, register_error: Exception) -> None:
    built.arm["register_error"] = register_error


def _advertised_address(info) -> str:
    return socket.inet_ntoa(info.addresses[0])


@pytest.fixture
def fake_mdns(monkeypatch):
    """An Advertiser wired to a fake Zeroconf, with the machine's identity under
    the test's control and the refresh timer off so ticks are explicit."""
    built: _Built = _Built()

    arm: dict = {"register_error": None}

    def _new_zeroconf(*_a, **_k):
        # A FRESH instance per construction, which is the whole point: the bug
        # was reusing one whose sockets were bound to an address that had gone.
        instance = _FakeZeroconf()
        instance.register_error = arm["register_error"]
        built.append(instance)
        return instance

    built.arm = arm

    monkeypatch.setattr(_zeroconf, "Zeroconf", _new_zeroconf)

    identity = {"address": "10.0.0.5", "host": "mac-one"}
    monkeypatch.setattr(mdns, "lan_ip", lambda: identity["address"])
    monkeypatch.setattr(mdns.socket, "gethostname", lambda: identity["host"])

    advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
    assert advertiser.start() is True
    assert len(built) == 1, "start() builds exactly one Zeroconf"
    fake = built[0]
    assert len(fake.registered) == 1
    try:
        yield advertiser, fake, identity, built
    finally:
        advertiser.stop()


class TestAdvertisementStaysTrue:
    """The record must keep naming the address this machine actually has.

    The bug: address and hostname were read once, at startup, and repeated for
    the life of the process. A LaunchAgent keeps this backend up for weeks, so
    the process that survives a Wi-Fi change, a tether, a DHCP move or a macOS
    rename is exactly the one still advertising where it used to be. That is
    worse than not advertising: a phone that finds nothing keeps looking, while a
    phone handed a dead address connects to nothing and reports the server down.
    """

    def test_a_quiet_tick_changes_nothing(self, fake_mdns):
        advertiser, fake, _, _built = fake_mdns
        assert advertiser.refresh() is False
        assert fake.updated == []
        assert fake.unregistered == []

    def test_a_new_dhcp_lease_REBINDS_rather_than_updating_in_place(self, fake_mdns):
        """This test used to assert `update_service`, and that was the bug.

        python-zeroconf binds its multicast sockets to the interface addresses
        present when the instance is constructed, and never rebinds them. So
        updating the record in place on an instance that has outlived a DHCP move
        re-announces over sockets bound to an address that is gone.

        What that looked like in the field, measured on a campus network that
        moved this Mac 38 times across 12 subnets: the SERVICE record kept being
        found -- macOS's own responder relays it -- while the STABLE_HOSTNAME A
        record inside it was answered on loopback only. A phone therefore found
        the service, failed to resolve the hostname in it, and said "Couldn't
        find a SensorDyme on this network". A Zeroconf instance built fresh at the
        same instant published the same name on en0 correctly, which is what
        identified the instance's age as the variable rather than the network.

        So a move now rebuilds. Clients see the service leave and come back
        instead of correcting itself, which is the right trade: a brief
        re-announcement is recoverable and an unresolvable hostname is not.
        """
        advertiser, fake, identity, built = fake_mdns
        identity["address"] = "192.168.4.22"

        assert advertiser.refresh() is True

        assert len(built) == 2, "the move must build a NEW Zeroconf, not reuse the old one"
        assert fake.closed, "and the old instance's dead sockets must be closed"
        assert fake.updated == [], "update_service on a stale instance is the bug itself"
        assert _advertised_address(built[-1].registered[-1]) == "192.168.4.22"
        assert built[-1].registered[-1].server == f"{mdns.STABLE_HOSTNAME}.local."

    def test_the_instance_name_does_not_churn(self, fake_mdns):
        """What a client keys on must stay put, or one server looks like several
        appearing and disappearing."""
        advertiser, fake, identity, _built = fake_mdns
        original = fake.registered[0].name

        identity["address"] = "192.168.4.22"
        advertiser.refresh()
        identity["host"] = "mac-one-4"
        advertiser.refresh()

        for info in fake.registered + fake.updated:
            assert info.name == original

    def test_a_macos_rename_changes_nothing_because_the_name_is_ours(self, fake_mdns):
        """This used to require a full re-registration, and that was the bug.

        macOS renames a Mac on any name collision — this one went from
        `Donovans-MacBook-Pro-2` to `-4`. The record carried that hostname in
        `server`, so every client holding the old name was pointed at a host that
        no longer existed. It cost ten days of a device dialling `…-Pro-2.local`
        and being reported offline while the backend was up the whole time.

        `server` is now `STABLE_HOSTNAME`, a name this backend publishes for
        itself, so a rename is not an event the record has to survive — it simply
        does not appear in it. The strongest form of the fix is that this test
        asserts *nothing happens*.
        """
        advertiser, fake, identity, _built = fake_mdns
        identity["host"] = "mac-one-4"

        assert advertiser.refresh() is False, "a rename must not disturb the record"

        assert fake.unregistered == []
        assert len(fake.registered) == 1
        assert fake.registered[0].server == f"{mdns.STABLE_HOSTNAME}.local."

    def test_losing_the_network_withdraws_rather_than_lying(self, fake_mdns):
        advertiser, fake, identity, _built = fake_mdns
        identity["address"] = None

        assert advertiser.refresh() is True

        assert len(fake.unregistered) == 1
        # Nothing new went out in its place: silence, not a dead address.
        assert len(fake.registered) == 1

    def test_the_announcement_comes_back_when_the_network_does(self, fake_mdns):
        advertiser, _fake, identity, built = fake_mdns
        identity["address"] = None
        advertiser.refresh()

        identity["address"] = "172.20.10.2"
        assert advertiser.refresh() is True

        # A DIFFERENT address came back, so this rebinds: the fresh announcement
        # lands on a new instance rather than the one bound to the old address.
        # That shape -- interface down, then up on a new address -- is how most
        # DHCP moves actually arrive, which is why this path has to rebind too.
        assert len(built) == 2
        assert len(built[-1].registered) == 1
        assert _advertised_address(built[-1].registered[-1]) == "172.20.10.2"

    def test_a_withdrawn_record_stays_withdrawn_on_a_quiet_tick(self, fake_mdns):
        advertiser, fake, identity, _built = fake_mdns
        identity["address"] = None
        advertiser.refresh()
        assert advertiser.refresh() is False
        assert len(fake.unregistered) == 1

    def test_a_rebind_that_cannot_register_reports_failure(self, fake_mdns):
        """A move that cannot be re-announced must say so rather than look fine.

        The old version of this test asserted the PREVIOUS address survived a
        failed update. That is no longer the right guarantee and keeping it would
        have been wrong: once the interface has moved, the old address is not a
        safer thing to keep advertising -- it is a dead address, and a phone
        handed one stops looking, where a phone handed nothing keeps searching.
        So the record is dropped and `refresh` returns False.
        """
        advertiser, _fake, identity, built = fake_mdns
        identity["address"] = "192.168.4.22"
        _zeroconf_failing(built, register_error=OSError("network down mid-register"))

        assert advertiser.refresh() is False
        assert advertiser._info is None, "a move that failed must not claim a live record"

    def test_a_rebind_that_cannot_open_mdns_at_all_reports_failure(self, fake_mdns, monkeypatch):
        """Constructing the replacement can itself fail while the network moves."""
        advertiser, _fake, identity, _built = fake_mdns
        identity["address"] = "192.168.4.22"
        monkeypatch.setattr(
            _zeroconf, "Zeroconf", lambda *a, **k: (_ for _ in ()).throw(OSError("no interface"))
        )

        assert advertiser.refresh() is False
        assert advertiser._info is None

    def test_refresh_before_start_is_a_no_op(self):
        assert mdns.Advertiser(8787).refresh() is False

    def test_refresh_after_stop_is_a_no_op(self, fake_mdns):
        advertiser, _, identity, _built = fake_mdns
        advertiser.stop()
        identity["address"] = "192.168.4.22"
        assert advertiser.refresh() is False


class TestAdvertisingRecoversOnItsOwn:
    """A failure to announce must be temporary, never terminal.

    The LaunchAgent starts this backend at login, which on a laptop is routinely
    before Wi-Fi has joined; and a DHCP move can make building a fresh Zeroconf
    fail for a moment. Either one used to leave `_zeroconf` as None, and every
    later refresh returned early on exactly that -- so the backend served
    happily for days while announcing nothing, and only a restart fixed it.
    """

    @pytest.fixture
    def world(self, monkeypatch):
        built: list = []
        state: dict = {"address": None, "zeroconf_fails": False}

        def _new_zeroconf(*_a, **_k):
            if state["zeroconf_fails"]:
                raise OSError("no multicast-capable interface")
            instance = _FakeZeroconf()
            built.append(instance)
            return instance

        monkeypatch.setattr(_zeroconf, "Zeroconf", _new_zeroconf)
        monkeypatch.setattr(mdns, "lan_ip", lambda: state["address"])
        advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
        try:
            yield advertiser, state, built
        finally:
            advertiser.stop()

    def test_started_before_the_network_it_announces_once_the_network_arrives(self, world):
        advertiser, state, built = world
        assert advertiser.start() is False, "nothing to announce yet"
        assert advertiser.refresh() is False, "still no network: nothing to do"

        state["address"] = "10.0.0.5"
        assert advertiser.refresh() is True
        assert len(built) == 1 and len(built[0].registered) == 1
        assert _advertised_address(built[0].registered[0]) == "10.0.0.5"

    def test_a_zeroconf_that_could_not_open_at_start_is_retried(self, world):
        advertiser, state, built = world
        state["address"] = "10.0.0.5"
        state["zeroconf_fails"] = True
        assert advertiser.start() is False

        state["zeroconf_fails"] = False
        assert advertiser.refresh() is True
        assert _advertised_address(built[-1].registered[-1]) == "10.0.0.5"

    def test_a_rebind_that_failed_mid_move_is_retried_on_the_next_tick(self, world):
        advertiser, state, built = world
        state["address"] = "10.0.0.5"
        assert advertiser.start() is True

        state["address"] = "192.168.4.22"
        state["zeroconf_fails"] = True
        assert advertiser.refresh() is False, "the move could not be announced"

        state["zeroconf_fails"] = False
        assert advertiser.refresh() is True, "...but it must not stay that way"
        assert _advertised_address(built[-1].registered[-1]) == "192.168.4.22"

    def test_stop_still_ends_retrying(self, world):
        advertiser, state, built = world
        assert advertiser.start() is False
        advertiser.stop()
        state["address"] = "10.0.0.5"
        assert advertiser.refresh() is False
        assert built == []


class TestTheInstanceNameIsOursNotTheNetworksx:
    """The service instance name must not be derived from the machine hostname.

    Found live, and it is why the app said "Couldn't find a SensorDyme on this
    network" even once discovery was otherwise working. This network is a
    university ResNet that assigns the hostname over DHCP as
    `dhcp-10-130-229-13.dyn.uncc.edu` -- the IP ADDRESS, inside the hostname. The
    instance name was `f"SensorDyme Cloud on {socket.gethostname()...}"`, so it
    became `SensorDyme Cloud on dhcp-10-130-229-13` and changed with every lease.

    The instance name is the client-facing key for a Bonjour service. One that
    moves means a browsing client watches its known server vanish and an
    unfamiliar one arrive on every DHCP move -- which is the opposite of what an
    instance name exists to provide, and the same class of bug already fixed once
    for `server` via STABLE_HOSTNAME.
    """

    def test_the_name_survives_a_hostname_that_embeds_the_ip(self, monkeypatch):
        monkeypatch.setattr(
            mdns.socket, "gethostname", lambda: "dhcp-10-130-229-13.dyn.uncc.edu"
        )
        name = mdns.Advertiser(8787).name
        assert "dhcp" not in name, "the DHCP hostname must not reach the instance name"
        assert "10-130" not in name, "and neither must the address inside it"

    def test_the_name_is_the_same_on_two_different_networks(self, monkeypatch):
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "dhcp-10-130-229-13")
        first = mdns.Advertiser(8787).name
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "dhcp-11-33-6-25")
        assert mdns.Advertiser(8787).name == first, \
            "the same backend on two networks is one service, not two"

    def test_the_name_still_distinguishes_two_backends(self, monkeypatch):
        """A single shared product name would make two Macs indistinguishable in
        the picker, which is the failure `chooseBackend` refuses to guess at."""
        monkeypatch.setattr(mdns, "backend_id", lambda: "bk_aaaaaaaaaaaa")
        mine = mdns.Advertiser(8787).name
        monkeypatch.setattr(mdns, "backend_id", lambda: "bk_bbbbbbbbbbbb")
        assert mdns.Advertiser(8787).name != mine

    def test_the_name_is_a_legal_mdns_label(self):
        name = mdns.Advertiser(8787).name
        assert "." not in name, "a dot splits the label into a name nobody published"
        assert 0 < len(name.encode()) <= 63  # RFC 6763 instance-name limit


class TestStableIdentity:
    """The service identity must be stable while the endpoint is disposable.

    That sentence is the whole requirement, and the old record got it backwards:
    the endpoint (address) was refreshed but the identity (`server`) was the OS
    hostname — the one part nobody controls. macOS renamed the Mac and every
    client holding the old name was pointed at nothing.
    """

    def test_the_advertised_hostname_is_one_we_own(self, fake_mdns):
        _, fake, _, _built = fake_mdns
        assert fake.registered[0].server == f"{mdns.STABLE_HOSTNAME}.local."

    def test_the_advertised_hostname_does_not_contain_the_machine_name(self, fake_mdns):
        """The specific regression: `mac-one` must not appear in the record."""
        _, fake, identity, _built = fake_mdns
        assert identity["host"] not in fake.registered[0].server

    def test_a_new_address_still_moves_the_record(self, fake_mdns):
        """A stable name must not mean a frozen address — the name resolves to
        wherever the machine currently is, so the address must still track."""
        advertiser, _fake, identity, built = fake_mdns
        identity["address"] = "192.168.9.9"
        assert advertiser.refresh() is True
        latest = built[-1].registered[-1]
        assert _advertised_address(latest) == "192.168.9.9"
        assert latest.server == f"{mdns.STABLE_HOSTNAME}.local.", \
            "the NAME is what clients keep; only the address it points at moves"

    def test_the_stable_name_is_short_and_dns_safe(self):
        """It becomes a DNS label, so it cannot carry dots, spaces or capitals
        without becoming a different name than the one clients are told to use."""
        name = mdns.STABLE_HOSTNAME
        assert name == name.lower()
        assert "." not in name and " " not in name and "_" not in name
        assert 0 < len(name.encode()) <= 63  # RFC 1035 label limit
        assert name.replace("-", "").isalnum()


class TestBackendIdentity:
    """Two SensorDyme backends on one network have to be distinguishable.

    Without an id the app takes whichever answers its health check first, which
    on a shared or campus network can be somebody else's Mac. The device side has
    had `id=SD-…` for a while and the app already refuses to silently adopt a
    different device; this is the same guarantee for the backend.
    """

    def test_the_id_is_advertised_in_the_txt_record(self, fake_mdns):
        advertiser, fake, _, _built = fake_mdns
        txt = fake.registered[0].properties
        assert txt[b"id"].decode() == advertiser.backend_id

    def test_the_id_is_stable_across_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(tmp_path / ".backend_id"))
        first = mdns.backend_id()
        assert mdns.backend_id() == first, "an identity that changes is not one"

    def test_the_id_is_persisted_so_it_survives_a_restart(self, tmp_path, monkeypatch):
        path = tmp_path / ".backend_id"
        monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(path))
        minted = mdns.backend_id()
        assert path.exists() and path.read_text().strip() == minted

    def test_two_backends_get_different_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(tmp_path / "a"))
        a = mdns.backend_id()
        monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(tmp_path / "b"))
        b = mdns.backend_id()
        assert a != b

    def test_an_unwritable_location_still_yields_an_id(self, monkeypatch):
        """Better a temporary identity than none: the backend must still start."""
        monkeypatch.setattr(mdns, "BACKEND_ID_FILE", "/proc/definitely/not/writable")
        assert mdns.backend_id().startswith("bk_")

    def test_the_id_is_not_derived_from_a_secret(self):
        """It identifies a backend, it does not authenticate one. Deriving it from
        the JWT secret would leak information about the secret into a record that
        every device on the network can read."""
        source = (REPO / "cloud" / "mdns.py").read_text()
        block = source[source.index("def backend_id"):source.index("def backend_id") + 1400]
        for forbidden in ("jwt", "JWT", "secret_file", "_load_jwt"):
            assert forbidden not in block


class TestRefreshTimer:
    def test_the_watcher_runs_refresh_without_being_asked(self, monkeypatch):
        """The correction is worthless if something has to remember to trigger it."""
        monkeypatch.setattr(_zeroconf, "Zeroconf", lambda *a, **k: _FakeZeroconf())
        monkeypatch.setattr(mdns, "lan_ip", lambda: "10.0.0.5")
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "mac-one")

        ticked = _threading.Event()
        advertiser = mdns.Advertiser(8787, refresh_interval_s=0.01)
        monkeypatch.setattr(advertiser, "refresh", ticked.set)
        try:
            assert advertiser.start() is True
            assert ticked.wait(5), "the refresh timer never fired"
        finally:
            advertiser.stop()

    def test_stop_joins_the_watcher(self, monkeypatch):
        """A thread still polling after shutdown keeps a Zeroconf instance alive
        and re-registers a service the server no longer backs."""
        monkeypatch.setattr(_zeroconf, "Zeroconf", lambda *a, **k: _FakeZeroconf())
        monkeypatch.setattr(mdns, "lan_ip", lambda: "10.0.0.5")
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "mac-one")

        advertiser = mdns.Advertiser(8787, refresh_interval_s=0.01)
        advertiser.start()
        watcher = advertiser._watcher
        assert watcher is not None
        advertiser.stop()
        assert not watcher.is_alive()

    def test_a_refresh_that_raises_does_not_kill_the_timer(self, monkeypatch):
        monkeypatch.setattr(_zeroconf, "Zeroconf", lambda *a, **k: _FakeZeroconf())
        monkeypatch.setattr(mdns, "lan_ip", lambda: "10.0.0.5")
        monkeypatch.setattr(mdns.socket, "gethostname", lambda: "mac-one")

        calls = []
        twice = _threading.Event()

        def exploding_refresh():
            calls.append(1)
            if len(calls) >= 2:
                twice.set()
            raise RuntimeError("boom")

        advertiser = mdns.Advertiser(8787, refresh_interval_s=0.01)
        monkeypatch.setattr(advertiser, "refresh", exploding_refresh)
        try:
            advertiser.start()
            assert twice.wait(5), "the timer stopped after the first failure"
        finally:
            advertiser.stop()


def test_the_zeroconf_pin_includes_the_nsec_fix():
    """zeroconf before 0.150.4 answered an AAAA query for our IPv4-only host with
    an NSEC owned by the service INSTANCE name, listing the missing type. Resolvers
    ignore that, so every lookup of sensordyme-backend.local that asked for IPv6
    stalled 5 s (measured 5.004 s; 0.001-0.013 s after the upgrade)."""
    import re
    from pathlib import Path

    from zeroconf import __version__ as installed

    pin = re.search(r"^zeroconf==0\.(\d+)\.\*", (Path(__file__).resolve().parent.parent / "cloud" / "requirements.txt").read_text(), re.M)
    assert pin and int(pin.group(1)) >= 151, "cloud/requirements.txt pins a zeroconf without the NSEC fix"
    major, minor, patch = (int(p) for p in installed.split(".")[:3])
    assert (major, minor, patch) >= (0, 150, 4), f"installed zeroconf {installed} predates the NSEC fix"
