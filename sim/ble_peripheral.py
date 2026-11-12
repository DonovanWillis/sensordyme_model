"""A virtual SensorDyme Bluetooth peripheral on localhost (development only).

    python -m sim.ble_peripheral [--port 8790] [--auth pop|legacy] [--state-dir DIR]
                                 [--network SSID[:PSK[:SIGNAL]]]... [--online SSID]
                                 [--factory DIR/factory.json --start-device]

The iOS Simulator cannot be a Bluetooth central for a real peripheral, and
nothing Bluetooth has run on the hardware yet. This runs the device's REAL
provisioning code -- services/provisioning.py (the window, the join worker,
the attempt counter, the outcome linger), services/ble_pop.py (proof of
possession, lockout, timeouts) and services/ble_gatt.py's
BLEProvisioningServer (status encoding, network list, auth replies, write
reassembly) -- with only two things replaced:

  * BlueZ/D-Bus: `SimulatedGattServer` serves the same GATT table from
    memory. Reads and writes go through BLEProvisioningServer.gatt_read /
    gatt_write -- what the D-Bus characteristics call on the Pi -- chunked at
    an ATT MTU the way a phone's stack does (long writes are reassembled by
    the real _WriteAssembler, long reads follow ReadValue's offset cache).
  * nmcli: `VirtualWifi` joins networks from a list, with a delay, refusing
    a wrong passphrase the way core/network.py words it.

A phone-side "central" reaches it over TCP on 127.0.0.1 (never another
interface: `serve` refuses a non-loopback bind) with one JSON object per line:

    central -> peripheral                  peripheral -> central
    {"op":"scan"}                          {"ev":"adv","advertising":true,"id","name","rssi","services"}
    {"op":"connect"}                       {"ev":"connected","id"} + {"ev":"services","characteristics":[{uuid,flags}]}
                                           | {"ev":"connect_failed","reason"}
    {"op":"subscribe","char":U}            {"ev":"subscribed","char":U}
    {"op":"read","char":U,"seq":N}         {"ev":"read","char":U,"seq":N,"value":<b64>} | {...,"error"}
    {"op":"write","char":U,"seq":N,        {"ev":"wrote","char":U,"seq":N} | {...,"error"}
     "value":<b64>}
    {"op":"disconnect"}                    (async) {"ev":"notify","char":U,"value":<b64>}
                                           (async) {"ev":"disconnected","reason"}

The iOS app speaks this only in DEBUG builds launched with
`-orynSimulatedBLE 127.0.0.1:8790` (ios/Sources/BLESimulatedTransport.swift);
a Release build contains no simulated transport at all. NOT a Bluetooth
validation: it proves the app and the device state machines agree message by
message, not that a radio carries them.

Secrets: the setup code is written 0600 to <state-dir>/ble_setup_code (the
real load_setup_code checks apply) and is printed only with --print-label. The
optional --transcript records every exchange with nonces, proofs, ciphertext
and passphrases replaced by their lengths.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import functools
import ipaddress
import json
import os
import shlex
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import uuid as uuidlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.network import credential_problem, explain_connect_failure  # noqa: E402
from services import ble_gatt  # noqa: E402
from services.ble_gatt import (  # noqa: E402
    CHAR_AUTH_UUID,
    CHAR_COMMAND_UUID,
    CHAR_DEVICE_INFO_UUID,
    CHAR_NETWORKS_UUID,
    CHAR_STATUS_UUID,
    CHAR_WIFI_CREDS_UUID,
    SERVICE_UUID,
    BLEProvisioningServer,
)
from services.ble_pop import Limits  # noqa: E402

WIRE_VERSION = 1
DEFAULT_PORT = 8790
#: What iOS typically negotiates with a BlueZ peripheral (185 bytes). A long
#: value therefore crosses several ATT packets, exactly the case the real
#: reassembly code exists for.
DEFAULT_MTU = 185
MAX_LINE_BYTES = 8192

CHARACTERISTIC_NAMES = {
    CHAR_DEVICE_INFO_UUID: "device_info",
    CHAR_WIFI_CREDS_UUID: "wifi_creds",
    CHAR_STATUS_UUID: "status",
    CHAR_COMMAND_UUID: "command",
    CHAR_NETWORKS_UUID: "networks",
    CHAR_AUTH_UUID: "auth",
}

#: Fields of the proof-of-possession messages that are key material or
#: ciphertext. A recorded proof is enough to guess a code offline, so none of
#: them is ever written down, only its length.
_REDACTED_FIELDS = ("na", "nd", "tag", "iv", "ct")


# ---------------------------------------------------------------------- transcript
def redact(characteristic: str, data: bytes) -> Any:
    """A loggable account of one GATT value: structure kept, secrets reduced to lengths."""
    try:
        value = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return f"<{len(data)} bytes, not JSON>"
    if isinstance(value, dict):
        value = dict(value)
        for key in _REDACTED_FIELDS:
            if isinstance(value.get(key), str):
                try:
                    size = len(base64.b64decode(value[key], validate=True))
                except (binascii.Error, ValueError):
                    size = len(value[key])
                value[key] = f"<{size} bytes>"
        if "psk" in value:  # a legacy plain credential write
            value["psk"] = None if value["psk"] is None else "<redacted>"
    return value


class Transcript:
    """Every exchange on the virtual link, redacted, optionally appended to a JSONL file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def add(self, direction: str, kind: str, **fields: Any) -> None:
        record = {"t": round(time.monotonic() - self._t0, 3), "dir": direction, "kind": kind, **fields}
        with self._lock:
            self.records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------- the radio
@dataclass
class SimNetwork:
    ssid: str
    psk: Optional[str] = None  # None: an open network
    signal: int = 70

    @classmethod
    def parse(cls, text: str) -> "SimNetwork":
        """SSID[:PSK[:SIGNAL]]; an empty PSK is an open network."""
        parts = text.split(":")
        ssid = parts[0]
        psk = parts[1] if len(parts) > 1 and parts[1] != "" else None
        signal = int(parts[2]) if len(parts) > 2 else 70
        if not ssid:
            raise ValueError(f"network {text!r} has no SSID")
        return cls(ssid, psk, signal)


DEFAULT_NETWORKS = (
    SimNetwork("SensorDyme Lab", "correct-horse-battery", 82),
    SimNetwork("Neighbour 5G", "not-yours-123", 41),
    SimNetwork("CoffeeShop Guest", None, 58),
)


class VirtualWifi:
    """The WifiManager surface ProvisioningService uses, without nmcli.

    `provision` takes `join_delay_s`, joins when the passphrase matches the
    network's, and otherwise fails with the same words core/network.py uses
    for NetworkManager's refusal. A network not in the list "cannot be seen".
    """

    def __init__(self, networks: list[SimNetwork], *, online_ssid: Optional[str] = None,
                 join_delay_s: float = 1.0, on_joined: Optional[Callable[[str], None]] = None) -> None:
        self.networks = {n.ssid: n for n in networks}
        self.join_delay_s = float(join_delay_s)
        self._ssid = online_ssid
        self._on_joined = on_joined
        self._watchers: list[Callable[[], None]] = []
        self.joins: list[tuple[str, bool]] = []  # (ssid, ok) -- never the passphrase

    def is_online(self) -> bool:
        return self._ssid is not None

    def current_ssid(self) -> Optional[str]:
        return self._ssid

    def ip_address(self) -> Optional[str]:
        return "192.0.2.10" if self._ssid else None  # TEST-NET-1: obviously not a real address

    def scan(self, limit: int = 20) -> list[dict]:
        found = sorted(self.networks.values(), key=lambda n: -n.signal)[:limit]
        return [{"ssid": n.ssid, "signal": n.signal, "secured": n.psk is not None} for n in found]

    def watch(self, on_change: Callable[[], None], stop: threading.Event) -> None:
        self._watchers.append(on_change)

    def _changed(self) -> None:
        for watcher in list(self._watchers):
            watcher()

    def provision(self, ssid: str, password: str) -> tuple[bool, str]:
        problem = credential_problem(ssid, password)
        if problem:
            return False, problem
        time.sleep(self.join_delay_s)
        network = self.networks.get(ssid)
        if network is None:
            ok, message = False, explain_connect_failure("No network with SSID found", ssid)
        elif (network.psk or "") != (password or ""):
            ok, message = False, explain_connect_failure("Secrets were required, but not provided", ssid)
        else:
            ok, message = True, f"joined '{ssid}'"
            self._ssid = ssid
        self.joins.append((ssid, ok))
        self._changed()
        if ok and self._on_joined is not None:
            self._on_joined(ssid)
        return ok, message


# ---------------------------------------------------------------------- GATT
class _Notifier:
    """Stands in for a D-Bus characteristic's push_value (PropertiesChanged)."""

    def __init__(self, peripheral: "VirtualPeripheral", uuid: str) -> None:
        self._peripheral, self._uuid = peripheral, uuid

    def push_value(self, data: bytes) -> None:
        self._peripheral.notify(self._uuid, bytes(data))


class SimulatedGattServer(BLEProvisioningServer):
    """BLEProvisioningServer with the BlueZ layer replaced by an in-memory GATT table.

    Everything above D-Bus is the production class: status encoding, the scan
    cache, the auth handler, the credential refusal under proof of possession.
    """

    def __init__(self, *args: Any, peripheral: "VirtualPeripheral", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The Pi reports its own hostname; this Mac's is not the device's (nor
        # anything a transcript should carry).
        if isinstance(self._device_info, dict) and "hostname" in self._device_info:
            self._device_info = {**self._device_info,
                                 "hostname": f"sensordyme-sim-{peripheral.device_id[-8:].lower()}"}
        self._peripheral = peripheral
        self._alive = False
        self._assemblers: dict[str, ble_gatt._WriteAssembler] = {}
        self._read_cache: dict[str, bytes] = {}

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._running = True
        self._alive = True
        self._status_chrc = _Notifier(self._peripheral, CHAR_STATUS_UUID)
        if self._on_auth is not None:
            self._auth_chrc = _Notifier(self._peripheral, CHAR_AUTH_UUID)
        self._peripheral.attach(self)
        self.refresh_networks()

    def stop(self) -> None:
        self._running = False
        self._alive = False
        self._status_chrc = None
        self._auth_chrc = None
        self._peripheral.detach(self)

    def is_alive(self) -> bool:
        return self._alive

    def is_advertising(self) -> bool:
        return self._alive

    # ---------------------------------------------------------------- ATT
    def flags(self, uuid: str) -> Optional[tuple[str, ...]]:
        return dict(self.gatt_characteristics()).get(uuid)

    def att_read(self, uuid: str, mtu: int) -> bytes:
        """A whole read as a phone performs it: Read, then Read Blob until a short chunk.

        Later chunks come from the value cached at offset 0, as ReadValue does.
        """
        chunk = max(1, mtu - 1)
        out = bytearray()
        offset = 0
        while True:
            if offset == 0 or uuid not in self._read_cache:
                self._read_cache[uuid] = bytes(self.gatt_read(uuid, offset))
            piece = self._read_cache[uuid][offset:offset + chunk]
            out.extend(piece)
            offset += len(piece)
            if len(piece) < chunk:
                return bytes(out)

    def att_write(self, uuid: str, value: bytes, mtu: int) -> Optional[str]:
        """A write with response as a phone performs it; an ATT error string, or None."""
        flags = self.flags(uuid) or ()
        if "write" not in flags:
            return "write not permitted"
        if len(value) > ble_gatt._GATT_MAX_ATTRIBUTE_BYTES:
            return "invalid attribute value length"
        assembler = self._assemblers.setdefault(uuid, ble_gatt._WriteAssembler())
        if len(value) <= mtu - 3:
            pieces = [(value, 0)]
        else:  # a long (prepared) write: pieces with offsets, reassembled by the device
            size = max(1, mtu - 5)
            pieces = [(value[i:i + size], i) for i in range(0, len(value), size)]
        for piece, offset in pieces:
            data = assembler.feed(piece, offset)
            if data is not None:
                self.gatt_write(uuid, data)
        return None


# ---------------------------------------------------------------------- the peripheral
class _Central:
    """One connected phone-side client."""

    def __init__(self, peripheral: "VirtualPeripheral", sock: socket.socket) -> None:
        self.peripheral = peripheral
        self.sock = sock
        self.connected = False
        self.subscribed: set[str] = set()
        self._send_lock = threading.Lock()
        self.closed = False

    def send(self, message: dict) -> None:
        if self.closed:
            return
        line = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._send_lock:
                self.sock.sendall(line)
        except OSError:
            self.closed = True

    def drop(self, reason: str) -> None:
        if self.connected:
            self.connected = False
            self.subscribed.clear()
            self.send({"ev": "disconnected", "reason": reason})


class VirtualPeripheral:
    """The device as a Bluetooth central sees it: advertising while its window is open."""

    def __init__(self, *, device_id: str, rssi: int = -52, mtu: int = DEFAULT_MTU,
                 transcript: Optional[Transcript] = None) -> None:
        self.device_id = device_id
        self.peripheral_uuid = str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f"sensordyme-sim-ble:{device_id}")).upper()
        self.rssi = int(rssi)
        self.mtu = int(mtu)
        self.transcript = transcript or Transcript()
        #: BlueZ runs every GATT operation on one GLib loop thread; this lock is that thread.
        self.loop_lock = threading.RLock()
        self._server: Optional[SimulatedGattServer] = None
        self._centrals: list[_Central] = []
        self._centrals_lock = threading.Lock()
        self.window_events: list[str] = []

    # ---------------------------------------------------------------- window
    @property
    def server(self) -> Optional[SimulatedGattServer]:
        return self._server

    @property
    def advertising(self) -> bool:
        server = self._server
        return server is not None and server.is_alive()

    def attach(self, server: SimulatedGattServer) -> None:
        self._server = server
        self.window_events.append("opened")
        self.transcript.add("device", "window", state="opened", name=server._device_name)

    def detach(self, server: SimulatedGattServer) -> None:
        if self._server is server:
            self._server = None
        self.window_events.append("closed")
        self.transcript.add("device", "window", state="closed")
        with self._centrals_lock:
            centrals = list(self._centrals)
        for central in centrals:
            central.drop("the device closed its Bluetooth window")

    def notify(self, uuid: str, data: bytes) -> None:
        with self._centrals_lock:
            targets = [c for c in self._centrals if c.connected and uuid in c.subscribed]
        if not targets:
            return
        name = CHARACTERISTIC_NAMES.get(uuid, uuid)
        self.transcript.add("device->app", "notify", char=name, value=redact(name, data))
        for central in targets:
            central.send({"ev": "notify", "char": uuid, "value": base64.b64encode(data).decode("ascii")})

    # ---------------------------------------------------------------- centrals
    def register(self, central: _Central) -> None:
        with self._centrals_lock:
            self._centrals.append(central)

    def unregister(self, central: _Central) -> None:
        with self._centrals_lock:
            if central in self._centrals:
                self._centrals.remove(central)

    def handle(self, central: _Central, message: dict) -> None:
        """One request from a central. Every answer is a message; nothing raises."""
        op = message.get("op")
        if op == "hello":
            central.send({"ev": "hello", "v": WIRE_VERSION, "transport": "sensordyme-sim-ble"})
        elif op == "scan":
            self._scan(central)
        elif op == "connect":
            self._connect(central)
        elif op == "disconnect":
            if central.connected:
                self.transcript.add("app", "disconnect")
            central.drop("the app disconnected")
        elif op in ("subscribe", "read", "write"):
            self._gatt(central, op, message)
        else:
            central.send({"ev": "error", "error": f"unknown op {op!r}"})

    def _scan(self, central: _Central) -> None:
        server = self._server
        if server is None or not server.is_alive():
            central.send({"ev": "adv", "advertising": False})
            return
        central.send({"ev": "adv", "advertising": True, "id": self.peripheral_uuid, "name": server._device_name,
                      "rssi": self.rssi, "services": [SERVICE_UUID]})

    def _connect(self, central: _Central) -> None:
        server = self._server
        if server is None or not server.is_alive():
            central.send({"ev": "connect_failed", "reason": "the device is not advertising"})
            return
        central.connected = True
        self.transcript.add("app", "connect")
        central.send({"ev": "connected", "id": self.peripheral_uuid})
        central.send({"ev": "services", "service": SERVICE_UUID,
                      "characteristics": [{"uuid": u, "flags": list(f)} for u, f in server.gatt_characteristics()]})

    def _gatt(self, central: _Central, op: str, message: dict) -> None:
        uuid = str(message.get("char", "")).lower()
        seq = message.get("seq")
        server = self._server
        reply_ev = {"subscribe": "subscribed", "read": "read", "write": "wrote"}[op]
        if not central.connected or server is None or not server.is_alive():
            central.send({"ev": reply_ev, "char": uuid, "seq": seq, "error": "not connected"})
            return
        flags = server.flags(uuid)
        name = CHARACTERISTIC_NAMES.get(uuid, uuid)
        if flags is None:
            central.send({"ev": reply_ev, "char": uuid, "seq": seq, "error": "no such characteristic"})
            return
        if op == "subscribe":
            if "notify" not in flags:
                central.send({"ev": reply_ev, "char": uuid, "seq": seq, "error": "notify not permitted"})
                return
            central.subscribed.add(uuid)
            central.send({"ev": reply_ev, "char": uuid, "seq": seq})
            return
        if op == "read":
            if "read" not in flags:
                central.send({"ev": reply_ev, "char": uuid, "seq": seq, "error": "read not permitted"})
                return
            with self.loop_lock:
                value = server.att_read(uuid, self.mtu)
            self.transcript.add("app->device", "read", char=name)
            self.transcript.add("device->app", "value", char=name, value=redact(name, value))
            central.send({"ev": reply_ev, "char": uuid, "seq": seq,
                          "value": base64.b64encode(value).decode("ascii")})
            return
        try:
            data = base64.b64decode(str(message.get("value", "")), validate=True)
        except (binascii.Error, ValueError):
            central.send({"ev": reply_ev, "char": uuid, "seq": seq, "error": "value is not base64"})
            return
        self.transcript.add("app->device", "write", char=name, value=redact(name, data))
        with self.loop_lock:
            error = server.att_write(uuid, data, self.mtu)
        reply = {"ev": reply_ev, "char": uuid, "seq": seq}
        if error:
            reply["error"] = error
        central.send(reply)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        peripheral: VirtualPeripheral = self.server.peripheral  # type: ignore[attr-defined]
        central = _Central(peripheral, self.connection)
        peripheral.register(central)
        try:
            while True:
                line = self.rfile.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    central.send({"ev": "error", "error": "line too long"})
                    break
                try:
                    message = json.loads(line)
                except ValueError:
                    central.send({"ev": "error", "error": "not JSON"})
                    continue
                if isinstance(message, dict):
                    peripheral.handle(central, message)
        finally:
            if central.connected:
                peripheral.transcript.add("app", "link_lost")
            central.connected = False
            central.closed = True
            peripheral.unregister(central)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def ensure_loopback(host: str) -> str:
    """`host` if it is a loopback address; the simulated link never leaves this machine."""
    resolved = "127.0.0.1" if host == "localhost" else host
    try:
        if ipaddress.ip_address(resolved).is_loopback:
            return resolved
    except ValueError:
        pass
    raise SystemExit(f"sim.ble_peripheral: refusing to listen on {host!r}; only loopback (127.0.0.1) is allowed")


# ---------------------------------------------------------------------- one virtual device
@dataclass
class DeviceSpec:
    state_dir: Path
    auth: str = "pop"
    networks: tuple[SimNetwork, ...] = DEFAULT_NETWORKS
    online_ssid: Optional[str] = None
    join_delay_s: float = 1.0
    rssi: int = -52
    mtu: int = DEFAULT_MTU
    limits: Limits = field(default_factory=Limits)
    outcome_linger_s: float = 10.0
    reconfigure_window_s: float = 300.0
    device_key: Optional[str] = None
    claim_code: Optional[str] = None
    on_joined: Optional[Callable[[str], None]] = None
    transcript_path: Optional[Path] = None
    deployment: str = "development"


class SimBLEDevice:
    """A virtual device's Bluetooth side: identity, setup code, radio, provisioning service, TCP link."""

    def __init__(self, spec: DeviceSpec) -> None:
        from services import ble_pop
        from services.provisioning import ProvisioningService
        from sim import identity as sim_identity

        self.spec = spec
        state_dir = Path(spec.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if spec.device_key is not None or (state_dir / sim_identity.FILE_NAME).exists():
            ident = sim_identity.load_or_create(state_dir, device_key=spec.device_key, claim_code=spec.claim_code)
        else:
            # Bluetooth-only: no backend row yet. A placeholder key, never used for an uplink.
            ident = sim_identity.load_or_create(state_dir, device_key="dk_sim_ble_only")
        self.identity = ident
        self.device_id = ident.hardware_id
        self.name = f"SensorDyme-{self.device_id[-4:]}"
        self.setup_code_path = state_dir / "ble_setup_code"
        if not self.setup_code_path.exists():
            ble_pop.write_setup_code(str(self.setup_code_path), ble_pop.generate_setup_code())
        self.transcript = Transcript(spec.transcript_path)
        self.peripheral = VirtualPeripheral(device_id=self.device_id, rssi=spec.rssi, mtu=spec.mtu,
                                            transcript=self.transcript)
        self.wifi = VirtualWifi(list(spec.networks), online_ssid=spec.online_ssid,
                                join_delay_s=spec.join_delay_s, on_joined=spec.on_joined)
        config = {
            "method": "ble", "auth": spec.auth, "setup_code_path": str(self.setup_code_path),
            "poll_s": 0.2, "grace_s": 0, "outcome_linger_s": spec.outcome_linger_s,
            "reconfigure_window_s": spec.reconfigure_window_s,
        }
        self.service = ProvisioningService(
            cast(Any, self.wifi), config, version="sim", deployment=spec.deployment,
            ble_server_factory=functools.partial(SimulatedGattServer, peripheral=self.peripheral),
            identity=(self.device_id, self.name), radio_available=lambda: True, pop_limits=spec.limits,
        )
        self._tcp: Optional[_Server] = None
        self._tcp_thread: Optional[threading.Thread] = None

    def qr_payload(self) -> str:
        from services import ble_pop

        return ble_pop.qr_payload(self.device_id, ble_pop.load_setup_code(str(self.setup_code_path)))

    def start(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> int:
        """Boot the provisioning service and listen; returns the bound port."""
        self._tcp = _Server((ensure_loopback(host), port), _Handler)
        self._tcp.peripheral = self.peripheral  # type: ignore[attr-defined]
        self._tcp_thread = threading.Thread(target=self._tcp.serve_forever, daemon=True, name="sim-ble-tcp")
        self._tcp_thread.start()
        self.service.start()
        return int(self._tcp.server_address[1])

    def wait_advertising(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.peripheral.advertising:
                return True
            time.sleep(0.02)
        return False

    def stop(self) -> None:
        self.service.stop(timeout=2.0)
        if self._tcp is not None:
            self._tcp.shutdown()
            self._tcp.server_close()
            self._tcp = None


# ---------------------------------------------------------------------- CLI
def _device_launcher(args: argparse.Namespace, state_dir: Path, ident_key: str, claim: str):
    """on_joined: power the virtual device's runtime once its Wi-Fi is up (--start-device)."""
    started: dict[str, subprocess.Popen] = {}
    lock = threading.Lock()

    def on_joined(ssid: str) -> None:
        with lock:
            if "device" in started:
                return
            argv = [sys.executable, "-m", "sim.virtual_device", "--backend", args.backend,
                    "--device-key", ident_key, "--claim-code", claim, "--state-dir", str(state_dir),
                    "--scenario", args.scenario, "--log-level", "warning", *shlex.split(args.device_args)]
            started["device"] = subprocess.Popen(argv, cwd=str(REPO))  # noqa: S603 - our own module
            print(f"sim.ble_peripheral: joined {ssid!r}; started the virtual device runtime "
                  f"(pid {started['device'].pid}) against {args.backend}", flush=True)

    return on_joined, started


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.ble_peripheral", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="loopback address to listen on (only loopback)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="0 picks a free port")
    parser.add_argument("--state-dir", default=None, help="identity + setup code (default: a fresh temp dir)")
    parser.add_argument("--factory", default=None, help="factory.json from `python -m sim.backend` (device key, claim code)")
    parser.add_argument("--auth", choices=("pop", "legacy"), default="pop")
    parser.add_argument("--network", action="append", default=None, metavar="SSID[:PSK[:SIGNAL]]")
    parser.add_argument("--online", default=None, metavar="SSID",
                        help="boot already on this network (an installed device: the reconfigure window)")
    parser.add_argument("--join-delay", type=float, default=1.5)
    parser.add_argument("--rssi", type=int, default=-52)
    parser.add_argument("--mtu", type=int, default=DEFAULT_MTU)
    parser.add_argument("--challenge-ttl", type=float, default=Limits.challenge_ttl_s)
    parser.add_argument("--proven-ttl", type=float, default=Limits.proven_ttl_s)
    parser.add_argument("--max-failures", type=int, default=Limits.max_failures)
    parser.add_argument("--lockout-base", type=float, default=Limits.lockout_base_s)
    parser.add_argument("--outcome-linger", type=float, default=10.0)
    parser.add_argument("--start-device", action="store_true",
                        help="after a successful join, start `python -m sim.virtual_device` (needs --factory)")
    parser.add_argument("--backend", default="http://127.0.0.1:8787")
    parser.add_argument("--scenario", default="attentive")
    parser.add_argument("--device-args", default="", help="extra arguments for sim.virtual_device")
    parser.add_argument("--transcript", default=None, help="append the redacted exchange to this JSONL file")
    parser.add_argument("--print-label", action="store_true",
                        help="print the setup code and QR text (a printed label; never logged otherwise)")
    args = parser.parse_args(argv)

    host = ensure_loopback(args.host)
    state_dir = Path(args.state_dir or tempfile.mkdtemp(prefix="sensordyme-sim-ble-")).resolve()
    device_key = claim_code = None
    if args.factory:
        entry = json.loads(Path(args.factory).read_text())["devices"][0]
        device_key, claim_code = entry["device_key"], entry["claim_code"]
    if args.start_device and device_key is None:
        raise SystemExit("sim.ble_peripheral: --start-device needs --factory (the device key it reports with)")
    on_joined = None
    launched: dict[str, subprocess.Popen] = {}
    if args.start_device:
        on_joined, launched = _device_launcher(args, state_dir, device_key or "", claim_code or "")

    networks = tuple(SimNetwork.parse(n) for n in args.network) if args.network else DEFAULT_NETWORKS
    spec = DeviceSpec(
        state_dir=state_dir, auth=args.auth, networks=networks, online_ssid=args.online,
        join_delay_s=args.join_delay, rssi=args.rssi, mtu=args.mtu,
        limits=Limits(challenge_ttl_s=args.challenge_ttl, proven_ttl_s=args.proven_ttl,
                      max_failures=args.max_failures, lockout_base_s=args.lockout_base),
        outcome_linger_s=args.outcome_linger, device_key=device_key, claim_code=claim_code,
        on_joined=on_joined, transcript_path=Path(args.transcript) if args.transcript else None,
    )
    device = SimBLEDevice(spec)
    port = device.start(host, args.port)
    device.wait_advertising(5.0)
    print("=" * 72)
    print("VIRTUAL SensorDyme Bluetooth peripheral (SIMULATED, localhost only)")
    print(f"  device       {device.device_id} as {device.name!r}, auth: {args.auth}")
    print(f"  listening    {host}:{port}")
    print(f"  app          launch a DEBUG build with: -orynSimulatedBLE {host}:{port}")
    print(f"  setup code   in {device.setup_code_path} (mode 0600)" + ("" if args.print_label else
                                                                       "; --print-label shows it"))
    if args.print_label:
        from services import ble_pop

        print(f"  label        {ble_pop.format_setup_code(ble_pop.load_setup_code(str(device.setup_code_path)))}")
        print(f"  QR text      {device.qr_payload()}")
    print("  networks     " + ", ".join(f"{n.ssid} ({'open' if n.psk is None else 'secured'})" for n in networks))
    if args.start_device:
        print(f"  on join      starts sim.virtual_device against {args.backend}")
    print("=" * 72)
    print(f"SIMBLE READY port={port} id={device.device_id}", flush=True)

    def _terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        device.stop()
        proc = launched.get("device")
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
