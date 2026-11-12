"""The phone side of the simulated Bluetooth link (sim/ble_peripheral.py), in Python.

For tests and scripts: the same wire protocol the iOS app's DEBUG-only
simulated transport speaks. `PopClient` is the app's half of proof of
possession built from services/ble_pop.py's primitives (the Swift client is
the real app's; tests/test_m8_onboarding_crosslang.py runs that one).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import socket
import threading
import time
from typing import Any, Optional

from services import ble_pop as pop
from services.ble_gatt import CHAR_AUTH_UUID, CHAR_STATUS_UUID


class LinkError(RuntimeError):
    pass


class Central:
    """One TCP link to a virtual peripheral: scan, connect, read, write, notifications."""

    def __init__(self, host: str, port: int, timeout_s: float = 10.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.settimeout(None)
        self.timeout_s = timeout_s
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self.notifications: "queue.Queue[tuple[str, bytes]]" = queue.Queue()
        self.disconnected: Optional[str] = None
        self.characteristics: dict[str, list[str]] = {}
        self._seq = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        with self.sock.makefile("rb") as stream:
            for line in stream:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get("ev") == "notify":
                    self.notifications.put((message["char"], base64.b64decode(message["value"])))
                elif message.get("ev") == "disconnected":
                    self.disconnected = message.get("reason") or "disconnected"
                    self._replies.put(message)
                else:
                    self._replies.put(message)
        if self.disconnected is None:
            self.disconnected = "link closed"
        self._replies.put({"ev": "disconnected", "reason": self.disconnected})

    def _send(self, message: dict) -> None:
        self.sock.sendall((json.dumps(message) + "\n").encode())

    def _wait(self, ev: str, seq: Optional[int] = None) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise LinkError(f"no {ev} within {self.timeout_s}s")
            try:
                message = self._replies.get(timeout=left)
            except queue.Empty as exc:
                raise LinkError(f"no {ev} within {self.timeout_s}s") from exc
            if message.get("ev") == "disconnected" and ev != "disconnected":
                raise LinkError(f"disconnected: {message.get('reason')}")
            if message.get("ev") in (ev, "connect_failed") and (seq is None or message.get("seq") == seq):
                return message

    # ---------------------------------------------------------------- GATT
    def scan(self) -> dict:
        self._send({"op": "scan"})
        return self._wait("adv")

    def connect(self) -> dict:
        self._send({"op": "connect"})
        reply = self._wait("connected")
        if reply.get("ev") == "connect_failed":
            raise LinkError(reply.get("reason", "connect failed"))
        services = self._wait("services")
        self.characteristics = {c["uuid"]: c["flags"] for c in services["characteristics"]}
        return reply

    def subscribe(self, char: str) -> None:
        self._seq += 1
        self._send({"op": "subscribe", "char": char, "seq": self._seq})
        reply = self._wait("subscribed", self._seq)
        if reply.get("error"):
            raise LinkError(reply["error"])

    def read(self, char: str) -> bytes:
        self._seq += 1
        self._send({"op": "read", "char": char, "seq": self._seq})
        reply = self._wait("read", self._seq)
        if reply.get("error"):
            raise LinkError(reply["error"])
        return base64.b64decode(reply["value"])

    def read_json(self, char: str) -> Any:
        return json.loads(self.read(char))

    def write(self, char: str, value: bytes) -> Optional[str]:
        """Write with response; the ATT error, or None."""
        self._seq += 1
        self._send({"op": "write", "char": char, "seq": self._seq,
                    "value": base64.b64encode(value).decode("ascii")})
        return self._wait("wrote", self._seq).get("error")

    def next_notification(self, char: str, timeout_s: Optional[float] = None) -> bytes:
        deadline = time.monotonic() + (timeout_s or self.timeout_s)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise LinkError(f"no notification on {char} in time")
            try:
                uuid, value = self.notifications.get(timeout=left)
            except queue.Empty as exc:
                raise LinkError(f"no notification on {char} in time") from exc
            if uuid == char:
                return value

    def wait_status(self, predicate, timeout_s: float = 15.0) -> dict:
        """The first status notification (or poll) for which predicate(status) holds."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                status = json.loads(self.next_notification(CHAR_STATUS_UUID, timeout_s=0.5))
            except LinkError:
                if self.disconnected:
                    raise
                status = self.read_json(CHAR_STATUS_UUID)
            if predicate(status):
                return status
        raise LinkError("status never matched")

    def disconnect(self) -> None:
        with contextlib.suppress(OSError):
            self._send({"op": "disconnect"})

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


class PopClient:
    """The app's half of proof of possession over a Central (mirrors BLEProofOfPossession.Session)."""

    def __init__(self, central: Central, code: str, device_id: str) -> None:
        self.central = central
        self.code_key = pop.derive_code_key(pop.normalize_setup_code(code), device_id)
        self.device_id = device_id
        self.keys: Optional[pop.SessionKeys] = None

    def exchange(self, message: dict) -> dict:
        error = self.central.write(CHAR_AUTH_UUID, json.dumps(message).encode())
        if error:
            raise LinkError(error)
        return json.loads(self.central.next_notification(CHAR_AUTH_UUID))

    def hello(self) -> dict:
        self.na = os.urandom(16)
        return self.exchange({"op": "hello", "v": 1, "na": pop.b64e(self.na)})

    def prove(self) -> dict:
        """hello + proof; the device's reply to the proof (or its refusal of the hello)."""
        challenge = self.hello()
        if challenge.get("op") != "challenge":
            return challenge
        nd = base64.b64decode(challenge["nd"])
        self.keys = pop.session_keys(self.code_key, challenge["id"], self.na, nd)
        reply = self.exchange({"op": "proof", "tag": pop.b64e(pop.app_proof(self.keys))})
        if reply.get("op") == "proven":
            tag = base64.b64decode(reply["tag"])
            if tag != pop.device_proof(self.keys):
                raise LinkError("the device could not prove the code")
        return reply

    def send_credentials(self, ssid: str, psk: Optional[str]) -> dict:
        assert self.keys is not None
        iv = os.urandom(12)
        ct = pop.seal_credentials(self.keys, ssid, psk, iv)
        return self.exchange({"op": "creds", "iv": pop.b64e(iv), "ct": pop.b64e(ct)})
