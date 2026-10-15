#!/usr/bin/env python3
"""Provision a SensorDyme over Bluetooth from this machine — what the app does.

Exists so the Bluetooth path can be *tested* rather than reasoned about. The
iOS app is the real client, but it cannot be driven from a terminal, and
"the code looks right" has already been wrong twice on this stack: the GATT
advertisement was silently failing for weeks, and a Wi-Fi scan was blocking the
whole event loop. Both were only visible by talking to the device.

    scripts/ble_provision_client.py scan
    scripts/ble_provision_client.py info
    scripts/ble_provision_client.py networks
    scripts/ble_provision_client.py provision --ssid HomeWiFi --password '...'

The passphrase is never printed or logged — only whether one was supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "5d1e0001-5344-4d45-b000-536e447a6d65"
CHAR_DEVICE_INFO = "5d1e0002-5344-4d45-b000-536e447a6d65"
CHAR_WIFI_CREDS = "5d1e0003-5344-4d45-b000-536e447a6d65"
CHAR_STATUS = "5d1e0004-5344-4d45-b000-536e447a6d65"
CHAR_COMMAND = "5d1e0005-5344-4d45-b000-536e447a6d65"
CHAR_NETWORKS = "5d1e0006-5344-4d45-b000-536e447a6d65"


def pick_outcome(updates: list[dict], baseline_attempt: Optional[int] = None) -> Optional[dict]:
    """The ok/error answer to THIS write among `updates`, or None while there is none.

    The device keeps its last outcome in every status it sends, so a status that
    arrives after the write can still describe the PREVIOUS attempt. Taking the
    newest ok/error in the list (which this did) reported a retry's result as
    whatever the last attempt said. An outcome counts only when it is numbered
    after `baseline_attempt` (the attempt the device reported before this write),
    or -- for firmware that does not number attempts -- once this write's own
    "connecting" has been seen.
    """
    seen_connecting = False
    for update in updates:
        provision = update.get("provision") or {}
        state = provision.get("state")
        attempt = provision.get("attempt")
        numbered = isinstance(attempt, int)
        if numbered and baseline_attempt is not None and attempt <= baseline_attempt:
            continue  # an earlier attempt's status
        if state == "connecting":
            seen_connecting = True
            continue
        if state in ("ok", "error") and (seen_connecting or numbered):
            return provision
    return None


async def find_device(timeout: float = 10.0):
    """Find one SensorDyme by its service UUID, not by name.

    Names are not identity — the same device can present an advertised name and
    a cached adapter name that differ, which is what produced duplicate entries
    in the app's list.
    """
    found = {}
    first_seen = {}

    def seen(device, advertisement):
        uuids = [u.lower() for u in (advertisement.service_uuids or [])]
        if SERVICE_UUID.lower() in uuids:
            if device.address not in found:
                first_seen[device.address] = time.monotonic()
            found[device.address] = (device, advertisement)

    scanner = BleakScanner(detection_callback=seen)
    started = time.monotonic()
    await scanner.start()
    deadline = started + timeout
    while time.monotonic() < deadline and not found:
        await asyncio.sleep(0.05)
    await scanner.stop()
    for address, stamp in first_seen.items():
        print(f"  (first advertisement from {address[:8]}… after {stamp - started:.2f}s)")
    return list(found.values())


async def cmd_scan(args) -> int:
    started = time.monotonic()
    devices = await find_device(args.timeout)
    elapsed = time.monotonic() - started
    if not devices:
        print(f"no SensorDyme advertising {SERVICE_UUID} within {args.timeout:.0f}s")
        return 1
    print(f"found {len(devices)} device(s) in {elapsed:.1f}s")
    for device, advertisement in devices:
        print(f"  {advertisement.local_name or device.name or '(no name)'}"
              f"  addr={device.address}  rssi={advertisement.rssi}dBm")
    return 0


async def _read_json(client, uuid, label):
    raw = await client.read_gatt_char(uuid)
    try:
        return json.loads(raw.decode())
    except Exception:  # noqa: BLE001 - any undecodable payload is reported, never fatal
        print(f"  {label}: unparseable: {raw!r}")
        return None


async def cmd_info(args) -> int:
    devices = await find_device(args.timeout)
    if not devices:
        print("no device found")
        return 1
    device, _ = devices[0]
    async with BleakClient(device) as client:
        print(f"connected to {device.address}")
        info = await _read_json(client, CHAR_DEVICE_INFO, "info")
        print("  device info:", json.dumps(info))
        status = await _read_json(client, CHAR_STATUS, "status")
        print("  status     :", json.dumps(status))
    return 0


async def cmd_networks(args) -> int:
    devices = await find_device(args.timeout)
    if not devices:
        print("no device found")
        return 1
    device, _ = devices[0]
    scan_done = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_status(_handle, data: bytearray):
        try:
            status = json.loads(data.decode())
        except Exception:  # noqa: BLE001 - an undecodable push is skipped, never fatal
            return
        print(f"    status push: scanning={status.get('scanning')}")
        if status.get("scanning") is False:
            loop.call_soon_threadsafe(scan_done.set)

    async with BleakClient(device) as client:
        await client.start_notify(CHAR_STATUS, on_status)

        # First read answers from cache and kicks a fresh scan. It must return
        # instantly: this runs on the device's BLE event loop, and it used to
        # scan inline, freezing every other Bluetooth operation for ~20s.
        started = time.monotonic()
        networks = await _read_json(client, CHAR_NETWORKS, "networks")
        print(f"  read 1 (cached): {time.monotonic() - started:.2f}s, "
              f"{len(networks or [])} network(s)")

        # The device reports when its scan finishes; the app re-reads then. This
        # is why a stale first answer is acceptable rather than a bug.
        try:
            await asyncio.wait_for(scan_done.wait(), timeout=40)
            print("  device reported scan complete")
        except TimeoutError:
            print("  no scan-complete push within 40s")

        networks = await _read_json(client, CHAR_NETWORKS, "networks")
        print(f"  read 2 (fresh) : {len(networks or [])} network(s)")
        for n in (networks or [])[:12]:
            print(f"      {n.get('ssid'):28} {n.get('signal'):>3}  "
                  f"{'secured' if n.get('secured') else 'open'}")
    return 0


async def cmd_provision(args) -> int:
    devices = await find_device(args.timeout)
    if not devices:
        print("no device found")
        return 1
    device, _ = devices[0]

    updates: list[dict] = []
    # Assigned before any callback can run: a status pushed between
    # start_notify() and the write used to hit an unassigned `t0`.
    t0 = time.monotonic()

    def record(status, how: str) -> None:
        if not isinstance(status, dict):
            return
        updates.append(status)
        stamp = time.monotonic() - t0
        provision = status.get("provision") or {}
        print(f"  [{stamp:5.1f}s] {how:6} connectivity={status.get('state')!r:15} "
              f"provision={provision.get('state')!r} attempt={provision.get('attempt')!r} "
              f"{provision.get('error') or ''}")

    def on_status(_handle, data: bytearray):
        try:
            status = json.loads(data.decode())
        except Exception:  # noqa: BLE001 - an undecodable push is skipped, never fatal
            return  # e.g. a notification cut at the MTU; the reads below cover it
        record(status, "notify")

    async with BleakClient(device) as client:
        print(f"connected to {device.address}")
        before = await _read_json(client, CHAR_STATUS, "status")
        baseline = ((before or {}).get("provision") or {}).get("attempt") if isinstance(before, dict) else None
        await client.start_notify(CHAR_STATUS, on_status)

        # Never log the passphrase. Whether one was supplied is enough to
        # diagnose an open-vs-secured mistake; the value itself is not.
        print(f"sending SSID={args.ssid!r} password={'RECEIVED' if args.password else 'NONE'}")
        payload = json.dumps({"ssid": args.ssid, "psk": args.password}).encode()

        t0 = time.monotonic()
        await client.write_gatt_char(CHAR_WIFI_CREDS, payload, response=True)
        print(f"  write acknowledged in {time.monotonic() - t0:.2f}s")

        # Watch the OUTCOME of the write, not the device's connectivity. A
        # device that refuses a network stays on its old one and keeps reporting
        # "online" — reading that as the result declares success for a setup
        # that failed, which is exactly what this client did at first.
        # Read as well as listen, like the app: a notification longer than the
        # MTU arrives truncated and unparseable, a read does not.
        deadline = time.monotonic() + args.wait
        next_read = time.monotonic()
        while time.monotonic() < deadline and pick_outcome(updates, baseline) is None:
            if time.monotonic() >= next_read:
                record(await _read_json(client, CHAR_STATUS, "status"), "read")
                next_read = time.monotonic() + 2.0
            await asyncio.sleep(0.25)

    final = pick_outcome(updates, baseline)
    if final is None:
        print(f"NO OUTCOME within {args.wait:.0f}s — this is the 'stuck on Sending' failure")
        return 2
    if final.get("state") == "ok":
        print(f"OK: device joined {args.ssid} at {final.get('ip')}")
        return 0
    print(f"FAILED: {final.get('error')}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.0, help="discovery timeout")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan")
    sub.add_parser("info")
    sub.add_parser("networks")

    provision = sub.add_parser("provision")
    provision.add_argument("--ssid", required=True)
    provision.add_argument("--password", default="")
    provision.add_argument("--wait", type=float, default=90.0,
                           help="how long to wait for a terminal state")

    args = parser.parse_args()
    handler = {
        "scan": cmd_scan, "info": cmd_info,
        "networks": cmd_networks, "provision": cmd_provision,
    }[args.command]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
