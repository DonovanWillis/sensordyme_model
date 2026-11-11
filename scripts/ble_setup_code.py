#!/usr/bin/env python3
"""Factory tool for the Bluetooth setup code (services/ble_pop.py).

    scripts/ble_setup_code.py generate --out /etc/sensordyme/ble_setup_code --device-id SD-1A2B3C4D
    scripts/ble_setup_code.py label --code-file /etc/sensordyme/ble_setup_code --device-id SD-1A2B3C4D
    scripts/ble_setup_code.py vectors --out tests/ble_pop_vectors.json

`generate` writes a fresh code once (mode 0600, atomic) and prints the label
text and QR payload. It refuses to replace an existing code: the printed label
is the owner's proof, and silently changing it would orphan that label. There
is no rotation command; replacing a code is a deliberate factory/recovery step
(delete the file by hand, generate, print a new label).

`label` reprints the label for an existing code. Run both ON the device, as the
service user, with the output going to the label printer only: the code is a
secret and must not land in shell history, CI logs or tickets.

`vectors` regenerates the fixed test vectors the Python and Swift tests share.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from services import ble_pop as pop


def _label(device_id: str, code: str) -> str:
    return (f"SensorDyme {device_id}\n"
            f"Setup code: {pop.format_setup_code(code)}\n"
            f"QR: {pop.qr_payload(device_id, code)}")


#: Fixed inputs for the shared vectors. Test-only codes: never on a device.
_VECTOR_CASES = [
    {"name": "basic", "device_id": "SD-1A2B3C4D", "code": "ABCD-EFGH-JKMN",
     "na": "00112233445566778899aabbccddeeff", "nd": "ffeeddccbbaa99887766554433221100",
     "iv": "000102030405060708090a0b", "ssid": "HomeWiFi", "psk": "correct horse battery staple"},
    {"name": "unicode_open_network", "device_id": "SD-00000000", "code": "0123-4567-89VW",
     "na": "0f" * 16, "nd": "f0" * 16, "iv": "a1" * 12, "ssid": "Café ☕ 5G", "psk": None},
]


def vectors() -> dict:
    cases = []
    for case in _VECTOR_CASES:
        na, nd, iv = (bytes.fromhex(case[k]) for k in ("na", "nd", "iv"))
        code_key = pop.derive_code_key(pop.normalize_setup_code(case["code"]), case["device_id"])
        keys = pop.session_keys(code_key, case["device_id"], na, nd)
        cases.append({
            **case,
            "qr": pop.qr_payload(case["device_id"], case["code"]),
            "code_key": code_key.hex(),
            "k_mac": keys.mac.hex(),
            "k_enc": keys.enc.hex(),
            "transcript": keys.transcript.hex(),
            "app_proof": pop.app_proof(keys).hex(),
            "device_proof": pop.device_proof(keys).hex(),
            "creds_ct": pop.seal_credentials(keys, case["ssid"], case["psk"], iv).hex(),
        })
    return {"protocol": pop.PROTOCOL_VERSION, "pbkdf2_iterations": pop.PBKDF2_ITERATIONS, "cases": cases}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", required=True)
    gen.add_argument("--device-id", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--code-file", required=True)
    lab.add_argument("--device-id", required=True)
    vec = sub.add_parser("vectors")
    vec.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    try:
        if args.cmd == "generate":
            code = pop.generate_setup_code()
            pop.write_setup_code(args.out, code)
            print(_label(args.device_id, code))
        elif args.cmd == "label":
            print(_label(args.device_id, pop.load_setup_code(args.code_file)))
        else:
            Path(args.out).write_text(json.dumps(vectors(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.out}")
    except pop.SetupCodeError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
