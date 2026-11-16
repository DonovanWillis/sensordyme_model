#!/usr/bin/env python3
"""Factory tool for the direct-mode Wi-Fi passphrase (core/direct_mode.py, docs/DIRECT_MODE.md).

    scripts/direct_mode_psk.py generate --out /etc/sensordyme/direct_mode_psk [--device-id SD-1A2B3C4D]
    scripts/direct_mode_psk.py label --psk-file /etc/sensordyme/direct_mode_psk [--device-id SD-1A2B3C4D]

`generate` writes a fresh random passphrase once (mode 0600, atomic) and prints
the label lines: the access point's name, the passphrase and a Wi-Fi QR payload
the iPhone camera can join from. It refuses to replace an existing passphrase:
the printed label is how the owner joins, and silently changing it would orphan
that label. Replacing it is a deliberate factory/recovery step (delete the file
by hand, generate, print a new label).

`label` reprints the label for an existing passphrase.

The passphrase is random (Python `secrets`), never derived from the MAC, the
serial or the device ID, all of which are broadcast. Run both ON the device, as
the service user (the file must be owned by the account that reads it), with
the output going to the label printer only: the passphrase is a secret and must
not land in shell history, CI logs or tickets. --device-id defaults to this
board's own ID (core/identity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from core import direct_mode as dm
from core.identity import device_id as board_device_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", required=True)
    gen.add_argument("--device-id")
    lab = sub.add_parser("label")
    lab.add_argument("--psk-file", required=True)
    lab.add_argument("--device-id")
    args = parser.parse_args(argv)
    device = args.device_id or board_device_id()

    try:
        if args.cmd == "generate":
            psk = dm.generate_psk()
            dm.write_psk(args.out, psk)
        else:
            psk = dm.load_psk(args.psk_file)
        print(dm.label_text(device, psk))
    except dm.PskError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
