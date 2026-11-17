#!/usr/bin/env python3
"""List or revoke the phones this device trusts locally (sd-local-v1, docs/LOCAL_ACCESS.md §9).

The physical recovery path: a lost phone and no other phone to revoke it from,
or a device changing hands. Run ON the device, as the service user, from the
install directory. The running service picks the change up at its next local
request (the store reloads when the file changes); no restart is needed:

    .venv/bin/python scripts/local_clients.py list
    .venv/bin/python scripts/local_clients.py revoke <client_id>
    .venv/bin/python scripts/local_clients.py revoke-all --yes

Prints ids, names and fingerprints only; never a key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import local_auth as la


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default="data/local_clients.json", help="default: %(default)s")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    one = sub.add_parser("revoke")
    one.add_argument("client_id")
    every = sub.add_parser("revoke-all")
    every.add_argument("--yes", action="store_true", help="required: every phone must enrol again")
    args = parser.parse_args(argv)

    store = la.ClientStore(args.store)
    if store.fault:
        print(f"refused: {store.fault} ({args.store})", file=sys.stderr)
        return 2
    if args.cmd == "list":
        if not store.clients:
            print("no phones are trusted locally")
        for c in sorted(store.clients.values(), key=lambda c: c.enrolled_at):
            v = c.view()
            print(f"{v['client_id']}  {v['name']!r}  via {v['source']}  key {v['key_fp']}")
        return 0
    if args.cmd == "revoke":
        if store.clients.pop(args.client_id, None) is None:
            print(f"{args.client_id} is not trusted on this device", file=sys.stderr)
            return 1
    else:
        if not args.yes:
            print("refusing without --yes: every phone would have to pair again", file=sys.stderr)
            return 2
        store.clients.clear()
    if not store.save():
        print(f"could not write {args.store}", file=sys.stderr)
        return 1
    print("done: the running service applies it at its next local request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
