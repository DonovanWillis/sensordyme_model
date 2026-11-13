"""Split-leakage check for a training run on the oi7-sd-v1 manifest (val/test must not reach train).

    python -m research.m84.leakage_check --data <oi7-sd-v1> --coco <dir with train.json, val.json> --out leak.json

Checks, failing (exit 3) on any hit:
  * manifest: an image id in more than one split; a photographer group (``subject``) in more
    than one split (the split is author-grouped, so a shared group would leak near-duplicates);
  * pixels: identical file sha256 across splits (exact re-uploads under different ids);
  * the COCO export the trainer reads: a train.json image that the manifest places in val or
    test, or that also appears in val.json.
It reads only the manifest, the export's JSON and file hashes; no labels from test are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import manifest as mf


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(data: Path, coco: Path | None) -> dict:
    manifest = mf.load(data / "manifest.json")
    split_of: dict[str, set[str]] = defaultdict(set)
    group_of: dict[str, set[str]] = defaultdict(set)
    by_sha: dict[str, set[str]] = defaultdict(set)
    for item in manifest.items:
        split_of[item.id].add(str(item.split))
        group_of[str(item.subject)].add(str(item.split))
        by_sha[item.sha256 or _sha(data / item.path)].add(str(item.split))
    rep: dict = {"manifest_id": manifest.id, "items": len(manifest.items),
                 "split_counts": {s: len(manifest.split(s)) for s in ("train", "val", "test")}}
    rep["ids_in_multiple_splits"] = sorted(i for i, s in split_of.items() if len(s) > 1)
    rep["groups_in_multiple_splits"] = sorted(g for g, s in group_of.items() if len(s) > 1)
    rep["sha_in_multiple_splits"] = sorted(h for h, s in by_sha.items() if len(s) > 1)
    if coco:
        train = json.loads((coco / "train.json").read_text())["images"]
        val = json.loads((coco / "val.json").read_text())["images"]
        tnames = {i.get("oi_id") or Path(i["file_name"]).stem for i in train}
        vnames = {i.get("oi_id") or Path(i["file_name"]).stem for i in val}
        rep["coco_train_images"], rep["coco_val_images"] = len(tnames), len(vnames)
        rep["coco_train_not_train"] = sorted(n for n in tnames if split_of.get(n) != {"train"})
        rep["coco_train_and_val"] = sorted(tnames & vnames)
    hits = [k for k in ("ids_in_multiple_splits", "groups_in_multiple_splits", "sha_in_multiple_splits",
                        "coco_train_not_train", "coco_train_and_val") if rep.get(k)]
    rep["leak"] = bool(hits)
    rep["failed_checks"] = hits
    return rep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--coco")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = check(Path(a.data), Path(a.coco) if a.coco else None)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}))
    return 3 if rep["leak"] else 0


if __name__ == "__main__":
    sys.exit(main())
