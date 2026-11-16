"""M13: CPU decode latency of the two product decoders on IDENTICAL real-shaped head tensors (THIS host).

No Pi, no INT8. The tensors are float32 heads of the shipping damoyolo_tinynasL20_T
(the float ONNX the .hef was compiled from) on the 6 known-good images and 10 Open
Images val images, written by research/m13/decoder_compare.py (fleet job
20261116-023006-932455). What is timed is the device's own decode,
core/accel.py HailoYOLO._decode_raw, with decoder "legacy" and "reference",
called exactly as predict() calls it: the driver_attention classes at conf 0.20,
the letterbox geometry of each frame. HailoYOLO is constructed behind
scripts/hw_validation/fake_hailo.py (no device is opened; nothing is inferred).

Order is alternated per frame (ABAB / BABA) and repeated, so drift on a shared
machine lands on both arms. Reported: per-frame median and p90 per decoder over
all repeats, and the detection differences on the same tensors (counts per class
group, IoU-matched boxes).

    python benchmarks/m13_decoder_latency.py --tensors ~/sensordyme-fleet/runs/20261116-023006-932455/artifacts \\
        --repeats 200 --out decoder_latency.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root(argv: list[str]) -> Path:
    for i, a in enumerate(argv):
        if a == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
    return HERE.parent


REPO = _repo_root(sys.argv)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import numpy as np  # noqa: E402

RUN_CONF = 0.20
PERSON, PHONE = 0, 67


def run_classes() -> list[int]:
    import yaml

    from core.accel import COCO_NAMES

    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    names = cfg["modules"]["driver_attention"]["yolo_classes"]
    return sorted({cid for cid, name in COCO_NAMES.items() if name in names} | {PERSON})


def load_sets(artifacts: Path) -> list[dict]:
    report = json.loads((artifacts / "report.json").read_text())
    sets = []
    for entry in report["tensors"]["files"]:
        with np.load(artifacts / "tensors" / f"{entry['name']}.npz") as z:
            heads = {k.replace("__", "/"): np.ascontiguousarray(z[k]) for k in z.files}
        sets.append({"name": entry["name"], "geometry": entry["geometry"], "heads": heads})
    return sets


def q(values: list[float], p: float) -> float:
    s = sorted(values)
    pos = p * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=str(HERE.parent))
    p.add_argument("--tensors", type=Path, required=True, help="the job's artifacts dir (report.json + tensors/)")
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    import fake_hailo
    import hwv_decode as hd

    from core import accel, stage_timing

    fake_hailo.install()
    accel.record_inference = lambda: None
    model = accel.HailoYOLO("damoyolo_tinynasL20_T.hef (SIMULATED: decode timing only)", decoder="legacy")
    sets = load_sets(args.tensors)
    classes = run_classes()

    def decode(s, decoder):
        model.decoder = decoder
        g = s["geometry"]
        t0 = time.perf_counter()
        (res,) = model._decode_raw(s["heads"], 1.0, g["left"], g["top"], g["w"], g["h"], classes, RUN_CONF)
        return (time.perf_counter() - t0) * 1000.0, res

    for s in sets:  # warm-up
        decode(s, "legacy")
        decode(s, "reference")
    times: dict[str, list[float]] = {"legacy": [], "reference": []}
    per_set: dict[str, dict[str, list[float]]] = {s["name"]: {"legacy": [], "reference": []} for s in sets}
    for r in range(args.repeats):
        for i, s in enumerate(sets):
            order = ("legacy", "reference") if (r + i) % 2 == 0 else ("reference", "legacy")
            for d in order:
                ms, _ = decode(s, d)
                times[d].append(ms)
                per_set[s["name"]][d].append(ms)

    def to_dets(res):
        boxes = list(res.boxes)
        if not boxes:
            return hd.Detections.empty()
        return hd.Detections(np.array([[float(v) for v in b.xyxy[0]] for b in boxes]),
                             np.array([float(b.conf[0]) for b in boxes]),
                             np.array([int(b.cls[0]) for b in boxes], dtype=np.int64),
                             np.zeros(len(boxes), dtype=np.int64))

    groups = {"person@0.50": ({PERSON}, 0.50), "phone@0.20": ({PHONE}, 0.20),
              "other product classes@0.20": (set(classes) - {PERSON, PHONE}, 0.20)}
    agree = {g: hd.Agreement(0.5) for g in groups}
    per_frame_counts = []
    for s in sets:
        _, a = decode(s, "legacy")
        _, b = decode(s, "reference")
        da, db = to_dets(a), to_dets(b)
        row = {"name": s["name"]}
        for g, (cls, conf) in groups.items():
            ka = [i for i in range(len(da)) if int(da.classes[i]) in cls and da.scores[i] >= conf]
            kb = [i for i in range(len(db)) if int(db.classes[i]) in cls and db.scores[i] >= conf]
            ga = da.take(ka) if ka else hd.Detections.empty()
            gb = db.take(kb) if kb else hd.Detections.empty()
            agree[g].add(ga, gb)
            row[g] = [len(ga), len(gb)]
        per_frame_counts.append(row)

    def stats(v):
        return {"n": len(v), "p50": round(statistics.median(v), 4), "p90": round(q(v, 0.9), 4),
                "mean": round(statistics.fmean(v), 4), "max": round(max(v), 4)}

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    result = {
        "schema": "sd-m13-decoder-latency/1",
        "label": "float tensors, desktop CPU (this host): NOT INT8, NOT a Raspberry Pi",
        "host_class": stage_timing.platform_tag(simulated=True)["class"],
        "conditions": {"platform": platform.platform(), "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
                       "python": platform.python_version(), "numpy": np.__version__,
                       "loadavg": list(__import__("os").getloadavg()),
                       "git_commit": sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]), "repo": str(REPO),
                       "tensor_source": str(args.tensors), "frames": len(sets), "repeats": args.repeats,
                       "run_conf": RUN_CONF, "run_classes": classes},
        "decode_ms": {d: stats(v) for d, v in times.items()},
        "decode_ms_by_frame_p50": {n: {d: round(statistics.median(v), 4) for d, v in arms.items()}
                                   for n, arms in per_set.items()},
        "reference_minus_legacy_p50_ms": round(statistics.median(times["reference"])
                                               - statistics.median(times["legacy"]), 4),
        "agreement_legacy_vs_reference": {g: a.summary() for g, a in agree.items()},
        "counts_per_frame_legacy_reference": per_frame_counts,
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({k: result[k] for k in ("label", "decode_ms", "reference_minus_legacy_p50_ms")}, indent=1))
    for g, a in result["agreement_legacy_vs_reference"].items():
        print(g, {k: a[k] for k in ("n_a", "n_b", "matched")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
