"""M10.1: float expectations for the known-good images under BOTH decoders, and a
repeatability check of the committed legacy expectations.

RUNS ON A WORKER (oryn, CPU), never on the Pi. Read-only: no training, no new
data (the 6 known-good images are files in this checkout), model files verified
by sha256.

1. Rebuilds the legacy expectations exactly as research/m83/inference_path_val.py
   made scripts/hw_validation/known_good/expected_float.json (same float ONNX,
   same simulated-Hailo path through core.accel.HailoYOLO, conf 0.05) and
   compares them with the committed file: input tensor sha256, per-head
   statistics and detections. Equal = the procedure is exact and repeatable.
2. The same images through the opt-in `reference` decoder, written in the same
   format with "decoder": "reference" -- the expectations hwv_stage.py parity
   --decoder reference compares the Pi's INT8 tensors against.

    python -m research.m101.known_good_decoders --onnx $D/models/damoyolo_tinynasL20_T.onnx \\
        --heads-onnx $D/m101/zoo.heads.onnx --out $SDJOB_OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

from research.m83 import inference_path_val as ipv  # noqa: E402

COMMITTED = ROOT / "scripts" / "hw_validation" / "known_good" / "expected_float.json"


def compare_expectations(made: dict, committed: dict, *, score_tol: float = 1e-4, box_tol: float = 0.05,
                         stat_tol: float = 1e-4) -> dict:
    """Image by image: identical input tensor, head statistics within `stat_tol`,
    the same detections (class, score within `score_tol`, box within `box_tol` px)."""
    by_file = {im["file"]: im for im in committed["images"]}
    report: dict = {"images": {}, "equal": True}
    for im in made["images"]:
        ref = by_file.get(im["file"])
        if ref is None:
            report["images"][im["file"]] = {"missing_in_committed": True}
            report["equal"] = False
            continue
        tensor = im["input_tensor_sha256"] == ref["input_tensor_sha256"]
        stats_worst = 0.0
        for name, entry in im["heads"].items():
            other = ref["heads"].get(name, {})
            for key in ("min", "max", "mean", "std"):
                if key in entry and key in other:
                    stats_worst = max(stats_worst, abs(entry[key] - other[key]))
        a, b = im["detections"], ref["detections"]
        same_count = len(a) == len(b)
        worst_score = worst_box = 0.0
        same_classes = same_count
        if same_count:
            for da, db in zip(sorted(a, key=lambda d: (-d["score"], d["cls"])),
                              sorted(b, key=lambda d: (-d["score"], d["cls"])), strict=True):
                same_classes = same_classes and da["cls"] == db["cls"]
                worst_score = max(worst_score, abs(da["score"] - db["score"]))
                worst_box = max(worst_box, *(abs(x - y) for x, y in zip(da["xyxy"], db["xyxy"], strict=True)))
        equal = (tensor and stats_worst <= stat_tol and same_count and same_classes
                 and worst_score <= score_tol and worst_box <= box_tol)
        report["images"][im["file"]] = {
            "input_tensor_identical": tensor, "head_stat_max_abs_diff": stats_worst,
            "detections": [len(a), len(b)], "classes_identical": same_classes,
            "score_max_abs_diff": worst_score, "box_max_abs_diff_px": worst_box, "equal": equal}
        report["equal"] = report["equal"] and equal
    report["equal"] = report["equal"] and len(made["images"]) == len(committed["images"])
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--heads-onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import fake_hailo
    import hwv_decode as hd
    import onnxruntime as ort

    from core import accel
    from core.accel import HailoYOLO

    got = ipv.sha256_file(args.onnx)
    if got != ipv.ONNX_SHA256:
        raise SystemExit(f"{args.onnx}: sha256 {got} != pinned {ipv.ONNX_SHA256}")
    sess = ipv.build_session(args.heads_onnx, args.threads)
    provenance = {"job": os.environ.get("SDJOB_ID"), "onnx_sha256": got,
                  "heads_onnx_sha256": ipv.sha256_file(args.heads_onnx), "onnxruntime": ort.__version__,
                  "core_accel_sha256": ipv.sha256_file(ROOT / "core" / "accel.py"),
                  "hwv_decode_sha256": ipv.sha256_file(ROOT / "scripts" / "hw_validation" / "hwv_decode.py")}
    captured: dict = {}
    base = fake_hailo.onnx_backend(sess)

    def backend(batch):
        heads = base(batch)
        captured["batch"], captured["heads"] = batch, heads
        return heads

    fake_hailo.install(backend)
    accel.record_inference = lambda: None
    made = {}
    # One model: the (fake) chip is exclusive, as the real one is. Its decoder is
    # switched between runs, as research/m83/inference_path_val.py does.
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (simulated: float ONNX)", decoder="legacy")
    for decoder in ("legacy", "reference"):
        model.decoder = decoder
        sub = args.out / decoder
        sub.mkdir(exist_ok=True)
        result = ipv.known_good(model, fake_hailo, hd, sub, captured)
        result["decoder"] = decoder
        result["provenance"] = provenance
        (args.out / f"expected_float_{decoder}.json").write_text(json.dumps(result, indent=1) + "\n")
        made[decoder] = result
    committed = json.loads(COMMITTED.read_text())
    repeat = compare_expectations(made["legacy"], committed)
    repeat["provenance"] = provenance
    (args.out / "repeatability.json").write_text(json.dumps(repeat, indent=1) + "\n")
    print(f"legacy expectations reproduce the committed file: {repeat['equal']}")
    for f, r in repeat["images"].items():
        print(f"  {f}: {r}")
    return 0 if repeat["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
