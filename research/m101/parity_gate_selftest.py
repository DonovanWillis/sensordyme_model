"""M10.1: self-test of the INT8 parity gate with FLOAT tensors standing in for the chip.

RUNS ON A WORKER (oryn, CPU). Read-only. The Pi procedure is
`hwv_stage.py capture` then `hwv_stage.py parity --decoder both`; here the same two
stages run with the float ONNX behind the simulated chip (fake_hailo.onnx_backend).
With no quantisation in the path, every check must pass with zero score delta
and IoU 1.0 for BOTH decoders. A disagreement here would be a defect of the gate
(preprocessing, decoding, unmapping, the expectations), not of INT8 -- so on the Pi
any disagreement the gate reports is the chip's.

    python -m research.m101.parity_gate_selftest --heads-onnx $D/m101/zoo.heads.onnx --out $SDJOB_OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--heads-onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import fake_hailo
    import hwv_stage as hs

    from research.m83 import inference_path_val as ipv

    sess = ipv.build_session(args.heads_onnx, 4)
    bundle = args.out / "bundle_float"
    ns = argparse.Namespace(fake=True, no_service_check=True, bundle=bundle, out=args.out,
                            install_dir=None, decoder="both")
    cap = hs.Report("capture", fake=True)
    with hs.fake_device(backend=fake_hailo.onnx_backend(sess)) as install:
        ns.install_dir = install
        code = hs.stage_capture(cap, ns)
    print("capture:", code, [(c["check"], c["ok"]) for c in cap.checks])
    if code != hs.PASS:
        return 2
    par = hs.Report("parity", fake=False)  # compare against the REAL float expectations
    ns.fake = False
    code = hs.stage_parity(par, ns)
    summary = {"capture_checks": cap.checks, "parity_checks": par.checks,
               "decoders": {d: {k: v for k, v in s.items() if k != "per_image"}
                            for d, s in par.values.get("decoders", {}).items()},
               "per_image": {d: s["per_image"] for d, s in par.values.get("decoders", {}).items()}}
    (args.out / "parity_gate_selftest.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("decoders:", json.dumps(summary["decoders"]))
    return 0 if code == hs.PASS and all(c["ok"] for c in par.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
