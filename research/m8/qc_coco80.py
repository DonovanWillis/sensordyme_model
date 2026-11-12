"""quantcompute driver (Windows, DirectML): all-class predictions of the shipping DAMO-T float ONNX
on Open Images val and train, then the M8 analyses on them. No training; test split untouched.

    sdrun submit -t quantcompute ... -- python -u research/m8/qc_coco80.py

The pretrained COCO model never saw Open Images, so the train split is a fair extra evaluation
set FOR THIS MODEL ONLY (about 3x more phones than val). It is not fair for EXP-08, which
trained on it. Predictions stay on the worker in $SDJOB_DATA/m8cv; only reports come back.
"""

from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path

D = Path(os.environ["SDJOB_DATA"])
OUT = Path(os.environ["SDJOB_OUT"])
os.environ["SD_PYLIB"] = str(D / "pylib" / "pillow-11.3.0")
sys.path.insert(0, os.environ["SD_PYLIB"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OI = D / "openimages" / "oi7-sd-v1"
PREDS = D / "m8cv"
SPLITS = [s for s in os.environ.get("M8_SPLITS", "val,train").split(",") if s]
STEPS = set(sys.argv[1:]) or {"ops", "paired", "phone"}  # e.g. `qc_coco80.py phone` re-runs only that


def run(module: str, argv: list[str]) -> bool:
    sys.argv = [module, *argv]
    print(">>", module, " ".join(argv), flush=True)
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as e:
        if e.code:
            print("exit", e.code, flush=True)
            return False
    except Exception:  # noqa: BLE001 - keep going: later steps are independent
        traceback.print_exc()
        return False
    return True


for split in SPLITS:
    done = (PREDS / f"coco80_{split}.complete.json").exists()  # written only after both files are whole
    if not done or os.environ.get("M8_REDO"):
        run("research.m8.coco80_preds", ["--data", str(OI), "--models", str(D / "models"), "--out-preds", str(PREDS),
                                         "--report", str(OUT / f"coco80_{split}"), "--split", split, "--provider", "dml"])
    leg, ref = PREDS / f"coco80_legacy_{split}.jsonl", PREDS / f"coco80_reference_{split}.jsonl"
    if "ops" in STEPS:
        run("research.m8.analysis", ["ops", "--data", str(OI), "--split", split, "--out", str(OUT),
                                     "--preds", f"legacy={leg}", "--preds", f"reference={ref}"])
    if "paired" in STEPS:
        run("research.m8.analysis", ["paired", "--data", str(OI), "--split", split, "--out", str(OUT),
                                     "--a", f"reference={ref}", "--b", f"legacy={leg}", "--n", "300"])
    for name, path in (("reference", ref), ("legacy", leg)) if "phone" in STEPS else ():
        run("research.m8.analysis", ["phone", "--data", str(OI), "--split", split, "--out", str(OUT), "--tag", name,
                                     "--preds", f"{name}={path}"])
print("done", flush=True)
