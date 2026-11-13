"""quantcompute driver (Windows, CPU): B2 look-alike error anatomy on the pretrained model's all-80-class
val predictions (written by research/m8/coco80_preds.py in M8) and train-only hard-negative mining.

    sdrun submit -t quantcompute ... -- python -u research/m84/qc_b2.py

Val for characterisation; TRAIN for mining (the pretrained COCO model never saw Open Images, so its
train-split errors are a fair source of hard negatives for a future fine-tune). Test never.
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
P = D / "m8cv"
W = D / "m84"
W.mkdir(parents=True, exist_ok=True)


def run(module: str, argv: list[str]) -> bool:
    sys.argv = [module, *argv]
    print(">>", module, " ".join(argv), flush=True)
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as e:
        if e.code:
            print("exit", e.code, flush=True)
            return False
    except Exception:  # noqa: BLE001 - later steps are independent
        traceback.print_exc()
        return False
    return True


st = W / "stats_val.csv"
if not st.exists():
    run("research.m84.characterise", ["imgstats", "--data", str(OI), "--split", "val", "--out", str(st)])
run("research.m84.characterise", ["errors", "--data", str(OI), "--split", "val", "--stats", str(st),
                                  "--preds", f"coco80_reference={P / 'coco80_reference_val.jsonl'}",
                                  "--preds", f"coco80_legacy={P / 'coco80_legacy_val.jsonl'}",
                                  "--out", str(OUT / "b2_errors_coco80_val.json")])
run("research.m84.characterise", ["mine", "--data", str(OI), "--split", "train",
                                  "--preds", f"coco80_reference={P / 'coco80_reference_train.jsonl'}",
                                  "--out", str(OUT / "hardneg_phone_train.csv")])
print("done", flush=True)
