"""quantcompute driver (Windows, CPU EP): research.m84.hostquant, then scoring of every variant on VAL.

    sdrun submit -t quantcompute ... -- python -u research/m84/qc_hostquant.py [--limit N]

HOST-SIMULATED quantisation only; not a Hailo result. Predictions stay in $SDJOB_DATA/m84_hostquant.
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
WORK = D / "m84_hostquant"
EXTRA = sys.argv[1:]


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


ok = run("research.m84.hostquant", ["--data", str(OI), "--models", str(D / "models"), "--work", str(WORK),
                                    "--out", str(OUT), *EXTRA])
if ok and not EXTRA:
    from research.m84.hostquant import EVAL

    for v in EVAL:
        for d in ("reference", "legacy"):
            run("research.detector_bench.bench", ["score", "--data", str(OI), "--models", str(D / "models"),
                                                  "--preds", str(WORK / f"preds_{v}_{d}_val.jsonl"), "--split", "val",
                                                  "--name", f"hq_{v}_{d}", "--out", str(OUT / "scores")])
    for v in EVAL[1:]:
        for d in ("reference", "legacy"):
            run("research.m8.analysis", ["paired", "--data", str(OI), "--split", "val", "--out", str(OUT / "paired"),
                                         "--tag", f"float_vs_{v}_{d}",
                                         "--a", f"float_{d}={WORK / f'preds_float_{d}_val.jsonl'}",
                                         "--b", f"{v}_{d}={WORK / f'preds_{v}_{d}_val.jsonl'}", "--n", "300"])
    run("research.m8.analysis", ["ops", "--data", str(OI), "--split", "val", "--out", str(OUT / "ops"), "--tag", "hq",
                                 *[x for v in ("float", "pc_minmax_o8") for d in ("reference", "legacy")
                                   for x in ("--preds", f"{v}_{d}={WORK / f'preds_{v}_{d}_val.jsonl'}")]])
print("done", flush=True)
