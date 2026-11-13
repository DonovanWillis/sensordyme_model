"""Apply EXP-09's pre-registered criteria (research/registry/experiments.jsonl) to its fetched artifacts.

    python -m research.m84.exp09_verdict <job artifacts dir>

Reads exp09/score_exp09_nms05_val.json, exp09/m8_paired_val_pre_vs_exp09.json,
exp09/m8_paired_val_exp09_vs_exp08.json, exp09/m8_ops_val_exp09.json and exp09/m8_phone_val_exp09_own_hands.json.
Prints one line per criterion (gating and secondary) and the decision. Never edits the registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: list[str]) -> int:
    a = Path(argv[0]) / "exp09"
    score = _load(a / "score_exp09_nms05_val.json")
    pre = _load(a / "m8_paired_val_pre_vs_exp09.json")
    e8 = _load(a / "m8_paired_val_exp09_vs_exp08.json")
    ops = _load(a / "m8_ops_val_exp09.json")
    if not (score and pre):
        print("missing artifacts")
        return 2
    cls = score["classes"]
    r = pre["classes"]["person"]["r_at_p0.80"]
    rows = [
        ("G1 person R@P0.80: pretrained - EXP-09 <= 0.010", r["delta"] <= 0.010,
         (f"pretrained {r['a']:.3f}, EXP-09 {r['b']:.3f}, delta {r['delta']:+.3f} "
          f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p(delta<=0) {r['p_delta_le_0']:.3f}")),
        ("G2 phone AP50 >= 0.79", cls["phone"]["ap50"] >= 0.79, f"{cls['phone']['ap50']:.3f} {cls['phone']['bootstrap']['ap50_ci95']}"),
        ("G3 hand AP50 >= 0.70", cls["hand"]["ap50"] >= 0.70, f"{cls['hand']['ap50']:.3f} {cls['hand']['bootstrap']['ap50_ci95']}"),
    ]
    p70 = pre["classes"]["person"]["p_at_r0.70"]
    ap = pre["classes"]["person"]["ap50"]
    sec = [("S1 person P@R0.70 >= 0.66", p70["b"] is not None and p70["b"] >= 0.66, f"{p70['b']}"),
           ("S2 person AP50 >= 0.703", ap["b"] >= 0.703, f"{ap['b']:.3f}")]
    if ops:
        ph = ops["models"]["exp09_nms05"]["classes"]["phone"]["recall_at_precision"]["0.8"]
        sec.append(("S3 phone R@P0.8 >= 0.75", (ph.get("recall") or 0) >= 0.75, f"{ph.get('recall')}"))
        g = {m: next((x for x in ops["models"][m]["classes"]["phone"]["grid"] if abs(x["threshold"] - 0.4) < 1e-9), {})
             for m in ("pretrained_reference", "exp09_nms05")}
        fa, fb = (g[m].get("negative_images_with_a_false_box") for m in ("pretrained_reference", "exp09_nms05"))
        sec.append(("S4 false-phone images @0.4 <= pretrained", fb is not None and fa is not None and fb <= fa, f"{fb} vs {fa}"))
    for name, ok, detail in rows + sec:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if e8:
        for c in ("person", "phone", "hand"):
            x = e8["classes"].get(c, {})
            for k in ("ap50", "r_at_p0.80", "p_at_r0.70"):
                if k in x and x[k]["delta"] is not None:
                    print(f"EXP-09 - EXP-08 {c} {k}: {x[k]['a']:.3f} vs {x[k]['b']:.3f}, delta {x[k]['delta']:+.3f} "
                          f"[{x[k]['ci95'][0]:+.3f}, {x[k]['ci95'][1]:+.3f}] p(delta<=0) {x[k]['p_delta_le_0']:.3f}")
    print("DECISION:", "retain (all gating criteria met)" if all(ok for _, ok, _ in rows) else "reject (a gating criterion failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
