"""Figures and tables for docs/research/M84_*.md, from small fetched JSON artifacts (no data, no model).

    python -m research.m84.figures --slices b2_slices_val.json [--hostquant-scores DIR] \
        [--epochs exp08=sd_epochs.jsonl --epochs exp09=sd_epochs.jsonl] --out docs/research/m84_figures

Every PNG has a CSV with the plotted numbers next to it. Colours: the dataviz reference
categorical palette, slots in fixed order per model (never by rank). Image-level Open Images val;
NOT in-cabin or distracted-driving accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ORDER = ["pretrained_reference", "pretrained_legacy", "exp08_nms05", "exp09_nms05"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _color(model: str, models: list[str]) -> str:
    known = [m for m in ORDER if m in models] + [m for m in models if m not in ORDER]
    return SLOTS[known.index(model) % len(SLOTS)]


def pr_and_reliability(slices: dict, out: Path) -> None:
    models = [m for m in ORDER if m in slices["models"]] + [m for m in slices["models"] if m not in ORDER]
    classes = ["person", "phone", "hand"]
    for kind in ("pr", "reliability"):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
        rows = []
        for ax, c in zip(axes, classes, strict=True):
            for m in models:
                cd = slices["models"][m]["classes"].get(c)
                if not cd:
                    continue
                if kind == "pr":
                    r, p = cd["pr"]["recall"], cd["pr"]["precision"]
                    ax.plot(r, p, color=_color(m, models), linewidth=2, label=m)
                    rows += [[m, c, x, y, s] for x, y, s in zip(r, p, cd["pr"]["score"], strict=True)]
                else:
                    b = cd["reliability"]["bins"]
                    xs = [x["mean_score"] for x in b]
                    ys = [x["precision"] for x in b]
                    ax.plot(xs, ys, color=_color(m, models), linewidth=2, marker="o", markersize=4,
                            label=f"{m} (ECE {cd['reliability']['ece']:.3f})")
                    rows += [[m, c, x["lo"], x["hi"], x["n"], x["mean_score"], x["precision"]] for x in b]
            if kind == "pr":
                ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle=":")
                _style(ax, f"{c}: precision vs recall", "recall", "precision")
            else:
                ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
                _style(ax, f"{c}: reliability", "mean score in bin", "fraction correct (IoU 0.5)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=6, frameon=False, loc="lower left")
        fig.suptitle("Open Images val (oi7-sd-v1), image level - not in-cabin accuracy", fontsize=8, color=MUTED, x=0.01,
                     ha="left")
        fig.tight_layout()
        fig.savefig(out / f"b2_{kind}.png")
        plt.close(fig)
        with (out / f"b2_{kind}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "class", "recall", "precision", "score"] if kind == "pr" else
                       ["model", "class", "bin_lo", "bin_hi", "n", "mean_score", "precision"])
            w.writerows(rows)


def luma(slices: dict, out: Path) -> None:
    models = [m for m in ORDER if m in slices["models"]]
    qs = ["luma_lt40", "luma_lt60", "Q1_dark", "Q2", "Q3", "Q4_bright"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
    rows = []
    for ax, c in zip(axes, ["person", "phone", "hand"], strict=True):
        for k, m in enumerate(models):
            cd = slices["models"][m]["classes"].get(c)
            if not cd:
                continue
            xs, ys, lo, hi = [], [], [], []
            for i, q in enumerate(qs):
                d = cd["ap50_by_luma_quartile"].get(q)
                if not d or d["ap50"] is None:
                    continue
                xs.append(i + (k - 1.5) * 0.12)
                ys.append(d["ap50"])
                ci = d["ci95"] or [d["ap50"], d["ap50"]]
                lo.append(d["ap50"] - ci[0])
                hi.append(ci[1] - d["ap50"])
                rows.append([m, c, q, d["npos"], d["ap50"], *ci])
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", markersize=4, color=_color(m, models), elinewidth=1,
                        capsize=0, label=m)
        ax.set_xticks(range(len(qs)), ["<40", "<60", "Q1", "Q2", "Q3", "Q4"])
        _style(ax, f"{c}: AP50 by image luminance (proxy)", "mean luma bucket (darkest left)", "AP50 (95% group CI)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "b2_luma.png")
    plt.close(fig)
    with (out / "b2_luma.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "bucket", "npos", "ap50", "ci_lo", "ci_hi"])
        w.writerows(rows)


def hostquant(scores: Path, out: Path) -> None:
    variants = ["float", "float_o8", "pc_minmax", "pc_minmax_o8", "pt_minmax_o8", "pc_pct_o8"]
    rows = []
    for v in variants:
        for d in ("reference", "legacy"):
            files = list(scores.glob(f"score_hq_{v}_{d}_val.json"))
            if not files:
                continue
            s = json.loads(files[0].read_text())
            for c in ("person", "phone"):
                cd = s.get("classes", s).get(c) or {}
                ap = cd.get("ap50")
                ci = cd.get("ap50_ci95") or cd.get("ci95") or [None, None]
                rows.append([v, d, c, ap, *ci[:2], cd.get("ap50_95") or cd.get("ap")])
    with (out / "b2_hostquant.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "decoder", "class", "ap50", "ci_lo", "ci_hi", "ap50_95"])
        w.writerows(rows)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for ax, c in zip(axes, ["person", "phone"], strict=True):
        for k, d in enumerate(("reference", "legacy")):
            pts = [(i, r[3], r[4], r[5]) for i, v in enumerate(variants) for r in rows
                   if r[0] == v and r[1] == d and r[2] == c and r[3] is not None]
            if not pts:
                continue
            xs = [p[0] + (k - 0.5) * 0.15 for p in pts]
            ys = [p[1] for p in pts]
            err = [[p[1] - (p[2] if p[2] is not None else p[1]) for p in pts],
                   [(p[3] if p[3] is not None else p[1]) - p[1] for p in pts]]
            ax.errorbar(xs, ys, yerr=err, fmt="o", markersize=4, color=SLOTS[k], elinewidth=1, capsize=0,
                        label=f"{d} decoder")
        ax.set_xticks(range(len(variants)), variants, rotation=30, ha="right")
        _style(ax, f"{c}: AP50, HOST-SIMULATED INT8 (not Hailo)", "", "AP50")
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "b2_hostquant.png")
    plt.close(fig)


def epochs(specs: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    rows = []
    for k, spec in enumerate(specs):
        name, _, path = spec.partition("=")
        recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        xs = [r["epoch"] for r in recs]
        ys = [r["metrics"].get("AP") for r in recs]
        ax.plot(xs, ys, color=SLOTS[2 + k], linewidth=2, label=name)
        ax.annotate(f"{name} {ys[-1]:.3f}", (xs[-1], ys[-1]), fontsize=7, color=INK, xytext=(3, 0),
                    textcoords="offset points")
        rows += [[name, r["epoch"], r["metrics"].get("AP"), r["metrics"].get("AP50")] for r in recs]
    _style(ax, "Val AP50:95 per epoch (DAMO COCO eval, 3 classes; learning signal only)", "epoch", "AP50:95")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "exp09_epochs.png")
    plt.close(fig)
    with (out / "exp09_epochs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "epoch", "AP", "AP50"])
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slices")
    p.add_argument("--hostquant-scores")
    p.add_argument("--epochs", action="append", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.slices:
        s = json.loads(Path(a.slices).read_text())
        pr_and_reliability(s, out)
        luma(s, out)
    if a.hostquant_scores:
        hostquant(Path(a.hostquant_scores), out)
    if a.epochs:
        epochs(a.epochs, out)
    print(sorted(x.name for x in out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
