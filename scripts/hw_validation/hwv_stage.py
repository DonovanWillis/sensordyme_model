#!/usr/bin/env python3
"""One command per hardware-validation stage, each ending in one PASS/FAIL line.

    PY=/home/pi/sensordyme/.venv/bin/python   V=~/hwval/sensordyme_model/scripts/hw_validation
    $PY $V/hwv_stage.py list                  # the stage table
    $PY $V/hwv_stage.py pins                  # model pins agree (runs anywhere)
    $PY $V/hwv_stage.py model                 # .hef sha + production load + head layout     [Pi, service stopped]
    $PY $V/hwv_stage.py parse-hef             # hailortcli parse-hef: 6 outputs, no NMS      [Pi]
    $PY $V/hwv_stage.py capture               # INT8 outputs on the known-good images        [Pi, service stopped]
    $PY $V/hwv_stage.py parity --bundle DIR   # INT8 vs float expectations (runs anywhere)
    $PY $V/hwv_stage.py latency               # per-frame latency + accelerator duty cycle   [Pi, service stopped]
    $PY $V/hwv_stage.py thermal               # 30-min soak under the real service          [Pi, service RUNNING]
    $PY $V/hwv_stage.py buzzer                # commanded pattern timing (+ a person listens) [Pi, service stopped]
    $PY $V/hwv_stage.py observe --minutes 10  # the runtime's own per-stage timings, recorded  [Pi, service RUNNING]
    $PY $V/hwv_stage.py arch --second-hef B.hef  # single vs dual detector, equal conditions   [Pi, service stopped]
    python hwv_stage.py all --fake            # every stage against fakes, for a dry run

Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED (a prerequisite is missing: no Hailo device,
the service holds the chip, a tool is not installed, an input file is absent),
3 SIMULATED (--fake: the stage's logic ran against stand-ins; NOT evidence).
A --fake run can never exit 0, so a dry run cannot be mistaken for a pass.

Every stage writes a JSON report (--out, default ~/hwval/stages/) holding the
checks, the measured values and the provenance (code commit, model sha256,
versions). The scripts follow docs/HAILO_HARDWARE_VALIDATION.md's ground rules:
nothing under the installed tree is written, no service is started or stopped
(the stage prints the operator command and exits BLOCKED), and frames are not
saved (the known-good images are files in this checkout, not camera frames).

PASS criteria marked "provisional" are engineering tolerances chosen before any
hardware run (docs/MONDAY_HARDWARE_SESSION.md explains each). They are not
product thresholds, and nothing here changes a threshold or the decoder.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402

KNOWN_GOOD = HERE / "known_good"
EXPECTED_FLOAT = KNOWN_GOOD / "expected_float.json"
#: M10.1: the same images through the opt-in reference decoder (research/m101/known_good_decoders.py).
EXPECTED_FLOAT_BY_DECODER = {"legacy": EXPECTED_FLOAT, "reference": KNOWN_GOOD / "expected_float_reference.json"}
PASS, FAIL, BLOCKED, SIMULATED = 0, 1, 2, 3
EXPECTED_LAYOUT = [(80, 8, 81), (40, 16, 81), (20, 32, 81)]  # (grid, stride, class channels)
EXPECTED_HEF_SHAPES = sorted([(80, 80, 68), (80, 80, 81), (40, 40, 68), (40, 40, 81), (20, 20, 68), (20, 20, 81)])

#: Provisional parity tolerances (INT8 vs float, known-good images, legacy decoder).
PARITY = {"score_floor": 0.30, "min_iou": 0.80, "max_score_delta": 0.10, "ghost_float_floor": 0.15,
          "negative_conf": 0.20}

#: name -> (needs hardware, service must be, one-line purpose)
STAGES: dict[str, tuple[str, str, str]] = {
    "pins": ("no", "any", "config / setup_pi.sh / model_registry.yaml pins agree; decoder legacy; EXP-08 inactive"),
    "model": ("yes", "stopped", ".hef present, sha256 == pin, loads through core.accel, 6-head layout"),
    "parse-hef": ("yes (HailoRT tools)", "any", "hailortcli parse-hef: HAILO8L, 6 outputs 80/40/20 x 68/81, no NMS"),
    "capture": ("yes", "stopped", "INT8 head tensors for the known-good images; inputs bit-identical to the float run"),
    "parity": ("no (needs a capture bundle)", "any",
               "INT8 vs float per decoder (--decoder legacy|reference|both), production == offline legacy"),
    "latency": ("yes", "stopped", "predict() p50/p95 per stage vs the production stride; accelerator duty cycle"),
    "thermal": ("yes", "running", "30-min soak: never throttled, ARM clock held, fps stable"),
    "buzzer": ("yes (+ a person)", "stopped", "every pattern commanded with the designed timing; audibility by ear"),
    "observe": ("yes", "running", "M10: record /api/health per-stage timings, fps, RSS, temp, throttle for N min"),
    "arch": ("yes (+ a 2nd .hef)", "stopped", "M10: Arch A (production .hef) vs Arch B (+ --second-hef), ABBA blocks"),
}
#: Stages `all` never runs: they need an input only the operator has (a second .hef)
#: or the service in the other state than the rest of `all`.
_NOT_IN_ALL = ("observe", "arch")


# ------------------------------------------------------------------ reporting
class Report:
    def __init__(self, stage: str, fake: bool) -> None:
        self.stage, self.fake = stage, fake
        self.checks: list[dict] = []
        self.values: dict[str, Any] = {}
        self.blocked: Optional[str] = None
        self.manual: list[str] = []

    def check(self, name: str, ok: bool, detail: Any = "", provisional: bool = False) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail, "provisional": provisional})
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {name}{' (provisional)' if provisional else ''}: {detail}", flush=True)
        return bool(ok)

    def block(self, why: str) -> int:
        self.blocked = why
        return BLOCKED

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    def verdict(self) -> tuple[str, int]:
        if self.blocked:
            return "BLOCKED", BLOCKED
        if self.fake:
            return ("SIMULATED-PASS (stand-ins; not evidence)" if self.passed else "SIMULATED-FAIL"), (
                SIMULATED if self.passed else FAIL)
        return ("PASS", PASS) if self.passed else ("FAIL", FAIL)

    def finish(self, out_dir: Path) -> int:
        label, code = self.verdict()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.stage}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        hc.write_json(path, {"stage": self.stage, "verdict": label, "exit_code": code, "simulated": self.fake,
                             "blocked": self.blocked, "checks": self.checks, "values": self.values,
                             "manual_checks": self.manual, "provenance": provenance(), "utc": hc.utc_now()})
        if self.blocked:
            print(f"  blocked: {self.blocked}")
        for m in self.manual:
            print(f"  MANUAL: {m}")
        print(f"STAGE {self.stage}: {label}   (report {path})", flush=True)
        return code


def provenance() -> dict:
    commit_file = hc.REPO_ROOT / "HWVAL_COMMIT"
    return {"git_commit": hc.git_commit(hc.REPO_ROOT) or (f"file:{commit_file.read_text().strip()}"
                                                          if commit_file.exists() else None),
            "environment": hc.environment(),
            "hwv_stage_sha256": hc.sha256_file(Path(__file__)),
            "core_accel_sha256": hc.sha256_file(hc.REPO_ROOT / "core" / "accel.py")}


# ------------------------------------------------------------------ shared helpers
def load_config(install_dir: Path) -> tuple[dict, Path]:
    path = install_dir / "config.yaml"
    if not path.exists():
        path = hc.REPO_ROOT / "config.yaml"
    return yaml.safe_load(path.read_text()) or {}, path


def hef_path_and_pin(install_dir: Path, config: dict) -> tuple[Path, str]:
    acc = config.get("accelerator") or {}
    path = Path(acc.get("yolo_hef") or "models/damoyolo_tinynasL20_T.hef")
    return (path if path.is_absolute() else install_dir / path), str(acc.get("yolo_hef_sha256") or "").lower()


def service_must_be(rep: Report, wanted: str, args) -> Optional[int]:
    """None when the service state allows the stage; otherwise BLOCKED with the operator command."""
    if args.fake or wanted == "any":
        return None
    state = hc.service_state()
    if wanted == "stopped":
        ok, why = hc.service_gate(state, allow_unknown=args.no_service_check)
        if not ok:
            print(hc.operator_commands())
            return rep.block(why)
        holders = hc.device_holders()
        if holders:
            return rep.block(f"/dev/hailo0 is open in pid(s) {holders}")
    elif wanted == "running" and state != "active":
        return rep.block(f"sensordyme.service is '{state}': the soak measures the production load, so start it "
                         "first (sudo systemctl start sensordyme.service)")
    return None


def known_good_frames() -> list[tuple[str, np.ndarray]]:
    import cv2

    sources = json.loads((KNOWN_GOOD / "SOURCES.json").read_text())
    frames = []
    for row in sources["images"]:
        path = KNOWN_GOOD / row["file"]
        if hc.sha256_file(path) != row["sha256"]:
            raise SystemExit(f"REFUSED: {path} does not match SOURCES.json; the known-good set was modified")
        frames.append((row["file"], cv2.imread(str(path), cv2.IMREAD_COLOR)))
    return frames


@contextlib.contextmanager
def fake_device(backend=None):
    """A stand-in chip, a temporary .hef and an install dir whose config pins it."""
    import fake_hailo

    tmp = Path(tempfile.mkdtemp(prefix="hwv-fake-"))
    hef = tmp / "models" / "damoyolo_tinynasL20_T.hef"
    hef.parent.mkdir()
    hef.write_bytes(b"SIMULATED HEF - not a model\n" * 64)
    config = yaml.safe_load((hc.REPO_ROOT / "config.yaml").read_text())
    config["accelerator"]["yolo_hef_sha256"] = hc.sha256_file(hef)
    (tmp / "config.yaml").write_text(yaml.safe_dump(config))
    planted = fake_hailo.planted_backend([(41, 0.78, 207, 152, 452, 444, 32), (15, 0.81, 109, 185, 511, 482, 32)])

    def default_backend(batch):
        # A uniform frame (the flat-grey known-good image) shows nothing; anything else
        # shows the two planted objects. Enough for every stage's logic to run.
        return fake_hailo.zero_backend(batch) if int(batch.min()) == int(batch.max()) else planted(batch)

    fake_hailo.install(backend or default_backend)
    try:
        yield tmp
    finally:
        fake_hailo.uninstall()
        shutil.rmtree(tmp, ignore_errors=True)


def load_production_model(install_dir: Path, config: dict):
    """The detector exactly as the runtime loads it: accel.configure + maybe_hailo_yolo (integrity check)."""
    sys.path.insert(0, str(hc.REPO_ROOT))
    from core import accel

    acc = dict(config.get("accelerator") or {})
    hef, _pin = hef_path_and_pin(install_dir, config)
    acc["yolo_hef"] = str(hef)
    info = None
    if "hailo_platform" in sys.modules and getattr(sys.modules["hailo_platform"], "SIMULATED", False):
        info = accel.AcceleratorInfo(True, True, "hailo", "SIMULATED Hailo (fake_hailo)")
    accel.configure(acc, info=info)
    return accel, accel.maybe_hailo_yolo("hwv_stage")


# ------------------------------------------------------------------ stages
def stage_pins(rep: Report, args) -> int:
    root = hc.REPO_ROOT
    config, cfg_path = load_config(args.install_dir)
    rep.values["config_path"] = str(cfg_path)
    acc = config.get("accelerator") or {}
    pin = str(acc.get("yolo_hef_sha256") or "")
    setup = re.search(r'^MODEL_SHA256="([0-9a-f]{64})"', (root / "setup_pi.sh").read_text(), re.M)
    registry = yaml.safe_load((root / "core" / "model_registry.yaml").read_text())
    reg_shas = [a.get("sha256") for m in registry.get("models", []) for a in m.get("artifacts", [])
                if str(a.get("path", "")).endswith("damoyolo_tinynasL20_T.hef")]
    rep.values.update({"config_pin": pin, "setup_pi_pin": setup.group(1) if setup else None,
                       "registry_pins": reg_shas})
    rep.check("config pin is a sha256", bool(re.fullmatch(r"[0-9a-f]{64}", pin)), pin[:16])
    rep.check("setup_pi.sh MODEL_SHA256 == config pin", bool(setup) and setup.group(1) == pin,
              setup.group(1)[:16] if setup else "missing")
    rep.check("model_registry.yaml .hef sha256 == config pin", reg_shas == [pin], reg_shas)
    rep.check("decoder is legacy (reference stays opt-in)", acc.get("yolo_decoder", "legacy") == "legacy",
              acc.get("yolo_decoder"))
    hef_name = Path(str(acc.get("yolo_hef") or "")).name
    rep.check("production detector is damoyolo_tinynasL20_T.hef (EXP-08 never active)",
              hef_name == "damoyolo_tinynasL20_T.hef" and "exp08" not in json.dumps(acc).lower(), hef_name)
    return PASS


def stage_model(rep: Report, args) -> int:
    if (b := service_must_be(rep, "stopped", args)) is not None:
        return b
    if not args.fake and not os.path.exists("/dev/hailo0"):
        return rep.block("no /dev/hailo0: the AI HAT+ is not seated or its driver is not loaded")
    config, _ = load_config(args.install_dir)
    hef, pin = hef_path_and_pin(args.install_dir, config)
    if not hef.exists():
        return rep.block(f"{hef} does not exist (setup_pi.sh provisions it)")
    actual = hc.sha256_file(hef)
    rep.values.update({"hef": str(hef), "sha256": actual, "pin": pin})
    rep.check(".hef sha256 == accelerator.yolo_hef_sha256", actual == pin, actual[:16])
    if actual != pin:
        return FAIL
    try:
        import hailo_platform  # noqa: F401
    except ImportError as exc:
        return rep.block(f"hailo_platform does not import ({exc}); sudo apt install hailo-all")
    t0 = time.perf_counter()
    accel, model = load_production_model(args.install_dir, config)
    rep.values["load_s"] = round(time.perf_counter() - t0, 3)
    rep.check("core.accel loads the model (integrity check passed)", model is not None,
              accel.failure_reason() or f"loaded in {rep.values['load_s']} s")
    if model is None:
        return FAIL
    try:
        layout = [(s["grid"], s["stride"], s["num_classes"]) for s in model._scales] if model._raw_outputs else []
        rep.values["layout"] = layout
        rep.check("raw outputs, no on-chip NMS", bool(model._raw_outputs))
        rep.check("heads pair into grids 80/40/20, strides 8/16/32, 81 class channels", layout == EXPECTED_LAYOUT,
                  layout)
        rep.check("input is 640x640", (model._input_h, model._input_w) == (640, 640),
                  (model._input_h, model._input_w))
        rep.check("class map: 0=person, 67=cell phone",
                  model.names.get(0) == "person" and model.names.get(67) == "cell phone")
        rep.check("model identity sha256 == pin", (model.identity or {}).get("sha256") == pin)
        (res,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2)
        rep.check("one inference completes on a black frame", True, f"{len(res.boxes)} boxes")
    finally:
        model.close()
        accel.release_hailo_models()
    rep.check(".hef unchanged after the load", hc.sha256_file(hef) == actual)
    return PASS


def parse_hef_text(text: str) -> dict:
    """What `hailortcli parse-hef` says about outputs, NMS and the target architecture."""
    outputs = []
    for line in text.splitlines():
        m = re.search(r"\bOutput\s+(\S+).*?\((\d+)x(\d+)x(\d+)\)", line)
        if m:
            outputs.append({"name": m.group(1), "shape": [int(m.group(i)) for i in (2, 3, 4)]})
    arch = re.search(r"compiled for:\s*(\S+)", text)
    return {"outputs": outputs, "nms": bool(re.search(r"\bNMS\b", text)),
            "architecture": arch.group(1) if arch else None}


#: What the doc says parse-hef must show; used ONLY for --fake (never as evidence).
SIMULATED_PARSE_HEF = "\n".join([
    "(SIMULATED parse-hef text, written from docs/HAILO_HARDWARE_VALIDATION.md; NOT tool output)",
    "Architecture HEF was compiled for: HAILO8L",
    "Network group name: damoyolo_tinynasL20_T, Multi Context - Number of contexts: 2",
    "    Network name: damoyolo_tinynasL20_T/damoyolo_tinynasL20_T",
    "        VStream infos:",
    "            Input  damoyolo_tinynasL20_T/input_layer1 UINT8, NHWC(640x640x3)",
    *[f"            Output damoyolo_tinynasL20_T/conv{n} UINT8, FCR({g}x{g}x{c})"
      for n, g, c in ((53, 80, 68), (54, 80, 81), (65, 40, 68), (66, 40, 81), (76, 20, 68), (77, 20, 81))],
])


def stage_parse_hef(rep: Report, args) -> int:
    config, _ = load_config(args.install_dir)
    hef, _pin = hef_path_and_pin(args.install_dir, config)
    if args.fake:
        text = SIMULATED_PARSE_HEF
    else:
        if shutil.which("hailortcli") is None:
            return rep.block("hailortcli is not installed (sudo apt install hailo-all)")
        if not hef.exists():
            return rep.block(f"{hef} does not exist")
        text = hc.run_text(["hailortcli", "parse-hef", str(hef)], timeout=60) or ""
    rep.values["raw"] = text[-4000:]
    parsed = parse_hef_text(text)
    rep.values["parsed"] = parsed
    shapes = sorted(tuple(o["shape"]) for o in parsed["outputs"])
    rep.check("six outputs", len(parsed["outputs"]) == 6, len(parsed["outputs"]))
    rep.check("shapes 80/40/20 x 68/81", shapes == EXPECTED_HEF_SHAPES, shapes)
    rep.check("no on-chip NMS", not parsed["nms"])
    rep.check("compiled for HAILO8L", (parsed["architecture"] or "").upper() == "HAILO8L", parsed["architecture"])
    return PASS


def stage_capture(rep: Report, args) -> int:
    if (b := service_must_be(rep, "stopped", args)) is not None:
        return b
    if not args.fake and not os.path.exists("/dev/hailo0"):
        return rep.block("no /dev/hailo0: the AI HAT+ is not seated or its driver is not loaded")
    import capture_hailo_outputs as cap

    out = args.bundle or (args.out / f"capture_known_good_{time.strftime('%Y%m%dT%H%M%S')}")
    argv = ["--images", str(KNOWN_GOOD), "--modes", "letterbox", "--conf", "0.05", "--warmup", "3",
            "--install-dir", str(args.install_dir), "--out", str(out)]
    if args.fake or args.no_service_check:
        argv.append("--no-service-check")
    print(f"  capture_hailo_outputs.py {' '.join(argv)}", flush=True)
    code = cap.main(argv)
    rep.values["bundle"] = str(out)
    if code != 0:
        return rep.block(f"capture_hailo_outputs.py refused (exit {code}); its message is above")
    manifest = hc.load_bundle(out)
    expected = json.loads(EXPECTED_FLOAT.read_text()) if EXPECTED_FLOAT.exists() else None
    rep.check("bundle complete", manifest.get("status") == "complete", manifest.get("status"))
    m = manifest["model"]
    rep.check("model sha256 before == after == pin", m["sha256"] == m.get("sha256_after") == (m["pin"] or "").lower())
    shapes = sorted(tuple(o["shape"]) for o in manifest["hef"]["outputs"])
    rep.check("six outputs, 80/40/20 x 68/81", shapes == EXPECTED_HEF_SHAPES, shapes)
    ranges = [r for s in manifest["samples"] for r in s["ranges"].values()]
    rep.check("class heads inside [0, 1] (on-chip sigmoid)",
              all(r["cls_min"] >= 0 and r["cls_max"] <= 1 for r in ranges))
    rep.check("box heads are logits (not inside [0, 1])", any(r["box_min"] < 0 or r["box_max"] > 1 for r in ranges))
    if expected:
        want = {i["file"]: i["input_tensor_sha256"] for i in expected["images"]}
        got = {}
        for s in manifest["samples"]:
            inp, _ = hc.load_sample(out, s)
            got[s["image_id"]] = hashlib.sha256(np.ascontiguousarray(inp).tobytes()).hexdigest()
        same = [k for k in want if got.get(k) == want[k]]
        # Real even under --fake: the input tensor is made by the real preprocessing code.
        rep.check("input tensors bit-identical to the float run's", len(same) == len(want), f"{len(same)}/{len(want)}")
    else:
        rep.check("float expectations present", False, f"{EXPECTED_FLOAT} missing")
    timing = manifest.get("timing_summary_ms", {})
    rep.values["timing_ms"] = timing
    return PASS


def _dets(rows: list[dict]) -> hd.Detections:
    if not rows:
        return hd.Detections.empty()
    return hd.Detections(np.array([r["xyxy"] for r in rows], float), np.array([r["score"] for r in rows], float),
                         np.array([r["cls"] for r in rows], int), np.zeros(len(rows), int))


def compare_to_float(expected: dict, device: dict[str, list[dict]], tol: dict = PARITY) -> dict:
    """Per-image INT8-vs-float agreement on decoded detections (legacy, frame pixels)."""
    per_image = {}
    for img in expected["images"]:
        f_rows = img["detections"]
        d_rows = device.get(img["file"])
        if d_rows is None:
            per_image[img["file"]] = {"missing": True}
            continue
        fd, dd = _dets(f_rows), _dets(d_rows)
        pairs = hd.match(fd, dd, iou_threshold=0.5)
        by_f = {i: (j, iou) for i, j, iou in pairs}
        strong = [i for i, r in enumerate(f_rows) if r["score"] >= tol["score_floor"]]
        missed = [f_rows[i] for i in strong if i not in by_f or by_f[i][1] < tol["min_iou"]
                  or abs(d_rows[by_f[i][0]]["score"] - f_rows[i]["score"]) > tol["max_score_delta"]]
        matched_d = {j for _, j, _ in pairs}
        ghosts = []
        for j, r in enumerate(d_rows):
            if r["score"] < tol["score_floor"]:
                continue
            fi = next((i for i, (jj, _) in by_f.items() if jj == j), None)
            if j not in matched_d or fi is None or f_rows[fi]["score"] < tol["ghost_float_floor"]:
                ghosts.append(r)
        per_image[img["file"]] = {
            "float_strong": len(strong), "missed_or_moved": missed, "ghosts": ghosts,
            "matched": len(pairs), "iou": [round(iou, 4) for _, _, iou in pairs],
            "score_delta": [round(d_rows[j]["score"] - f_rows[i]["score"], 4) for i, j, _ in pairs]}
    return per_image


def _rows(dets: hd.Detections) -> list[dict]:
    return [{"cls": int(c), "score": round(float(x), 5), "xyxy": [round(float(v), 2) for v in b]}
            for b, x, c in zip(dets.boxes, dets.scores, dets.classes, strict=True)]


def stage_parity(rep: Report, args) -> int:
    """INT8 (the bundle's device tensors) vs float expectations, per decoder.

    legacy: the detections the PRODUCTION decoder made on the device (and a check
    that they equal the offline legacy decoder on the same tensors). reference:
    the opt-in decoder run offline on the SAME device tensors -- the production
    decoder is never switched. M10.1 H1: is the shipping .hef's INT8 output as
    good as the float model under either decoder?"""
    decoders = ("legacy", "reference") if args.decoder == "both" else (args.decoder,)
    expected_by: dict[str, dict] = {}
    for dec in decoders:
        path = EXPECTED_FLOAT_BY_DECODER[dec]
        if not path.exists():
            return rep.block(f"{path} missing (made on the fleet by research/m101/known_good_decoders.py)")
        expected_by[dec] = json.loads(path.read_text())
        if expected_by[dec].get("decoder") != dec:
            return rep.block(f"{path} says decoder {expected_by[dec].get('decoder')!r}, not {dec!r}")
    bundle = args.bundle
    if bundle is None:
        return rep.block("--bundle DIR (from the capture stage) is required")
    manifest = hc.load_bundle(bundle)
    sys.path.insert(0, str(hc.REPO_ROOT))
    device_by: dict[str, dict[str, list[dict]]] = {dec: {} for dec in decoders}
    offline_equal = []
    for s in manifest["samples"]:
        if s["mode"] != "letterbox":
            continue
        _inp, outs = hc.load_sample(bundle, s)
        scales = hd.group_heads({o["name"]: tuple(o["shape"]) for o in s["outputs"]}, 640)
        conf = expected_by[decoders[0]]["conf"]
        offline = hd.unmap(hd.decode(outs, scales, hd.LEGACY, conf), s["geometry"])
        prod = s.get("production_detections") or []
        if "legacy" in device_by:
            device_by["legacy"][s["image_id"]] = prod
        if "reference" in device_by:
            # The product's reference decoder (core.accel), not the upstream DAMO spec (M10.1).
            ref = hd.unmap(hd.decode(outs, scales, hd.DEVICE_REFERENCE, expected_by["reference"]["conf"]),
                           s["geometry"])
            device_by["reference"][s["image_id"]] = _rows(ref)
        a = np.array(sorted((round(r["score"], 4), r["cls"]) for r in prod))
        b = np.array(sorted((round(float(x), 4), int(c)) for x, c in zip(offline.scores, offline.classes,
                                                                         strict=True)))
        offline_equal.append(a.shape == b.shape and (len(a) == 0 or np.allclose(a, b, atol=2e-4)))
    rep.check("production decoder == offline legacy decoder on the device tensors",
              bool(offline_equal) and all(offline_equal), f"{sum(offline_equal)}/{len(offline_equal)}")
    rep.values["decoders"] = {}
    for dec in decoders:
        device, expected = device_by[dec], expected_by[dec]
        if args.fake:
            # The fake chip is not the float model: compare against expectations made by the same fake.
            expected = {"images": [{"file": k, "detections": v} for k, v in device.items()], "conf": expected["conf"]}
        per_image = compare_to_float(expected, device)
        tag = f"[{dec}] "
        missing = [k for k, v in per_image.items() if v.get("missing")]
        rep.check(tag + "every known-good image captured", not missing, missing or "all")
        missed = {k: len(v["missed_or_moved"]) for k, v in per_image.items() if not v.get("missing")}
        ghosts = {k: len(v["ghosts"]) for k, v in per_image.items() if not v.get("missing")}
        rep.check(tag + f"every float detection >= {PARITY['score_floor']} found on INT8 with IoU >= "
                  f"{PARITY['min_iou']} and |score delta| <= {PARITY['max_score_delta']}",
                  sum(missed.values()) == 0, missed, provisional=True)
        rep.check(tag + f"no INT8 detection >= {PARITY['score_floor']} without a float counterpart >= "
                  f"{PARITY['ghost_float_floor']}", sum(ghosts.values()) == 0, ghosts, provisional=True)
        grey = device.get("kg_flat_grey.png", [])
        rep.check(tag + f"flat grey frame: nothing >= {PARITY['negative_conf']}",
                  not [r for r in grey if r["score"] >= PARITY["negative_conf"]], len(grey))
        deltas = [d for v in per_image.values() if not v.get("missing") for d in v["score_delta"]]
        ious = [i for v in per_image.values() if not v.get("missing") for i in v["iou"]]
        summary = {"per_image": per_image,
                   "matched_score_delta": hd.percentiles(np.abs(deltas).tolist()) if deltas else None,
                   "matched_iou": hd.percentiles(ious) if ious else None}
        rep.values["decoders"][dec] = summary
        if dec == "legacy":  # the pre-M10.1 report keys, unchanged
            rep.values.update({k: summary[k] for k in ("per_image", "matched_score_delta", "matched_iou")})
        print(f"  [{dec}] matched |score delta| {summary['matched_score_delta']}  IoU {summary['matched_iou']}")
    rep.manual.append("for head-level numbers (cosine, per-channel LSB error) run compare_float_vs_int8.py on a "
                      "worker with this bundle (docs/HAILO_HARDWARE_VALIDATION.md step 12)")
    return PASS


def stage_latency(rep: Report, args) -> int:
    if (b := service_must_be(rep, "stopped", args)) is not None:
        return b
    if not args.fake and not os.path.exists("/dev/hailo0"):
        return rep.block("no /dev/hailo0: the AI HAT+ is not seated or its driver is not loaded")
    import stage_timing as st

    config, _ = load_config(args.install_dir)
    hef, pin = hef_path_and_pin(args.install_dir, config)
    if not args.fake and (not hef.exists() or hc.sha256_file(hef) != pin):
        return rep.block(f"{hef} missing or not the pinned model: run the model stage first")
    driver = (config.get("modules") or {}).get("driver_attention") or config.get("driver_attention") or {}
    every = int(driver.get("yolo_every_n_frames", 3))
    fps = float((config.get("camera") or {}).get("fps", 30))
    budget_ms = 1000.0 * every / fps
    if args.fake:  # a dry run only needs to execute, not to measure
        args.iterations, args.seconds = min(args.iterations, 30), min(args.seconds, 2)
    else:
        os.environ.setdefault("HAILO_MONITOR", "1")  # lets `hailortcli monitor` see this process
    accel, model = load_production_model(args.install_dir, config)
    if model is None:
        return rep.block(f"the model did not load: {accel.failure_reason()}")
    monitor = None
    if not args.fake and shutil.which("hailortcli"):
        monitor = subprocess.Popen(["hailortcli", "monitor"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)
    frames = [f for _, f in known_good_frames()]
    rec = st.StageRecorder()
    st.instrument_detector(model, rec)
    try:
        for _ in range(5):
            model.predict(frames[0], conf=0.2)
        rec.frames.clear()
        rec._current = {}
        # Burst: back-to-back, the worst case for the host side.
        for i in range(args.iterations):
            model.predict(frames[i % len(frames)], conf=0.2)
            rec.end_frame()
        burst = rec.summary()
        # Paced: the production rate (camera fps / yolo_every_n_frames) for --seconds.
        rec.frames.clear()
        period = budget_ms / 1000.0
        t_end = time.perf_counter() + args.seconds
        t_start = time.perf_counter()
        n = late = 0
        nxt = t_start
        while time.perf_counter() < t_end:
            model.predict(frames[n % len(frames)], conf=0.2)
            rec.end_frame()
            n += 1
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                late += 1
        wall = time.perf_counter() - t_start
        infer_s = sum(f.get("hailo_infer", 0.0) for f in rec.frames) / 1000.0
    finally:
        model.close()
        accel.release_hailo_models()
        if monitor is not None:
            monitor.terminate()
            try:
                text = monitor.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                monitor.kill()
                text = monitor.communicate()[0]
            rep.values["hailortcli_monitor_tail"] = (text or "")[-3000:]
    rep.values.update({"budget_ms": budget_ms, "camera_fps": fps, "yolo_every_n_frames": every,
                       "burst": burst, "paced_frames": n, "paced_late": late, "paced_wall_s": round(wall, 2),
                       "accelerator_duty_cycle": round(infer_s / wall, 4) if wall else None,
                       "note": "duty cycle = time inside pipeline.infer / wall time at the production rate "
                               "(host-observed: includes PCIe transfer, not on-chip utilisation)"})
    total = (burst.get("detector_total") or {})
    infer = (burst.get("hailo_infer") or {})
    print(f"  detector_total p50 {total.get('p50')} p95 {total.get('p95')} ms; hailo_infer p50 {infer.get('p50')} ms;"
          f" decode p50 {(burst.get('decode') or {}).get('p50')} ms; duty {rep.values['accelerator_duty_cycle']}")
    rep.check("the detector is the Hailo path (infer and decode were timed)", "hailo_infer" in (burst or {})
              and bool(infer))
    rep.check(f"predict() p95 <= the production stride budget {budget_ms:.0f} ms",
              bool(total) and total["p95"] <= budget_ms, total.get("p95"))
    if args.fake:  # a 2-second run on a busy laptop says nothing about pacing; record, do not judge
        rep.values["paced_check"] = "not judged under --fake"
    else:
        rep.check("the paced run kept up (fewer than 1% late)", n > 0 and late / n < 0.01, f"{late}/{n} late")
    rep.manual.append("end-to-end camera->alert latency: docs/HAILO_HARDWARE_VALIDATION.md step 11 (ms_clock.html, "
                      "240 fps phone video, stage_timing.py latency --trials)")
    return PASS


def analyse_soak(path: Path) -> dict:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def col(k):
        out = []
        for r in rows:
            with contextlib.suppress(TypeError, ValueError):
                out.append(float(r.get(k) or ""))
        return out

    throttled = sorted({(r.get("throttled") or "").strip() for r in rows} - {""})
    arm = col("arm_hz")
    fps = col("processed_fps")
    head = fps[: max(1, len(fps) // 10)]
    tail = fps[-max(1, len(fps) // 10):]
    return {"rows": len(rows), "throttled_values": throttled, "peak_soc_c": max(col("soc_temp_c"), default=None),
            "arm_hz_first": arm[0] if arm else None, "arm_hz_min": min(arm, default=None),
            "fps_first_10pct": float(np.mean(head)) if fps else None,
            "fps_last_10pct": float(np.mean(tail)) if fps else None,
            "states": sorted({r.get("monitoring_state") or "" for r in rows} - {""})}


def stage_thermal(rep: Report, args) -> int:
    csv_path = args.csv
    if csv_path is None:
        if args.fake:
            csv_path = args.out / "soak_SIMULATED.csv"
            args.out.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w") as f:
                f.write("utc,elapsed_s,soc_temp_c,throttled,arm_hz,core_v,fan_rpm,load1,hailo_temp_c,processed_fps,"
                        "camera_fps,processed_p95_ms,hailo_inferences,monitoring_state\n")
                for i in range(12):
                    f.write(f"SIMULATED,{i * 5},{55 + i * 0.5},0x0,2400000000,0.9,3000,1.2,,29.8,30,40,10,ready\n")
        else:
            if (b := service_must_be(rep, "running", args)) is not None:
                return b
            if shutil.which("vcgencmd") is None:
                return rep.block("vcgencmd not found: not a Raspberry Pi")
            csv_path = args.out / f"soak_{time.strftime('%Y%m%dT%H%M%S')}.csv"
            code = subprocess.call(["bash", str(HERE / "thermal_soak.sh"), "--minutes", str(args.minutes),
                                    "--interval", "5", "--out", str(csv_path)])
            if code != 0:
                return rep.block(f"thermal_soak.sh exited {code}")
    s = analyse_soak(Path(csv_path))
    rep.values.update(s)
    rep.check("rows recorded", s["rows"] >= 10, s["rows"])
    rep.check("never throttled (get_throttled == 0x0 throughout)", s["throttled_values"] == ["0x0"],
              s["throttled_values"])
    rep.check("ARM clock never below its starting value",
              s["arm_hz_first"] is not None and s["arm_hz_min"] >= s["arm_hz_first"], (s["arm_hz_first"],
                                                                                       s["arm_hz_min"]))
    if s["fps_first_10pct"]:
        rep.check("processed fps at the end within 10% of the start", s["fps_last_10pct"] >= 0.9 *
                  s["fps_first_10pct"], (round(s["fps_first_10pct"], 2), round(s["fps_last_10pct"], 2)),
                  provisional=True)
    else:
        rep.check("processed fps recorded (SENSORDYME_TOKEN set?)", False, "no fps column values")
    rep.check("monitoring stayed ready", s["states"] == ["ready"], s["states"])
    return PASS


def stage_buzzer(rep: Report, args) -> int:
    if (b := service_must_be(rep, "stopped", args)) is not None:
        return b
    import buzzer_check

    out = args.out / f"buzzer_{time.strftime('%Y%m%dT%H%M%S')}.json"
    argv = ["run", "--out", str(out), "--install-dir", str(args.install_dir), "--gap", "0" if args.fake else "3"]
    if args.patterns:
        argv += ["--patterns", args.patterns]
    if args.fake or args.no_service_check:
        argv.append("--no-service-check")
    if args.fake:
        import types

        class _Buzzer:  # records nothing audible: SIMULATED
            def __init__(self, pin):
                self.pin = pin

            def on(self):
                pass

            def off(self):
                pass

            def close(self):
                pass

        sys.modules["gpiozero"] = types.SimpleNamespace(Buzzer=_Buzzer)  # type: ignore[assignment]
    else:
        try:
            import gpiozero  # noqa: F401
        except ImportError:
            return rep.block("gpiozero is not installed: not a Raspberry Pi image")
    try:
        code = buzzer_check.main(argv)
    finally:
        if args.fake:
            sys.modules.pop("gpiozero", None)
    if code != 0 or not out.exists():
        return rep.block(f"buzzer_check.py run exited {code}")
    result = json.loads(out.read_text())
    rep.values["timeline_file"] = str(out)
    for name, pat in result["patterns"].items():
        cmp = pat["comparison"]
        tol = 30.0
        rep.check(f"pattern {name}: every pulse commanded, timing within {tol:.0f} ms",
                  bool(cmp.get("ok")) and cmp.get("max_abs_error_ms", 0) <= tol, cmp, provisional=True)
    rep.manual.append("a person confirms every pattern is heard and distinguishable at 50 cm and 1 m (phone audio "
                      "recording, SPL at 1 m), then: buzzer_check.py analyse --timeline <file> --annotations audio.csv")
    return PASS


def stage_observe(rep: Report, args) -> int:
    """The runtime's own always-on stage timings (core/stage_timing.py), recorded."""
    import stage_timing as st

    out_dir = args.out / f"observe_{'SIMULATED_' if args.fake else ''}{time.strftime('%Y%m%dT%H%M%S')}"
    if args.fake:
        # The real recording code against a real dashboard and a real performance
        # block whose only samples come from the production HailoYOLO on the FAKE
        # chip: every number in it is SIMULATED and labelled so by the platform tag.
        sys.path.insert(0, str(hc.REPO_ROOT))
        from core import stage_timing as core_st
        from core.dashboard import DashboardServer

        with fake_device() as install:
            config, _ = load_config(install)
            accel, model = load_production_model(install, config)
            try:
                for _, frame in known_good_frames():
                    model.predict(frame, conf=0.2)
            finally:
                model.close()
                accel.release_hailo_models()
        import secrets

        token = secrets.token_hex(8)  # this dry run's own throwaway token
        server = DashboardServer(host="127.0.0.1", port=0, token=token, health_provider=lambda: {
            "state": "ok", "device": {"monitoring": {"state": "simulated", "model": None}},
            "performance": core_st.report(simulated=True)})
        url = server.start() + "/api/health"
        try:
            code, summary = st.record(url, token, 3 / 60, 1.0, out_dir, args.install_dir, hc.REPO_ROOT)
        finally:
            server.close()
    else:
        if (b := service_must_be(rep, "running", args)) is not None:
            return b
        token = os.environ.get("SENSORDYME_TOKEN") or None
        code, summary = st.record(args.url, token, args.minutes, args.interval, out_dir, args.install_dir,
                                  hc.REPO_ROOT)
        if code == BLOCKED:
            return rep.block("the service's /api/health did not answer with a `performance` block (see above)")
    rep.values.update({"out_dir": str(out_dir), "summary": summary})
    rep.check("polls recorded", bool(summary) and summary.get("polls", 0) > 0, summary.get("polls") if summary else 0)
    stages = (summary or {}).get("stages") or {}
    rep.check("Hailo inference timed by the running runtime", bool(stages.get("hailo_inference")),
              (stages.get("hailo_inference") or {}).get("window_p95_ms"))
    if args.fake:
        rep.values["platform_check"] = "not judged under --fake (SIMULATED by construction)"
    else:
        rep.check("the numbers are a Raspberry Pi measurement (platform class 'pi')",
                  bool(summary.get("is_pi_measurement")), summary.get("platform_classes"))
        rep.check("never throttled during the recording", not summary.get("throttled_any"),
                  summary.get("throttle_values"), provisional=True)
    return PASS


def _arch_frames(args, config: dict):
    """A frame source that gives every block the same frames: the known-good images
    (--fake), a video rewound per block (--video), or the live camera in its configured
    mode. Returns (reset, next_frame, close)."""
    import cv2

    if args.video:
        video = str(Path(args.video).expanduser())
        state: dict = {}

        def reset():
            if state.get("cap") is not None:
                state["cap"].release()
            state["cap"] = cv2.VideoCapture(video)

        def nxt():
            ok, frame = state["cap"].read()
            if not ok:  # loop the clip: a block longer than the clip sees it again
                reset()
                ok, frame = state["cap"].read()
            return frame if ok else None

        return reset, nxt, lambda: state.get("cap") and state["cap"].release()
    if args.fake:
        frames = [f for _, f in known_good_frames()]
        pos = {"i": 0}

        def reset_list():
            pos["i"] = 0

        def nxt_list():
            frame = frames[pos["i"] % len(frames)]
            pos["i"] += 1
            return frame

        return reset_list, nxt_list, lambda: None
    from core.camera import Camera

    camera = Camera(config.get("camera") or {})
    return (lambda: None), (lambda: camera.read(timeout_s=5.0)), camera.release


def stage_arch(rep: Report, args) -> int:
    """Arch A (production: one .hef) vs Arch B (production .hef + a second .hef) at equal
    operating conditions: same frames / camera mode, same detector stride, same pacing,
    each block started in the same thermal state, ABBA order to cancel drift."""
    second = Path(args.second_hef).expanduser() if args.second_hef else None
    if second is None:
        return rep.block("--second-hef PATH is required: Arch B's compiled .hef. This tool never creates one")
    if not second.is_file():
        return rep.block(f"{second} does not exist: Arch B needs its own compiled .hef, and this tool never "
                         "fabricates one")
    second_sha = hc.sha256_file(second)
    try:  # M10.1: a 3-class candidate's hand channel (e.g. 80:hand), read as the device would
        second_extra = hd.parse_extra_classes(args.second_extra_classes)
    except ValueError as exc:
        return rep.block(f"--second-extra-classes: {exc}")
    second_pin = (args.second_hef_sha256 or "").strip().lower() or None
    if second_pin and second_sha != second_pin:
        return rep.block(f"{second} sha256 {second_sha[:16]} != --second-hef-sha256 {second_pin[:16]}: refusing it")
    if (b := service_must_be(rep, "stopped", args)) is not None:
        return b
    if not args.fake and not os.path.exists("/dev/hailo0"):
        return rep.block("no /dev/hailo0: the AI HAT+ is not seated or its driver is not loaded")
    config, cfg_path = load_config(args.install_dir)
    hef, pin = hef_path_and_pin(args.install_dir, config)
    if not hef.exists() or hc.sha256_file(hef) != pin:
        return rep.block(f"{hef} missing or not the pinned production model: run the model stage first")
    if not args.fake:
        try:
            import hailo_platform  # noqa: F401
        except ImportError as exc:
            return rep.block(f"hailo_platform does not import ({exc}); sudo apt install hailo-all")

    import stage_timing as st

    sys.path.insert(0, str(hc.REPO_ROOT))
    from core import accel, detectors
    from core import stage_timing as core_st
    from modules import get_module_class

    module_name = args.module or config.get("module") or "driver_attention"
    module_cfg = dict((config.get("modules") or {}).get(module_name) or {})
    every = max(1, int(module_cfg.get("yolo_every_n_frames", 3)))
    camera = dict(config.get("camera") or {})
    fps = float(camera.get("fps", 30))
    seconds = min(args.block_seconds, 2.0) if args.fake else float(args.block_seconds)
    order = list("ABBA") * max(1, args.rounds)
    acc = dict(config.get("accelerator") or {})
    acc["yolo_hef"] = str(hef)
    info = (accel.AcceleratorInfo(True, True, "hailo", "SIMULATED Hailo (fake_hailo)")
            if getattr(sys.modules.get("hailo_platform"), "SIMULATED", False) else None)
    accel.configure(acc, info=info)
    platform_tag = core_st.platform_tag(simulated=args.fake)
    rep.values.update({
        "conditions": {"module": module_name, "camera_mode": camera, "camera_fps": fps,
                       "detector_stride_frames": every, "block_seconds": seconds, "order": "".join(order),
                       "frames": "known-good images (SIMULATED)" if args.fake else (args.video or "live camera"),
                       "arch_a_scheduler": "HailoRT ROUND_ROBIN" if args.a_scheduled else "none (production)",
                       "arch_b_scheduler": "HailoRT ROUND_ROBIN (two network groups, one VDevice)",
                       "temp_tolerance_c": args.temp_tolerance_c},
        "models": {"production": {"path": str(hef), "sha256": pin, "pin": pin, "decoder": acc.get("yolo_decoder")},
                   "second": {"path": str(second), "sha256": second_sha, "pin": second_pin or "UNPINNED",
                              "extra_classes": second_extra}},
        "platform": platform_tag, "config_sha256": hc.sha256_file(cfg_path),
    })
    reset, next_frame, close_frames = _arch_frames(args, config)
    blocks: list[dict] = []
    baseline = st.thermal_now()
    target = args.start_temp_c if args.start_temp_c is not None else baseline["soc_temp_c"]
    try:
        for index, arm in enumerate(order):
            before = st.wait_for_thermal_state(target, args.temp_tolerance_c,
                                               0.0 if args.fake else args.cooldown_max_s)
            reset()
            with contextlib.ExitStack() as stack:
                if arm == "B" or args.a_scheduled:
                    stack.enter_context(st.scheduled_chip())
                module = get_module_class(module_name)(module_cfg)
                stack.callback(detectors.release_all)
                stack.callback(getattr(module, "close", lambda: None))
                second_model = None
                if arm == "B":
                    second_model = accel.HailoYOLO(str(second), sha256=second_sha,
                                                   decoder=acc.get("yolo_decoder", "legacy"),
                                                   extra_classes=second_extra or None)
                    stack.callback(second_model.close)
                result = st.run_block(module=module, second=second_model, next_frame=next_frame, every=every,
                                      fps=fps, seconds=seconds, warmup=max(2, every + 1) + args.warmup)
            after = st.thermal_now()
            result.update({"block": index, "arm": arm, "thermal_start": before, "thermal_end": after})
            blocks.append(result)
            s = result["summary"]
            print(f"  block {index} arch {arm}: {result['achieved_fps']} fps, frame p95 "
                  f"{(s.get('frame_total') or {}).get('p95')} ms, detector p95 {(s.get('detector_total') or {}).get('p95')}"
                  f" ms, B p95 {(s.get('b_detector_total') or {}).get('p95')} ms, temp {before['soc_temp_c']}->"
                  f"{after['soc_temp_c']}", flush=True)
    finally:
        close_frames()
        detectors.release_all()
    arms = {arm: _arch_aggregate([b for b in blocks if b["arm"] == arm]) for arm in ("A", "B")}
    deltas = {k: (None if arms["A"].get(k) is None or arms["B"].get(k) is None
                  else round(arms["B"][k] - arms["A"][k], 3))
              for k in ("frame_total_p50", "frame_total_p95", "process_total_p95", "achieved_fps", "rss_mb_max")}
    rep.values.update({"blocks": blocks, "arms": arms, "b_minus_a": deltas})
    label = "SIMULATED_" if args.fake else ""
    csv_path = args.out / f"arch_blocks_{label}{time.strftime('%Y%m%dT%H%M%S')}.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["simulated", "platform_class", "block", "arm", "stage", "n", "p50_ms", "p95_ms", "max_ms",
                         "achieved_fps", "late_frames", "rss_mb", "temp_start_c", "temp_end_c", "throttled_end"])
        for b in blocks:
            for stage, stats in b["summary"].items():
                if isinstance(stats, dict) and "p50" in stats:
                    writer.writerow([args.fake, platform_tag["class"], b["block"], b["arm"], stage, stats["n"],
                                     round(stats["p50"], 3), round(stats["p95"], 3), round(stats["max"], 3),
                                     b["achieved_fps"], b["late_frames"], b["rss_mb"],
                                     b["thermal_start"]["soc_temp_c"], b["thermal_end"]["soc_temp_c"],
                                     b["thermal_end"]["throttled_raw"]])
    rep.values["csv"] = str(csv_path)
    print(f"  A vs B (B - A): {deltas}   [{platform_tag['note']}]")
    rep.check("every block of both arms processed frames", all(b["frames"] > 0 for b in blocks),
              [b["frames"] for b in blocks])
    rep.check("Arch B ran its second detector on every stride", all(
        (b["summary"].get("b_detector_total") or {}).get("n", 0) > 0 for b in blocks if b["arm"] == "B"))
    if args.fake:
        rep.values["thermal_check"] = "not judged under --fake (no SoC sensor; SIMULATED)"
    else:
        rep.check(f"every block started within {args.temp_tolerance_c} C of the same temperature",
                  all(b["thermal_start"]["within_target"] is not False for b in blocks),
                  [b["thermal_start"]["soc_temp_c"] for b in blocks], provisional=True)
        rep.check("never throttled during the comparison",
                  all(not b["thermal_end"]["throttle_now"] for b in blocks),
                  [b["thermal_end"]["throttled_raw"] for b in blocks])
        rep.check("measured on a Raspberry Pi", platform_tag["class"] == "pi", platform_tag["class"])
    rep.manual.append("Arch B's scheduler path (HailoRT ROUND_ROBIN) is unverified until this first Pi run; if "
                      "the second network group fails to configure, the report says so and nothing is inferred")
    return PASS


def _arch_aggregate(blocks: list[dict]) -> dict:
    """One arm across its blocks: the mean of each block's p50 / p95, the worst max."""
    if not blocks:
        return {}
    out: dict = {"blocks": len(blocks)}
    for stage in ("capture", "detector_total", "hailo_infer", "decode", "preprocess", "facemesh", "hands",
                  "module_logic", "process_total", "frame_total", "b_detector_total", "b_hailo_infer", "b_decode"):
        stats = [b["summary"].get(stage) for b in blocks if b["summary"].get(stage)]
        if stats:
            out[f"{stage}_p50"] = round(float(np.mean([s["p50"] for s in stats])), 3)
            out[f"{stage}_p95"] = round(float(np.mean([s["p95"] for s in stats])), 3)
            out[f"{stage}_max"] = round(float(max(s["max"] for s in stats)), 3)
    fps = [b["achieved_fps"] for b in blocks if b["achieved_fps"] is not None]
    out["achieved_fps"] = round(float(np.mean(fps)), 2) if fps else None
    rss = [b["rss_mb"] for b in blocks if b["rss_mb"] is not None]
    out["rss_mb_max"] = max(rss) if rss else None
    temps = [b["thermal_end"]["soc_temp_c"] for b in blocks if b["thermal_end"]["soc_temp_c"] is not None]
    out["soc_temp_end_max_c"] = max(temps) if temps else None
    out["throttled_any"] = any(bool(b["thermal_end"]["throttle_now"]) for b in blocks)
    return out


RUNNERS: dict[str, Callable[[Report, Any], int]] = {
    "pins": stage_pins, "model": stage_model, "parse-hef": stage_parse_hef, "capture": stage_capture,
    "parity": stage_parity, "latency": stage_latency, "thermal": stage_thermal, "buzzer": stage_buzzer,
    "observe": stage_observe, "arch": stage_arch,
}


def run_stage(name: str, args) -> int:
    print(f"== STAGE {name}{' [--fake: stand-ins, NOT evidence]' if args.fake else ''}", flush=True)
    rep = Report(name, args.fake)
    try:
        if args.fake and name in ("model", "capture", "latency", "buzzer", "arch"):
            with fake_device() as install:
                saved = args.install_dir
                args.install_dir = install
                try:
                    code = RUNNERS[name](rep, args)
                finally:
                    args.install_dir = saved
        else:
            code = RUNNERS[name](rep, args)
    except SystemExit as exc:  # a helper refused with a message
        rep.block(str(exc))
        code = BLOCKED
    except Exception as exc:  # noqa: BLE001 - a crash is a FAIL with its reason, never a bare traceback
        import traceback

        traceback.print_exc()
        rep.values["exception"] = traceback.format_exc()[-4000:]
        rep.check("stage ran to completion", False, repr(exc))
        code = FAIL
    if code == BLOCKED and not rep.blocked:
        rep.block("refused")
    return rep.finish(args.out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("stage", choices=["list", "all", *STAGES])
    p.add_argument("--fake", action="store_true", help="run against stand-ins (dry run; exit 3, never 0)")
    p.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR)
    p.add_argument("--out", type=Path, default=None, help="report directory (default ~/hwval/stages)")
    p.add_argument("--bundle", type=Path, default=None, help="capture: output dir; parity: the capture bundle")
    p.add_argument("--decoder", choices=("legacy", "reference", "both"), default="both",
                   help="parity: which decoder's INT8-vs-float agreement to check (reference runs offline on the "
                        "device tensors; the production decoder is never switched)")
    p.add_argument("--csv", type=Path, default=None, help="thermal: analyse this soak CSV instead of running one")
    p.add_argument("--minutes", type=int, default=30, help="thermal soak length")
    p.add_argument("--iterations", type=int, default=300, help="latency: back-to-back frames")
    p.add_argument("--seconds", type=int, default=60, help="latency: paced run length")
    p.add_argument("--patterns", default=None, help="buzzer: comma list (default: every production pattern)")
    p.add_argument("--no-service-check", action="store_true", help="ONLY on a machine with no systemd service")
    # M10 observe / arch
    p.add_argument("--url", default="http://127.0.0.1:8899/api/health", help="observe: the service's health URL "
                   "(token from SENSORDYME_TOKEN, never printed)")
    p.add_argument("--interval", type=float, default=10.0, help="observe: seconds between polls")
    p.add_argument("--second-hef", default=None, help="arch: Arch B's second .hef (must exist; never created)")
    p.add_argument("--second-hef-sha256", default=None, help="arch: pin for --second-hef (refused on mismatch)")
    p.add_argument("--second-extra-classes", default=None,
                   help="arch: the second .hef's class channels beyond COCO-80, e.g. 80:hand (M10.1)")
    p.add_argument("--block-seconds", type=float, default=120.0, help="arch: length of one timed block")
    p.add_argument("--rounds", type=int, default=2, help="arch: ABBA rounds (2 = ABBAABBA)")
    p.add_argument("--warmup", type=int, default=30, help="arch: extra untimed frames per block")
    p.add_argument("--video", default=None, help="arch: a video instead of the live camera (same frames per block)")
    p.add_argument("--module", default=None, help="arch: module to run (default: config.yaml's)")
    p.add_argument("--a-scheduled", action="store_true",
                   help="arch: run Arch A under the HailoRT scheduler too (isolates the second network's cost)")
    p.add_argument("--start-temp-c", type=float, default=None,
                   help="arch: start every block at or below this SoC temperature (default: the temperature at start)")
    p.add_argument("--temp-tolerance-c", type=float, default=2.0, help="arch: allowed start-temperature spread")
    p.add_argument("--cooldown-max-s", type=float, default=900.0, help="arch: longest wait for the start temperature")
    args = p.parse_args(argv)
    sys.dont_write_bytecode = True
    if args.out is None:
        args.out = (Path(tempfile.gettempdir()) / "hwval-fake" if args.fake else Path.home() / "hwval" / "stages")
    ok, why = hc.ensure_outside(args.out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return BLOCKED
    if args.stage == "list":
        print(f"{'stage':<10} {'hardware':<28} {'service':<8} purpose")
        for name, (hw, svc, what) in STAGES.items():
            print(f"{name:<10} {hw:<28} {svc:<8} {what}")
        return PASS
    if args.stage != "all":
        return run_stage(args.stage, args)
    results = {}
    for name in STAGES:
        if name in _NOT_IN_ALL:
            print(f"== STAGE {name}: not part of `all` (run it on its own; see `list`)")
            continue
        if name == "parity" and args.bundle is None:
            results[name] = BLOCKED
            print("== STAGE parity: needs --bundle (in `all`, it uses the capture stage's bundle)")
            continue
        if name == "capture" and args.bundle is None:
            args.bundle = args.out / f"capture_known_good_{time.strftime('%Y%m%dT%H%M%S')}"
        if name == "thermal" and not args.fake:
            print("== STAGE thermal: skipped in `all` (needs the service RUNNING; run it on its own)")
            results[name] = BLOCKED
            continue
        results[name] = run_stage(name, args)
    labels = {PASS: "PASS", FAIL: "FAIL", BLOCKED: "BLOCKED", SIMULATED: "SIMULATED"}
    print("\nSUMMARY")
    for name, code in results.items():
        print(f"  {name:<10} {labels[code]}")
    codes = set(results.values())
    return FAIL if FAIL in codes else BLOCKED if BLOCKED in codes else SIMULATED if SIMULATED in codes else PASS


if __name__ == "__main__":
    raise SystemExit(main())
