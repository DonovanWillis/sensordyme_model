"""M13 performance changes: each one measured in docs/research/M13_PERFORMANCE.md, each pinned here.

1. The uplink sends a queued alert at once instead of after the rest of the poll
   interval -- without ever shortening the reconnection backoff, and without
   changing what is sent.
2. Both raw-head decoders in core/accel.py skip a redundant full pass over the
   (anchors x classes) scores. Their output is pinned BIT FOR BIT (same boxes,
   same float32 scores, same order, same refusals) against the pre-M13
   formulation, frozen below, on random and planted heads, with and without a
   class filter and the extra-class map, at every threshold edge.
3. core/eye_signal.AdaptiveBaseline keeps its window sorted as samples arrive
   and leave, instead of sorting up to 1,800 floats twice per frame. Its
   reference() is pinned to the pre-M13 sort on random streams with ties,
   signed zeros, infinities, NaNs, irregular time steps and resets.
4. The isolated reference-decoder development config (deploy/dev overlay +
   scripts/hw_validation/dev_config.py) changes exactly one key, validates, and
   is refused inside the installed tree; config.yaml keeps the legacy default.
   `hwv_stage.py latency --latency-decoder` times either or both decoders in
   process, and its default is the pre-M13 stage.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import requests

from core.uplink import CloudUplink

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))
import fake_hailo  # noqa: E402

from core import accel  # noqa: E402
from core.accel import COCO_NAMES, HailoYOLO, _nms, _ShimBox, _ShimResult  # noqa: E402


# ---------------------------------------------------------------------- 1. uplink wake
class _Backend:
    def __init__(self) -> None:
        self.down = False
        self.posts: list[tuple[float, list[str]]] = []
        self.lock = threading.Lock()

    def post(self, url, json=None, **_kwargs):
        with self.lock:
            self.posts.append((time.monotonic(), [e["type"] for e in json["events"]]))
        if self.down:
            raise requests.ConnectionError("SIMULATED: backend unreachable")
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"command": None})

    def count(self) -> int:
        with self.lock:
            return len(self.posts)


def _event(kind: str):
    return types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "2026-11-16T08:00:00"),
                                 type=kind, confidence=0.9, contributing_signals={"ear": 0.1}, duration=2.0)


def _uplink(backend, poll_interval_s: float = 5.0) -> CloudUplink:
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_" + "x" * 32,
                          "poll_interval_s": poll_interval_s, "command_auth": "off"})
    uplink._requests = backend
    return uplink


def _until(predicate, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _worker(uplink: CloudUplink) -> threading.Thread:
    thread = threading.Thread(target=uplink._run, daemon=True)
    thread.start()
    return thread


def test_an_alert_is_sent_at_once_not_after_the_rest_of_the_poll_interval():
    backend = _Backend()
    uplink = _uplink(backend, poll_interval_s=5.0)
    thread = _worker(uplink)
    try:
        assert _until(lambda: backend.count() >= 1)  # the first (empty) heartbeat went up
        pushed = time.monotonic()
        uplink.push_event(_event("drowsiness"), "loud")
        assert _until(lambda: any("drowsiness" in types_ for _, types_ in backend.posts), timeout=1.0), \
            "the alert waited for the 5 s poll interval"
        sent_at = next(t for t, types_ in backend.posts if "drowsiness" in types_)
        assert sent_at - pushed < 1.0
        assert uplink.backlog()["queued"] == 0
    finally:
        uplink.close()
        thread.join(2.0)
    assert not thread.is_alive(), "close() must end a worker that is waiting for its next sync"


def test_without_an_alert_the_worker_still_waits_the_poll_interval():
    backend = _Backend()
    uplink = _uplink(backend, poll_interval_s=5.0)
    thread = _worker(uplink)
    try:
        assert _until(lambda: backend.count() >= 1)
        time.sleep(0.3)
        assert backend.count() == 1  # no extra heartbeats
    finally:
        uplink.close()
        thread.join(2.0)


def test_alerts_never_shorten_the_reconnection_backoff():
    backend = _Backend()
    backend.down = True
    uplink = _uplink(backend, poll_interval_s=5.0)
    thread = _worker(uplink)
    try:
        assert _until(lambda: backend.count() >= 1)  # the first sync failed: backing off
        for _ in range(5):
            uplink.push_event(_event("drowsiness"), "loud")
        time.sleep(0.4)
        assert backend.count() == 1, "an alert during an outage must not turn the backoff into a retry storm"
        assert uplink.backlog()["queued"] == 5  # kept for the reconnection
    finally:
        uplink.close()
        thread.join(2.0)
    assert not thread.is_alive()


def test_an_alert_queued_while_a_sync_is_in_flight_goes_up_on_the_next_cycle_at_once():
    backend = _Backend()
    uplink = _uplink(backend, poll_interval_s=5.0)
    gate = threading.Event()
    original = backend.post

    def slow_post(url, json=None, **kw):
        if not gate.is_set():
            gate.set()
            uplink.push_event(_event("phone_use"), "loud")  # arrives after this batch was taken
        return original(url, json=json, **kw)

    backend.post = slow_post  # type: ignore[method-assign]
    thread = _worker(uplink)
    try:
        assert _until(lambda: any("phone_use" in t for _, t in backend.posts), timeout=1.0)
    finally:
        uplink.close()
        thread.join(2.0)


# ---------------------------------------------------------------------- 2. decoders, bit for bit
def _pre_m13_legacy(m: HailoYOLO, raw, scale, left, top, w, h, classes, conf):
    """core/accel.py HailoYOLO._decode_raw (legacy) as it was before M13, verbatim but for comments."""
    wanted = {int(c) for c in classes} if classes is not None else None
    all_boxes, all_scores, all_classes = [], [], []
    for spec in m._scales:
        box_raw, cls_raw = m._scale_heads(raw, spec)
        grid, stride = spec["grid"], spec["stride"]
        scores, channels = m._class_scores(cls_raw)
        best_cls = scores.argmax(axis=1)
        if channels is not None:
            best_cls = channels[best_cls]
        best_score = scores.max(axis=1)
        m._finite_or_raise(best_score, f"class scores at grid {grid}")
        keep = best_score >= conf
        if wanted is not None:
            keep &= np.isin(best_cls, list(wanted))
        if not keep.any():
            continue
        dfl = box_raw.reshape(-1, 4, m._DFL_BINS)[keep]
        dfl = dfl - dfl.max(axis=2, keepdims=True)
        prob = np.exp(dfl)
        prob /= prob.sum(axis=2, keepdims=True)
        dist = (prob * np.arange(m._DFL_BINS, dtype=np.float32)).sum(axis=2)
        m._finite_or_raise(dist, f"box logits at grid {grid}")
        ys, xs = np.divmod(np.nonzero(keep)[0], grid)
        cx, cy = (xs + 0.5) * stride, (ys + 0.5) * stride
        all_boxes.append(np.stack([cx - dist[:, 0] * stride, cy - dist[:, 1] * stride,
                                   cx + dist[:, 2] * stride, cy + dist[:, 3] * stride], axis=1))
        all_scores.append(best_score[keep])
        all_classes.append(best_cls[keep])
    return _pre_m13_finish(m, all_boxes, all_scores, all_classes, scale, left, top, w, h, cap=None)


def _pre_m13_reference(m: HailoYOLO, raw, scale, left, top, w, h, classes, conf):
    """core/accel.py HailoYOLO._decode_reference as it was before M13, verbatim but for comments."""
    num_classes = m._class_head_width()
    wanted = {int(c) for c in classes} if classes is not None else None
    all_boxes, all_scores, all_classes = [], [], []
    for spec in m._scales:
        box_raw, cls_raw = m._scale_heads(raw, spec)
        if cls_raw.shape[-1] < num_classes:
            raise RuntimeError("narrow class head")
        grid, stride = spec["grid"], spec["stride"]
        scores, channels = m._class_scores(cls_raw)
        m._finite_or_raise(scores.sum(), f"class scores at grid {grid}")
        anchor_idx, col_idx = np.nonzero(scores >= conf)
        class_idx = col_idx if channels is None else channels[col_idx]
        if wanted is not None:
            keep = np.isin(class_idx, list(wanted))
            anchor_idx, col_idx, class_idx = anchor_idx[keep], col_idx[keep], class_idx[keep]
        if anchor_idx.size == 0:
            continue
        dfl = box_raw.reshape(-1, 4, m._DFL_BINS)[anchor_idx]
        dfl = dfl - dfl.max(axis=2, keepdims=True)
        prob = np.exp(dfl)
        prob /= prob.sum(axis=2, keepdims=True)
        dist = (prob * np.arange(m._DFL_BINS, dtype=np.float32)).sum(axis=2)
        m._finite_or_raise(dist, f"box logits at grid {grid}")
        ys, xs = np.divmod(anchor_idx, grid)
        px, py = xs * stride, ys * stride
        all_boxes.append(np.stack([px - dist[:, 0] * stride, py - dist[:, 1] * stride,
                                   px + dist[:, 2] * stride, py + dist[:, 3] * stride], axis=1))
        all_scores.append(scores[anchor_idx, col_idx])
        all_classes.append(class_idx)
    return _pre_m13_finish(m, all_boxes, all_scores, all_classes, scale, left, top, w, h,
                           cap=m._REFERENCE_MAX_DETECTIONS)


def _pre_m13_finish(m, all_boxes, all_scores, all_classes, scale, left, top, w, h, cap):
    if not all_boxes:
        return [_ShimResult([])]
    boxes_xyxy = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    class_ids = np.concatenate(all_classes)
    keep_idx: list = []
    for cls_id in np.unique(class_ids):
        sel = np.nonzero(class_ids == cls_id)[0]
        keep_idx.extend(sel[_nms(boxes_xyxy[sel], scores[sel], m._nms_iou)])
    order = sorted(keep_idx, key=lambda j: -scores[j])
    if cap is not None:
        order = order[:cap]
    shim = []
    for i in order:
        x1, y1, x2, y2 = boxes_xyxy[i]
        shim.append(_ShimBox(np.asarray([min(max((x1 - left) / scale, 0.0), w), min(max((y1 - top) / scale, 0.0), h),
                                         min(max((x2 - left) / scale, 0.0), w), min(max((y2 - top) / scale, 0.0), h)]),
                             int(class_ids[i]), float(scores[i])))
    return [_ShimResult(shim)]


def _rows(result) -> list[tuple]:
    return [(b.cls.tobytes(), b.conf.tobytes(), b.xyxy.tobytes()) for b in result[0].boxes]


@pytest.fixture
def chip():
    fake_hailo.install()
    try:
        yield
    finally:
        fake_hailo.uninstall()


def _heads(rng, *, density: float, channels: int = 81, nan: bool = False) -> dict:
    out = {}
    for name, (g, _, c) in fake_hailo.OUTPUTS:
        if c == fake_hailo.BOX_CHANNELS:
            out[name] = rng.normal(0.0, 2.0, (1, g, g, c)).astype(np.float32)
        else:
            scores = (rng.random((1, g, g, channels)) ** (1.0 / density)).astype(np.float32) * 0.6
            # exact threshold edges and ties: inclusive >= must keep these
            scores.reshape(-1, channels)[::97, 67] = np.float32(0.2)
            scores.reshape(-1, channels)[::89, 0] = np.float32(0.5)
            scores.reshape(-1, channels)[::83, :3] = np.float32(0.3)
            if nan:
                scores[0, g // 2, g // 2, 5] = np.nan
            out[name] = scores
    return out


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("decoder", ["legacy", "reference"])
@pytest.mark.parametrize("extra", [None, {80: "hand"}])
def test_the_decoders_are_bit_identical_to_their_pre_m13_formulation(chip, seed, decoder, extra):
    rng = np.random.default_rng(seed)
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (SIMULATED)", decoder=decoder, extra_classes=extra)
    before = _pre_m13_legacy if decoder == "legacy" else _pre_m13_reference
    some = sorted({0, 39, 41, 67, 80} if extra else {0, 39, 41, 67})
    for density in (0.001, 0.004):  # ~0.5% / ~2% of (anchor, class) pairs above 0.01: real-head-like
        raw = _heads(rng, density=density)
        for conf in (0.01, 0.05, 0.2, 0.3, 0.5, 1.0):
            for classes in (None, some, [67], []):
                for geo in ((1.0, 0, 80, 640, 480), (0.5, 0, 160, 1280, 640)):
                    got = model._decode_raw(raw, *geo, classes, conf)
                    want = before(model, raw, *geo, classes, conf)
                    assert _rows(got) == _rows(want), (decoder, density, conf, classes, geo)


@pytest.mark.parametrize("decoder", ["legacy", "reference"])
def test_a_non_finite_class_score_is_refused_exactly_as_before(chip, decoder):
    rng = np.random.default_rng(0)
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (SIMULATED)", decoder=decoder)
    before = _pre_m13_legacy if decoder == "legacy" else _pre_m13_reference
    for poison in (np.nan, np.inf):
        raw = _heads(rng, density=0.05)
        name = next(n for n, (_, _, c) in fake_hailo.OUTPUTS if c != fake_hailo.BOX_CHANNELS)
        raw[name][0, 3, 4, 7] = poison
        with pytest.raises(RuntimeError, match="non-finite class scores") as new:
            model._decode_raw(raw, 1.0, 0, 80, 640, 480, None, 0.2)
        with pytest.raises(RuntimeError, match="non-finite class scores") as old:
            before(model, raw, 1.0, 0, 80, 640, 480, None, 0.2)
        assert str(new.value) == str(old.value)


def test_a_planted_phone_beside_a_stronger_person_survives_both_decoders(chip):
    """The reference decoder's reason to exist (multilabel) is intact after the M13 prefilter."""
    raw = {name: np.zeros((1, g, g, c), np.float32) for name, (g, _, c) in fake_hailo.OUTPUTS}
    cls_name = next(n for n, (g, _, c) in fake_hailo.OUTPUTS if g == 40 and c != fake_hailo.BOX_CHANNELS)
    raw[cls_name][0, 10, 12, 0] = 0.9   # person
    raw[cls_name][0, 10, 12, 67] = 0.4  # phone on the same anchor: loses the argmax
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (SIMULATED)", decoder="reference")
    labels = sorted(COCO_NAMES[int(b.cls[0])] for b in model._decode_raw(raw, 1.0, 0, 80, 640, 480, None, 0.2)[0].boxes)
    assert labels == ["cell phone", "person"]
    model.decoder = "legacy"
    labels = [COCO_NAMES[int(b.cls[0])] for b in model._decode_raw(raw, 1.0, 0, 80, 640, 480, None, 0.2)[0].boxes]
    assert labels == ["person"]
    assert accel.HailoYOLO.DECODERS == ("legacy", "reference")


# ---------------------------------------------------------------------- 3. adaptive baseline
class _PreM13Baseline:
    """core/eye_signal.AdaptiveBaseline as it was before M13 (sort per reference())."""

    def __init__(self, window_s=60.0, percentile=0.90, band_low=0.6, band_high=1.4, min_samples=30):
        from collections import deque

        self._window_s, self._percentile = window_s, percentile
        self._band_low, self._band_high, self._min_samples = band_low, band_high, min_samples
        self._samples = deque()

    def observe(self, value, now):
        self._samples.append((now, float(value)))
        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def reference(self, calibrated):
        if len(self._samples) < self._min_samples:
            return calibrated
        ordered = sorted(value for _, value in self._samples)
        index = min(len(ordered) - 1, int(len(ordered) * self._percentile))
        session = ordered[index]
        if calibrated is None:
            return session
        return max(self._band_low * calibrated, min(self._band_high * calibrated, session))

    def reset(self):
        self._samples.clear()


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if a != a or b != b:  # NaN
        return (a != a) and (b != b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("calibrated", [None, 0.30])
def test_adaptive_baseline_reference_is_identical_to_the_pre_m13_sort(seed, calibrated):
    from core.eye_signal import AdaptiveBaseline

    rng = np.random.default_rng(seed)
    kwargs = {"window_s": float(rng.choice([2.0, 5.0, 60.0])), "min_samples": int(rng.choice([1, 10, 30])),
              "percentile": float(rng.choice([0.5, 0.9, 1.0, 0.0]))}
    new, old = AdaptiveBaseline(**kwargs), _PreM13Baseline(**kwargs)
    now = 1000.0
    pool = [0.0, -0.0, 0.25, 0.25, 0.31, float("inf"), float("-inf")]
    for i in range(3000):
        now += float(rng.choice([0.0, 1 / 30, 0.05, 0.5, 3.0], p=[0.05, 0.75, 0.1, 0.07, 0.03]))
        r = rng.random()
        if seed % 2 and r < 0.01:
            value = float("nan")  # odd seeds only: exercises the fallback to the original sort
        elif r < 0.3:
            value = float(rng.choice(pool))
        else:
            value = float(np.round(rng.normal(0.28, 0.05), int(rng.choice([2, 6]))))
        new.observe(value, now)
        old.observe(value, now)
        assert _same(new.reference(calibrated), old.reference(calibrated)), i
        if i == 1500:
            new.reset()
            old.reset()


def test_adaptive_baseline_window_still_evicts_by_time():
    from core.eye_signal import AdaptiveBaseline

    b = AdaptiveBaseline(window_s=1.0, min_samples=1, percentile=1.0)
    b.observe(0.9, 0.0)
    b.observe(0.2, 0.5)
    assert b.reference(None) == 0.9
    b.observe(0.3, 1.6)  # the 0.9 at t=0 and the 0.2 at t=0.5 are older than 1 s
    assert b.reference(None) == 0.3
    assert b._ordered == [0.3]
    assert b._nans == 0


# ---------------------------------------------------------------------- 4. isolated dev config + latency option
def test_the_reference_decoder_dev_config_changes_exactly_one_key_and_validates(tmp_path):
    import dev_config
    import yaml

    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert base["accelerator"]["yolo_decoder"] == "legacy"  # the shipped default is untouched
    overlay = yaml.safe_load(dev_config.DEFAULT_OVERLAY.read_text())
    merged, diff, errors, _warnings = dev_config.build(base, overlay)
    assert errors == []
    assert diff == [("accelerator.yolo_decoder", "legacy", "reference")]
    assert merged["accelerator"]["yolo_hef_sha256"] == base["accelerator"]["yolo_hef_sha256"]
    out = tmp_path / "dev" / "config.reference.yaml"
    install = tmp_path / "install"
    install.mkdir()
    assert dev_config.main(["--install-dir", str(install), "--out", str(out)]) == 0
    written = yaml.safe_load(out.read_text())
    assert written["accelerator"]["yolo_decoder"] == "reference"
    assert out.read_text().startswith("# DEVELOPMENT CONFIG -- NOT INSTALLED")
    # never inside the installed tree
    assert dev_config.main(["--install-dir", str(install), "--out", str(install / "config.yaml")]) == 2
    assert not (install / "config.yaml").exists()


def test_a_dev_overlay_that_breaks_the_schema_writes_nothing(tmp_path):
    import dev_config

    bad = tmp_path / "bad.yaml"
    bad.write_text("accelerator:\n  yolo_decoder: fastest\n")
    out = tmp_path / "out.yaml"
    assert dev_config.main(["--install-dir", str(tmp_path / "none"), "--overlay", str(bad), "--out", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("which", ["installed", "both"])
def test_latency_stage_times_the_asked_decoders_in_process_only(tmp_path, which):
    import json
    import subprocess

    hwv = ROOT / "scripts" / "hw_validation"
    r = subprocess.run([sys.executable, str(hwv / "hwv_stage.py"), "latency", "--fake",  # noqa: S603
                        "--latency-decoder", which, "--iterations", "8", "--seconds", "1", "--out", str(tmp_path)],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out
    assert r.returncode == 3, out  # SIMULATED: a --fake run is never a PASS
    report = json.loads(next(tmp_path.glob("latency_*.json")).read_text())
    values = report["values"]
    assert values["latency_decoders"]["installed"] == "legacy"
    if which == "installed":
        assert values["latency_decoders"]["timed"] == ["legacy"]
        assert "burst_by_decoder" not in values
    else:
        assert set(values["burst_by_decoder"]) == {"legacy", "reference"}
        assert all((s.get("decode") or {}).get("n") for s in values["burst_by_decoder"].values())
