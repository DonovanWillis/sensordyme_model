"""Shared helpers for the hardware-validation scripts: safety gates, hashing,
preprocessing identical to production, the capture-bundle format, tensor stats.

Nothing here writes outside the directory a caller names, and nothing here
starts, stops or edits a service. The gates exist so a script run on the wrong
machine, or at the wrong moment, refuses instead of interfering with the
running product.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SERVICE = "sensordyme.service"
DEFAULT_INSTALL_DIR = Path("/home/pi/sensordyme")
BUNDLE_FORMAT = "sensordyme-hailo-capture/1"

#: systemd states in which the service may hold (or be about to take) the chip,
#: the camera or the buzzer GPIO. All three are single-user devices.
_BUSY_STATES = {"active", "activating", "reloading", "deactivating", "refreshing"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ safety gates
#: Modules that exist only on the Pi image. Missing one means "not on the device".
HARDWARE_MODULES = ("hailo_platform", "picamera2", "gpiozero", "libcamera")


def run_cli(main, argv=None) -> int:
    """Run a script's main() so that a missing input or missing hardware is a
    one-line REFUSED with exit 2, never a traceback (M8.3).

    Only these are translated: a file or directory that does not exist or
    cannot be read (the install tree, a bundle, an image directory) and an
    import of a Pi-only module. Anything else is a genuine fault and keeps its
    traceback, so a bug cannot hide behind a tidy message.
    """
    try:
        return main(argv)
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        print(f"REFUSED: cannot read {getattr(exc, 'filename', None) or exc}: {exc.strerror or exc}. "
              "On a Pi, check --install-dir / the input path; on a Mac this step needs the device.",
              file=sys.stderr)
        return 2
    except ImportError as exc:
        name = getattr(exc, "name", "") or ""
        if name.split(".")[0] in HARDWARE_MODULES:
            print(f"BLOCKED: {name} is not importable ({exc}); this step runs only on the Pi image "
                  "(sudo apt install hailo-all / python3-picamera2 / python3-gpiozero).", file=sys.stderr)
            return 2
        raise


def operator_commands(service: str = SERVICE) -> str:
    """The reversible stop/start an OPERATOR runs. Printed, never executed."""
    return "\n".join([
        "  The Hailo chip, the camera and the buzzer GPIO are single-user. This script",
        f"  will not run while {service} may hold them, and it will not stop the",
        "  service for you. On a DEVELOPMENT Pi, an operator may run:",
        "",
        f"    systemctl is-active {service}          # note the state to restore",
        f"    sudo systemctl stop {service}          # explicit stop: Restart=always does not fire",
        f"    systemctl is-active {service}          # expect: inactive",
        "    # ... run the validation script ...",
        f"    sudo systemctl start {service}         # restore",
        f"    systemctl is-active {service}          # expect: active",
        "    curl -s -H \"X-Device-Token: $SENSORDYME_TOKEN\" localhost:8899/api/health \\",
        "      | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"device\"][\"monitoring_active\"])'",
        "",
        "  Never do this on a unit protecting a driver: while stopped it monitors nothing.",
    ])


def service_state(service: str = SERVICE) -> Optional[str]:
    """`systemctl is-active` output, or None when systemctl is unavailable."""
    if shutil.which("systemctl") is None:
        return None
    try:
        out = subprocess.run(["systemctl", "is-active", service], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or "").strip() or None


def service_gate(state: Optional[str], *, allow_unknown: bool = False) -> tuple[bool, str]:
    """(ok, reason). Fail closed: an unknown state is a refusal unless overridden."""
    if state is None:
        if allow_unknown:
            return True, "service state unknown (systemctl unavailable); override given"
        return False, (f"cannot determine whether {SERVICE} is running (systemctl unavailable). "
                       "Refusing; pass --no-service-check only on a machine without the service.")
    if state in _BUSY_STATES:
        return False, f"{SERVICE} is '{state}': it may hold the device this script needs."
    return True, f"{SERVICE} is '{state}'"


def device_holders(device: str = "/dev/hailo0") -> Optional[list[int]]:
    """PIDs with `device` open, from /proc (read-only). None if /proc is absent.

    Processes of other users are invisible without root, so an empty list means
    "none that this user can see", not "none".
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    holders = []
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
            continue
        try:
            for fd in (pid_dir / "fd").iterdir():
                if os.readlink(fd) == device:
                    holders.append(int(pid_dir.name))
                    break
        except OSError:
            continue
    return holders


def verify_pin(actual: str, pinned: Optional[str]) -> tuple[bool, str]:
    """The .hef must match accelerator.yolo_hef_sha256. No pin is a refusal too."""
    pin = str(pinned or "").strip().lower()
    if not pin:
        return False, ("accelerator.yolo_hef_sha256 is not set in the config: the model on this "
                       "device cannot be identified, so nothing it outputs can be attributed.")
    if actual.lower() != pin:
        return False, (f"model sha256 {actual} does not match the pinned "
                       f"accelerator.yolo_hef_sha256 {pin}. Refusing (the runtime refuses it too).")
    return True, f"model sha256 matches the pin ({pin[:12]}...)"


def ensure_outside(path: Path, protected: Path) -> tuple[bool, str]:
    """Refuse to write inside the installed tree (or at it)."""
    p, root = Path(path).resolve(), Path(protected).resolve()
    if p == root or root in p.parents:
        return False, f"output {p} is inside the installed tree {root}; choose a directory outside it"
    return True, "ok"


def estimate_bundle_bytes(samples: int, input_hw=(640, 640), native: bool = False) -> int:
    """Upper bound for the uncompressed bundle (float32 heads at strides 8/16/32)."""
    h, w = input_hw
    cells = sum((h // s) * (w // s) for s in (8, 16, 32))
    per = h * w * 3 + cells * (68 + 81) * 4
    if native:
        per += cells * (68 + 81) * 2
    return samples * per


def check_free_space(path: Path, needed: int, margin: int = 1 << 30) -> tuple[bool, str]:
    target = Path(path)
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    if free < needed * 1.5 + margin:
        return False, (f"{free / 1e9:.1f} GB free at {target}; this capture needs about "
                       f"{needed / 1e9:.1f} GB (+50% and 1 GB margin). Use fewer images.")
    return True, f"{free / 1e9:.1f} GB free"


# ------------------------------------------------------------------ preprocessing
def letterbox_bgr_to_rgb(frame_bgr: np.ndarray, input_w: int, input_h: int):
    """EXACTLY the preprocessing in core.accel.HailoYOLO.predict (pinned by a test
    that captures the batch predict() hands to the pipeline).

    Returns (rgb uint8 HxWx3, geometry dict).
    """
    import cv2

    from core.accel import letterbox_transform

    h, w = frame_bgr.shape[:2]
    scale, left, top = letterbox_transform(w, h, input_w, input_h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = input_w - new_w, input_h - new_h
    boxed = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return rgb, {"mode": "letterbox", "scale": float(scale), "left": int(left), "top": int(top),
                 "frame_w": int(w), "frame_h": int(h), "input_w": int(input_w),
                 "input_h": int(input_h), "pad_value": 114, "interpolation": "INTER_LINEAR"}


def stretch_bgr_to_rgb(frame_bgr: np.ndarray, input_w: int, input_h: int):
    """The DAMO-YOLO reference preprocessing (keep_ratio False): plain resize, RGB."""
    import cv2

    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb, {"mode": "stretch", "frame_w": int(w), "frame_h": int(h),
                 "input_w": int(input_w), "input_h": int(input_h),
                 "interpolation": "INTER_LINEAR"}


PREPROCESSORS = {"letterbox": letterbox_bgr_to_rgb, "stretch": stretch_bgr_to_rgb}


# ------------------------------------------------------------------ provenance
def git_commit(root: Path) -> Optional[str]:
    """HEAD of a git checkout; for a `git archive` copy (no .git), the commit
    recorded next to it in HWVAL_COMMIT, prefixed "file:" (unverified claim)."""
    marker = Path(root) / "HWVAL_COMMIT"
    if not (Path(root) / ".git").exists() or shutil.which("git") is None:
        return f"file:{marker.read_text().strip()}" if marker.is_file() else None
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_dirty(root: Path) -> Optional[bool]:
    if not (Path(root) / ".git").exists() or git_commit(root) is None:
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def run_text(cmd: list[str], timeout: float = 15.0) -> Optional[str]:
    """stdout+stderr of a read-only command, or None if it is not available."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<failed: {exc}>"
    return ((out.stdout or "") + (out.stderr or "")).strip()


def environment() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None
    model = Path("/proc/device-tree/model")
    if model.exists():
        info["board"] = model.read_bytes().rstrip(b"\x00").decode(errors="replace")
    return info


def hailo_runtime_info() -> dict:
    """HailoRT version and device identity, read-only. Call BEFORE opening a VDevice."""
    info: dict = {}
    try:
        import hailo_platform

        info["hailo_platform_version"] = getattr(hailo_platform, "__version__", None)
        try:
            device_ids = hailo_platform.Device.scan()
            info["device_ids"] = [str(d) for d in device_ids]
        except Exception as exc:  # noqa: BLE001 - informational only
            info["device_ids_error"] = repr(exc)
    except ImportError as exc:
        info["hailo_platform_error"] = repr(exc)
    info["hailortcli_version"] = run_text(["hailortcli", "--version"])
    info["fw_control_identify"] = run_text(["hailortcli", "fw-control", "identify"])
    return info


def describe_vstream_info(info) -> dict:
    """Name, shape, format and quantisation of a HailoRT vstream info, defensively."""
    out = {"name": str(getattr(info, "name", "?")),
           "shape": [int(v) for v in getattr(info, "shape", ())]}
    fmt = getattr(info, "format", None)
    if fmt is not None:
        out["format"] = {k: str(getattr(fmt, k)) for k in ("type", "order", "flags")
                         if hasattr(fmt, k)}
    quant = getattr(info, "quant_info", None)
    if quant is not None:
        q = {}
        for k in ("qp_zp", "qp_scale", "limvals_min", "limvals_max"):
            if hasattr(quant, k):
                try:
                    q[k] = float(getattr(quant, k))
                except (TypeError, ValueError):
                    q[k] = str(getattr(quant, k))
        out["quant_info"] = q
    return out


# ------------------------------------------------------------------ bundle I/O
def write_json(path: Path, data) -> None:
    tmp = Path(str(path) + ".part")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False, default=str))
    tmp.replace(path)


def save_sample(path: Path, input_tensor: np.ndarray, outputs: list[np.ndarray],
                *, compressed: bool = True, prefix: str = "out") -> None:
    arrays = {"input": np.ascontiguousarray(input_tensor)}
    for i, arr in enumerate(outputs):
        arrays[f"{prefix}{i}"] = np.asarray(arr)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)


def load_bundle(bundle_dir: Path) -> dict:
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text())
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{bundle_dir}: not a {BUNDLE_FORMAT} bundle "
                         f"(format={manifest.get('format')!r})")
    return manifest


def load_sample(bundle_dir: Path, sample: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(input uint8 (1,H,W,3), {hef output name: float32 array as captured})."""
    names = [o["name"] for o in sample["outputs"]]
    with np.load(Path(bundle_dir) / sample["file"]) as data:
        inp = data["input"]
        outs = {name: data[f"out{i}"] for i, name in enumerate(names)}
    return inp, outs


# ------------------------------------------------------------------ tensor stats
def tensor_parity(a: np.ndarray, b: np.ndarray) -> dict:
    """a = device (int8 path, dequantised), b = float reference. Same shape."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = a - b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else None
    sa, sb = a.std(), b.std()
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else None
    return {
        "n": int(a.size),
        "mean_abs_err": float(np.abs(diff).mean()),
        "max_abs_err": float(np.abs(diff).max()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "mean_signed_err": float(diff.mean()),
        "cosine": cos,
        "pearson": corr,
    }


def range_check(arr: np.ndarray, role: str) -> dict:
    """Is a head carrying probabilities (class: sigmoid, in [0,1]) or logits (box)?"""
    a = np.asarray(arr, dtype=np.float64)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "fraction_in_0_1": float(((a >= 0.0) & (a <= 1.0)).mean()),
            "verdict": range_verdict(float(a.min()), float(a.max()), role)}


def range_verdict(lo: float, hi: float, role: str) -> str:
    inside = lo >= 0.0 and hi <= 1.0
    if role == "cls":
        return ("ok: probabilities (sigmoid applied)" if inside
                else "FAIL: class head has values outside [0,1] - not sigmoid'd")
    return ("SUSPECT: box head entirely in [0,1] - looks like probabilities" if inside
            else "ok: logits (values outside [0,1])")


class ParityAccumulator:
    """Streaming tensor_parity over many samples (no need to hold them all).

    a = device, b = float reference. Compares the first min(C_a, C_b) channels
    of the last axis; also keeps per-channel error, the worst single-sample
    cosine, and the value range of each side.
    """

    def __init__(self) -> None:
        self.n = 0
        self.sum_abs = self.sum_sq = self.max_abs = self.sum_signed = 0.0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0
        self.worst_cosine: Optional[float] = None
        self.worst_sample: Optional[str] = None
        self.ch_abs: Optional[np.ndarray] = None
        self.ch_max: Optional[np.ndarray] = None
        self.ch_n = 0
        self.ranges = {"a": [np.inf, -np.inf, 0, 0], "b": [np.inf, -np.inf, 0, 0]}

    def add(self, a: np.ndarray, b: np.ndarray, sample: str = "") -> None:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        c = min(a.shape[-1], b.shape[-1])
        a2 = a.reshape(-1, a.shape[-1])[:, :c]
        b2 = b.reshape(-1, b.shape[-1])[:, :c]
        if a2.shape != b2.shape:
            raise ValueError(f"{sample}: spatial shape mismatch {a.shape} vs {b.shape}")
        for key, arr in (("a", a2), ("b", b2)):
            r = self.ranges[key]
            r[0], r[1] = min(r[0], float(arr.min())), max(r[1], float(arr.max()))
            r[2] += int(((arr >= 0) & (arr <= 1)).sum())
            r[3] += int(arr.size)
        diff = a2 - b2
        abs_err = np.abs(diff)
        self.n += diff.size
        self.sum_abs += float(abs_err.sum())
        self.sum_sq += float((diff ** 2).sum())
        self.sum_signed += float(diff.sum())
        self.max_abs = max(self.max_abs, float(abs_err.max()))
        self.sa += float(a2.sum())
        self.sb += float(b2.sum())
        self.saa += float((a2 ** 2).sum())
        self.sbb += float((b2 ** 2).sum())
        self.sab += float((a2 * b2).sum())
        cos = tensor_parity(a2, b2)["cosine"]
        if cos is not None and (self.worst_cosine is None or cos < self.worst_cosine):
            self.worst_cosine, self.worst_sample = cos, sample
        if self.ch_abs is None or self.ch_max is None:
            self.ch_abs, self.ch_max = np.zeros(c), np.zeros(c)
        self.ch_abs += abs_err.sum(axis=0)
        self.ch_max = np.maximum(self.ch_max, abs_err.max(axis=0))
        self.ch_n += abs_err.shape[0]

    def summary(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        n = self.n
        cos_den = np.sqrt(self.saa) * np.sqrt(self.sbb)
        var_a = self.saa / n - (self.sa / n) ** 2
        var_b = self.sbb / n - (self.sb / n) ** 2
        cov = self.sab / n - (self.sa / n) * (self.sb / n)

        def _range(key: str) -> dict:
            lo, hi, inside, total = self.ranges[key]
            return {"min": lo, "max": hi, "fraction_in_0_1": inside / max(1, total)}

        return {
            "n": n,
            "mean_abs_err": self.sum_abs / n,
            "max_abs_err": self.max_abs,
            "rmse": float(np.sqrt(self.sum_sq / n)),
            "mean_signed_err": self.sum_signed / n,
            "cosine": float(self.sab / cos_den) if cos_den > 0 else None,
            "pearson": float(cov / np.sqrt(var_a * var_b)) if var_a > 0 and var_b > 0 else None,
            "worst_sample_cosine": self.worst_cosine,
            "worst_sample": self.worst_sample,
            "range_device": _range("a"),
            "range_float": _range("b"),
        }

    def per_channel(self, qp_scale: Optional[float] = None) -> list[dict]:
        if self.ch_abs is None or self.ch_max is None:
            return []
        rows = []
        for ch, (s, m) in enumerate(zip(self.ch_abs, self.ch_max, strict=True)):
            row = {"channel": ch, "mean_abs_err": float(s / self.ch_n), "max_abs_err": float(m)}
            if qp_scale:
                row["mean_abs_err_lsb"] = row["mean_abs_err"] / qp_scale
                row["max_abs_err_lsb"] = row["max_abs_err"] / qp_scale
            rows.append(row)
        return rows
