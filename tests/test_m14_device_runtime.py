"""M14: device-runtime truths that a Mac (and the fakes) used to hide.

* The `sound` sink on Linux plays through `aplay`. Its exit status used to be
  ignored, so a failed playback (a Pi 5 has no analog audio out) reported
  "played" and kept the sink -- and the device's audible-alert capability --
  ready. A non-zero exit is now a failed playback, reported by readiness().
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from core import alerts


@pytest.fixture
def linux_aplay(monkeypatch):
    """SoundSink as it is on the Pi: Linux, no simpleaudio, `aplay` on PATH."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(alerts.shutil, "which", lambda name: "/usr/bin/aplay" if name == "aplay" else None)
    calls: list[list[str]] = []
    result = {"code": 0, "stderr": ""}

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        assert kwargs.get("timeout"), "aplay must not be able to hang the alert thread"
        return subprocess.CompletedProcess(argv, result["code"], "", result["stderr"])

    monkeypatch.setattr(alerts.subprocess, "run", fake_run)
    sink = alerts.SoundSink()
    sink._simpleaudio = None
    return sink, calls, result


def test_a_failed_aplay_is_a_failed_playback_and_the_sink_is_not_ready(linux_aplay):
    sink, calls, result = linux_aplay
    result.update(code=1, stderr="ALSA lib pcm.c: Unknown PCM default\naplay: main:831: audio open error: No such file")
    why = sink._play_tone_safe(0.05)
    assert calls and calls[0][:2] == ["aplay", "-q"]
    assert why is not None and "aplay exited 1" in why and "audio open error" in why
    ready, detail = sink.readiness()
    assert ready is False and "last playback failed: aplay exited 1" in detail


def test_a_successful_aplay_is_played_and_clears_the_error(linux_aplay):
    sink, _calls, result = linux_aplay
    result.update(code=1)
    assert sink._play_tone_safe(0.05) is not None
    result.update(code=0, stderr="")
    assert sink._play_tone_safe(0.05) is None
    assert sink.readiness() == (True, "playback commands succeed (not physically verified)")


def test_the_temp_wav_is_removed_even_when_aplay_fails(linux_aplay, tmp_path, monkeypatch):
    sink, _calls, _result = linux_aplay
    monkeypatch.setattr(alerts.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        subprocess.TimeoutExpired(cmd="aplay", timeout=5)))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert sink._play_tone_safe(0.05) is not None
    assert list(tmp_path.glob("*.wav")) == []
    assert sink.readiness()[0] is False


# ---------------------------------------------------------------------- camera: which board is this?
class _BrokenPicamera2Finder:
    """A `picamera2` whose import raises `exc` (as a broken apt/numpy install can)."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def find_spec(self, name, path=None, target=None):
        if name != "picamera2":
            return None
        import importlib.util

        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise self.exc


@pytest.fixture
def broken_picamera2(monkeypatch):
    def install(exc: BaseException) -> None:
        monkeypatch.delitem(sys.modules, "picamera2", raising=False)
        monkeypatch.setattr(sys, "meta_path", [_BrokenPicamera2Finder(exc), *sys.meta_path])
    return install


def _board(monkeypatch, tmp_path, model: str | None) -> None:
    from core import camera

    path = tmp_path / "model"
    if model is not None:
        path.write_text(model + "\x00")
    monkeypatch.setattr(camera, "DT_MODEL", path)


@pytest.mark.parametrize("exc", [ImportError("No module named 'libcamera'"),
                                 ValueError("numpy.dtype size changed, may indicate binary incompatibility")])
def test_on_a_pi_a_broken_picamera2_is_the_cameras_reason_not_a_webcam_fallback(monkeypatch, tmp_path,
                                                                                 broken_picamera2, exc):
    from core import camera

    _board(monkeypatch, tmp_path, "Raspberry Pi 5 Model B Rev 1.0")
    broken_picamera2(exc)
    assert camera.detect_pi() is True
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *_a: pytest.fail("a Pi never opens the webcam path"))
    with pytest.raises(camera.CameraError) as raised:
        camera.Camera({"width": 640, "height": 480, "fps": 30})
    assert "could not start the Pi camera" in str(raised.value) and str(exc) in str(raised.value)
    assert "macOS" not in str(raised.value)


def test_off_a_pi_any_picamera2_import_failure_means_not_a_pi_and_never_crashes(monkeypatch, tmp_path,
                                                                                 broken_picamera2):
    from core import camera

    _board(monkeypatch, tmp_path, None)  # no device tree (a Mac, a CI box)
    broken_picamera2(RuntimeError("libcamera: no pipeline handler"))
    assert camera.detect_pi() is False
    _board(monkeypatch, tmp_path, "QEMU Virtual Machine")
    assert camera.detect_pi() is False
