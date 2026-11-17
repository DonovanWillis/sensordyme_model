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
