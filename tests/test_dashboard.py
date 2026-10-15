"""Tests for the Drive Dashboard server (ephemeral port, no camera)."""

import json
import time
import urllib.request

import numpy as np
import pytest

from core.dashboard import FLASH_DURATION_S, DashboardServer
from core.events import Event


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def test_dashboard_serves_page_state_and_updates():
    server = DashboardServer(host="127.0.0.1", port=0, module_name="driver_attention_max")
    url = server.start()
    try:
        status, ctype, body = _get(url + "/")
        assert status == 200 and "text/html" in ctype
        assert b"SensorDyme" in body

        # feed it like the main loop does
        server.push_frame(np.zeros((120, 160, 3), dtype=np.uint8))
        server.push_event(
            Event.now("phone_use_while_distracted", 1.5, 0.95, "driver_attention_max", "phone + gaze_off"),
            severity="loud",
        )
        server.push_event(Event.now("vlm_judgment", 0.0, 0.9, "driver_attention_max", "vlm:attentive"), severity="log")
        server.push_status(alert_active=True, snapshot={"risk_score": 0.8, "ear": 0.21})

        status, ctype, body = _get(url + "/api/state")
        assert status == 200 and "application/json" in ctype
        state = json.loads(body)
        assert state["module"] == "driver_attention_max"
        assert state["alert_active"] is True
        assert state["total_events"] == 2
        assert state["events"][0]["type"] == "vlm_judgment"  # newest first
        assert state["events"][1]["severity"] == "loud"
        assert state["counts"] and state["status"]["risk_score"] == 0.8
        assert state["risk"] and state["risk"][-1] == 0.8

        jpeg, seq = server.latest_jpeg()
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8" and seq == 1  # valid JPEG magic
    finally:
        server.close()


def test_no_viewers_until_a_client_reads_stream():
    """The main loop uses has_viewers() to skip the annotate+encode path entirely."""
    server = DashboardServer(host="127.0.0.1", port=0)
    url = server.start()
    try:
        assert server.has_viewers() is False  # nobody watching -> main loop skips encoding

        resp = urllib.request.urlopen(url + "/stream", timeout=5)
        try:
            deadline = time.time() + 5
            while time.time() < deadline and not server.has_viewers():
                time.sleep(0.05)
            assert server.has_viewers() is True
        finally:
            resp.close()

        deadline = time.time() + 5  # closing the connection releases the viewer slot
        while time.time() < deadline and server.has_viewers():
            server.push_frame(np.zeros((32, 32, 3), dtype=np.uint8))  # unblock the writer
            time.sleep(0.05)
        assert server.has_viewers() is False
    finally:
        server.close()


def test_jpeg_quality_and_stream_fps_are_configurable():
    low = DashboardServer(host="127.0.0.1", port=0, jpeg_quality=20, stream_fps=5)
    high = DashboardServer(host="127.0.0.1", port=0, jpeg_quality=95)
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    low.push_frame(frame)
    high.push_frame(frame)
    assert len(low.latest_jpeg()[0]) < len(high.latest_jpeg()[0])  # quality actually applied
    assert low.stream_interval == 0.2  # 5 fps
    assert DashboardServer(host="127.0.0.1", port=0).stream_interval == 1.0 / 12  # default


def _post(url, payload, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_preset_api_lists_switches_and_rejects_unknown():
    presets = {
        "driver_distraction": {"title": "Driver Distraction", "module": "driver_attention"},
        "elderly_monitoring": {"title": "Elderly Monitoring", "module": "fall_detection"},
    }
    server = DashboardServer(host="127.0.0.1", port=0, presets=presets, active_preset="driver_distraction")
    url = server.start()
    try:
        _, _, body = _get(url + "/api/presets")
        state = json.loads(body)
        assert state["active"] == "driver_distraction"
        assert {p["name"] for p in state["presets"]} == set(presets)

        status, resp = _post(url + "/api/preset", {"preset": "elderly_monitoring"})
        assert status == 202 and resp["switching_to"] == "elderly_monitoring"
        assert server.pop_pending_preset() == "elderly_monitoring"  # main loop picks it up
        assert server.pop_pending_preset() is None  # consumed

        status, _ = _post(url + "/api/preset", {"preset": "nope"})
        assert status == 400
    finally:
        server.close()


def test_device_token_gates_api_when_set():
    server = DashboardServer(host="127.0.0.1", port=0, token="s3cret", presets={"a": {"module": "m"}})
    url = server.start()
    try:
        try:
            urllib.request.urlopen(url + "/api/state", timeout=5)
            assert False, "should have been rejected"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        req = urllib.request.Request(url + "/api/state", headers={"X-Device-Token": "s3cret"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
        status, _ = _post(url + "/api/preset", {"preset": "a"}, headers={"X-Device-Token": "s3cret"})
        assert status == 202
    finally:
        server.close()


def test_untokened_dashboard_fails_closed_on_a_network_bind():
    """A LAN-reachable dashboard with no token must refuse, not serve the camera.

    This is the regression guard for the open-camera default. `token: null` used
    to mean "no authentication", which on the Pi's `host: 0.0.0.0` meant the live
    feed and the whole /api surface were served to every stranger on the Wi-Fi --
    verified at the time by two unauthenticated 200s from another machine.

    The fix is deliberately NOT "bind 127.0.0.1": the phone fetches /stream from
    this dashboard directly, so refusing to listen on the LAN would delete the
    live feed rather than secure it. So the two halves are asserted together --
    loopback still serves without a token (a real convenience), a network bind
    without a token refuses everything (/api AND /stream, which is the one that
    actually shows a picture).
    """
    loopback = DashboardServer(host="127.0.0.1", port=0)
    url = loopback.start()
    try:
        status, _, _ = _get(url + "/api/state")
        assert status == 200, "loopback with no token should still be a convenience"
    finally:
        loopback.close()

    exposed = DashboardServer(host="0.0.0.0", port=0)  # noqa: S104 - that IS the case under test
    url = exposed.start()
    try:
        for path in ("/api/state", "/api/health", "/stream"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url + path, timeout=5)  # noqa: S310 - fixed http:// URL
            code = caught.value.code
            assert code == 401, f"{path} answered {code} to an unauthenticated caller"
        # The page itself is harmless and stays reachable, so the operator can
        # see *why* -- a blank refusal with no explanation is how this gets
        # debugged as "the feed is broken" instead of "set a token".
        status, _, _ = _get(url + "/")
        assert status == 200
    finally:
        exposed.close()


def test_history_reads_events_csv(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "timestamp,type,duration_s,confidence,module,contributing_signals\n"
        "2026-07-16T10:00:00+00:00,drowsiness,2.000,0.900,driver_attention,sustained\n"
        "2026-07-16T10:01:00+00:00,phone_use,1.500,0.950,driver_attention,phone+gaze\n"
    )
    server = DashboardServer(host="127.0.0.1", port=0, events_csv_path=str(csv_path))
    url = server.start()
    try:
        _, _, body = _get(url + "/api/history?limit=10")
        events = json.loads(body)["events"]
        assert len(events) == 2
        assert events[0]["type"] == "phone_use"  # newest first
    finally:
        server.close()


def test_health_endpoint_serves_the_monitor_snapshot():
    server = DashboardServer(
        host="127.0.0.1", port=0,
        health_provider=lambda: {"state": "ok", "components": {"camera": {"state": "ok"}}},
    )
    url = server.start()
    try:
        _, ctype, body = _get(url + "/api/health")
        assert "application/json" in ctype
        health = json.loads(body)
        assert health["state"] == "ok"
        assert health["components"]["camera"]["state"] == "ok"
    finally:
        server.close()


def test_health_endpoint_without_provider_is_unknown():
    server = DashboardServer(host="127.0.0.1", port=0)
    url = server.start()
    try:
        _, _, body = _get(url + "/api/health")
        assert json.loads(body) == {"state": "unknown"}
    finally:
        server.close()


def test_apply_wifi_dry_runs_off_pi():
    ok, message = DashboardServer(host="127.0.0.1", port=0).apply_wifi("HomeNet", "pw")
    assert ok and "dry-run" in message


def test_flash_is_edge_triggered_not_continuous():
    """Every push_event() call arms a brief flash window; is_alert_active-style
    continuous state must never drive it -- see services/runtime.py's own
    edge-triggered overlay for the video-frame counterpart. push_event uses
    the real wall clock internally (like push_status's risk sampling already
    does), so this asserts relative to a timestamp captured around the call
    rather than injecting now=0.0."""
    server = DashboardServer(host="127.0.0.1", port=0)
    try:
        before = time.time()
        assert server.flash_active(now=before) is False
        server.push_event(Event.now("holding_object", 1.0, 0.9, "baby_monitoring", "object=phone"), "log")
        assert server.flash_active(now=before) is True  # armed the instant the event lands
        assert server.flash_active(now=before + FLASH_DURATION_S - 0.05) is True
        assert server.flash_active(now=before + FLASH_DURATION_S + 0.5) is False  # flashed, then back to normal
    finally:
        server.close()


def test_flash_is_not_gated_on_severity():
    """The product requirement is 'every new event flashes', not just loud
    ones -- a log-severity event must arm the flash exactly like a loud one."""
    server = DashboardServer(host="127.0.0.1", port=0)
    try:
        server.push_event(Event.now("person_count_change", 0.0, 0.9, "baby_monitoring"), "log")
        assert server.flash_active(now=0.0) is True
    finally:
        server.close()


def test_state_exposes_flash_active():
    server = DashboardServer(host="127.0.0.1", port=0)
    url = server.start()
    try:
        _, _, body = _get(url + "/api/state")
        assert json.loads(body)["flash_active"] is False

        server.push_event(Event.now("fall", 0.0, 0.9, "fall_detection"), "loud")
        _, _, body = _get(url + "/api/state")
        assert json.loads(body)["flash_active"] is True
    finally:
        server.close()


def test_a_second_event_within_the_flash_window_extends_it_not_stacks_it():
    """Two events in quick succession must read as one continuous flash window
    (extended), not stack/queue -- the spec is 'flash, then return to normal'
    per event, and extension is what keeps a burst of events from looking
    broken (flickering on/off between two flashes 50ms apart)."""
    server = DashboardServer(host="127.0.0.1", port=0)
    try:
        before = time.time()
        server.push_event(Event.now("holding_object", 1.0, 0.9, "baby_monitoring"), "log")
        server.push_event(Event.now("holding_object", 1.0, 0.9, "baby_monitoring"), "log")
        assert server.flash_active(now=before + FLASH_DURATION_S - 0.05) is True
    finally:
        server.close()


def test_dashboard_unknown_path_is_404():
    server = DashboardServer(host="127.0.0.1", port=0)
    url = server.start()
    try:
        try:
            urllib.request.urlopen(url + "/nope", timeout=5)
            raised = False
        except urllib.error.HTTPError as exc:
            raised = exc.code == 404
        assert raised
    finally:
        server.close()


def _status(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as resp:  # noqa: S310
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_a_numeric_token_from_yaml_still_authenticates():
    """`token: 123456` loads as an int; compare_digest(str, int) used to raise on
    every request, dropping the connection instead of answering."""
    server = DashboardServer(host="127.0.0.1", port=0, token=123456)
    url = server.start()
    try:
        assert _status(url + "/api/state", {"X-Device-Token": "123456"}) == 200
        assert _status(url + "/api/state", {"X-Device-Token": "nope"}) == 401
    finally:
        server.close()


def test_a_non_ascii_token_header_is_refused_not_crashed():
    server = DashboardServer(host="127.0.0.1", port=0, token="s3cret")
    url = server.start()
    try:
        assert _status(url + "/api/state", {"X-Device-Token": "café"}) == 401
    finally:
        server.close()


def test_the_query_token_is_url_decoded():
    """The page sends encodeURIComponent(TOKEN); a token with + / = must match."""
    import urllib.parse

    token = "ab+c/d= e"
    server = DashboardServer(host="127.0.0.1", port=0, token=token)
    url = server.start()
    try:
        assert _status(url + "/api/state?token=" + urllib.parse.quote(token, safe="")) == 200
    finally:
        server.close()


def test_event_text_is_escaped_before_it_reaches_innerhtml():
    """Event signals carry rule descriptions written by the AI rule interpreter.
    Rendered raw into innerHTML they were markup the page would execute, on the
    origin that holds the dashboard token in its URL."""
    import re

    from core import dashboard

    page = dashboard._PAGE
    assert "function esc(" in page
    # Every field interpolated into the event, count and mode templates is
    # escaped. The one bare ${...} left picks between two constant CSS values.
    for field in ("e.type", "e.signals", "e.time", "e.severity", "c.type", "m.title", "m.description", "m.name"):
        assert re.search(r"\$\{" + re.escape(field), page) is None, f"{field} reaches innerHTML unescaped"
        assert f"esc({field})" in page
    assert "setMode('${" not in page, "a preset name must not be spliced into an inline JS string"
