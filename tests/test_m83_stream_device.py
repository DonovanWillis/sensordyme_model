"""M8.3: the device's live stream carries evidence the app can verify.

"A visible frame is not proof the stream is live." Before M8.3 every /stream
part was a bare JPEG: the app could not tell a frame captured 80 ms ago from
one captured an hour ago, and the device made that worse in two ways:

  * a new viewer was sent the CACHED last frame immediately, however old --
    after monitoring was stopped, or the camera froze, the phone got one frame
    from whenever, and labelled it LIVE;
  * while monitoring was stopped (or the module could not annotate) the
    runtime stopped pushing frames, so the stream froze on its last picture
    with the connection open (docs/PHASE7_COMPETITION_FREEZE.md: "the video is
    not proof of life").

These pin the device half: per-part X-Frame-Seq / X-Frame-Age-Ms /
X-Frame-Source / X-Stream-Id, stale-frame suppression, opt-in heartbeat parts
that say WHY no frame is coming, the camera fallback, auth unchanged, and the
bounded, drop-not-block behaviour toward slow clients.
"""

from __future__ import annotations

import itertools
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from core.dashboard import DashboardServer
from core.frame_bus import FrameBus
from sim.stream_client import StreamClient

TOKEN = "m83-token"


_NOISE = np.random.default_rng(83).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def _frame(value: int = 90, size: tuple[int, int] = (48, 64)) -> np.ndarray:
    if size[0] * size[1] > 10_000:
        # Large frames are noise: a flat frame compresses to ~2 KB and a few
        # seconds of them fit in the loopback socket buffers, so a "slow"
        # client would never actually make the server's send block.
        frame = _NOISE[: size[0], : size[1]].copy()
    else:
        frame = np.zeros((*size, 3), dtype=np.uint8)
        frame[:, :, 1] = value % 256
    frame[0, 0, 0] = value % 256  # every frame differs
    return frame


def _server(**kwargs) -> DashboardServer:
    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN, stream_fps=50, **kwargs)
    server.start()
    return server


class _Pusher:
    """Feeds push_frame (the runtime's annotated output) at a fixed rate."""

    def __init__(self, server: DashboardServer, fps: float = 30.0, age_s: float = 0.0,
                 size: tuple[int, int] = (48, 64)) -> None:
        self.server, self.interval, self.age_s, self.size = server, 1.0 / fps, age_s, size
        self.stop = threading.Event()
        self.durations: list[float] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        n = 0
        while not self.stop.is_set():
            n += 1
            started = time.perf_counter()
            self.server.push_frame(_frame(n, self.size), captured_at=time.monotonic() - self.age_s)
            self.durations.append(time.perf_counter() - started)
            self.stop.wait(self.interval)

    def __enter__(self) -> "_Pusher":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


class _Camera:
    """Publishes to a FrameBus like CaptureService does, without the runtime."""

    def __init__(self, bus: FrameBus, fps: float = 30.0) -> None:
        self.bus, self.interval = bus, 1.0 / fps
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        n = 0
        while not self.stop.is_set():
            n += 1
            self.bus.publish(_frame(n))
            self.stop.wait(self.interval)

    def __enter__(self) -> "_Camera":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def _status(url: str, headers: dict | None = None) -> int:
    request = urllib.request.Request(url, headers=headers or {})  # noqa: S310 - local test server
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------- metadata
def test_every_part_carries_a_monotonic_seq_its_capture_age_and_the_stream_id():
    server = _server()
    try:
        with _Pusher(server, fps=30, age_s=0.25), StreamClient(server._port, TOKEN) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 8)
        assert client.status == 200
        parts = client.jpeg_parts()
        seqs = [p.int_header("X-Frame-Seq") for p in parts]  # KeyError if any part lacks it
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"seq not strictly increasing: {seqs}"
        ages = [p.int_header("X-Frame-Age-Ms") for p in parts]
        # captured 250 ms before push: the age is the CAPTURE age, not the encode age
        assert all(250 <= a < 2000 for a in ages), ages
        stream_ids = {p.headers.get("x-stream-id") for p in parts}
        assert len(stream_ids) == 1 and client.headers.get("x-stream-id") in stream_ids
        assert {p.headers.get("x-frame-source") for p in parts} == {"annotated"}
    finally:
        server.close()


def test_old_clients_still_get_plain_well_framed_jpeg_parts_and_no_heartbeats():
    """Backwards compatible: extra part headers only; nothing new unless asked for."""
    server = _server()
    try:
        with StreamClient(server._port, TOKEN) as client:
            # no frames at all for 2.5 s: an opted-out client must see silence, as before
            client.read_for(2.5)
            assert client.status == 200
            assert client.parts == [], "a client that did not opt in received a non-image part"
            with _Pusher(server, fps=30):
                assert client.read_until(lambda c: len(c.parts) >= 5)
        assert client.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
        for part in client.parts:
            assert part.is_jpeg
            assert int(part.headers["content-length"]) == len(part.body)
            assert part.body[:2] == b"\xff\xd8" and part.body[-2:] == b"\xff\xd9"
    finally:
        server.close()


def test_a_new_viewer_is_never_sent_a_stale_cached_frame():
    """The old stream sent the last cached JPEG to every new viewer, however old.

    After a stop or a camera freeze the phone received exactly one picture from
    whenever and, with nothing to compare it to, called it LIVE.
    """
    server = _server()
    try:
        server.push_frame(_frame(), captured_at=time.monotonic() - 30.0)  # a 30 s old picture
        with StreamClient(server._port, TOKEN) as client:
            client.read_for(1.5)
        assert client.status == 200
        assert client.jpeg_parts() == [], "a 30 s old frame was served as if it were live"
        assert server.stream_stats()["stale_frames_withheld"] >= 1
    finally:
        server.close()


def test_frames_older_than_the_limit_are_withheld_mid_stream_too():
    server = _server()
    try:
        with StreamClient(server._port, TOKEN) as client:
            with _Pusher(server, fps=20, age_s=0.0):
                assert client.read_until(lambda c: len(c.jpeg_parts()) >= 3)
            fresh = len(client.jpeg_parts())
            with _Pusher(server, fps=20, age_s=DashboardServer.MAX_FRAME_AGE_S + 1.0):
                client.read_for(1.0)
        assert len(client.jpeg_parts()) == fresh, "frames past the age limit were still sent"
    finally:
        server.close()


# ---------------------------------------------------------------------- heartbeats
def test_heartbeats_say_the_camera_is_stale_when_the_viewer_opted_in():
    bus = FrameBus()
    bus.publish(_frame())  # one frame, then the camera stops
    server = _server()
    server.attach_frame_source(bus)
    try:
        time.sleep(DashboardServer.CAMERA_STALE_S + 0.2)
        with StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.heartbeats()) >= 2, timeout=5)
        assert client.headers.get("x-stream-heartbeat-interval-ms") is not None
        beat = client.heartbeats()[-1]
        assert beat.headers["x-stream-status"] == "camera_stale"
        assert beat.int_header("X-Camera-Age-Ms") >= DashboardServer.CAMERA_STALE_S * 1000
        assert beat.headers["x-stream-id"] == client.headers["x-stream-id"]
        assert client.jpeg_parts() == []
    finally:
        server.close()


def test_heartbeats_without_a_camera_source_say_no_fresh_frames():
    server = _server()
    try:
        with StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.heartbeats()) >= 1, timeout=5)
        assert client.heartbeats()[0].headers["x-stream-status"] == "no_fresh_frames"
    finally:
        server.close()


def test_no_heartbeat_while_frames_flow():
    server = _server()
    try:
        with _Pusher(server, fps=30), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            client.read_for(2.5)
        assert len(client.jpeg_parts()) >= 20
        assert client.heartbeats() == []
    finally:
        server.close()


# ---------------------------------------------------------------------- camera fallback
def test_the_camera_keeps_streaming_when_the_runtime_stops_annotating():
    """Monitoring stopped / module cannot annotate: video stays live, labelled as such."""
    bus = FrameBus()
    server = _server()
    server.attach_frame_source(bus)
    try:
        with _Camera(bus, fps=30), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 10, timeout=8)
        parts = client.jpeg_parts()
        assert {p.headers["x-frame-source"] for p in parts} == {"camera"}
        seqs = [p.int_header("X-Frame-Seq") for p in parts]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        assert all(p.int_header("X-Frame-Age-Ms") < 1000 for p in parts)
    finally:
        server.close()


def test_annotated_frames_win_while_the_runtime_is_pushing():
    bus = FrameBus()
    server = _server()
    server.attach_frame_source(bus)
    try:
        with _Camera(bus, fps=30), _Pusher(server, fps=20), StreamClient(server._port, TOKEN) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 10, timeout=8)
        sources = [p.headers["x-frame-source"] for p in client.jpeg_parts()]
        assert sources.count("annotated") == len(sources), sources
        assert server.stream_stats()["camera_fallback_encodes"] == 0
    finally:
        server.close()


def test_the_camera_fallback_is_encoded_once_for_all_viewers():
    bus = FrameBus()
    server = _server()
    server.attach_frame_source(bus)
    try:
        with _Camera(bus, fps=15):
            clients = [StreamClient(server._port, TOKEN) for _ in range(3)]
            readers = [threading.Thread(target=c.read_for, args=(2.0,)) for c in clients]
            for reader in readers:
                reader.start()
            for reader in readers:
                reader.join()
            for client in clients:
                client.close()
            published = bus.sequence
        encodes = server.stream_stats()["camera_fallback_encodes"]
        delivered = sum(len(c.jpeg_parts()) for c in clients)
        assert delivered >= 3 * 10
        # one encode per camera frame, shared -- never one per viewer
        assert encodes <= published, (encodes, published, delivered)
        assert delivered > encodes, "each encoded frame reached more than one viewer"
    finally:
        server.close()


def test_the_fallback_can_be_switched_off():
    bus = FrameBus()
    server = _server(camera_fallback=False)
    server.attach_frame_source(bus)
    try:
        with _Camera(bus, fps=30), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.heartbeats()) >= 1, timeout=5)
        assert client.jpeg_parts() == []
        assert client.heartbeats()[0].headers["x-stream-status"] == "no_fresh_frames"
    finally:
        server.close()


# ---------------------------------------------------------------------- auth (unchanged)
def test_stream_auth_is_unchanged():
    server = _server()
    try:
        base = f"http://127.0.0.1:{server._port}"
        with StreamClient(server._port, token=None) as anon:
            anon.read_for(0.5)
        assert anon.status == 401
        with StreamClient(server._port, token="wrong") as wrong:
            wrong.read_for(0.5)
        assert wrong.status == 401
        with StreamClient(server._port, token=None, query_token=TOKEN) as query:
            query.read_for(0.3)
        assert query.status == 200, "the <img> query-token path on GET /stream still works"
        # the new stats endpoint is /api: header only, never ?token=
        assert _status(base + "/api/stream/stats") == 401
        assert _status(base + f"/api/stream/stats?token={TOKEN}") == 401
        assert _status(base + "/api/stream/stats", {"X-Device-Token": TOKEN}) == 200
        # the heartbeat opt-in header grants nothing on its own
        with StreamClient(server._port, token=None, heartbeat=True) as beat:
            beat.read_for(0.5)
        assert beat.status == 401 and beat.parts == []
    finally:
        server.close()


# ---------------------------------------------------------------------- backpressure
def test_a_slow_client_never_blocks_the_producer_and_is_served_latest_frame_only():
    """Drop frames, not block capture: one slot, no per-client queue.

    A client with a tiny receive buffer that stops reading fills its socket; its
    handler thread blocks in send. The producer (push_frame, called by the
    inference loop) must not notice, a healthy client alongside must keep its
    frame rate, and when the slow client resumes it gets the LATEST frame --
    the ones in between are skipped, never queued.
    """
    server = _server()
    try:
        slow = StreamClient(server._port, TOKEN, recv_buffer=4096)
        fast = StreamClient(server._port, TOKEN)
        with _Pusher(server, fps=30, size=(240, 320)) as pusher:
            fast.read_for(3.0)  # the slow client reads nothing for 3 s
            slow.read_for(1.0)
        assert max(pusher.durations) < 0.25, f"push_frame blocked: {max(pusher.durations):.3f}s"
        assert len(fast.jpeg_parts()) >= 40, len(fast.jpeg_parts())
        seqs = [p.int_header("X-Frame-Seq") for p in slow.jpeg_parts()]
        assert seqs, "the slow client never got a frame"
        gaps = [b - a for a, b in itertools.pairwise(seqs)]
        assert any(g > 1 for g in gaps) or seqs[0] > 1, "frames were queued for the slow client, not dropped"
        stats = server.stream_stats()
        assert stats["frames_skipped"] > 0
        assert stats["max_pending_frames_per_client"] == 1
        slow.close()
        fast.close()
    finally:
        server.close()


def test_a_client_that_stops_reading_is_dropped_after_the_send_timeout():
    server = _server(stream_send_timeout_s=1.0)
    try:
        stuck = StreamClient(server._port, TOKEN, recv_buffer=2048)
        with _Pusher(server, fps=30, size=(480, 640)):
            assert _wait(server.has_viewers, 3)
            assert _wait(lambda: not server.has_viewers(), 10), "a stuck client held its viewer slot"
        assert server.stream_stats()["client_send_timeouts"] >= 1
        stuck.close()
    finally:
        server.close()


# ---------------------------------------------------------------------- lifecycle
def test_reconnect_keeps_the_stream_id_and_a_restart_resets_it():
    server = _server()
    try:
        with _Pusher(server, fps=30):
            with StreamClient(server._port, TOKEN) as first:
                assert first.read_until(lambda c: len(c.jpeg_parts()) >= 3)
            with StreamClient(server._port, TOKEN) as second:
                assert second.read_until(lambda c: len(c.jpeg_parts()) >= 3)
        assert first.headers["x-stream-id"] == second.headers["x-stream-id"]
        # same process: the seq carries on (the newest frame may be re-served once)
        assert second.jpeg_parts()[0].int_header("X-Frame-Seq") >= first.jpeg_parts()[-1].int_header("X-Frame-Seq")
    finally:
        server.close()
    rebooted = _server()  # a device reboot: new process, new server
    try:
        with _Pusher(rebooted, fps=30), StreamClient(rebooted._port, TOKEN) as third:
            assert third.read_until(lambda c: len(c.jpeg_parts()) >= 1)
        assert third.headers["x-stream-id"] != first.headers["x-stream-id"]
        assert third.jpeg_parts()[0].int_header("X-Frame-Seq") < first.jpeg_parts()[-1].int_header("X-Frame-Seq")
    finally:
        rebooted.close()


def test_closing_the_server_ends_open_streams_so_viewers_reconnect():
    """close() used to stop only ACCEPTING: an open viewer kept a silent socket
    to a server nobody fed (a health-monitor dashboard restart, the sim's
    in-process reboot) until its own idle timeout."""
    server = _server()
    with _Pusher(server, fps=30):
        client = StreamClient(server._port, TOKEN, heartbeat=True)
        assert client.read_until(lambda c: len(c.jpeg_parts()) >= 2)
        server.close()
        assert client.read_until(lambda c: c.closed_by_peer, timeout=5), "the stream outlived its server"
    client.close()
    assert _wait(lambda: not server.has_viewers())


def test_stream_stats_count_concurrent_connections():
    server = _server()
    try:
        assert server.stream_stats()["viewers"] == 0
        clients = [StreamClient(server._port, TOKEN) for _ in range(2)]
        assert _wait(lambda: server.stream_stats()["viewers"] == 2)
        assert server.stream_stats()["connections_total"] == 2
        for client in clients:
            client.close()
        assert _wait(lambda: server.stream_stats()["viewers"] == 0, 10)
        assert server.stream_stats()["connections_total"] == 2
    finally:
        server.close()
