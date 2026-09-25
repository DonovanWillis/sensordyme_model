"""The /stream endpoint's viewer accounting must survive a client that leaves quietly.

`has_viewers()` is the gate the inference loop uses to decide whether to annotate
and JPEG-encode at all (services/runtime.py). On the Pi that work competes with
the detector for the same cores, so a viewer slot that is never released does not
merely mis-report a number -- it permanently spends CPU the product needs for
inference, for the rest of the device's uptime.

The failure these tests pin down: the streaming loop only ever learned that a
client had gone away by having a write FAIL. During any window with no fresh
frames -- a camera stall, a degraded detector, or simply the moment before the
first frame is encoded for a newly connected viewer -- there is no write, so a
departed client is never noticed. The handler thread spins forever and its
viewer slot is never returned.
"""

import socket
import threading
import time

import numpy as np

from core.dashboard import DashboardServer


def _stream_socket(server: DashboardServer) -> socket.socket:
    """Connect to /stream the way the app does, and wait until it is registered."""
    sock = socket.create_connection(("127.0.0.1", server._port), timeout=5)
    sock.sendall(b"GET /stream HTTP/1.1\r\nHost: device\r\n\r\n")
    deadline = time.time() + 5
    while time.time() < deadline and not server.has_viewers():
        time.sleep(0.02)
    assert server.has_viewers() is True, "the client never registered as a viewer"
    return sock


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_viewer_slot_is_released_when_a_client_leaves_during_a_camera_stall():
    """No frames are flowing -- exactly when a departed client used to go unnoticed."""
    server = DashboardServer(host="127.0.0.1", port=0)
    server.start()
    try:
        sock = _stream_socket(server)
        # Deliberately push NOTHING: this models a stalled camera, a degraded
        # detector, or the gap before the first encode for a new viewer.
        sock.close()

        assert _wait_until(lambda: not server.has_viewers()), (
            "viewer slot was never released: the device will annotate and JPEG-encode "
            "every frame forever, competing with inference on the Pi's cores"
        )
    finally:
        server.close()


def test_stream_handler_thread_exits_when_a_client_leaves_during_a_stall():
    """The leaked viewer slot is also a leaked thread -- one per abandoned connection."""
    server = DashboardServer(host="127.0.0.1", port=0)
    server.start()

    def handler_threads() -> int:
        return sum(1 for t in threading.enumerate() if t.name.startswith("Thread-"))

    try:
        before = handler_threads()
        socks = [_stream_socket(server) for _ in range(3)]
        assert handler_threads() > before
        for sock in socks:
            sock.close()

        assert _wait_until(lambda: handler_threads() <= before), (
            "stream handler threads never exited; an app that reconnects repeatedly "
            "parks one thread per attempt on the device"
        )
    finally:
        server.close()


def test_viewer_slot_is_released_on_a_normal_disconnect_while_frames_flow():
    """The path that already worked must keep working."""
    server = DashboardServer(host="127.0.0.1", port=0)
    server.start()
    stop = threading.Event()

    def feeder():
        while not stop.is_set():
            if server.has_viewers():
                server.push_frame(np.zeros((32, 32, 3), dtype=np.uint8))
            time.sleep(0.02)

    thread = threading.Thread(target=feeder, daemon=True)
    thread.start()
    try:
        sock = _stream_socket(server)
        time.sleep(0.3)
        sock.close()
        assert _wait_until(lambda: not server.has_viewers())
    finally:
        stop.set()
        thread.join(timeout=5)
        server.close()


def test_repeated_connect_and_abandon_does_not_accumulate_viewers():
    """Models the app reconnecting: each attempt is cancelled before the first frame."""
    server = DashboardServer(host="127.0.0.1", port=0)
    server.start()
    try:
        for _ in range(8):
            sock = socket.create_connection(("127.0.0.1", server._port), timeout=5)
            sock.sendall(b"GET /stream HTTP/1.1\r\nHost: device\r\n\r\n")
            time.sleep(0.02)
            sock.close()

        assert _wait_until(lambda: not server.has_viewers()), (
            "abandoned reconnect attempts accumulated viewer slots that are never returned"
        )
    finally:
        server.close()
