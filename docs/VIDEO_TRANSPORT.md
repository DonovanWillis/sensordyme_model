# Video transport: camera → app (M8.3)

The live picture travels **directly from the device to the phone over the LAN**.
The backend never relays or stores video (privacy by design). This document
traces every hop with file:line references (as of the M8.3 commits on
`feat/m83-stream`), says what each hop has been proven on, and specifies the
stream-liveness evidence added in M8.3.

> "A visible frame is not proof the stream is live." Before M8.3 the app's LIVE
> badge followed "the last frame decoded", and the device could freeze the
> stream with the connection open. `docs/PHASE7_COMPETITION_FREEZE.md` recorded
> it on hardware: *"the live MJPEG stream freezes on the last frame rather than
> going dark when degraded, so the video is not proof of life"*.

## 1. The path

```
Camera Module 3 ─Picamera2─▶ Camera.read()            core/camera.py:158
   CaptureService.run() ── publish ──▶ FrameBus        services/capture.py:137, :177 · core/frame_bus.py:61
   inference loop: wait_for_fresh_stamped()           services/runtime.py:2851
      annotate() only while has_viewers()             services/runtime.py:2965, :2986
      push_frame(view, captured_at)  → JPEG (q60)     services/runtime.py:3009 · core/dashboard.py:779, :788
   [M8.3] camera fallback: FrameBus → JPEG (shared)   core/dashboard.py:694, :717
DashboardServer (ThreadingHTTPServer, ≤64 conns)      core/dashboard.py:77
   GET /stream → _authorized() → serve_stream()       core/dashboard.py:501, :319, :621
      one shared latest-frame slot, per-viewer thread, part headers   core/dashboard.py:156, :187
──────────── LAN (HTTP/1.1, multipart/x-mixed-replace; boundary=frame) ────────────
iPhone: DeviceLink (Bonjour + cached address)         ios/Sources/DeviceLink.swift:152 (resolve), :102 (path change)
   CameraSource → MJPEGView(url, session)             ios/Sources/CameraViews.swift:54, :110, :247
   LiveFeedSession → SharedFeedCoordinator (leases)    ios/Sources/FeedViews.swift:197 · ios/Sources/StreamLiveness.swift:364
   MJPEGReader: URLSession, X-Device-Token header      ios/Sources/FeedViews.swift:444, :569 · ios/Sources/MJPEGStream.swift:413
   per-part response → MJPEGResponse.classify (seq, age, heartbeat)   ios/Sources/FeedViews.swift:684 · MJPEGStream.swift:195
   MJPEGFrameParser.appendParts (SOI/EOI; in-band headers if any)    ios/Sources/MJPEGStream.swift:274
   FeedLivenessTracker → FeedLivenessModel            ios/Sources/StreamLiveness.swift:79, :196 · FeedViews.swift:318
   UIImage(data:) → MJPEGFrame; MJPEGLiveIndicator     ios/Sources/FeedViews.swift:726, :326, :390
```

### Hop by hop

| # | Hop | Where | Proven on | Evidence |
|---|---|---|---|---|
| 1 | Camera capture (Picamera2, 640×480) | `core/camera.py:52`, `:158` | **hardware** | `docs/PERFORMANCE_BUDGET.md` §1: Pi 5 + Camera Module 3 Wide NoIR, live pipeline 9.5 min sustained. The virtual device replaces this hop with a renderer (`sim/camera.py`). |
| 2 | Capture pump → FrameBus (latest slot, reopen on failure) | `services/capture.py:137–177`, `core/frame_bus.py:61` | **hardware** (same 9.5 min run) + tests | `tests/test_capture_service.py`, `tests/test_frame_bus.py` |
| 3 | Inference → annotate → JPEG encode (only while someone watches, ≤ `stream_fps`) | `services/runtime.py:2965–3009`, `core/dashboard.py:779` | **hardware** (pre-M8.3 code) | PERFORMANCE_BUDGET: "MJPEG stream — 79 real JPEG frames in 8 s (~10/s) against `stream_fps: 12`". The encode's CPU share was **not** separated from inference on the Pi. |
| 3b | **M8.3** camera fallback (unannotated, while the runtime is not drawing) | `core/dashboard.py:694–729` | **simulator only** | `tests/test_m83_stream_runtime.py`, `scripts/sim/stream_soak.py` (`monitoring_stopped`) |
| 4 | `/stream` auth: `X-Device-Token`, `?token=` on GET /stream only, constant-time compare, fail closed on a LAN bind without a token | `core/dashboard.py:319–360` | **hardware** (Phase 8) + tests | `docs/PHASE8_85_POINT_HARDENING.md`: from another machine `/stream` without token **401**, `/stream?token=` **200**. The *header* path (what the app uses) is covered by tests only (`tests/test_m6_security_device.py`, `tests/test_m83_stream_device.py::test_stream_auth_is_unchanged`). |
| 5 | Multipart writer, per-viewer thread, one shared slot | `core/dashboard.py:621–692` | **hardware** (pre-M8.3 writer) / **simulator only** (M8.3 headers, heartbeats, stale withholding, send timeout) | as 3; `tests/test_m83_stream_device.py` |
| 6 | LAN discovery (Bonjour `_sensordyme._tcp`, avahi on the Pi) | `ios/Sources/DeviceLink.swift`, `Discovery.swift` | **simulator only** | `docs/VIRTUAL_DEVICE.md`: `--mdns` visible to `dns-sd -B` on the Mac. "avahi advertising on a real LAN" is listed under *Still needs the physical Pi*. |
| 7 | App connection: URLSession, header token, retry/backoff, re-resolve | `ios/Sources/FeedViews.swift:444–770`, `MJPEGStream.swift` | **never verified end to end** (no real iPhone ↔ Pi run; the 2026-11-11 Simulator e2e explicitly did not cover the live feed) | Logic: `ios/Tests/MJPEGStreamTests.swift`, `StreamLivenessTests.swift`. M8.3: the same Foundation-only logic over a real URLSession on macOS against the virtual device (`scripts/sim/stream_probe.sh`). |
| 8 | Parse + decode (`UIImage(data:)`) + presentation | `MJPEGStream.swift:274`, `FeedViews.swift:669–745`, `:326`, `:390` | **never verified** on a device; parser logic tested | UIKit decoding and SwiftUI drawing are not exercised by any test here (XCUITest has no live-feed journey). |

What only the Pi and a real iPhone can prove is listed in §7.

## 2. The wire format (M8.3)

Response headers on `GET /stream` (200):

| Header | Meaning |
|---|---|
| `Content-Type: multipart/x-mixed-replace; boundary=frame` | unchanged |
| `Cache-Control: no-store` | never cached (also on every `/api` JSON answer) |
| `X-Stream-Id: 1a2b3c4d` | random per dashboard process; a change = the device restarted, seq restarts |
| `X-Stream-Max-Frame-Age-Ms: 5000` | frames older than this are never sent |
| `X-Stream-Heartbeat-Interval-Ms: 1000` | only when the request asked for heartbeats |

Each JPEG part — the first three lines are exactly what the stream always sent,
so pre-M8.3 clients (old app, browser `<img>`) parse it unchanged and ignore the rest
(`core/dashboard.py:156`):

```
--frame
Content-Type: image/jpeg
Content-Length: 31207
X-Frame-Seq: 4812          strictly increasing per X-Stream-Id (annotated and camera frames share it)
X-Frame-Age-Ms: 87         camera capture → this write, on the device's monotonic clock (transit not included)
X-Frame-Source: annotated  "annotated" = the analysed view; "camera" = the camera without analysis
X-Stream-Id: 1a2b3c4d

<JPEG>
```

Heartbeat part, **only** if the request carried `X-Stream-Heartbeat: 1` (the
app does; a browser `<img>` and the old app do not), after
`HEARTBEAT_INTERVAL_S` (1 s) with no frame sent (`core/dashboard.py:187`):

```
--frame
Content-Type: text/plain
Content-Length: 0
X-Stream-Heartbeat: 1
X-Stream-Id: 1a2b3c4d
X-Stream-Status: camera_stale | camera_not_started | no_fresh_frames
X-Frame-Seq: 4812          last seq this viewer was sent
X-Camera-Age-Ms: 6150      age of the camera's newest frame (when a camera source is attached)
```

The opt-in header grants nothing: auth is unchanged (header everywhere,
`?token=` on GET `/stream` only — `docs/SECURITY_REVIEW_M6.md` S1). Pages
whose URL can carry `?token=` (`/`, `/setup`) send `Referrer-Policy: no-referrer`.

### Device-side behaviour

| Rule | Constant | Why |
|---|---|---|
| Never send a frame captured > 5 s ago | `DashboardServer.MAX_FRAME_AGE_S` | the slot held the last picture from before a stop / freeze and handed it to every new viewer, which called it LIVE |
| While the runtime pushes no annotated frame for 1.5 s, stream the camera unannotated (`X-Frame-Source: camera`), encoded **once** for all viewers, non-blocking | `ANNOTATION_GAP_S`, `_current_frame()` | monitoring stopped (the inference loop returns before reading frames, `services/runtime.py:2822`), a degraded module, `annotate()` failing — the stream froze with the socket open. Video can be live while monitoring is stopped; the header says so |
| `camera_stale` once the camera's newest frame is > 2 s old | `CAMERA_STALE_S` | device reachable, camera not producing ≠ network problem |
| One shared latest-frame slot; a viewer thread sends the newest frame newer than its last; frames published while it was blocked are **skipped** (`frames_skipped`), never queued | `serve_stream()` | a slow phone must not grow memory or slow capture; `push_frame` only swaps a reference under a lock |
| A write blocked for 30 s drops the viewer (thread + slot freed) | `_STREAM_SEND_TIMEOUT_S` | a client that stopped reading entirely |
| `close()` ends every open stream | `_closing` | after a dashboard restart viewers reconnect at once instead of holding a silent socket |
| One write per part (header + JPEG + CRLF) | | the handler's `wfile` is unbuffered: four writes were four syscalls per frame |
| Counters at `GET /api/stream/stats` (token header) | `stream_stats()` | viewers, connections, frames sent/skipped, heartbeats, stale withheld, send timeouts, fallback encodes |

Overhead of the metadata: ~120 bytes per part (the four `X-` lines), against
a 640×480 q60 JPEG of tens of KB — measured below. CPU: one f-string per part.

## 3. App-side liveness model

`ios/Sources/StreamLiveness.swift` (Foundation-only, `ios/Tests/StreamLivenessTests.swift`).
`FeedLivenessTracker` takes the transport state (`MJPEGStreamState`) and the
device's evidence and decides:

| `FeedLiveness` | When | Badge (`FeedIndicator`) |
|---|---|---|
| `.live` | connected and a frame advanced (new seq, `X-Frame-Age-Ms` ≤ 2000) within the last 2 s | **LIVE**; + "Monitoring paused" when the device reports an explicit stop; + "Not analysed" when frames are `camera` while monitoring runs |
| `.stalled(seconds)` | connected, no advancing frame for > 2 s: no part, the **same seq** repeated (frozen source), or frames too old | "NO NEW FRAMES · N s old"; the last picture is faded |
| `.cameraUnavailable` | connected, a recent heartbeat says `camera_stale` / `camera_not_started` | "CAMERA NOT SENDING" |
| `.unreachable(reason)` | the transport is reconnecting (the reason from `MJPEGStreamDiagnosis`) | the picture area's own message ("reconnecting… — …") |
| `.refused` | 401/403 or an address that is not a URL | the picture area's message (needs a person) |
| `.connecting` | opening, or connected < 2 s with no fresh frame yet | spinner |
| `.idle` / `.disabled` | not watching / no live source for this device (`CameraSource.none` → `CameraPlaceholder`) | placeholder |

Only `.live` may say LIVE (`FeedIndicator.resolve`, tested for every other
state). Video and monitoring are separate facts: **video can be live while
monitoring is stopped**, and the badge says "Monitoring paused" (the stop comes
from the device's monitoring state, `InstrumentState.Kind.stopped`, passed by
`CameraViews.swift:120/:273`).

Further rules:

- **Old firmware** (no `X-Frame-Seq`): one frame proves nothing (the old
  device sends its cached frame to every new viewer); two frames within 2 s do.
- **Reboot**: a new `X-Stream-Id` (or a backwards seq without one) is a new
  baseline, counted in `streamResets`, never a stall.
- **Evidence does not survive a reconnect**: after a drop, not live until a new
  fresh frame arrives on the new connection.
- **Duplicates** (same seq) are not even decoded.
- **Silence** from a device that advertised heartbeats (nothing at all for 5 s)
  means a dead socket (network change, reboot without FIN): reconnect at once
  instead of the 15 s idle timeout. Old firmware is left to the idle timeout.

### Instrumentation

`StreamMetrics` (plain struct inside the reader — never `@Published`; read
with `MJPEGReader.metrics` / `metricsSummary()`): connection opens, reconnects,
transport errors, frames received / advanced / duplicate / stale, frames the
device skipped for us, stream resets, heartbeats, decode failures, bytes,
parser high-water, last and median frame age, delivered fps over a 32-frame
window. Liveness is published through `FeedLivenessModel` **only when it
changes**; the badge observes that, not the per-frame reader.

## 4. Lifecycle

| Concern | Implementation |
|---|---|
| Bounded connect | `MJPEGReaderPolicy.connectTimeout` = 6 s from request to response headers (`FeedViews.swift:614`); URLSession's only timeout was the 15 s idle one |
| Explicit errors | unchanged `MJPEGStreamState` messages (token, address, unreachable with reason, camera stalled) + liveness badge |
| Teardown | `invalidateAndCancel`, parser reset, connect-timeout task cancelled; stale callbacks dropped by `CurrentConnection` |
| **Exactly one connection** | `LiveFeedSession` handles (`FeedViews.swift:197`) all share one static `MJPEGReader` and one `SharedFeedCoordinator`. Leases (not a bare counter), 0.6 s grace for view swaps. Before M8.3 the session was per screen: the zone editor's own `MJPEGView` opened a **second** connection while the device screen's stayed open under the sheet. Also `httpMaximumConnectionsPerHost = 1` per session. |
| Rapid view changes | 200 inline↔full-screen swaps in either appear/disappear order: 1 open, 0 closes (`StreamLivenessTests`); on the virtual device, 40 swaps leave **1** device viewer (probe, §5) |
| Backgrounding | `scenePhase == .background` → `coordinator.suspend()` (one close, whatever the view count); `.active` → `resume()` (one open) |
| Network change | `DeviceLink` re-resolves on a path change and writes the new address → `MJPEGView.onChange(of: url)` → `retarget`; a socket that silently died is replaced after 5 s of silence |
| Device reboot | the stream ends (process exit, or `close()` now ends open streams), backoff reconnect 0.5→8 s, new `X-Stream-Id`, seq from 1 — handled as a reset |
| Backpressure | §2: latest-frame slot, skip not queue, 30 s send timeout; app parser buffer capped at 4 MB, header/heartbeat bodies at 4 KB |

## 5. Measurements — SIMULATOR (virtual device on a Mac)

See §6 for the numbers. All produced by `scripts/sim/stream_soak.py` and
`scripts/sim/stream_probe.sh` against `python -m sim.virtual_device`
(scripted 640×480 camera at 15 fps, real runtime + dashboard). **None of these
are Raspberry Pi numbers**: the Pi's JPEG encode competes with inference on 4
A76 cores, and Wi-Fi is not modelled (loopback).

## 6. Results (SIMULATOR, 2026-11-13)

Raw data: `~/sensordyme-audit/2026-11-10/m83-evidence/stream/`
(`stream_measurements.json`, `swift_probe*.json`, `header_overhead.txt`,
`urlsession_multipart_behaviour.txt`). The host was an 8-core Mac shared with
other workstreams' processes (load average 8–12 during the run,
`host_load_during_soak.txt`), so CPU figures are indicative only.

**Delivery and age** (`scripts/sim/stream_soak.py`, 15 s per phase):

| Phase | Delivered fps per viewer | Inter-arrival p50 / p95 | Frame age p50 / p95 / max | Device CPU (1 core = 100 %) | Device RSS |
|---|---|---|---|---|---|
| no viewer | — | — | — | 3.0 % | 66.6 MB |
| 1 viewer | 7.3 | 171 / 178 ms | 52 / 93 / 123 ms | 7.9 % | 61.6 MB |
| 3 viewers | 7.2 each | 171 / 177 ms | 52 / 90 / 107 ms | 6.7 % | 60.7 MB |
| monitoring **stopped** (camera fallback) | 10.0 | 92 / 104 ms | 39 / 73 / 163 ms | — | — |
| resumed | 7.2 | 172 / 187 ms | 56 / 93 / 142 ms | — | — |

- The ~7.2 fps is the virtual device's annotate cadence (its scripted
  perception runs in the inference loop), below `stream_fps: 12`. While
  stopped, the fallback runs at the stream cadence instead (10 fps). All parts
  had `X-Frame-Source: camera` while stopped and `annotated` after resume (the
  first part after a viewer connects may be `camera`, before the runtime has
  drawn for it).
- Viewers 1 → 3 did not raise device CPU measurably: the annotate + encode is
  done once per frame for all viewers; each viewer only costs a socket write.
- **Metadata overhead**: header 149 B vs 60 B before (+89 B, **0.46 %** of a
  19.5 KB frame); building it costs 0.84 µs per part (`header_overhead.txt`).

**Reconnect**: connect → first fresh frame p50 **1.4 ms**, p95 8.9 ms (20
reconnects, loopback — Wi-Fi not modelled).

**Slow client** (8 KB receive buffer, reads once per second) beside a normal
one: the normal viewer kept 7.1 fps (gap p95 179 ms), device RSS −2.3 MB, 0
send timeouts. On loopback the 1 s pauses were absorbed by socket buffers
(client buffer peaked at 156 KB), so no frames needed skipping there; skipping
under real backpressure is proven by
`tests/test_m83_stream_device.py::test_a_slow_client_never_blocks_the_producer_and_is_served_latest_frame_only`
(640×480 noise frames: producer never blocked > 250 ms, slow viewer's seq
jumps, `frames_skipped > 0`, at most 1 pending frame per client by construction).

**Soak**: 10 minutes, one viewer: 4 212 frames (7.02 fps), device RSS
41.2 → 38.1 MB (max 41.4; no growth), median CPU 8.5 %, client buffer max
19.9 KB, 0 send timeouts.

**Camera freeze** (`camera_freeze` scenario, 20 s without frames): the
connection stayed open; 19 heartbeats, `camera_stale` from a camera age of
2.5 s; longest gap between JPEG parts 19.1 s; frames resumed on the same
connection. The app's own logic (`swift_probe_camera_freeze.json`): LIVE →
**CAMERA NOT SENDING** → LIVE, never LIVE during the freeze.

**Device reboot** (`reboot` scenario, in-process restart): stream id
`cb361913` → `06196246`, seq 259 → 1, drop → first frame on the new stream
**0.51 s** (2 attempts on the app's 0.5 s/1 s backoff).

**One connection** (`swift_probe.json`, the app's coordinator driving a real
URLSession; viewers counted **on the device** via `/api/stream/stats`):
inline + full screen → **1** device viewer, 1 transport open; after 40
rapid view swaps → **1**; backgrounded → **0**; foreground → **1**; all
views gone (plus 200 stray detaches) → **0**. While monitoring was stopped the
app's liveness stayed `live` with source `camera` ("LIVE · Monitoring
paused"); after resume, `annotated`. 0 transport errors, 0 duplicates, parser
high-water 19.4 KB.

### A finding the probe made: URLSession splits multipart parts

URLSession does not deliver `multipart/x-mixed-replace` as raw bytes: it calls
`didReceive response` for the HTTP response **and again for every part**
(an `NSHTTPURLResponse`, status 200, headers = the part's own merged with the
top-level ones), then `didReceive data` with that part's JPEG alone
(`urlsession_multipart_behaviour.txt`, macOS Foundation; iOS uses the same
CFNetwork loader, to be confirmed on a phone). Consequences:

- The per-frame evidence reaches the app through the per-part responses, so
  `MJPEGReader` classifies each response (`MJPEGResponse.classify`,
  `MJPEGStream.swift`) and hands the frame's metadata to the parser
  (`expectNextFrame`). A first integration that only parsed headers out of the
  bytes saw none of them and re-"opened" the connection on every part — the
  probe showed 184 opens and 0 advancing frames, and it was fixed before commit.
- The pre-M8.3 reader was not broken by this: it allowed every 200 response
  and framed JPEGs by SOI/EOI, so it never needed the part headers.
- The in-band header parsing in `MJPEGFrameParser` stays for transports that do
  not split (the Python client, a future raw-socket reader).

## 7. What only the real Pi and iPhone can prove

- Delivered fps and frame age on the Pi with the Hailo pipeline running (the
  encode was never measured separately from inference — PERFORMANCE_BUDGET §4),
  and the CPU cost of the camera fallback encode while stopped.
- Frame age including Wi-Fi transit (the header excludes transit; the phone
  cannot measure it without clock sync).
- `UIImage` decode cost and memory on an iPhone over a long session; the
  badge and fading as drawn.
- Real network changes (Wi-Fi roam, DHCP renewal, phone switching to
  cellular), iOS backgrounding with the socket held by the kernel, Local
  Network permission prompts.
- avahi/Bonjour discovery on a real LAN; the header-token path from a real
  phone (Phase 8 verified `?token=` and 401s from another machine, not the app).
- A real reboot (systemd restart, power loss) — the sim reboots in-process.
