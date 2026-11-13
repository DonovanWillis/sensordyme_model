# ORYN EDGE iOS design system: Liquid Precision

The iPhone app (`ios/`) is the instrument panel for a SensorDyme. This document
covers how it looks, moves and is put together. It sits on top of the truth
logic (ConnectionTruth, DeviceStatus, PerceptionStatus, ProtectionControl,
ModeSwitchState) and never second-guesses it: the design decides how a state
is *shown* and never what the state *is*.

## 1. What was wrong (baseline critique, M8 RC `618880c`)

Baseline renders: `~/sensordyme-audit/2026-11-10/design-evidence/before/`.

1. **Everything is a card.** Device home stacked six bordered rectangles
   (feed, status chips, Ask Jupiter bar, mode card, custom rules card, manage
   card, health card, capabilities card). Every one used the same 14 pt radius
   and hairline, so nothing was more important than anything else.
2. **Status said three times, never once clearly.** An `ONLINE` pill, an
   `ALL CLEAR` pill, a green check in the mode row, a "Monitoring" row in the
   health card and a green refresh tick in the device-list header all competed.
   None of them was large enough to read at a glance.
3. **The camera was a dead rectangle.** A rule-of-thirds grid behind a
   paragraph of instructions, the same weight as the cards below it.
4. **Mono-caps everywhere.** Tracked SF Mono eyebrows were used for chips,
   section titles, validation badges and the tab bar. With one register for
   every label, it read as a terminal rather than an instrument.
5. **Mode library = six identical rows**, each with a `NOT YET VALIDATED` amber
   badge in caps. The flagship mode (Driver Safety, the Distracted Driving
   product) looked no different from a mode the device's role forbids.
6. **Ask Jupiter was a search field** plus a sentence, with no sense of what it
   can and cannot do. Its examples deliberately mixed supported and refused
   requests without saying which was which.
7. **Activity was a list in a box** under a 24-hour scrubber with one dot on it.
   There was no time hierarchy, and every row had the same weight whatever its
   severity.
8. **No continuity.** Every change of context was a push or a sheet sliding up
   over unrelated content.
9. **Noise passed off as liveness.** The device-list header flashed a green
   check every 10 s, meaning "the list refreshed", which is easy to read as "your
   device is fine".

## 2. Direction

**Liquid Precision**: a calm, camera-first instrument that feels alive because
it responds to interaction and to *real* device state, and to nothing else.

- **Composed, not assembled.** One composition per screen. Hierarchy comes from
  scale, weight, spacing and position. Containers are used only where they mean
  something: a control surface, the camera, a floating dock.
- **The camera is the centrepiece.** It is the largest thing on the device
  screen, with its controls floating over it, and it opens into full screen
  with a zoom.
- **One status readout.** A single instrument line says presence, monitoring
  state and mode, at a size readable at arm's length. It is never green before
  the device has confirmed `ready` on fresh telemetry.
- **Continuity.** A device card zooms into its home. A mode row zooms into its
  detail. The camera zooms into full screen. The Jupiter dock expands in place
  and its review sheet zooms out of it. All of these use native iOS 18
  navigation transitions and fall back to the standard ones on iOS 17.
- **Restraint.** Bone (`#F2F1ED`) is the single interactive accent. Signal
  orange is reserved for the ORYN/Jupiter brand moment and the live indicator.
  Data colours mean state and nothing else.

Out of scope by design: moving gradients, glowing orbs, breathing animations on
an unverified device, neon, card grids, decorative charts, and glass for its
own sake.

## 3. Tokens (`ios/Sources/Theme.swift`, `SD.*`)

### Colour

ORYN EDGE stays dark by design (the collaborator's brand, kept). Contrast below
is on `background #0A0B0C`.

| Token | Value | Use | Contrast |
|---|---|---|---|
| `background` | `#0A0B0C` ink-900 | page | n/a |
| `surface` | `#0F1112` ink-850 | grouped surfaces (rare) | n/a |
| `surfaceRaised` | `#141618` ink-800 | controls, keys | n/a |
| `field` | `#191C1E` ink-750 | inputs, dock | n/a |
| `canvas` | `#000000` | behind video | n/a |
| `textPrimary` | `#F2F1ED` bone-50 | titles, values | 17.9:1 |
| `textSecondary` | `#B6BABE` steel-200 | body, details | 10.2:1 |
| `textTertiary` | `#8F959B` steel-300 | captions, metadata | 6.4:1 |
| `textDisabled` | `#6C7278` steel-400 | placeholders only | 3.9:1 |
| `accent` | bone | the one interactive colour | n/a |
| `signal` | `#FF6A2B` | ORYN/Jupiter brand moment, LIVE dot | n/a |
| `success` | `#8DBF63` | device-confirmed ready / working only | n/a |
| `warning` | `#E9B949` | starting, degraded, stale, waiting | n/a |
| `danger` | `#E5544B` | failed, refused, critical alert | n/a |
| `offline` | `#6C7278` | offline, stopped, unknown | n/a |

Rule: **colour is never the only carrier.** Every state also has a word and a
distinct SF Symbol (section 6).

### Type

Apple-native (SF Pro / SF Mono) with Dynamic Type throughout. The scale is short:

| Role | Font | Notes |
|---|---|---|
| `display` | largeTitle-relative 32 semibold, -0.6 kern | screen titles (Your Devices) |
| `title` | title2 semibold | device name, sheet headers |
| `readout` | title3 semibold | the status readout ("All clear") |
| `headline` | headline semibold | row titles, flagship mode name |
| `body` | body | reading text |
| `callout` | callout | supporting sentences |
| `label` | subheadline medium | buttons, keys |
| `caption` | caption | metadata |
| `eyebrow` | caption medium mono, uppercase | section labels **only**; never on chips or badges |
| `metric` / `mono` | monospaced digits | technical readings: fps, ids, timings, counts |

Badges and chips now use sentence case in `label`/`caption` weight. Mono caps
are kept for section labels and machine values, where they carry meaning.

### Spacing (4 pt grid)

`xxs 4 · xs 8 · sm 12 · md 16 · lg 24 · xl 32 · xxl 48`. The page gutter is
`md` (16). Sections are separated by `xl` (32) of open space, not by borders.
Rows inside a section use `sm`.

### Radius (continuous corners throughout)

| Token | Value | Use |
|---|---|---|
| `sm` | 6 | small tags |
| `md` | 10 | buttons, fields |
| `lg` | 14 | control surfaces, keys |
| `xl` | 18 | sheets' inner groups |
| `camera` | 22 | camera viewport (nested 22 → 14 → 10 keeps concentric spacing) |
| capsule | full | chips, the Jupiter dock, floating camera controls |

### Materials and depth

- `SDSurface.floating`: the floating dock and the camera's floating controls.
  On iOS 26 this is native Liquid Glass (`glassEffect`), on iOS 17/18
  `.ultraThinMaterial` over a tinted ink fill. **Reduce Transparency** swaps
  it for an opaque `field` fill with a hairline, and the screen stays fully
  legible.
- Nothing else is translucent. Depth is shown by surface steps (ink-900 → 850 →
  800 → 750), a single top highlight hairline on floating surfaces, and one
  soft shadow (y 8, radius 24, 35 % black) on floating surfaces only.
- No blur over live video. Overlays on the camera are opaque-ish black plates
  (60 %) so their contrast does not depend on the picture.

## 4. Motion (`SD.Motion`)

Native SwiftUI springs. Animation is applied at the **container** with
`.animation(_:value:)` keyed to a real state value, never per element and never
on a timer.

| Token | Definition | Use |
|---|---|---|
| `feedback` | `.spring(response: 0.18, dampingFraction: 0.9)` | press states, toggles of chrome |
| `standard` | `.spring(response: 0.34, dampingFraction: 0.88)` | most in-place changes |
| `spatial` | `.spring(response: 0.5, dampingFraction: 0.86)` | expand/collapse, dock expansion |
| `sheet` | `.spring(response: 0.42, dampingFraction: 0.9)` | content arriving in a sheet |
| `status` | `.easeInOut(duration: 0.35)` | status readout cross-fades (no bounce on truth) |
| `reveal` | `.easeOut(duration: 0.28)` + 40 ms stagger, max 8 items | capabilities in mode detail |
| `quick` | `.easeOut(duration: 0.12)` | legacy alias of feedback for buttons |

`SD.Motion.resolved(_:reduceMotion:)` returns `nil` (an instant change) for
spatial, sheet and reveal motion under Reduce Motion. The status cross-fade is
kept as an opacity fade. Zoom transitions are not applied under Reduce Motion,
so the standard push/sheet is used and full-screen camera presents without
animation.

Haptics (`.sensoryFeedback`) are tied to outcomes:

- `.impact(.light)` when a primary control is pressed (Activate, Stop, Resume,
  send to Jupiter), because the press is the real event;
- `.success` only when the **device** confirms (mode outcome confirmed,
  protection confirmed, Jupiter rule confirmed running);
- `.error` when the device or server refuses.

There is never a success haptic or animation on "sent".

### Live-state motion

| State | Visual | Motion |
|---|---|---|
| Connecting (nothing loaded) | steel readout "Connecting", camera placeholder | indeterminate hairline sweep under the readout; it stops as soon as there is an answer |
| Starting | amber hourglass, "Starting" | none (the device has not confirmed) |
| Ready ("All clear") | green check-shield; the camera frame stays neutral (a green box around a picture reads as "safe") | one-shot symbol bounce on the transition *into* ready from another state; nothing continuous |
| Monitoring alert (event < 60 s) | red bell, "Alert", camera frame danger 2 pt | the frame cross-fades in, with no pulsing |
| Degraded | amber triangle, camera frame tint goes amber | status cross-fade only |
| Stopped | steel pause, "Stopped" | status cross-fade; Resume becomes the primary key |
| Failed ("Not monitoring") | red octagon | status cross-fade |
| Offline / stale / unknown | steel wifi.slash / clock / questionmark; camera placeholder desaturated and dimmed; the LIVE badge is removed | cross-fade; active visuals are interrupted at once |

No state uses a continuous breathing or pulse animation. The old presence-pill
pulse is removed.

## 5. Information architecture

Tabs: **Devices** (home), **Shop**, **Settings** (unchanged, so the journeys
and deep links hold).

1. **Your Devices**: device cards, each led by a camera-shaped tile with the
   presence chip. A tap zooms into the device home.
2. **Device home** (the signature screen), top to bottom:
   camera viewport (expand → full-screen camera) → identity + **status
   readout** (presence, monitoring state, mode, contextual Stop/Resume) →
   degraded/perception notice → action keys (Activity, Areas, Report) →
   recent activity (three latest) → **Monitoring modes** (flagship + library)
   → running-on-device nodes → custom rules → device health (expand in place;
   folded to one line only while the device reports and every component
   works, open by itself otherwise).
   The **Ask Jupiter dock** floats above the tab bar.
3. **Mode detail** (sheet, zooms from the row): identity, live state if
   running, capabilities as the device reports them (staggered reveal),
   validation, when it alerts, limitations, settings, Activate with pending →
   device-confirmed outcome.
4. **Ask Jupiter** (sheet, zooms from the dock): request, what Jupiter
   understood, editable rules, the policy preview (will / won't watch for),
   activation with device confirmation.
5. **Activity** (push): a chronological timeline grouped by day, with severity
   markers on a hairline spine. A row opens the event's explanation sheet.
   Notification taps still open the exact event (DevicesView →
   DeviceDetailView routing unchanged).
6. **Settings, setup and onboarding**: native grouped forms in ORYN chrome, and
   a calm stepped progress for setup.

## 6. Components (small set, `ios/Sources/Instrument.swift` and `Theme.swift`)

| Component | What it is for |
|---|---|
| `InstrumentState` | Presentation of presence + monitoring + alert as one of: connecting, ready, alert, starting, degraded, stopped, failed, offline, stale, unknown. It has a word, a symbol, a tone and a spoken label, and is Foundation-only and unit-tested (`ios/Tests/InstrumentStateTests.swift`). It **reads** ConnectionTruth/MonitoringStatusPresentation and never decides truth. |
| `StatusReadout` | The one compact status presentation on the device home (keeps `detail.online`, `detail.monitoring`). |
| `SDStatusPill` | Presence/state chip (sentence case, no pulse). |
| `CameraViewport` | Camera container: real MJPEG, demo feed or a truthful placeholder, floating controls, state frame, zoom source. |
| `LiveCameraView` | Full-screen camera, sharing the same stream (no second connection). |
| `InstrumentKey` | Contextual action key (icon + label, 44 pt+). |
| `SDSectionHeading` | Section label + optional trailing action, set on open space. |
| `ModeRow` / flagship treatment | Mode library rows (keep `preset.row.*`, `preset.info.*`, `preset.variant.*`). |
| `CapabilityLine` | A capability as the device reports it, with word and glyph. |
| `TimelineRow` | Activity row on the timeline spine (keeps `event.row`). |
| `JupiterDock` | Floating prompt surface (keeps `jupiter.input`, `jupiter.send`; new `jupiter.dock`). |
| `PolicyPreview` | Will / won't watch for, with no ticks before activation. |
| `ConnectionBanner` | Unchanged API (`banner.connection`, `banner.message`, `banner.retry`), restyled. |
| `SetupStepper` | Onboarding progress. |
| `SDSurface` | Floating material with a Reduce Transparency fallback. |

## 7. Accessibility

Kept from M8: Dynamic Type everywhere (AX sizes re-flow horizontal layouts into
vertical ones), 44 pt targets, VoiceOver labels that carry the state words,
`isSelected` traits, error announcements, and focus-safe typing in the Jupiter
field. Added: every state has a symbol and a word as well as a colour; Reduce
Motion removes spatial motion; Reduce Transparency removes all blur.

The accessibility audit's clear band also excludes the floating Jupiter dock,
because text scrolled under it is sampled through it, exactly like the tab bar
(`ios/UITests/AccessibilityAuditUITests.swift`).

## 8. Performance method

- **Isolation of high-frequency state.** `MJPEGReader` publishes a frame up to
  12–15 times a second. It is held by the device screen in `@State`, which
  keeps the reference without subscribing, and is observed only by the small
  frame views (`MJPEGFrame`). The device screen's body re-evaluates on its 4 s
  poll and on user interaction, never per frame.
- **No continuous animation** on the device screen, apart from the connecting
  hairline while nothing has loaded (it stops on the first answer) and the demo
  feed (demo mode only, paused under Reduce Motion).
- **Measured**, not assumed:
  - transitions: `xcrun simctl io <udid> recordVideo` while the journeys (and
    DEBUG launch hooks `-orynOpenCamera`, `-orynOpenMode <id>`) drive them, then
    frame sequences (`ffmpeg … fps=N,tile=8x2`) inspected frame by frame. That
    is how the capability reveal animating *during* the mode zoom was caught
    (it read as scattered text) and moved to after the zoom settles;
  - idle cost: the app's CPU sampled every 0.5 s for 20 s on each screen
    against the real backend and virtual device (results below);
  - Instruments' SwiftUI and Animation Hitches templates on a real iPhone are
    the hardware follow-up.
  Evidence: `~/sensordyme-audit/2026-11-10/design-evidence/` (`transitions/`,
  `perf/`, `after/`).

Results (iPhone Simulator, iOS 26.1, real backend + virtual device, app CPU
sampled every 0.5 s for 20 s after the screen settled):

| Screen | Mean | Median | Max |
|---|---|---|---|
| Your Devices (10 s poll) | 0.3 % | 0.0 % | 6.5 % |
| Device home, ready (4 s poll, Jupiter dock, camera placeholder) | 1.3 % | 0.1 % | 8.3 % |
| Demo device home (simulated feed, 30 fps then) | 9.0 % | 8.8 % | 22.8 % |
| Demo full-screen camera (same feed) | 14.3 % | 15.4 % | 22.2 % |

The device home is idle between polls: no continuous animation and no
per-telemetry redraw (the spikes are the 4 s poll). The only continuous cost
is the demo's simulated feed, which has since been halved to 15 fps; it is
paused under Reduce Motion. A real MJPEG stream redraws only `MJPEGFrame`.

Transitions verified frame by frame in the Simulator: device card → home
(zoom), camera → full screen (zoom, shared stream), mode row → detail (zoom),
Jupiter dock collapsed → expanded (keyboard-avoiding) and review sheet → back
into the dock, policy preview → "Sent… waiting for the device" (never active
before the device confirms), timeline row → in-place details, stale →
interrupted visuals (banner, dimmed camera, Stale pill, "Status unknown") →
recovered, stopped → "Resuming…" → device-confirmed "All clear".
