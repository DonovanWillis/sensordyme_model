# Investor demonstration package (M11, updated M12, M13 and M14)

**Status:** prepared and rehearsed **in software only.** The iOS Simulator ran
against the virtual device, and the module harness ran the real driver module on
scripted faces. **No step has run on the Raspberry Pi 5 + Hailo-8L or on a
physical iPhone.** Every step below is designed for the real unit once
`docs/FRIEND_HARDWARE_SETUP.md` steps 1–25 have passed. Until then, the only
honest live demo is the **labelled simulator** (§6) or the **labelled fallback
recording** (§5).

Contents:

1. The setup that works today
2. The 90-second demo
3. The 5-minute demo
4. Preflight and operator checklist
5. The labelled fallback recording
6. If the hardware is not ready: the labelled simulator
7. Performance evidence (what is measured, and where)
8. What the product does, and what it does not
9. Edge vs cloud: where each part runs

---

## 1. The setup that works today (`docs/PORTABLE_OPERATION.md` §4)

**Venue network:**

- Use home or office Wi-Fi, **or** a travel router or phone hotspot that the Pi
  and the iPhone (and the MacBook, if used) all join.
- Guest Wi-Fi with client isolation does not work.
- Bonjour on a hotspot hosted by the iPhone itself is **unverified**. Rehearse
  it, or use a travel router.

**The phone reaches the camera directly (M12, sd-local-v1).**

- Live view, stop/resume and mode switching do not need the MacBook.
- History sync, push and account features use the backend. It runs on the
  MacBook (`scripts/install_backend_mac.sh`); **no hosted backend is deployed.**

**Ask Jupiter:**

- The deterministic parser (`jupiter/policy.py`) handles the demo phrasings
  with no language model.
- A local model server (Ollama) on the backend's machine is optional.
- No external AI is used unless explicitly enabled.

**Device role:**

- The shipping unit is `role: driver_safety`. Driver Safety is its mandatory
  floor, and only its variants can be selected (Standard / Strict / Teen Driver
  / Long Haul).
- A Jupiter rule runs **beside** Driver Safety, never instead of it.
- Switching to non-driving modes would need `role: general` on the demo unit.
  That is an **owner decision**; nothing has changed it.

**M13 app features (Simulator-verified only):**

- **AI view.** A toggle on the live camera draws what the device's AI saw on
  that frame, from the sd-overlay/1 stream (`docs/LIVE_OVERLAY.md`):
  - the driver box and other people;
  - phones marked "driver" or "other";
  - the face and eye outline;
  - the dashed driver-seat region and hand boxes;
  - the behaviour strip (amber while developing, alert colour once confirmed);
  - measured fps and per-model milliseconds.

  It shows no confidence percentages, never shows eye gaze ("Head" pose only),
  and nothing animates on its own.
- **Presentation mode.** The **Present** key on the device screen opens it:
  - full screen, camera first, landscape-aware;
  - the distracted-driving panel;
  - an alert banner in the device's own words;
  - a one-tap mode switch that is confirmed only by the device.
- **Distracted-driving panel.** It shows the monitoring state, the behaviours
  with their durations, eyes closed, head direction, and the last event's
  "why" line.

**M13.1 (Simulator-verified only):** the **AI view** toggle and **Present** are
also on the paired-camera screen, the one the app opens from the phone's local
pairing. They work with the backend **stopped**: live video, the overlay, the
panel and a device-confirmed mode switch all go phone ↔ device over the signed
local path (sd-local-v1). UI journey 31 (`test31_LocalPresentation`) proves
this with the backend process stopped. On the device, drawing the face mesh
into the stream costs 0.90 ms instead of 2.31 ms (p50, Mac M3; pixel-identical
to the previous drawing).

## 2. The 90-second demo

For a meeting slot, a booth or a video call. One presenter (a consenting
adult), seated, **stationary**. The iPhone is on the paired camera's screen,
in presentation mode with the AI view on.

**Before the slot (M14):**

- Draw one area named `console` beside the presenter, outside their body.
- Activate the Jupiter rule "Alert me when a cell phone stays in this area for
  10 seconds" (§3.1).
- Wait until the device reports it active. The rule is never activated live
  inside the 90 seconds, because the device's acknowledgement takes a step.

| Time | Say | Do | The audience sees |
|---|---|---|---|
| 0:00 | "This is SensorDyme: a camera with its own AI chip. That's its live picture on the phone." | Point at the Pi, then the phone | The live picture, LIVE, "Protecting" |
| 0:15 | "What you're seeing is what the device itself detected, on this exact frame." | Sit normally | The driver box, the face outline, the dashed seat region, "Head: Forward", measured fps |
| 0:30 | "A long look away is what causes crashes." | Turn your head ~40° and hold 4 s | "Head turned away" **developing** (amber), then **confirmed** at ~3 s |
| 0:50 | "The alert happens on the device, and it tells you *why*, in measured terms." | Turn back. Point at the buzzer, then the banner | **The buzzer sounds**; red banner in the device's own words ("turned 40 deg … + held 3.0 s"); the event in history |
| 1:05 | "You can also add your own rule in plain English. This one is already on, beside Driver Safety: 'alert me when a phone stays in this area for 10 seconds'." | Eyes forward. Put the phone upright in the `console` area and let go | The phone box; then, at ~10 s, **one** alert: "Cell phone — A cell phone stays in this area for 10 seconds" |
| 1:20 | "All of it runs on this device: camera, AI and alert. Our development computers are not involved." | — | — |

The 0:30 head turn is the step to keep if time runs short. Never shorten the
Jupiter rule's 10 s to fit the slot: it is the rule as written.

If anything fails, do not improvise a claim. Say what the screen says (it is
truthful by design), or switch to the fallback recording (§5) and **say it is a
simulated recording.**

## 3. The 5-minute demo

| # | Say | Do | The audience sees | If it goes wrong |
|---|---|---|---|---|
| 1 | "This is SensorDyme: a Raspberry Pi 5 with a Hailo AI accelerator and an infrared camera. Everything runs on the device." | Plug in the power (or start already running) | The device boots. Nothing to press | A boot takes ~60–90 s: talk through §9 |
| 2 | "I just open the app." | Open the app (already signed in) | The device card goes from starting to protecting | It still says starting: calibration takes ~30 s after a start. Never say "all clear" before the app does |
| 3 | "Live view, straight from the device over local Wi-Fi. The phone has its own key for this camera; there was nothing to type." | Tap the device. The clean live view appears | LIVE, the clean consumer picture | "Needs pairing again": re-pair over Bluetooth. Fallback: §5 |
| 4 | "Now the AI view: what the device saw on this frame, drawn by the app from data the device sends with the frame." | Toggle **AI view** | Driver box, face and eye outline, seat region, measured fps / model ms | `Detection unavailable` or `Not working: …` is shown honestly: say so, and never call it "clear" |
| 5 | "Presentation mode." | Tap **Present** | Full screen; the distracted-driving panel | — |
| 6 | "A normal glance does not alert." | Glance ~1.5 s | Developing, then cleared | — |
| 7 | "Watch what happens when I look away." | Turn the head ~40° for 4 s | Developing → confirmed at ~3 s | Too short a turn: hold it 4 s |
| 8 | "The alert is on the device itself." | — | **The buzzer sounds**; red outline and banner | Silent: hard stop; do not fake it |
| 9 | "And it explains itself." | Read the banner and panel | e.g. "turned 40 deg right of forward (limit 35 deg) + held 3.0 s" | — |
| 10 | "It arrives in the app's history too, once, with the same id the device recorded." | Open history | One event, with what it means and the device's own words | Late: the uplink delivers exactly once. Wait 30 s |
| 11 | "Phones: the device knows whose phone it is." | Hold a phone in the lap and look at it 3 s | `phone_use`, and the phone marked "driver" | — |
| 12 | "Eyes closing." | Close the eyes ~3 s | `drowsiness` with "eyes closed 2.0 s + eye openness … under …" | Needs a calibrated driver: calibrate in the demo seat beforehand |
| 13 | "One tap changes the policy, and the device confirms it." | Tap **Strict** in presentation mode | The chip selects only when the device reports it | A refusal names its reason: show it, as truthfulness is the point |
| 14 | "Custom rules in plain English, beside Driver Safety, never instead of it." | Ask Jupiter: "Alert me when a cell phone stays in this area for 10 seconds." (one area, `console`, beside the presenter) → review → **Activate**. Wait for the device. Then put the phone upright in the area, eyes forward | The rule shows active once the device reports it; **one** alert ~10 s after the phone is placed; history, explain and the AI view's last event name the rule | §3.1. SIMULATED on a `driver_safety` twin only (M14). **Rehearse on the unit.** If the phone flickers: "Alert me when a cell phone is in this area." |
| 15 | "None of this needs our development computers or the internet." | Close the MacBook lid, then repeat step 7 | **The buzzer still sounds.** The app shows "Direct connection"; the event reads "Not yet backed up" | Only once rehearsed on hardware (friend guide step 24b) |

**Never in a demo:**

- driving, or any moving vehicle;
- a person who has not consented;
- an untested alert;
- "accuracy" figures. **No real-world accuracy figure exists yet.** The only
  numbers are the Open Images float results (L1); say so if asked;
- a simulated event or recording presented as real. A simulated event is
  labelled "Simulated" and never sounds.

### 3.1 The Jupiter step on the real `role: driver_safety` unit (M14)

**Status: SIMULATED only.** It ran end to end on the digital twin: the real
backend, the real device runtime, scripted camera and models
(`tests/test_m14_jupiter_driver_safety.py`). It has **not** run on the Pi 5 +
Hailo-8L or on a physical iPhone. Rehearse it on the unit before showing it.

#### Why not the M13 rule


M13's showcase ("Alert me when a person remains in this area for 10 seconds")
ran on a simulated `role: general` device. On `general`, and on every role
except `driver_safety`, a Jupiter rule **replaces** the running mode (`ai_custom`).
On `driver_safety` it runs **beside** Driver Safety (`standard_plus_custom`).
In that case it may only name things the driver module's own detector reports:
`person` and the `yolo_classes` in `config.yaml` (cell phone, cup, bottle, food
and cutlery).

The M13 rule does run beside Driver Safety (the test pins this). It is still the
wrong demo here, for three reasons:

- it needs a second person to stand in the area;
- the demo rules say nobody but the consenting presenter may be in view;
- a second person in view also touches driver association.

The rule below needs only the presenter, who stays seated and keeps their eyes
forward.

#### The step

**Before:**

- Draw exactly **one** area in the app and name it `console`.
- Put it on the desk or console beside the presenter, **outside** their body,
  not over their lap. A phone inside the presenter's box is treated as the
  driver's phone; looking at it for 2 s is then `phone_use` (Driver Safety
  working as designed).
- Prop the phone upright in that spot. How well the detector sees a phone lying
  flat, in IR, at desk distance has **not been measured**.

**The exact prompt (type it in Ask Jupiter):**

> Alert me when a cell phone stays in this area for 10 seconds.

Say "cell phone". "A phone stays in this area…" is not in the deterministic
grammar, so it would go to a language model (none on the twin).

With more than one area drawn, "this area" waits for an area named "marked
area". In that case use the name instead: "Alert me when a cell phone stays in
the console for 10 seconds."

| Say | Do | The audience sees (twin) |
|---|---|---|
| "Driver Safety stays on. Jupiter adds a rule beside it, in plain English." | Type the prompt. Show the review. Tap **Activate** | Review: "Alert when a cell phone stays inside the area 'console' for 10 seconds or longer without leaving. Severity: warning; notification: push.", plus what it does and does not monitor |
| "It's active only when the camera says it is." | Wait for the device | The rule shows active once the device reports it (on the twin, within one 4 s step: rule `lingering cell phone`, mode `standard_plus_custom`). Over a direct connection, presentation mode says "Rules set by Jupiter are running" |
| "Watch the road. I'll put my phone down over here." | Eyes forward. Place the phone upright in the area and let go. **Don't look at it** | The phone box in the AI view; nothing else yet |
| — | Wait ~10 s, still facing forward | **One alert at ~10 s** (on the twin, +10.25 s after the phone first appeared in the area). The buzzer plays the *warning* pattern ("double"). AI view "Last event": **Cell phone**, "A cell phone stays in this area for 10 seconds". The history row has the same words and the same `event_key`. Explain: "You were alerted because your rule alerts after 10 seconds", Area: console |
| "And Driver Safety is untouched." | Take the phone away. Turn the head ~40° and hold 4 s (step 7) | `distraction`, exactly as without the rule |

A phone in the area for less than 10 s gives no alert (6 s on the twin). The
alert fires once per visit. A detector miss longer than 1 s restarts the
10 s clock. The rule then waits 60 s before it can alert again.

**Fallback if the phone flickers in rehearsal:** "Alert me when a cell phone
is in this area." This is presence held for 2 s instead of a 10 s dwell. Only
the parser has been tested on it; the device accepts the same label.

**Never:**

- change the unit's role for the pitch;
- send the rule together with a mode switch (refused as `not_beside_mode`;
  activate it on its own);
- ask for something the driver module does not detect (see the table).

#### What the test proves (SIMULATED)

`tests/test_m14_jupiter_driver_safety.py` runs the same scripted timeline with
the rule and without it. The timeline is:

- the phone in the area for 6 s, then for 20 s;
- the eyes closed for 5 s of the 20 s visit, so drowsiness and the rule fire
  within the same second;
- later, a 6 s head turn.

It checks the following, all through the app's API calls:

- The language model is never asked: the deterministic parser reads the
  prompt into one typed policy.
- The capability check passes against what **this** device reports, and the
  driver module serves `cell phone`.
- The device acknowledges the rule only after Activate. The mode is
  `standard_plus_custom`, `driver_attention` runs throughout, and the role and
  preset are unchanged.
- **No second model or pipeline:** one module build in both runs, and the same
  detector call count.
- One rule alert at 10–11.5 s, and none for the 6 s visit.
- **Driver Safety is not affected:** the same alerts, at the same device times,
  each `alerted` with the buzzer `played` and nothing `skipped`, and the same
  buzzer patterns. The driver thresholds equal `config.yaml`.
- The alert is the same `event_key` in the backend history and in the
  device's signed `/api/events`, and `/explain` answers for it.
- sd-overlay/1 `last_event` is the rule's alert, with its `event_key`.

#### Which existing policies run beside Driver Safety

Pinned by `test_which_existing_policies_run_beside_driver_safety`:

| Policy | On `driver_safety` | Why |
|---|---|---|
| Person dwell (M13; app suggestion "someone stays inside the marked area > 10 s") | runs | `person` is reported by the driver module. Needs a second person to demo; an area over the seat fires on the driver |
| Person enters the area (`in_zone`) | runs | Same label. Same demo caveat |
| Nobody in the area for N (`absent`) | runs | Same label. Overlaps Driver Safety's own critical `attention_lost` for an empty seat |
| More than N people in the area (`count`) | runs | Same label. Needs several people |
| **A cell phone stays in the area for 10 s (chosen)** | **runs** | `cell phone` is a driver-module detector class. One seated presenter can show it |
| A bottle / cup / food enters the area | runs | Driver-module classes. Not scriptable in the twin's scenes (only phones and people) |
| A dog enters the area (app suggestion) | **device refuses** | `dog` is not a driver-module label. The device refuses the whole set and keeps the running rule. The backend preview still says "ready" (it cannot see the device's labels) |
| Pose compositions: resting/sleeping, waving, hand raised, lying down | **device refuses** | They are served by the pose module, which Driver Safety does not run. Running it would be a second model |
| Identity / emotion / intent / sound / medical / "high risk" / "erratic hands" | refused up front | Refused on every role before any rule exists |
| "Alert me if a phone is visible for 5 seconds" (M11 phrasing) | nothing proposed on the twin | Not in the deterministic grammar. It needs a configured language model; the rule it would produce (`cell phone`) is a driver-module label |

#### What remains hardware-only

- **Hardware, unmeasured:**
  - whether the Hailo detector sees the propped phone steadily enough (in IR,
    at desk distance) for a 10 s dwell;
  - the real alert latency;
  - that the buzzer's warning pattern is audible.
- **App, not run:** what the iPhone app shows. This stream did not run the iOS
  Simulator.
  - The presentation banner's header reads "Driver alert" for **any** alert,
    including a Jupiter rule's (`ios/Sources/PresentationMode.swift`). Say
    "this is the rule's alert", or have the iOS stream change the header
    copy.
- **Buzzer queue: fixed in M14 (`2061794`).** A rule's pattern now ranks below
  every mandatory pattern on the one buzzer, so it can never displace or jump
  ahead of a waiting driver-safety pattern (`tests/test_m14_buzzer_precedence.py`).

## 4. Preflight and operator checklist

**The day before:**

- [ ] `docs/FRIEND_HARDWARE_SETUP.md` steps 14–27 all PASS on this unit, with evidence (`collect_evidence.py` sha256 recorded).
- [ ] The iPhone app was reinstalled from Xcode **within the last 7 days**. A free personal team's signature expires after 7 days.
- [ ] Driver calibration was done in the demo seat position, with only the presenter in view (README Phase 4).
- [ ] The buzzer was heard at 1 m in the demo room (step 21).
- [ ] The Jupiter step (§3.1, cell phone in the `console` area) was rehearsed on this unit, beside Driver Safety.
- [ ] The fallback recording (§5) is on the presenting laptop, and its sha256 matches the sidecar.

**30 minutes before:**

- [ ] `make demo-check DEVICE=<user>@<pi-host> DIR=<install dir>`. It is read-only and answers DEMO READY or DEMO NOT READY.
  - **New in M13:** a "live AI view (overlay)" check samples the sd-overlay/1 stream for 4 s.
  - It FAILs when there are no analysed frames, or when the source is simulated.
  - It WARNs when monitoring is stopped.
  - **Do not run `make demo-cycle`**: it refuses on a driver-safety device by design.
- [ ] `$PY $V/collect_evidence.py --install-dir $D --minutes 1` on the Pi: `throttled 0x0`; fps and temperature normal.
- [ ] The phone and the Pi are on the same network. The live feed shows LIVE, and the AI view shows fps.
- [ ] One rehearsal of the 40° head turn (steps 7–9).
- [ ] Nobody in the camera's view but the consenting presenter.

## 5. The labelled fallback recording

- **File:** `~/sensordyme-audit/2026-11-10/m13-evidence/fallback-recording-SIMULATED.mp4` (34 s, 900×1958, H.264, 2.2 MB).
- **Sidecar:** `fallback-recording-SIMULATED.txt`, with a second-by-second description and the file's sha256.
- **What it shows:**
  1. the clean device screen;
  2. presentation mode with the AI view;
  3. a simulated head-turn going from developing to confirmed, with the alert banner;
  4. landscape and back;
  5. a tap on Strict, confirmed by the device;
  6. exit.
- **Labels:**
  - A SIMULATED badge is on every live frame.
  - A burned-in bar says it is a virtual device in the iOS Simulator, not a Pi.
  - **Say so out loud when you play it.**
- **Provenance:** one continuous cut from UI-test journey 30 of the M13 final
  XCUITest run (30/30 at faae3de; M13.1 added journey 31).

## 6. If the hardware is not ready: the labelled simulator

- `make sim SCENARIO=tour` runs the real backend and a **virtual device**.
- `make sim SCENARIO=showcase` (M13) repeats a simulated distraction, so the AI
  view and the panel have something to show.
- Every report says `simulated: true`, and the device id starts `SD-SIM-`.
  The overlay carries `sim: true`, and the app shows SIMULATED on every frame.
- Show it only with a visible "SIMULATED — virtual device, no camera" caption.
- Never present its alerts or timings as measurements.
- The buzzer and the live camera do not exist in this mode.

## 7. Performance evidence (what is measured, and where)

**Source:** `docs/research/M13_PERFORMANCE.md`. **Every number is a desktop
measurement** (MacBook M3, or the x86 fleet worker), labelled with its host.
**None of it is Raspberry Pi performance.** The Pi was unreachable during M13.

| What | Measured (host) | Use in a pitch |
|---|---|---|
| Capture → decision in the real inference loop, Hailo replay | ~8.4 ms p50 (Mac M3, desktop) | Only as "our software adds single-digit milliseconds on a desktop". Not Pi latency |
| Behaviour reasoning per frame | −26–39 % after M13 (Mac) | Engineering detail |
| Reference decoder | −77 % p50 (3.06 → 0.70 ms, Mac); −50 % (x86) | Engineering detail |
| Alert → backend stored | p50 1,285 ms → 10.7 ms (Mac, software) | "Alerts reach the app history almost instantly" (software measurement) |
| Decisive frame → buzzer commanded | ~1 ms (SIMULATED, Mac) | "The device alerts locally, without a round trip" |
| Camera fault → "not protecting" in the app | 2.25 s (SIMULATED) | "It tells you when it can't see" |
| Face mesh drawn into the streamed frame | 2.31 → 0.90 ms p50 (Mac M3; M13.1, pixel-identical) | Engineering detail |
| Hailo-8L inference, Pi fps, thermal behaviour, INT8 parity | **not measured** (the Pi was also unreachable throughout M13.1) | Do not quote. Hardware gates H1–H3 |

The overlay itself shows **live, measured** fps and per-model run times on the
device. That is the only performance number to show during a live demo, and it
is the real unit's own measurement.

## 8. What the product does, and what it does not

**Does (software-verified; hardware gates pending):**

- Distraction:
  - the head turned away beyond the limit and held;
  - looking down or up, held;
  - phone use by the driver (a hand on the phone, experimental hand tracking);
  - drowsiness (calibrated eye closure; fatigue by PERCLOS).
- Each alert carries a measured "why".
- Driver association: the configured or calibrated seat region; a passenger's
  phone is "other" and never alerts.
- Honest states:
  - face not visible, so face rules abstain;
  - camera unavailable leads to FAILED / not protecting within ~2 s;
  - a component that is not working is named.
- A local buzzer that is independent of the phone and the network.
- Local, signed phone access with no backend (sd-local-v1). History is merged
  without duplicates.
- Plain-English custom rules through Jupiter, as reviewable data (never code).
  Identity requests such as "a stranger" are refused.
- In the simulated flagship scenarios: **10 / 10 PASS** (`docs/DEMO_SCENARIOS.md`).

**Does not (and must not be claimed):**

- **No real-world accuracy figure.** Simulated scenarios are not accuracy
  evidence.
- No eye-gaze tracking: it measures head pose and eye openness.
- No physical validation yet: Pi latency, thermal, INT8 parity and an audible
  buzzer are gates H1–H3 and step 21.
- No guaranteed iPhone alert while the phone is locked, silent or closed. The
  device buzzer is the safety alert.
- No hosted cloud backend, and no customers, pilots or commercial results.
- No driving demonstrations: every scenario is stationary.

## 9. Edge vs cloud: where each part runs

| Part | Runs on | Needs the internet? |
|---|---|---|
| Camera capture, object detection (Hailo-8L), face and hand landmarks, tracking | **the device** | no |
| Behaviour decisions (distraction, phone use, drowsiness), the "why" text | **the device** | no |
| The buzzer alert | **the device** | no |
| Live view, the AI view data (sd-overlay/1), stop / resume / mode | device ↔ phone, directly on the local network, signed per phone | no |
| Local event history (`/api/events`) | **the device** | no |
| Account, sign-in, history backup, push to the phone when away | the backend (today: the owner's MacBook as a home server; no hosted service) | only to reach the phone when away from the LAN |
| Jupiter rule parsing | the backend: a deterministic parser; optional local Ollama | no; no external AI unless explicitly enabled |
| Video | **never leaves the device** except to an authorised phone's live view | — |

**The pitch line:** "The intelligence is in the car. The cloud is for your
account and your history, not for the safety decision."
