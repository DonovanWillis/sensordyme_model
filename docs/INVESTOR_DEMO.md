# Investor demonstration package (M11, updated M12 and M13)

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

## 2. The 90-second demo

For a meeting slot, a booth or a video call. One presenter (a consenting
adult), seated, **stationary**. The app is already open on the device screen,
in presentation mode with the AI view on.

| Time | Say | Do | The audience sees |
|---|---|---|---|
| 0:00 | "This camera runs its own AI. Nothing here goes to a cloud." | Point at the Pi | The live picture, the "Protecting" state, fps |
| 0:10 | "What you're seeing is what the device itself detected, on this exact frame." | Sit normally | The driver box, the face outline, the dashed seat region, "Head: Forward" |
| 0:25 | "A short glance is normal driving." | Glance sideways ~1.5 s | "Head turned away" **developing** (amber), then it clears. No alert |
| 0:40 | "A long one isn't." | Turn your head ~40° and hold 4 s | Developing → **confirmed** at ~3 s; **the buzzer sounds**; red banner: "turned 40 deg … + held 3.0 s" |
| 1:00 | "The alert happens on the device, so it doesn't depend on the phone. It also tells you *why*, in measured terms." | Point at the banner and the panel | The device's own sentence; the event in history |
| 1:15 | "Same for a phone in your hand, or eyes closing." | Optional: hold a phone in the lap and look at it, 3 s | `phone_use`, or `eyes_down` with its reason |
| 1:30 | "All of it runs on the device in the car: the camera, the AI and the alert." | — | — |

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
| 14 | "Custom rules in plain English." | Ask Jupiter: "Alert me when a person remains in this area for 10 seconds." Draw the area → review → **Activate** | The rule appears beside Driver Safety; an alert ~10 s after someone stands in the area | Rehearsed only on a simulated general-role device. **Rehearse it on the unit beside Driver Safety before showing it.** Otherwise use the M11 rule "alert me if a phone is visible for 5 seconds" |
| 15 | "None of this needs our development computers or the internet." | Close the MacBook lid, then repeat step 7 | **The buzzer still sounds.** The app shows "Direct connection"; the event reads "Not yet backed up" | Only once rehearsed on hardware (friend guide step 24b) |

**Never in a demo:**

- driving, or any moving vehicle;
- a person who has not consented;
- an untested alert;
- "accuracy" figures. **No real-world accuracy figure exists yet.** The only
  numbers are the Open Images float results (L1); say so if asked;
- a simulated event or recording presented as real. A simulated event is
  labelled "Simulated" and never sounds.

## 4. Preflight and operator checklist

**The day before:**

- [ ] `docs/FRIEND_HARDWARE_SETUP.md` steps 14–27 all PASS on this unit, with evidence (`collect_evidence.py` sha256 recorded).
- [ ] The iPhone app was reinstalled from Xcode **within the last 7 days**. A free personal team's signature expires after 7 days.
- [ ] Driver calibration was done in the demo seat position, with only the presenter in view (README Phase 4).
- [ ] The buzzer was heard at 1 m in the demo room (step 21).
- [ ] The Jupiter step was rehearsed on this unit, or the M11 phrasing is used.
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
- **Provenance:** one continuous cut from UI-test journey 30 of the final 30/30
  XCUITest run.

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
| Hailo-8L inference, Pi fps, thermal behaviour, INT8 parity | **not measured** | Do not quote. Hardware gates H1–H3 |

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
