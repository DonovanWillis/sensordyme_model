# Investor demonstration: script and operator checklist (M11)

**Status:** prepared, **not yet rehearsed on the hardware.** Every step runs on the real Pi 5 + Hailo-8L once
`docs/FRIEND_HARDWARE_SETUP.md` steps 1–25 have passed. Until then, the only honest live demo is the
**simulator**, labelled as such (§4).

## 1. The setup that works today (`docs/PORTABLE_OPERATION.md` §4)

- **Venue network:** home or office Wi-Fi, **or** a travel router or phone hotspot that the Pi, the iPhone and the MacBook all join.
  - Guest Wi-Fi with client isolation does not work.
  - Bonjour from the phone that is hosting its own hotspot is **unverified**: rehearse it, or use a travel router.
- **Backend:** runs on the MacBook (`scripts/install_backend_mac.sh`). In this deployment the MacBook is the home server, not a development PC. **No hosted backend is deployed.**
- **Ask Jupiter:** uses a local model server (Ollama) on the backend's machine. Without it, Jupiter falls back to a fixed rule parser. No external AI is used unless explicitly enabled.
- **Do not demo a cold app launch with the MacBook off.** The app's device list comes from the backend.

## 2. Script (≈ 6 minutes)

| # | Say | Do | The audience sees | If it goes wrong |
|---|---|---|---|---|
| 1 | "This is SensorDyme: a Raspberry Pi 5 with a Hailo AI accelerator and an infrared camera. Everything runs on the device." | Plug in the power | The device boots. Nothing to press | A boot takes ~60–90 s: talk through §1 |
| 2 | "I just open the app." | Open the app (already signed in) | The device card goes from starting to protecting | It still says starting: calibration takes ~30 s after a start. Never say "all clear" before the app does |
| 3 | "It found its camera by itself; nothing to reconnect." | Tap the device | Device screen, true state | Offline: check the MacBook is awake and on the network |
| 4 | "Live view, straight from the device over local Wi-Fi." | Open the live feed | LIVE badge, annotated video | Same LAN? Token entered? Use the device dashboard on the MacBook instead |
| 5 | "Distracted-driving protection is one tap." | Select **Driver Safety** | Confirmed only when the device reports it running | A refusal names its reason: show it, as truthfulness is the point |
| 6 | "Watch what happens when I look away." | Presenter (a consenting adult) turns their head ~40° for 4 s, seated, stationary | `distraction` at ~3 s | Too short a turn: hold it 4 s. A mirror-length glance is ignored by design: say so |
| 7 | "The alert is on the device itself." | — | **The buzzer sounds** | Silent: hard stop; do not fake it |
| 8 | "And it arrives in the app." | Show the timeline | One event, once, at the same time | Late: the uplink queues and delivers exactly once. Wait 30 s |
| 9 | "It explains itself." | Open the event | The device's reason text and signals (e.g. head-pose distraction, the threshold crossed) | — |
| 10 | "Custom rules in plain English." | Ask Jupiter: "alert me if a phone is visible for 5 seconds" → review → **Activate**; hold a phone in view for 5 s | Rule active beside Driver Safety; an alert after 5 s | Without Ollama the fallback parser handles this phrasing. Never claim it "understood" something it refused |
| 11 | "None of this needs our development computers, or the internet." | Close the MacBook lid (or switch it off), then repeat step 6 | **The buzzer still sounds.** The app says truthfully that the server cannot be reached; the history catches up when the server is back | Do **not** cold-launch the app in this state |

**Never in a demo:**
- driving;
- a person who has not consented;
- an untested alert;
- "accuracy" figures. **No real-world accuracy figure exists yet.** The only numbers are the Open Images float results (L1), and say so if asked.

## 3. Operator checklist (the day before, and 30 minutes before)

**The day before:**
- [ ] `docs/FRIEND_HARDWARE_SETUP.md` steps 14–27 all PASS on this unit, with evidence (`collect_evidence.py` sha256 recorded).
- [ ] The iPhone app was reinstalled from Xcode **within the last 7 days** (a free personal team's signature expires after 7 days).
- [ ] The app session was used within the last month (it renews on launch).
- [ ] Driver calibration was done in the demo seat position, with only the presenter in view (README Phase 4).
- [ ] The buzzer was heard at 1 m in the demo room (step 21).
- [ ] Ollama is installed on the MacBook, if step 10 should use the model rather than the fallback.

**30 minutes before:**
- [ ] The MacBook is on the demo network, awake, with the backend running: `.venv/bin/python scripts/diagnose_backend_mac.py <pi-hostname>`.
- [ ] The device is ready: `make demo-check DEVICE=<user>@<pi-host> DIR=<install dir>`, a read-only check that answers DEMO READY or DEMO NOT READY. **Do not run `make demo-cycle`**: it refuses on a driver-safety device by design.
- [ ] `$PY $V/collect_evidence.py --install-dir $D --minutes 1` on the Pi: `throttled 0x0`; fps and temperature normal.
- [ ] The phone and the Pi are on the same Wi-Fi; the live feed shows LIVE.
- [ ] One rehearsal of steps 6–8.
- [ ] Nobody in the camera's view but the consenting presenter.

## 4. If the hardware is not ready: the labelled simulator

`make sim SCENARIO=tour` runs the real backend and a **virtual device**:
- Every report it produces says `simulated: true`, and its device id starts `SD-SIM-`.
- Show it only with a visible "SIMULATED — virtual device, no camera" caption.
- Never present its alerts or timings as measurements.
- The buzzer and the live camera do not exist in this mode.
