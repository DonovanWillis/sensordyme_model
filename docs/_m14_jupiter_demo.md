# M14: the Jupiter step on the real `role: driver_safety` unit (snippet for INVESTOR_DEMO.md)

> **For the integrator:** this replaces step 14 of `docs/INVESTOR_DEMO.md` §3
> (and its "Rehearsed only on a simulated general-role device" note). Merge,
> then delete this file.

**Status: SIMULATED only.** It ran end to end on the digital twin: the real
backend, the real device runtime, scripted camera and models
(`tests/test_m14_jupiter_driver_safety.py`). It has **not** run on the Pi 5 +
Hailo-8L or on a physical iPhone. Rehearse it on the unit before showing it.

## Why not the M13 rule

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

## The step

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

## What the test proves (SIMULATED)

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

## Which existing policies run beside Driver Safety

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

## What remains hardware-only

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
- **Buzzer queue, read in code, not tested:** a rule's warning pattern and a
  driver warning pattern are the same rank on the one buzzer queue
  (`core/alerts.py _PatternPlayer`). With two patterns already waiting, a new
  one displaces the *older* waiting one of equal rank. That could be a
  driver's pattern; the driver episode repeats 5 s later. It needs three
  overlapping patterns.
