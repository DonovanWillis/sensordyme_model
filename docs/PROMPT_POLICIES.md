# Prompt policies

How a sentence becomes something the device monitors, and what it never
becomes. Code: `jupiter/policy.py`, `core/rules.py` (dwell), `core/taxonomy.py`
(`REFUSALS`), `cloud/app.py` (`_interpret_request`, `_validate_proposal`).
Tests: `tests/test_m6_prompt_policies.py`, `tests/test_m6_dwell_rules.py`,
`tests/test_m6_policy_api.py`.

No executable code is generated. No language model runs per frame. A policy
is data, compiled to the same rule dict the device's deterministic engine
already runs.

## Pipeline (`POST /v1/ai/configure`)

1. **Refusals** (`refusal_for`). Requests about identity (recognise, stranger,
   known or unknown person), intent (suspicious), emotion, sound (crying,
   barking, shouting), intoxication, medical states, or repeated occurrences
   in a window ("3 times") are refused **before** any interpreter sees them.
   So are the driver module's **experimental** families (M8.4: the combined
   risk score `high_risk` / `elevated_risk`, `erratic_hand_movement`; "high
   risk", "risk score", "erratic hands", "hand movement", "fidgeting"):
   they are recorded as experimental observations and never sound or notify,
   and no interpreter may offer them back as an alert (`experimental_signal`).
   The reason comes from `core/taxonomy.py REFUSALS`, with `support:
   "unsupported"` and `category`. Nothing is set up (`rules: []`,
   `source: "policy_guard"`). This order is what guarantees "a stranger at the
   door" never becomes "a person at the door".
2. **Deterministic parser** (`parse`). It reads a bounded grammar of area and
   duration requests. It claims a request only when it reads **all** of it;
   otherwise it returns None and step 3 runs (`source: "policy_parser"`).
3. **Language model** (Jupiter / Anthropic, unchanged).
4. **Keyword fallback** (unchanged).

Every rule from every step then passes `_validate_proposal`: the capability
registry, the zone and secondary checks, clamps, and the **typed policy
schema** (`policy_from_rule`). A rule that does not fit the schema (a dwell
with no area, a condition the device does not evaluate, an object it cannot
detect) moves to `unsupported` with the reason. That holds for model output
too.

**Typed reasons (M8.4).** Every `unsupported` entry the preview returns
carries a `code` from `jupiter/policy.py UNSUPPORTED_CODES`, whatever produced
it: the refusal categories above (`identity`, `emotion`, `intent`, `audio`,
`intoxication`, `medical`, `repetition_window`, `experimental_signal`),
`not_detectable` (the detector does not know it; `support` says what it would
take), `needs_area`, `needs_second_object`, `no_target`, `invalid_rule`,
`not_beside_mode`, `interpreter_unavailable`, and `not_supported` for anything
else. The app renders one headline per code (`ios/Sources/JupiterRefusal.swift`)
and, for a policy refusal, says why rather than "to enable". The field is
additive; an older app ignores it.

The keyword fallback (step 4) matches whole words and their plain
inflections only, never "car" or "phone" (a car in the driveway is not a
driver to watch), and its explanation starts "No custom rule was set up ...
this is the closest built-in mode".

The owner confirms before anything runs. The response is a draft
(`draft_version`), and only Activate (`POST .../preset` with `preset: ""`)
promotes it.

## The typed policy

`/v1/ai/configure` returns `policies` beside `rules` (same order, `rule_id`
links them):

```json
{
  "rule_id": "r1a2b3c4d5e", "schema_version": 1,
  "targets": ["person"], "condition": "lingering",
  "zone": {"name": "front door", "drawn": true},
  "status": "ready",
  "temporal": {"dwell_s": 10.0},
  "required_capabilities": ["camera", "inference", "objects"],
  "severity": "warning", "notify": "push", "schedule": null,
  "explanation": {
    "summary": "Alert when a person stays inside the area 'front door' for 10 seconds or longer without leaving. ...",
    "monitors": ["the object detector's 'person' class", "whether a subject's feet are inside ...", "..."],
    "does_not_monitor": ["who the person is (no face recognition)", "why anyone is there (intent is never inferred)", "..."],
    "status": null
  }
}
```

- `zone` always means an area the owner drew. If none with that name exists,
  `status` is `needs_zone` and `explanation.status` says what to draw. The
  rule is delivered anyway and stays inert ("waiting") until the area exists.
  It never fires frame-wide. Generic phrases ("this marked area") bind to the
  only drawn area when there is exactly one. Otherwise the rule waits for an
  area named "marked area".
- `temporal` is exactly one of `{"dwell_s"}` (one subject continuously
  inside), `{"sustain_s"}` (the condition held that long), or
  `{"count", "sustain_s"}` (at least N, held). A count over a time window
  ("3 times in 10 minutes") is refused.
- `required_capabilities` uses the `core/monitoring_state.py` vocabulary.

## Grammar the parser reads

Each clause starts with a trigger: an optional `please` / `can you` / `could
you`, then `alert me | notify me | tell me | warn me | ping me | buzz me | let me
know | I want to know | I want an alert | send me an alert / a notification /
a push` + `if | when | whenever | as soon as | any time | every time`, or a
bare `if | when`. Clauses are split on sentence ends and on `and` + a new
trigger. M8.4 widened the phrasings, never the capabilities: every row below
still compiles to the same four conditions.

| Clause | Policy |
|---|---|
| `<subject> stays / remains / lingers / waits / is  in / inside / at / on  <area>  [for] [more than / longer than / at least] N s/min/h [or longer]` | `lingering` + `dwell_s` |
| `<subject> stays / spends / lingers [more than] N s/min/h in <area>` | `lingering` + `dwell_s` |
| `<subject> enters <area>` / `goes into <area>` / `is in <area>` / `shows up at <area>` / `arrives at <area>` / `is seen / detected / spotted in <area>` | `in_zone`, sustain 2 s |
| `there is / there's <subject> in <area>` | `in_zone`, sustain 2 s |
| `[there is] nobody / no one in <area> [for N]` / `<area> is empty [for N]` | `absent` person, zone |
| `[there are] more than N / at least N / N or more <things> [are] in <area> [for N]` | `count` |

Not read (left to the model, whose output is validated): posture verbs ("sits
in the kitchen"), relations to an object ("near the car", "on the couch"),
time qualifiers ("after dark"). Tests: `tests/test_m84_jupiter_grammar.py`
(including every suggestion the app offers).

`<subject>` is someone / anyone / a person / people (`person`) or `a|an|any|the
<label>` for a label the device detects. A label the registry declares but
cannot serve (package, keys, helmet, smoke...) is refused with its reason. An
unknown word ("intruder") is not claimed: the model gets it, and the
registry still checks the result. `<area>` is refused, and the request left
to the model, when it names an object (couch, bed, car: that is a relation,
not an area) or carries a qualifier the grammar does not read (after dark,
between, unless, ...).

## Dwell on the device (`core/rules.py`)

A zone rule with `dwell_s` (predicates `lingering`, `in_zone` or `present`)
alerts when **one subject** has been inside the zone continuously for
`dwell_s`:

- each in-zone box is matched to the nearest earlier subject (centre distance
  in box diagonals, scaled by the time since that subject was last seen);
- a subject's clock starts when it first appears inside the zone;
- a gap of up to 1 s (occlusion) keeps the clock. A longer one ends the visit,
  and coming back starts a new one;
- the dwell then goes through the shared `EventEngine`. It confirms on the
  frame after the threshold, stays one episode while the subject stays,
  resolves after 5 s of absence, and then honours the rule's cooldown;
- outside the rule's hours, or with its area undrawn, every visit is
  forgotten. Time not observed never counts;
- a dwell with no zone, or an unreadable `dwell_s`, is refused (fail closed).

Verified with a deterministic clock (10 fps). Leaving at 9 s gives no alert.
Staying 10 s gives exactly one alert, between 10.0 s and 10.15 s. A succession
of people adding up to 10 s gives no alert. A 0.5 s occlusion keeps the clock;
a 2 s absence restarts it.

The backend also sets `persist_s = dwell_s`. Firmware without per-subject
dwell (no `dwell_in_zone` feature) then runs "the area occupied for that
long" instead of refusing the set.

**Not identity.** Two people crossing inside the area can swap clocks. The
explanation says so. No clock can exceed the longer of the two real visits.

## Coexistence and role floors

Policies are additive to each other: N rules run together (all or nothing on
the device). On a `driver_safety`-role device they run beside the driver
preset, and the role's floor is untouched, because rules can add alerts and
cannot change module thresholds. A preset or settings switch keeps them
there (see docs/PRESET_LIBRARY.md, `keep_rules`). On other roles, custom
rules replace the preset, and the app is told so.

## Known limits

- The deterministic grammar is deliberately small. Anything outside it goes
  to the language model as before.
- Refusal matching is lexical. It over-refuses some benign phrasings ("the
  sound of the door", "a happy dog"); that costs a rephrase, and
  under-refusing would cost honesty.
- "Child" and "kid" map to `person`. The explanation states that adults and
  children are not told apart.
