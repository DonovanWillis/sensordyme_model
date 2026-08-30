# Jupiter

Jupiter is SensorDyme's AI layer. It is a **system**, not a model — the models it
coordinates stay specialised, and most of them are the ones that were already
here.

```
JUPITER
├── Vision        DAMO-YOLO on the Hailo-8L · MediaPipe FaceMesh/Hands/Pose ·
│                 openpilot dmonitoring — unchanged, each still specialised
├── Reasoning     a small local instruct model, text only
├── Config        natural language → validated SensorDyme rules
└── Safety        core/capabilities.py + _validate_proposal + core/rules.py
                  — deterministic, and Jupiter never overrides them
```

**Vision does not run through Jupiter's language model.** The camera path stays
`camera → YOLO/MediaPipe → structured observation → deterministic rule engine`
at full framerate. A language model has no place in a real-time safety loop.

## What Jupiter is responsible for

Exactly one thing today: turning a sentence someone typed into a structured
monitoring rule. That narrowness is what makes a 1.5B local model a realistic
replacement for a frontier API here.

Jupiter does **not** decide what the device can do. `core/capabilities.py` does.
Anything Jupiter proposes is re-checked against the live registry, clamped, and
either executed or moved to `unsupported` with the specific gap. A wrong answer
from Jupiter produces a rejected rule, never a camera watching the wrong thing.

## Why this works with a small model

The answer space is closed and generated, not open:

```
84 targets × 9 predicates = 756 executable combinations
```

Jupiter is picking from a menu it is handed in the prompt, not inventing from
world knowledge. Combined with grammar-constrained decoding, the failure mode
"emits a plausible object the camera cannot see" is structurally impossible
rather than something to detect afterwards.

## Composing activities from what's already there

Early Configure-with-AI refused "alert if I'm sleeping" outright: `sleeping` is
not a predicate any registered capability provides, and rightly so — no
action-recognition model ships on the device. But `pose.body`'s `lying_down`
label and the rule engine's generic `persist_s` temporal confirmation
(core/rules.py) were already both available, and lying down + still for a
while is a real, if imperfect, proxy for sleep. Nothing needed to be built —
Jupiter's prompt just never taught the model to reach for that composition
instead of refusing.

`jupiter/prompt.py` now allows exactly two such compositions, because in both
the substitute is literally, physically part of what was asked, not a guess at
association:

  * **resting/sleeping** → `resting` held with a long `persist_s`. `resting`
    (core/activity.py) is a settled, still body — sitting OR lying, so a chair,
    desk, couch and bed all work; head droop raises its confidence for the
    asleep-at-a-desk case. Earlier versions used `lying_down`, which silently
    missed everyone who falls asleep sitting up.
  * **a person eating/drinking something specific** → `person` `near_mouth` a
    food/drink label from the sheet (at most 2 rules, the most common members).
    `near_mouth` (core/rules.py) requires the object to overlap the person's
    face region, not merely their box — the difference between "has a drink"
    and "raised it to drink" at box-detector fidelity. `holding` keeps its
    broader any-overlap meaning for requests that really do mean holding.

Both carry an honest caveat in `description` — "sustained stillness, not
confirmed sleep" — because the composition is real evidence, not a verified
fact. Gestures need no composition at all: `waving` and `hand_raised`
(core/activity.py, from wrist trajectories over ~1.2s — a wave is a raised
wrist with ≥3 wide direction reversals, so reaching out and back does not
count) are direct labels served by the pose module. Everything else — a
clothing item, an environmental condition (smoke, water), an unnamed object —
still refuses, now with a registry entry stating specifically what model would
enable it (see `scene.fire_smoke`, `scene.water`, `appearance.headwear`,
`objects.package`, `objects.small_personal` in core/capabilities.py).
Early iterations that phrased this more loosely had the model inventing
substitutes for unrelated requests too (a "helmet" request producing a rule
for baseball gear, "raising a hand" producing one for a handbag) — the two
allowed patterns above are deliberately named explicitly rather than
described generally, because a small model given a general principle
generalises past where the evidence actually holds.

**This does not by itself make a `lying_down` rule fire on a device.** AI
Custom Monitoring's detector selection (`core/monitoring_mode.py`) has to
choose a module that actually runs pose estimation for that — `occupancy`
(the default, no MediaPipe) does not. See `select_custom_module` and
`services/runtime.py`'s `_apply_monitoring_mode`.

## Layout

| File | Job |
|---|---|
| `jupiter/schema.py` | The rule schema, **derived from the capability registry** |
| `jupiter/grammar.py` | GBNF generated from that registry, for constrained decoding |
| `jupiter/prompt.py` | System prompt + the generated capability sheet + few-shot |
| `jupiter/security.py` | Request sanitising, injection flagging, response bounds |
| `jupiter/providers.py` | Provider registry — Jupiter (local), OpenAI-compat, Anthropic |
| `jupiter/router.py` | Which model answers which role — see Model routing below |
| `jupiter/interpret.py` | The `configure` pipeline: sanitise → prompt → provider → parse → validate |
| `jupiter/explain.py` | The `explain` pipeline: event + rule → facts → optional paraphrase |
| `jupiter/eval/synth.py` | Dataset generated from the registry, with refusals |
| `jupiter/eval/harness.py` | Scoring and provider comparison |

Nothing hardcodes the vocabulary. Register a can-detection model and the schema,
the grammar, the prompt and the dataset all widen together, because all four read
`core.capabilities.registry()`.

## Model routing

Jupiter has exactly two roles today, and `jupiter/router.py` resolves each one
independently rather than every call site sharing one hardcoded model string:

| Role | Used by | Shape of the task | Default |
|---|---|---|---|
| `configure` | Configure-with-AI (`jupiter/interpret.py`) | One sentence → structured rule. Once per user action, not per frame. | `jupiter` / `qwen2.5:7b-instruct` |
| `explain` | "Why did I get this alert?" (`jupiter/explain.py`) | Paraphrase an already-computed fact list. Once per fired event, latency is user-visible. | `jupiter` / `qwen2.5:1.5b-instruct` |

Resolution order (highest priority first): a role-specific env var
(`JUPITER_ROLE_CONFIGURE_MODEL` / `JUPITER_ROLE_EXPLAIN_MODEL`, plus
`_PROVIDER` / `_BASE_URL`) → an optional `jupiter_config.yaml` (see
`jupiter_config.yaml.example`) → the repo-wide `SENSORDYME_AI_PROVIDER` /
`JUPITER_MODEL` / `JUPITER_BASE_URL` every deployment already sets → the
built-in defaults above. Nothing here requires the YAML file to exist; a
fresh install gets the defaults with zero configuration.

## Running Jupiter

Jupiter speaks HTTP to a model server rather than loading weights in-process,
which keeps the API server light and lets the model be restarted independently.

```bash
# 1. install a model server (macOS/Linux)
#    https://ollama.com  — or any OpenAI-compatible server (vLLM, LM Studio)
ollama serve

# 2. pull a model
ollama pull qwen2.5:1.5b-instruct

# 3. check Jupiter can reach it
python scripts/jupiter_eval.py --check

# 4. run the backend — no API key needed
.venv/bin/uvicorn cloud.app:app --host 0.0.0.0 --port 8787
```

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SENSORDYME_AI_PROVIDER` | `jupiter` | repo-wide provider; a role-specific `JUPITER_ROLE_*_PROVIDER` overrides it |
| `JUPITER_MODEL` | role default (see Model routing) | repo-wide model; a role-specific `JUPITER_ROLE_*_MODEL` overrides it |
| `JUPITER_BASE_URL` | `http://localhost:11434` | model server address |
| `SENSORDYME_ALLOW_EXTERNAL_AI` | unset (off) | permit the `configure` role to fall back to Anthropic when the local model fails |
| `SENSORDYME_AI_MODEL` | `claude-sonnet-5` | cloud fallback model, only used when the fallback above is enabled |
| `ANTHROPIC_API_KEY` | unset | **optional** — needed only if the cloud fallback or `--provider anthropic` benchmarking is used |

External providers are off unless explicitly enabled. A user's words are never
sent off the box because a key happens to exist in the environment.

## Model choice

Both roles default to the local, zero-cost `jupiter` provider — Ollama serving
a quantised model over HTTP, no API key, nothing leaves the box unless the
optional cloud fallback is explicitly turned on.

**Measured, not assumed** (`jupiter/eval/harness.py`, 40 generated cases, seed
1, on an Apple M5 / 16GB Mac):

| Model | target accuracy | predicate accuracy | refusal accuracy | latency mean | latency p95 |
|---|---|---|---|---|---|
| `qwen2.5:1.5b-instruct` | 78.1% | 78.1% | 97.5% | 1.0s | 1.6s |
| `qwen2.5:7b-instruct` | **100%** | 93.75% | **100%** | 5.4s | 6.5s |

The `configure` role runs once per user action, not per frame, so the ~5s
round trip is not on any latency-sensitive path — it is the difference between
7 wrong rules and 0 in that sample, for a feature whose entire value depends on
the rule actually matching what was asked. `qwen2.5:7b-instruct` is the
default for `configure` for that reason. `explain` stays on
`qwen2.5:1.5b-instruct`: it only ever paraphrases a fact list `jupiter/explain.py`
has already computed correctly with no model at all, so speed matters more
than the last percent of fluency, and a fired alert should not wait 5 seconds
to say why.

Both ship under Apache 2.0 and are widely available quantised. Re-run the
comparison above — with a different model, a different sample size, or
against your own dataset — before trusting these numbers on new hardware; see
`scripts/jupiter_eval.py` and `jupiter_config.yaml.example`.

### The optional cloud fallback is not Opus

`SENSORDYME_ALLOW_EXTERNAL_AI=1` lets the `configure` role retry once against
Anthropic when the local model genuinely fails to answer. If enabled, it
defaults to `claude-sonnet-5`, not `claude-opus-5` — Configure-with-AI is
schema-constrained rule extraction against a fixed, generated vocabulary
(`jupiter/schema.py`), the same task the local model handles above. Opus's
extra reasoning is not spent on anything this task asks for; at roughly 1.7x
Sonnet's per-token price for a job this narrow, it is the wrong default. Set
`SENSORDYME_AI_MODEL=claude-opus-5` to use it anyway.

## Where Jupiter should run

**On the backend, not the Pi** — at least to begin with:

- Configure-with-AI is already a backend endpoint; the prompt never touches the Pi.
- The Pi's CPU is what holds the camera at framerate. A 1.5B model would compete
  with the detector for seconds at a time.
- The backend is infrastructure you control, so the privacy and cost goals are met
  either way.

The Pi has ~3.2GB free with the full stack running (measured: 847MB used of
4049MB). `qwen2.5:1.5b-instruct` (~1GB at Q4) would fit; `qwen2.5:7b-instruct`
(~4.7GB at Q4) would not, on top of everything else already running. Either
way, fitting was never the constraint — the camera is, and this whole question
is moot as long as `configure` runs on the backend, which has no framerate to
protect.

**The Hailo AI HAT+ should stay on vision.** It is a dataflow CNN accelerator;
autoregressive decoding with a KV cache is a poor architectural fit, its compiler
targets vision op-sets, and it is already running YOLO.

## Benchmarking

```bash
# Jupiter alone
python scripts/jupiter_eval.py --provider jupiter --cases 200

# head to head against Claude
pip install anthropic
ANTHROPIC_API_KEY=sk-ant-... python scripts/jupiter_eval.py \
    --provider jupiter --provider anthropic --cases 200

# what constrained decoding is worth
python scripts/jupiter_eval.py --provider jupiter --no-grammar --cases 200

# save a baseline to compare future changes against
python scripts/jupiter_eval.py --provider jupiter --out data/jupiter-baseline.json
```

Metrics: `parse_rate`, `schema_rate`, `target_accuracy`, `predicate_accuracy`,
`refusal_accuracy`, `hallucination_rate`, latency mean/p95.

`hallucination_rate` is the one to watch. A model that declines a possible
request is a nuisance; a model that proposes a rule for an impossible one leaves
someone believing they are protected when nothing is watching.

## Security model

Jupiter has no tools, no shell, no filesystem, and no network. It reads a string
and returns JSON. **Prompt injection cannot make Jupiter do anything — only say
something**, and everything it says is re-validated before it can act.

- Requests are length-capped and refused rather than truncated.
- The request fence is stripped so a prompt cannot escape its own block.
- Blatant override attempts are flagged and logged, not blocked (flagging is a
  signal; validation is the control).
- Responses are bounded: max 8 rules, capped explanation length.
- Every rule is re-assessed against the capability registry and clamped.

## Training data

`jupiter/eval/synth.py` generates ground truth from the registry — enumerate the
executable combinations, template language around them. Labels cannot drift from
the device's real capabilities because both come from the same source, and
`validate_dataset()` asserts it.

Two honest limits: templated language is not user language, and every generated
supported case is answerable — which is why ~20% of every dataset is refusals
built from labels the registry explicitly lacks.

**Do not fine-tune first.** Try base model + capability sheet + few-shot +
constrained decoding, and measure. Fine-tune only if the harness says the gap is
real.

## Temperature 0 is not the same as deterministic

`qwen2.5:7b-instruct` is run at `temperature: 0`, which is often assumed to mean
"same input, same output, every time." Observed while building the activity
composition above: it is not, in practice, on this stack. Identical requests sent
seconds apart occasionally produced different rules — most often on longer or
less literal phrasings ("someone appears to be sleeping" vs. "I am sleeping"),
and the direction of the difference wasn't always toward the better answer. This
is consistent with known batching/kernel non-associativity in llama.cpp-family
runtimes, not a bug in this repo's code, but it means "temperature 0" cannot be
the safety argument — the registry-backed validation in `jupiter/schema.py`
(`RuleDraft.validate`) and `cloud/app.py`'s `_validate_proposal` are, because
the model's own consistency is not guaranteed. Treat any specific accuracy
number in this document as a sample from a distribution, not a guarantee, and
prefer re-running `scripts/jupiter_eval.py` over trusting a handful of manual
checks — the eval harness's repeat-and-aggregate approach is the correct way to
measure this; a few one-off calls are not.

## Status

Built and tested: architecture, provider abstraction, registry-derived schema and
grammar, prompt, security, dataset generator, eval harness, backend integration,
per-role model routing. 65 Jupiter tests, 710 in the full suite.

**Measured, not assumed** (see Model choice above): `qwen2.5:1.5b-instruct` vs
`qwen2.5:7b-instruct` scored on this repo's own eval harness against a live
Ollama server, on an Apple M5/16GB Mac — the numbers, and the reasoning for
choosing one per role, are in that section rather than repeated here. Re-run
`scripts/jupiter_eval.py` whenever the hardware, model, or capability registry
changes; nothing here is a permanent conclusion.

**What Jupiter still is not**, stated plainly rather than implied by omission:
no tool use, no memory across requests, no web access, no vision routing (the
camera pipeline is deliberately separate — see the top of this doc), and no
verification pass beyond the schema/registry validation every rule already
gets. It turns one sentence into one structured rule, safely, for
free. Anything beyond that is future work, not a claim this document is making.
