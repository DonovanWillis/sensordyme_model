# Jupiter

Jupiter is SensorDyme's AI layer. It is a **system**, not a model — the models it
coordinates stay specialised, and most of them are the ones that were already
here.

```
JUPITER
├── Vision        YOLO11 (Hailo/CPU) · MediaPipe FaceMesh/Hands/Pose ·
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

## Layout

| File | Job |
|---|---|
| `jupiter/schema.py` | The rule schema, **derived from the capability registry** |
| `jupiter/grammar.py` | GBNF generated from that registry, for constrained decoding |
| `jupiter/prompt.py` | System prompt + the generated capability sheet + few-shot |
| `jupiter/security.py` | Request sanitising, injection flagging, response bounds |
| `jupiter/providers.py` | Provider registry — Jupiter (local), OpenAI-compat, Anthropic |
| `jupiter/interpret.py` | The pipeline: sanitise → prompt → provider → parse → validate |
| `jupiter/eval/synth.py` | Dataset generated from the registry, with refusals |
| `jupiter/eval/harness.py` | Scoring and provider comparison |

Nothing hardcodes the vocabulary. Register a can-detection model and the schema,
the grammar, the prompt and the dataset all widen together, because all four read
`core.capabilities.registry()`.

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
| `SENSORDYME_AI_PROVIDER` | `jupiter` | `jupiter`, `jupiter_openai_compat`, `anthropic`, `scripted` |
| `JUPITER_MODEL` | `qwen2.5:1.5b-instruct` | model name on the server |
| `JUPITER_BASE_URL` | `http://localhost:11434` | model server address |
| `SENSORDYME_ALLOW_EXTERNAL_AI` | unset (off) | permit falling back to an external API |
| `ANTHROPIC_API_KEY` | unset | **optional** — benchmark provider only |

External providers are off unless explicitly enabled. A user's words are never
sent off the box because a key happens to exist in the environment.

## Model choice

`qwen2.5:1.5b-instruct` is the default because it is small enough to leave the
Pi's RAM alone (~1GB at Q4), has strong instruction-following for its size, ships
under Apache 2.0, and is widely available quantised. **This is a starting point,
not a measured winner** — the eval harness exists so the choice can be settled
with numbers instead of reputation. `JUPITER_MODEL` changes it without code.

## Where Jupiter should run

**On the backend, not the Pi** — at least to begin with:

- Configure-with-AI is already a backend endpoint; the prompt never touches the Pi.
- The Pi's CPU is what holds the camera at framerate. A 1.5B model would compete
  with the detector for seconds at a time.
- The backend is infrastructure you control, so the privacy and cost goals are met
  either way.

The Pi has ~3.2GB free with the full stack running (measured: 847MB used of
4049MB), so the weights would *fit*. Fitting is not the constraint; the camera is.

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

## Status

Built and tested: architecture, provider abstraction, registry-derived schema and
grammar, prompt, security, dataset generator, eval harness, backend integration.
43 Jupiter tests, 402 in the suite.

**Not yet measured: Jupiter's accuracy, latency, or memory use** — no local model
runtime was available in the environment where this was built. `--check` will
tell you the moment that changes, and the harness produces real numbers the first
time you run it against a served model.
