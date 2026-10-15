"""GBNF grammar generated from the capability registry.

llama.cpp-family runtimes (llama.cpp, Ollama, LM Studio) accept a GBNF grammar
that constrains sampling, so tokens which would break the grammar are never
emitted. That converts the biggest risk of a small local model — plausible
output in the wrong shape — from something to detect afterwards into something
that cannot happen.

The grammar is generated from `core.capabilities.registry()`, so the set of
targets it will accept is exactly the set the device can detect. This is the
difference between "the model was asked not to invent objects" and "the model
cannot emit a token spelling one".

Two things this deliberately does NOT do:

  * It does not replace validation. A grammar guarantees shape, never sense: a
    model can emit `{"target": "person", "predicate": "absent"}` for "alert me
    when someone arrives" and be perfectly grammatical about it.
  * It does not encode numeric bounds. GBNF can express digits but not ranges,
    so persist_s could still arrive as 99999. `RuleDraft.validate` clamps.
"""

from __future__ import annotations

from core.capabilities import registry
from jupiter.schema import NO_TARGET, NOTIFY_KINDS, SEVERITIES


def _alternation(values) -> str:
    """A GBNF alternation of quoted string literals."""
    escaped = [v.replace("\\", "\\\\").replace('"', '\\"') for v in sorted(values)]
    return " | ".join(f'"\\"{value}\\""' for value in escaped)


def build_grammar(reg=None) -> str:
    """A GBNF grammar for one Jupiter response.

    Targets and predicates are closed alternations drawn from the registry;
    free text (description, zone, explanation) stays an open string, because
    those are for the human and constraining them would only make the model
    fight the grammar. The target also admits NO_TARGET, for the same reason
    schema_for_registry does: without a legal "cannot", constrained decoding
    turns an honest refusal into a confident wrong label.
    """
    reg = reg or registry()
    labels = sorted(reg.labels())
    predicates = sorted(reg.predicates())

    if not labels or not predicates:
        # A registry with nothing in it means no vocabulary to constrain to.
        # Refusing is better than emitting a grammar that matches everything.
        raise ValueError("capability registry is empty; cannot build a grammar")

    return f"""
root        ::= "{{" ws "\\"understood\\"" ws ":" ws string ws ","
                ws "\\"rules\\"" ws ":" ws rules ws ","
                ws "\\"unsupported\\"" ws ":" ws unsupported ws "}}"

rules       ::= "[" ws (rule (ws "," ws rule)*)? ws "]"
unsupported ::= "[" ws (gap (ws "," ws gap)*)? ws "]"

rule        ::= "{{" ws "\\"target\\"" ws ":" ws target ws ","
                ws "\\"predicate\\"" ws ":" ws predicate ws ","
                ws "\\"secondary\\"" ws ":" ws secondary ws ","
                ws "\\"description\\"" ws ":" ws string ws ","
                ws "\\"zone\\"" ws ":" ws string ws ","
                ws "\\"persist_s\\"" ws ":" ws number ws ","
                ws "\\"min_count\\"" ws ":" ws integer ws ","
                ws "\\"confidence\\"" ws ":" ws number ws ","
                ws "\\"severity\\"" ws ":" ws severity ws ","
                ws "\\"cooldown_s\\"" ws ":" ws number ws ","
                ws "\\"schedule\\"" ws ":" ws string ws ","
                ws "\\"notify\\"" ws ":" ws notify ws "}}"

gap         ::= "{{" ws "\\"request\\"" ws ":" ws string ws ","
                ws "\\"reason\\"" ws ":" ws string ws "}}"

target      ::= label | {_alternation([NO_TARGET])}
secondary   ::= "\\"\\"" | label
label       ::= {_alternation(labels)}
predicate   ::= {_alternation(predicates)}
severity    ::= {_alternation(SEVERITIES)}
notify      ::= {_alternation(NOTIFY_KINDS)}

string      ::= "\\"" char* "\\""
char        ::= [^"\\\\\\x00-\\x1f] | "\\\\" ["\\\\/bfnrt]
number      ::= "-"? digit+ ("." digit+)?
integer     ::= "-"? digit+
digit       ::= [0-9]
ws          ::= [ \\t\\n]*
""".strip()


def grammar_stats(reg=None) -> dict:
    """What the grammar actually constrains — for docs and the eval report."""
    reg = reg or registry()
    labels = reg.labels()
    predicates = reg.predicates()
    return {
        "targets": len(labels),
        "predicates": len(predicates),
        # The executable answer space. Small enough that a 1.5B model is being
        # asked to pick from a menu, not to invent — which is the whole reason
        # a local model is viable for this task.
        "combinations": len(labels) * len(predicates),
    }
