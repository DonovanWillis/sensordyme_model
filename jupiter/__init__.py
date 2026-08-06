"""Jupiter: SensorDyme's AI layer.

Jupiter is the SYSTEM that coordinates SensorDyme's models — not a model itself,
and not a rename of the ones already here. The distinction matters:

    Jupiter
      ├─ Vision      YOLO11 (Hailo/CPU), MediaPipe FaceMesh/Hands/Pose,
      │              openpilot dmonitoring — each stays specialised
      ├─ Reasoning   a small local instruct model, text only
      ├─ Config      natural language -> validated SensorDyme rules
      └─ Safety      the deterministic engine, which Jupiter never overrides

What Jupiter is responsible for is narrow on purpose: turning a sentence a
person typed into a structured monitoring rule. Everything that makes the result
trustworthy — capability assessment, validation, clamping, temporal confirmation
— stays in deterministic code that treats Jupiter's output as untrusted input.

That boundary is what makes a 1.5B local model a realistic replacement for a
frontier API here. The model does not need to be right about what the device can
do; core/capabilities.py decides that, and anything the model invents is rejected
before it reaches a camera.

Vision deliberately does NOT run through Jupiter's reasoning model. A language
model has no place in a real-time safety loop: the camera path stays
YOLO/MediaPipe -> structured observation -> deterministic rule engine, at
30fps, with no LLM in it.
"""

from jupiter.schema import RULE_SCHEMA, RuleDraft, schema_for_registry
from jupiter.providers import (
    JupiterUnavailable,
    ProviderResult,
    available_providers,
    get_provider,
)

__all__ = [
    "RULE_SCHEMA",
    "RuleDraft",
    "schema_for_registry",
    "JupiterUnavailable",
    "ProviderResult",
    "available_providers",
    "get_provider",
]

VERSION = "0.1"
