"""Behaviour reasoning for driver attention (experimental, OFF by default).

Small pure components, each reporting ``unknown`` with a reason when its
inputs cannot support a claim:

  evidence.py   Assessment/UNKNOWN, FrameGapGuard (stale frames credit no
                time), KOfN persistence, box geometry (intersection over area)
  phone.py      PhoneInteraction: none / visible_only / held / near_face / in_use
  glance.py     GlanceAnalyzer: cabin zones from calibrated yaw+pitch, long
                glance, off-road budget, repeated glances
  eye_state.py  open / closed / looking_down / unknown, and DrowsinessEvidence
                that coasts through a brief face loss
  reasoner.py   BehaviorReasoner: the above composed per frame
  config.py     parse/validate ``modules.driver_attention.behavior_v2`` / ``.behavior``

M8.4 (always on, SHADOW: changes no validated alert; docs/BEHAVIOR_ENGINE.md):

  signals.py         typed per-signal evidence (freshness, provenance, missing reason) + Verdict
  engine.py          BehaviorEngine: shadow verdicts with abstention, phone evidence level
  phone_evidence.py  owner decision B: phone evidence levels; only held + eyes-off-road is risk;
                     M8.5: handling cues + supports_phone_use, the PRODUCTION phone_use gate
  association.py     which person is the driver, stated as a method (unevaluated)
  experimental.py    owner decision A: experimental alert families are observations
  evaluation.py      event-level metrics over labelled (synthetic) traces

Enabled only by ``behavior_v2: true`` in modules.driver_attention; nothing
here is validated on in-cabin data. See docs/BEHAVIOUR_REASONING.md.
"""

from core.behavior.config import BehaviorConfig, parse_behavior, validate_behavior
from core.behavior.evidence import UNKNOWN, Assessment
from core.behavior.reasoner import BehaviorEvent, BehaviorFrame, BehaviorReasoner

__all__ = [
    "UNKNOWN",
    "Assessment",
    "BehaviorConfig",
    "BehaviorEvent",
    "BehaviorFrame",
    "BehaviorReasoner",
    "parse_behavior",
    "validate_behavior",
]
