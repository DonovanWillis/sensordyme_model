"""The detection taxonomy: every detection each mode offers, with HONEST support levels.

This is the single source of truth for what SensorDyme can and cannot detect.
It powers three things:
  * the AI-configure validator (cloud/app.py) — natural-language requests are
    checked against this before the system claims it can do something;
  * docs/DETECTIONS.md — the generated capability matrix;
  * the app/dashboard (via the cloud API) when listing per-mode capabilities.

Support levels:
  SUPPORTED    — implemented and running on-device today.
  PARTIAL      — a related signal exists but with meaningful caveats (listed).
  PLANNED      — designed for (config keys/zones exist) but not implemented.
  UNSUPPORTED  — cannot be done reliably with the current camera/models;
                 `requires` says what would be needed. The system must say so
                 instead of pretending.

Honesty rules encoded in the entries:
  * never claim intoxication, identity, or intent from appearance;
  * a zone-based detection ("near the pool") requires the USER to define the
    zone — the camera cannot know what a pool is;
  * audio-dependent detections are unsupported (no microphone in the build);
  * "unknown person" needs face recognition, which the product deliberately
    does not ship (privacy design).
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED = "supported"
PARTIAL = "partial"
PLANNED = "planned"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Detection:
    id: str
    title: str
    support: str
    how: str = ""        # which signal/model provides it (SUPPORTED/PARTIAL)
    caveat: str = ""     # honest limitation (PARTIAL)
    requires: str = ""   # what's missing (PLANNED/UNSUPPORTED)


def _d(id: str, title: str, support: str, how: str = "", caveat: str = "", requires: str = "") -> Detection:  # noqa: A002 - mirrors the Detection.id field
    return Detection(id=id, title=title, support=support, how=how, caveat=caveat, requires=requires)


TAXONOMY: dict[str, list[Detection]] = {
    # ------------------------------------------------------------------ driver_safety
    "driver_safety": [
        _d("eyes_off_road", "Eyes/head away from road (sustained)", SUPPORTED,
           how="FaceMesh yaw vs calibrated baseline; a wide-angle turn shaped like a "
               "mirror/blind-spot check gets more patience than an ordinary glance"),
        _d("looking_down", "Looking down (sustained)", SUPPORTED,
           how="FaceMesh pitch vs calibrated baseline (eyes_down event)"),
        _d("phone_visible", "Phone visible", SUPPORTED, how="YOLO 'cell phone' class"),
        _d("phone_use", "Phone use (phone + gaze off road, sustained)", SUPPORTED,
           how="YOLO phone + gaze-off fusion, sustained"),
        _d("texting_vs_reading", "Distinguish texting / typing / reading / social media", UNSUPPORTED,
           requires="screen-content inference the camera can't see; all surface as phone_use"),
        _d("eating", "Eating (sustained)", SUPPORTED, how="YOLO food classes near face, sustained"),
        _d("drinking", "Drinking (sustained)", SUPPORTED, how="YOLO cup/bottle classes, sustained"),
        _d("smoking_vaping", "Smoking or vaping", UNSUPPORTED,
           requires="a smoking-action model; COCO has no cigarette/vape class"),
        _d("grooming", "Applying makeup / grooming / adjusting hair", UNSUPPORTED,
           requires="an action-recognition model (temporal video classifier)"),
        _d("reaching", "Reaching to passenger seat / back seat / floor", PARTIAL,
           how="extreme head-pose excursion + face loss surface as distraction/attention_lost",
           caveat="cannot distinguish WHERE the driver reached; needs pose estimation for that"),
        _d("object_search", "Searching a bag / picking something up", UNSUPPORTED,
           requires="action recognition; surfaces only as generic distraction if gaze leaves road"),
        _d("excessive_movement", "Dancing / excessive body movement", PARTIAL,
           how="head-pose instability contributes to the risk-fusion score",
           caveat="no dedicated body-movement metric; needs pose estimation"),
        _d("frequent_glancing", "Turning/glancing away repeatedly", SUPPORTED,
           how="GlanceCounter: N off-road glances within a window"),
        _d("eyes_closed", "Sleeping / eyes closed unsafe duration", SUPPORTED,
           how="EAR below calibrated threshold, sustained (drowsiness event)"),
        _d("drowsiness_pattern", "Drowsiness (repeated prolonged closure)", SUPPORTED,
           how="PERCLOS window (fatigue_warning) + EAR events + risk fusion"),
        _d("yawning", "Yawning", UNSUPPORTED,
           requires="mouth-aspect-ratio analysis; landmarks exist, detector not implemented"),
        _d("head_nodding", "Head nodding / posture loss", UNSUPPORTED,
           requires="pitch-oscillation detector; not implemented"),
        _d("intoxication", "Intoxication indicators", UNSUPPORTED,
           requires="never claimed from appearance by policy; observable behaviors (eye closure, "
                    "erratic head movement) surface as their own events instead"),
        _d("driver_position", "Driver not properly positioned / left position", PARTIAL,
           how="face lost from frame fires attention_lost",
           caveat="cannot verify seating posture, only face presence/absence"),
        _d("seatbelt", "Seatbelt usage", UNSUPPORTED, requires="a custom-trained seatbelt model"),
        _d("occupant_count", "Multiple occupants", PARTIAL,
           how="YOLO person class can count",
           caveat="not wired into driver mode's event set; enabled in security mode"),
        _d("hands_on_wheel", "Hands on/off steering wheel", UNSUPPORTED,
           requires="hand + wheel detection; wheel rarely in frame at recommended mounting"),
        _d("custom", "User-configured distraction behavior", PARTIAL,
           how="AI-configure maps requests onto the supported detections above",
           caveat="requests outside this taxonomy are declined with the reason"),
    ],
    # ----------------------------------------------------------------- baby_monitoring
    "baby_monitoring": [
        _d("presence", "Baby/child present / absent from view", SUPPORTED,
           how="YOLO person detection + event lifecycle (occlusion-tolerant)"),
        _d("safe_zone_exit", "Leaving a configured safe area (crib, play area)", SUPPORTED,
           how="user-drawn safe zone + feet-point membership + persistence"),
        _d("restricted_zone", "Approaching configured danger areas (stairs, pool, stove, outlets…)", SUPPORTED,
           how="user-drawn restricted zones; the camera can't know what a pool is — "
               "the USER defines the areas, the device enforces them"),
        _d("unusual_stillness", "Unusually still for an abnormal period", SUPPORTED,
           how="motion energy inside the person box below threshold for a configured duration"),
        _d("position_change", "Changing position / rolling / motion bursts", PARTIAL,
           how="motion-energy spikes are logged as activity observations",
           caveat="cannot classify the movement type (rolling vs settling); needs pose/action models"),
        _d("flailing", "Frantic / flailing arm movement", SUPPORTED,
           how="MediaPipe Pose wrist trajectories over a rolling ~1.2s window: peak speed, peak "
               "acceleration, and direction-reversal frequency (core/activity.py) -- not a two-frame "
               "position delta, and not gated on the hand being raised"),
        _d("sleeping_vs_awake", "Baby sleeping vs awake", UNSUPPORTED,
           requires="reliable infant eye-state models; adult FaceMesh underperforms on infants"),
        _d("crying", "Baby crying", UNSUPPORTED, requires="a microphone + audio classifier (none fitted)"),
        _d("face_covered", "Face covered by blanket / face-down position", UNSUPPORTED,
           requires="infant-specific face/occlusion model; adult face detectors are not reliable "
                    "enough on infants for a safety-critical claim — we refuse to fake this one"),
        _d("climbing_falling", "Climbing / falling", PARTIAL,
           how="large rapid motion surfaces as an activity burst observation",
           caveat="the pose-based fall detector is tuned for adults, not infants"),
        _d("trapped", "Trapped or wedged in unsafe position", UNSUPPORTED,
           requires="3D pose + scene understanding beyond current models"),
        _d("unattended", "Adult leaving baby unattended", PARTIAL,
           how="person-count drop while presence continues in the safe zone",
           caveat="cannot reliably distinguish adult from child by detection alone"),
        _d("unknown_person", "Unknown person approaching baby", UNSUPPORTED,
           requires="face recognition — deliberately not shipped (privacy); person-count "
                    "changes ARE detected"),
        _d("multiple_people", "Multiple people entering the area", SUPPORTED,
           how="person-count change events"),
        _d("holding_bottle", "Bottle overlapping the child's box (feeding context)", PARTIAL,
           how="YOLO bottle class + box overlap with the child, logged (not an alert by default)",
           caveat="approximates 'holding' from overlap, same as core/rules.py's `holding` "
                  "predicate elsewhere in the product — a bottle merely propped nearby can "
                  "also match; bare bottle presence away from the child is NOT reported"),
        _d("holding_object", "Phone / cup / food / small utensil overlapping the child's box", PARTIAL,
           how="the same YOLO pass + box-overlap approximation as holding_bottle, generalized "
               "to any COCO class in config.yaml's object_classes (phone, cup, food items, "
               "utensils, scissors, remote by default), logged (not an alert by default)",
           caveat="same overlap approximation as holding_bottle; 'keys' specifically cannot be "
                  "detected — COCO (what the object detector actually knows) has no such class, "
                  "and this module refuses to claim a detection the model can't perform"),
        _d("choking_hazard", "A hand-held object on a closing trajectory toward the baby's mouth", PARTIAL,
           how="hand-anchored object association (core/hand_perception.py) + a geometric mouth-"
               "region proxy (core/detectors.py's upper_region, the same one holding_bottle/"
               "near_mouth already use) + trajectory (is distance to that region decreasing) + "
               "a size factor weighting objects small relative to the holding hand — critical, "
               "immediate alert when all combine past a risk threshold, never on a single frame",
           caveat="approximates 'mouth' from a box region, not a facial landmark (same infant-"
                  "face-model reasoning as face_covered below); does not and cannot NAME the "
                  "object — a coin and an unclassified UNKNOWN_OBJECT of the same relative size "
                  "score identically, honestly, rather than guessing a class the detector doesn't "
                  "have (see objects.small_personal); when more than one person is in frame the infant "
                  "is positively identified from crib-zone containment, temporal identity, "
                  "size/position stability and relative scale (modules/baby_monitoring.py's "
                  "InfantIdentifier); when no person earns that identification the module "
                  "DECLINES to score and raises infant_unidentified rather than scoring an "
                  "adult. Identification is weakest with no safe zone drawn, and says so"),
        _d("medical", "Medical conditions / diagnoses", UNSUPPORTED,
           requires="out of scope by policy — the system reports observable safety conditions only"),
    ],
    # -------------------------------------------------------------- elderly_monitoring
    "elderly_monitoring": [
        _d("fall", "Fall (rapid drop then stillness)", SUPPORTED,
           how="MediaPipe Pose hip/shoulder velocity + stillness confirmation (temporal, "
               "not single-frame; sitting/lying deliberately does NOT trigger it)"),
        _d("on_floor", "Person remaining on floor after fall", SUPPORTED,
           how="fall alert stays active until significant movement resumes"),
        _d("presence", "Person present / absent", PARTIAL,
           how="MediaPipe Pose: a tracked skeleton reads as a person, a label AI-configured "
               "present/absent rules can target (this mode runs no object detector)",
           caveat="no built-in presence/absence alert in this mode, and pose tracking -- not a "
                  "person detector -- decides presence, so someone the pose model loses (far away, "
                  "heavily occluded) reads as absent"),
        _d("zone_events", "Restricted areas / stairs / wandering", SUPPORTED,
           how="user-drawn zones tested against the ankle midpoint + persistence"),
        _d("bed_exit", "Getting out of bed (a user-drawn 'bed' safe zone)", SUPPORTED,
           how="safe_zone_exit -- the same mechanism baby_monitoring.py uses for a crib, "
               "applied to a bed zone; soft severity by default (getting up is often benign; "
               "a real fall afterward is caught separately by the fall detector above)"),
        _d("stillness", "Motionless for an unusual period", SUPPORTED,
           how="hip/shoulder centroid movement over a long window (deliberately "
               "10 minutes by default, to not flag ordinary resting/reading)"),
        _d("posture", "Standing / sitting / lying classification", PARTIAL,
           how="core/posture.py classifies standing / sitting / lying_down (plus on_floor) from "
               "pose landmarks; each is a label AI-configured rules can target",
           caveat="posture raises no built-in alert: falls are detected by trajectory, not "
                  "posture, precisely to avoid lying-down false alarms"),
        _d("struggling", "Struggling to stand / loss of balance / stumble / gait analysis", UNSUPPORTED,
           requires="temporal gait models; current pose pipeline detects falls, not gait quality"),
        _d("schedule", "Time-conditioned rules (nighttime activity, outside hours)", PARTIAL,
           how="AI-configured rules carry an HH:MM-HH:MM window, enforced on the device "
               "(core/rules.py) in the rule's timezone when it names one, else device local time",
           caveat="applies to AI-configured rules only; this mode's own alerts are not "
                  "time-conditioned"),
        _d("smoke_fire_flood", "Smoke / fire / flooding visible", UNSUPPORTED,
           requires="dedicated smoke/fire segmentation models (COCO has no such classes); "
                    "a smoke/heat sensor is the honest hardware answer"),
        _d("unknown_person", "Unknown person / caregiver identification", UNSUPPORTED,
           requires="face recognition — deliberately not shipped (privacy); person-count "
                    "changes ARE detected"),
        _d("multiple_people", "Multiple people present", SUPPORTED, how="person-count events"),
    ],
    # ------------------------------------------------------------------ security_monitoring
    "security_monitoring": [
        _d("person_detected", "Person detected / count change", SUPPORTED,
           how="YOLO person + confirm_frames debounce (occupancy_change)"),
        _d("zone_violation", "Entering prohibited zone / crossing boundary / approaching door", SUPPORTED,
           how="user-drawn zones + feet-point membership + persistence"),
        _d("lingering", "Person lingering in an area", SUPPORTED,
           how="zone membership with a longer persistence requirement"),
        _d("crowd", "Unusual crowd / activity level", SUPPORTED, how="person-count threshold"),
        _d("vehicle", "Vehicle entering/leaving a zone", SUPPORTED,
           how="YOLO car/truck/bus/motorcycle/bicycle/train (core.capabilities' 'vehicle' "
               "category) + zone membership + persistence -- vehicle_in_zone, same "
               "zone engine as person zone_violation, requested automatically once any "
               "zone is configured"),
        _d("animal", "Animal / pet in a zone", SUPPORTED,
           how="YOLO dog/cat/bird/... (core.capabilities' 'animal' category) + zone "
               "membership + persistence -- animal_in_zone, same mechanism as vehicle"),
        _d("package", "Package placed / removed / object moved", UNSUPPORTED,
           requires="object permanence + change detection against a scene baseline; "
                    "COCO 'package' isn't a class — needs a custom model"),
        _d("door_window_state", "Door/window open or closed", UNSUPPORTED,
           requires="per-installation scene understanding or contact sensors (the honest answer)"),
        _d("smoke_fire", "Smoke / fire visible", UNSUPPORTED,
           requires="dedicated detection models; a smoke detector is the honest hardware answer"),
        _d("known_person", "Known vs unknown person", UNSUPPORTED,
           requires="face recognition — deliberately not shipped (privacy)"),
        _d("mask_or_weapon", "Face covering / carrying weapon / climbing barrier", UNSUPPORTED,
           requires="specialized models with serious false-positive stakes; we refuse to "
                    "approximate these with the current stack — actions like 'person entered "
                    "restricted area' are reported instead of labels like 'burglar'"),
        _d("schedule", "Armed hours / motion during configured times", PARTIAL,
           how="AI-configured rules carry an HH:MM-HH:MM window, enforced on the device "
               "(core/rules.py) in the rule's timezone when it names one, else device local time",
           caveat="applies to AI-configured rules only; this mode's own zone/lingering/crowd "
                  "alerts are not time-conditioned"),
    ],
}


def find(mode: str, detection_id: str) -> Detection | None:
    for det in TAXONOMY.get(mode, []):
        if det.id == detection_id:
            return det
    return None


def supported_ids(mode: str) -> list[str]:
    return [d.id for d in TAXONOMY.get(mode, []) if d.support in (SUPPORTED, PARTIAL)]


def summary_for_llm() -> str:
    """Compact capability sheet injected into the AI-configure prompt so the model
    grounds its answers in what the device can actually do."""
    lines = []
    for mode, detections in TAXONOMY.items():
        lines.append(f"## {mode}")
        for det in detections:
            note = det.how if det.support in (SUPPORTED, PARTIAL) else det.requires
            caveat = f" CAVEAT: {det.caveat}" if det.caveat else ""
            lines.append(f"- {det.id} [{det.support.upper()}]: {det.title}. {note}{caveat}")
    return "\n".join(lines)
