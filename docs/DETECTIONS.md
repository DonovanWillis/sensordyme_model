# SensorDyme Detection Capability Matrix

*Generated from `core/taxonomy.py` by `scripts/generate_detections_doc.py` — edit the
taxonomy, not this file.* This is the honest contract of what each mode can and cannot
detect. ✅ supported today · 🟡 partial (caveats listed) · 🔵 planned · ❌ not possible
with the current camera/models (what it would need is listed).

## driver_safety

| | Detection | How / Caveat / Requires |
|---|---|---|
| ✅ | Eyes/head away from road (sustained) (`eyes_off_road`) | FaceMesh yaw vs calibrated baseline; a wide-angle turn shaped like a mirror/blind-spot check gets more patience than an ordinary glance |
| ✅ | Looking down (sustained) (`looking_down`) | FaceMesh pitch vs calibrated baseline (eyes_down event) |
| ✅ | Phone visible (`phone_visible`) | YOLO 'cell phone' class |
| ✅ | Phone use (phone + gaze off road, sustained) (`phone_use`) | YOLO phone + gaze-off fusion, sustained |
| ❌ | Distinguish texting / typing / reading / social media (`texting_vs_reading`) | screen-content inference the camera can't see; all surface as phone_use |
| ✅ | Eating (sustained) (`eating`) | YOLO food classes near face, sustained |
| ✅ | Drinking (sustained) (`drinking`) | YOLO cup/bottle classes, sustained |
| ❌ | Smoking or vaping (`smoking_vaping`) | a smoking-action model; COCO has no cigarette/vape class |
| ❌ | Applying makeup / grooming / adjusting hair (`grooming`) | an action-recognition model (temporal video classifier) |
| 🟡 | Reaching to passenger seat / back seat / floor (`reaching`) | extreme head-pose excursion + face loss surface as distraction/attention_lost **Caveat:** cannot distinguish WHERE the driver reached; needs pose estimation for that |
| ❌ | Searching a bag / picking something up (`object_search`) | action recognition; surfaces only as generic distraction if gaze leaves road |
| 🟡 | Dancing / excessive body movement (`excessive_movement`) | head-pose instability contributes to the risk-fusion score **Caveat:** no dedicated body-movement metric; needs pose estimation |
| ✅ | Turning/glancing away repeatedly (`frequent_glancing`) | GlanceCounter: N off-road glances within a window |
| ✅ | Sleeping / eyes closed unsafe duration (`eyes_closed`) | EAR below calibrated threshold, sustained (drowsiness event) |
| ✅ | Drowsiness (repeated prolonged closure) (`drowsiness_pattern`) | PERCLOS window (fatigue_warning) + EAR events + risk fusion |
| ❌ | Yawning (`yawning`) | mouth-aspect-ratio analysis; landmarks exist, detector not implemented |
| ❌ | Head nodding / posture loss (`head_nodding`) | pitch-oscillation detector; not implemented |
| ❌ | Intoxication indicators (`intoxication`) | never claimed from appearance by policy; observable behaviors (eye closure, erratic head movement) surface as their own events instead |
| 🟡 | Driver not properly positioned / left position (`driver_position`) | face lost from frame fires attention_lost **Caveat:** cannot verify seating posture, only face presence/absence |
| ❌ | Seatbelt usage (`seatbelt`) | a custom-trained seatbelt model |
| 🟡 | Multiple occupants (`occupant_count`) | YOLO person class can count **Caveat:** not wired into driver mode's event set; enabled in security mode |
| ❌ | Hands on/off steering wheel (`hands_on_wheel`) | hand + wheel detection; wheel rarely in frame at recommended mounting |
| 🟡 | User-configured distraction behavior (`custom`) | AI-configure maps requests onto the supported detections above **Caveat:** requests outside this taxonomy are declined with the reason |

## baby_monitoring

| | Detection | How / Caveat / Requires |
|---|---|---|
| ✅ | Baby/child present / absent from view (`presence`) | YOLO person detection + event lifecycle (occlusion-tolerant) |
| ✅ | Leaving a configured safe area (crib, play area) (`safe_zone_exit`) | user-drawn safe zone + feet-point membership + persistence |
| ✅ | Approaching configured danger areas (stairs, pool, stove, outlets…) (`restricted_zone`) | user-drawn restricted zones; the camera can't know what a pool is — the USER defines the areas, the device enforces them |
| ✅ | Unusually still for an abnormal period (`unusual_stillness`) | motion energy inside the person box below threshold for a configured duration |
| 🟡 | Changing position / rolling / motion bursts (`position_change`) | motion-energy spikes are logged as activity observations **Caveat:** cannot classify the movement type (rolling vs settling); needs pose/action models |
| ✅ | Frantic / flailing arm movement (`flailing`) | MediaPipe Pose wrist trajectories over a rolling ~1.2s window: peak speed, peak acceleration, and direction-reversal frequency (core/activity.py) -- not a two-frame position delta, and not gated on the hand being raised |
| ❌ | Baby sleeping vs awake (`sleeping_vs_awake`) | reliable infant eye-state models; adult FaceMesh underperforms on infants |
| ❌ | Baby crying (`crying`) | a microphone + audio classifier (none fitted) |
| ❌ | Face covered by blanket / face-down position (`face_covered`) | infant-specific face/occlusion model; adult face detectors are not reliable enough on infants for a safety-critical claim — we refuse to fake this one |
| 🟡 | Climbing / falling (`climbing_falling`) | large rapid motion surfaces as an activity burst observation **Caveat:** the pose-based fall detector is tuned for adults, not infants |
| ❌ | Trapped or wedged in unsafe position (`trapped`) | 3D pose + scene understanding beyond current models |
| 🟡 | Adult leaving baby unattended (`unattended`) | person-count drop while presence continues in the safe zone **Caveat:** cannot reliably distinguish adult from child by detection alone |
| ❌ | Unknown person approaching baby (`unknown_person`) | face recognition — deliberately not shipped (privacy); person-count changes ARE detected |
| ✅ | Multiple people entering the area (`multiple_people`) | person-count change events |
| 🟡 | Bottle overlapping the child's box (feeding context) (`holding_bottle`) | YOLO bottle class + box overlap with the child, logged (not an alert by default) **Caveat:** approximates 'holding' from overlap, same as core/rules.py's `holding` predicate elsewhere in the product — a bottle merely propped nearby can also match; bare bottle presence away from the child is NOT reported |
| 🟡 | Phone / cup / food / small utensil overlapping the child's box (`holding_object`) | the same YOLO pass + box-overlap approximation as holding_bottle, generalized to any COCO class in config.yaml's object_classes (phone, cup, food items, utensils, scissors, remote by default), logged (not an alert by default) **Caveat:** same overlap approximation as holding_bottle; 'keys' specifically cannot be detected — COCO (what the object detector actually knows) has no such class, and this module refuses to claim a detection the model can't perform |
| ❌ | Medical conditions / diagnoses (`medical`) | out of scope by policy — the system reports observable safety conditions only |

## elderly_monitoring

| | Detection | How / Caveat / Requires |
|---|---|---|
| ✅ | Fall (rapid drop then stillness) (`fall`) | MediaPipe Pose hip/shoulder velocity + stillness confirmation (temporal, not single-frame; sitting/lying deliberately does NOT trigger it) |
| ✅ | Person remaining on floor after fall (`on_floor`) | fall alert stays active until significant movement resumes |
| ✅ | Person present / absent (`presence`) | YOLO person + event lifecycle |
| ✅ | Restricted areas / stairs / wandering (`zone_events`) | user-drawn zones tested against the ankle midpoint + persistence |
| ✅ | Getting out of bed (a user-drawn 'bed' safe zone) (`bed_exit`) | safe_zone_exit -- the same mechanism baby_monitoring.py uses for a crib, applied to a bed zone; soft severity by default (getting up is often benign; a real fall afterward is caught separately by the fall detector above) |
| ✅ | Motionless for an unusual period (`stillness`) | hip/shoulder centroid movement over a long window (deliberately 10 minutes by default, to not flag ordinary resting/reading) |
| 🟡 | Standing / sitting / lying classification (`posture`) | pose landmarks exist in the fall module **Caveat:** explicit posture classification not implemented; falls are detected by trajectory, not posture, precisely to avoid lying-down false alarms |
| ❌ | Struggling to stand / loss of balance / stumble / gait analysis (`struggling`) | temporal gait models; current pose pipeline detects falls, not gait quality |
| 🔵 | Time-conditioned rules (nighttime activity, outside hours) (`schedule`) | rule schedules — representable in AI-configure rules, enforcement lands with the scheduler |
| ❌ | Smoke / fire / flooding visible (`smoke_fire_flood`) | dedicated smoke/fire segmentation models (COCO has no such classes); a smoke/heat sensor is the honest hardware answer |
| ❌ | Unknown person / caregiver identification (`unknown_person`) | face recognition — deliberately not shipped (privacy); person-count changes ARE detected |
| ✅ | Multiple people present (`multiple_people`) | person-count events |

## security_monitoring

| | Detection | How / Caveat / Requires |
|---|---|---|
| ✅ | Person detected / count change (`person_detected`) | YOLO person + confirm_frames debounce (occupancy_change) |
| ✅ | Entering prohibited zone / crossing boundary / approaching door (`zone_violation`) | user-drawn zones + feet-point membership + persistence |
| ✅ | Person lingering in an area (`lingering`) | zone membership with a longer persistence requirement |
| ✅ | Unusual crowd / activity level (`crowd`) | person-count threshold |
| ✅ | Vehicle entering/leaving a zone (`vehicle`) | YOLO car/truck/bus/motorcycle/bicycle/train (core.capabilities' 'vehicle' category) + zone membership + persistence -- vehicle_in_zone, same zone engine as person zone_violation, requested automatically once any zone is configured |
| ✅ | Animal / pet in a zone (`animal`) | YOLO dog/cat/bird/... (core.capabilities' 'animal' category) + zone membership + persistence -- animal_in_zone, same mechanism as vehicle |
| ❌ | Package placed / removed / object moved (`package`) | object permanence + change detection against a scene baseline; COCO 'package' isn't a class — needs a custom model |
| ❌ | Door/window open or closed (`door_window_state`) | per-installation scene understanding or contact sensors (the honest answer) |
| ❌ | Smoke / fire visible (`smoke_fire`) | dedicated detection models; a smoke detector is the honest hardware answer |
| ❌ | Known vs unknown person (`known_person`) | face recognition — deliberately not shipped (privacy) |
| ❌ | Face covering / carrying weapon / climbing barrier (`mask_or_weapon`) | specialized models with serious false-positive stakes; we refuse to approximate these with the current stack — actions like 'person entered restricted area' are reported instead of labels like 'burglar' |
| 🔵 | Armed hours / motion during configured times (`schedule`) | rule schedules — representable in AI-configure rules, enforcement lands with the scheduler |
