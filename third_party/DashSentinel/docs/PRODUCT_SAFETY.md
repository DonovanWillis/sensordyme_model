# Product safety and human factors

DashSentinel is a prototype warning system. It should not prevent a person from driving, lock a vehicle, disable ignition, or make emergency access decisions.

The system now includes guardrails to reduce false positives:

- frame-quality checks for dark, overexposed, low-contrast, or blurry frames
- eye and mouth visibility gates so hidden landmarks do not create fake drowsiness events
- pose-only fallback when eyes or mouth are not visible
- multi-face selection for scenes where a passenger appears in the frame
- optional local deep-learning fusion for more robust classification
- status hold frames so the status does not jump from one bad frame

A real deployment still needs human-factor work:

- clear audible/visual warnings instead of vehicle control
- driver override and emergency-use assumptions
- documented failure modes such as sunglasses, masks, glare, camera movement, night driving, and multiple occupants
- validation on diverse drivers, lighting, cameras, vehicle cabins, and road conditions
- privacy review for video handling and logs

The ESP8266 module should remain an alert/display module only. It should not be wired into safety-critical vehicle controls.
