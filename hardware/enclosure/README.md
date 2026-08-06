# SensorDyme Enclosure

Two-piece FDM-printable enclosure for the SensorDyme device. Generated
parametrically by [sensordyme_enclosure.py](sensordyme_enclosure.py) — every
dimension is a named parameter and the script refuses to export unless all 24
clearance checks pass and both meshes are watertight.

| File | Part | Volume |
|---|---|---|
| `sensordyme_bottom.stl` | Bottom housing, 108 × 100 × 45 mm | 87.6 cm³ |
| `sensordyme_lid.stl` | Lid, 108 × 100 × 3.2 mm + skirt | 30.4 cm³ |

## Component fit (all verified in-script)

- **Raspberry Pi 5** on four Ø6 standoffs (M2.5 self-tapping, 6 mm under-board
  clearance for solder joints). USB-C/HDMI edge faces the rear wall, microSD
  edge faces the right wall, USB/Ethernet enclosed on the left with 10 mm spare.
- **Active Cooler**: 8.4 mm free air above the fan, 3.2 mm-slot vent field in
  the lid directly above it, plus 22 side/rear wall vents (all below the lid
  skirt so airflow is never blocked).
- **USB-C**: 16 × 8 mm opening + 1.6 mm exterior recess — connector is reached
  through the opening without removing the lid; any overmold ≤ 16 × 8 mm seats
  fully (all Pi-branded and typical USB-C cables fit).
- **microSD**: 20 mm slot in the right wall at card height with a finger scoop
  — insert/remove without opening the enclosure.
- **Camera Module 3 Wide NoIR**: four Ø5 pads (M2 self-tap) behind a centered
  conical opening (Ø14 → Ø24, 59° half-angle) — clears the 120° diagonal FOV.
- **IR illuminator**: Ø20 chamfered window + 28 × 28 mm post grid to the right
  of the camera. Boards vary — tune `IR_*` params in the script if yours
  differs, then re-run.
- **Buzzer**: Ø12.6 socket ring on the front-left floor (wire slot at rear of
  ring); 7 × Ø2.5 sound holes in the lid directly above.
- **Camera cable (200–300 mm)**: slack coils in the 28 mm front bay between
  two pairs of guide posts — no sharp bends, never near the fan.

## Hardware

- 4 × M3 × 10 self-tapping (or coarse-thread) screws — lid corners
- 4 × M2.5 × 8 self-tapping screws — Pi 5
- 4 × M2 × 6 self-tapping screws — Camera Module 3 (+ 2–4 for the IR board)
- 4 × Ø10 adhesive rubber feet (recesses on the underside)

## Print settings

- Material: PETG or ASA recommended (near hot Pi); matte PLA acceptable indoors
- Layer 0.2 mm, 3–4 perimeters, 25 %+ infill, no supports needed
- **Bottom**: print floor-down as oriented
- **Lid**: print top-face-down (logo against a textured plate gives the matte
  premium finish); skirt prints upward, no supports
- First lid screw insertion cuts the threads — drive slowly, don't overtighten

## Assembly

1. Press the buzzer into its ring (wires out the rear slot), route to GPIO.
2. Screw the camera to the front-wall pads (lens through the center opening),
   IR board to its pads.
3. Connect the camera cable to the Pi's CAM/DISP port, coil the slack through
   the front-bay guide posts.
4. Mount the Pi (cooler already fitted) on the standoffs, USB-C toward the
   rear opening. Insert the microSD through the right-side slot.
5. Seat the lid — the chamfered skirt self-aligns — and fit the four M3s.

## Regenerating

```
uv venv cadenv --python 3.12 && uv pip install --python cadenv/Scripts/python.exe cadquery trimesh
cadenv/Scripts/python.exe sensordyme_enclosure.py
```
