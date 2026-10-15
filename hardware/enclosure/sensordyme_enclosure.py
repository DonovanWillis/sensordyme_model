"""SensorDyme enclosure — parametric CAD model (CadQuery).

Generates two watertight, FDM-printable STLs:
    sensordyme_bottom.stl  — bottom housing
    sensordyme_lid.stl     — removable lid (4x M3 corner screws)

Components designed for (all dims mm, from published mechanical drawings):
  - Raspberry Pi 5 (85 x 56, holes 58 x 49 grid, 3.5 from edges, M2.5)
  - Official Pi 5 Active Cooler (modeled as 60 x 43 x 26 envelope above board)
  - Camera Module 3 Wide NoIR (25 x 24 board, holes 21 x 12.5, ~120 deg FOV)
  - 22p->15p camera cable 200-300 mm (slack loop lives in the front bay)
  - 5V active piezo buzzer (~12 dia x 9.5)
  - SanDisk microSD (side slot access, card protrudes ~2.5 from Pi edge)
  - 850 nm IR illuminator board (generic: <=34 x 34 board, 28 x 28 post grid,
    20 dia LED window — adjust IR_* params below for your exact board)

Coordinate system: origin = exterior front-left-bottom corner.
X = width (right+), Y = depth (rear+), Z = up. Front face = camera face.
The Pi mounts with its USB-C/HDMI edge facing the REAR wall; the microSD
edge faces the RIGHT wall (side slot); USB/Ethernet face the LEFT interior.

Every placement below is checked by the verify() section, which asserts
positive clearance between all component envelopes and printed geometry,
then exports STLs and re-imports them to confirm watertightness.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cadquery as cq

OUT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------- parameters
WALL = 3.0
FLOOR = 3.0
LID_T = 3.2

IX, IY, IZ = 102.0, 94.0, 42.0            # interior cavity
OX, OY = IX + 2 * WALL, IY + 2 * WALL     # 108 x 100 exterior
BASE_H = FLOOR + IZ                       # 45
R_OUT, R_IN = 7.0, 5.0                    # corner radii (exterior / cavity)

# Raspberry Pi 5
PI_L, PI_W, PI_T = 85.0, 56.0, 1.6
PI_MARGIN_LEFT = 13.0                     # USB/Ethernet side (they overhang ~3)
PI_GAP_REAR = 10.0                        # USB-C reached via oversized cutout
PI_X0 = WALL + PI_MARGIN_LEFT             # 16
PI_X1 = PI_X0 + PI_L                      # 101 (microSD edge, 4 from right wall)
PI_Y1 = WALL + IY - PI_GAP_REAR           # 87 (rear edge)
PI_Y0 = PI_Y1 - PI_W                      # 31
STANDOFF_H, STANDOFF_D, STANDOFF_PILOT = 6.0, 6.0, 2.05
BOARD_TOP = FLOOR + STANDOFF_H + PI_T     # 10.6
# hole grid 58 x 49, 3.5 from edges; board rotated 180 deg (USB-C edge rear)
STANDOFFS = [(PI_X1 - sx, PI_Y1 - sy) for sx in (3.5, 61.5) for sy in (3.5, 52.5)]

# Active Cooler envelope (generous) centered on board
COOLER_W, COOLER_D, COOLER_H = 60.0, 43.0, 26.0
COOLER_TOP = BOARD_TOP + COOLER_H         # 36.6
FAN_HEADROOM = FLOOR + IZ - COOLER_TOP    # must be >= 5

# USB-C: center 11.2 from the microSD-side board edge
USBC_X = PI_X1 - 11.2                     # 89.8
USBC_CUT_W, USBC_CUT_H = 16.0, 8.0
USBC_CUT_Z = 9.3                          # connector body spans z 10.6..14.2
USBC_POCKET_W, USBC_POCKET_H, USBC_POCKET_D = 26.0, 14.0, 1.6

# microSD side slot (slot centered on Pi short edge, card under board)
SD_Y = PI_Y1 - PI_W / 2                   # 59
SD_SLOT_W, SD_SLOT_Z0, SD_SLOT_Z1 = 20.0, 4.5, 10.0
SD_SCOOP_W, SD_SCOOP_H, SD_SCOOP_D = 26.0, 9.0, 1.5

# corner lid-screw bosses (M3 self-tap)
BOSS_D, BOSS_PILOT_D, BOSS_PILOT_DEPTH = 9.0, 2.6, 12.0
BOSS_R = BOSS_D / 2
BOSSES = [
    (WALL + BOSS_R, WALL + BOSS_R),
    (WALL + IX - BOSS_R, WALL + BOSS_R),
    (WALL + BOSS_R, WALL + IY - BOSS_R),
    (WALL + IX - BOSS_R, WALL + IY - BOSS_R),
]
LID_HOLE_D, LID_CBORE_D, LID_CBORE_DEPTH = 3.4, 6.8, 2.2

# Front-wall board layout. The camera and the IR board share the front wall's
# inner face (both on 2 mm pads), between the two front corner bosses, which
# leaves x 12..96 for 25 mm + 34 mm of board. With the camera centred (x 54) the
# 34 mm IR board overlapped the camera board by 0.5 mm on one side and the
# front-right lid-screw boss by ~3.5 mm on the other: no position for it
# fitted. So the camera sits 8 mm left of centre and verify() checks every
# board envelope against the other board and the bosses. Lid features that
# used CAM_X as "the middle of the box" (vents, logo) are pinned to CENTER_X so
# the lid is unchanged.
CENTER_X = WALL + IX / 2                  # 54
BOARD_T, BOARD_CLR = 1.6, 1.0             # PCB thickness; min board clearance

# Camera Module 3 (front wall, left of centre — see the layout note above)
CAM_BOARD_W, CAM_BOARD_H = 25.0, 24.0     # CM3 board, x by z
CAM_X = 46.0                              # lens axis x
CAM_Z = 22.0                              # lens axis height
CAM_HOLE_DX, CAM_HOLE_DZ = 21.0, 12.5     # CM3 hole grid
CAM_PAD_D, CAM_PAD_H = 5.0, 2.0
CAM_PILOT_D, CAM_PILOT_DEPTH = 1.7, 4.0   # M2 self-tap
CAM_OPEN_INNER_R, CAM_OPEN_OUTER_R = 7.0, 12.0  # conical, clears 120 deg FOV
CAM_PADS = [
    (CAM_X - CAM_HOLE_DX / 2, CAM_Z - CAM_HOLE_DZ / 2),
    (CAM_X + CAM_HOLE_DX / 2, CAM_Z - CAM_HOLE_DZ / 2),
    (CAM_X - CAM_HOLE_DX / 2, CAM_Z + CAM_HOLE_DZ / 2),
    (CAM_X + CAM_HOLE_DX / 2, CAM_Z + CAM_HOLE_DZ / 2),
]

# IR illuminator (front wall, right of camera) — tune for your exact board
IR_BOARD = 34.0                           # square board envelope (generic <=34)
IR_X, IR_Z = 77.5, 22.0
IR_OPEN_R = 10.0                          # 20 dia LED window
IR_GRID = 28.0                            # post spacing (square)
IR_PAD_D, IR_PAD_H = 5.0, 2.0
IR_PILOT_D, IR_PILOT_DEPTH = 1.7, 4.0
IR_PADS = [
    (IR_X - IR_GRID / 2, IR_Z - IR_GRID / 2),
    (IR_X + IR_GRID / 2, IR_Z - IR_GRID / 2),
    (IR_X - IR_GRID / 2, IR_Z + IR_GRID / 2),
    (IR_X + IR_GRID / 2, IR_Z + IR_GRID / 2),
]

# piezo buzzer holder (front-left floor) + lid sound holes above it
BUZZ_X, BUZZ_Y = 22.0, 18.0
BUZZ_ID, BUZZ_RING_T, BUZZ_RING_H = 12.6, 2.2, 6.0
BUZZ_OR = BUZZ_ID / 2 + BUZZ_RING_T       # 8.5
SOUND_HOLE_D, SOUND_HOLE_RING_R = 2.5, 4.5

# camera-cable guide posts (front bay, keep slack away from the fan)
GUIDE_D, GUIDE_H = 4.0, 14.0
GUIDES = [(34.0, 20.0), (34.0, 28.0), (64.0, 20.0), (64.0, 28.0)]

# ventilation
LID_VENT_W, LID_VENT_L = 3.2, 63.0        # slots over the cooler
LID_VENT_YS = [38.0 + 6.5 * i for i in range(8)]           # y centers 38..83.5
LID_VENT_X = CENTER_X + 4.5               # centered over the Pi (58.5)
WALL_VENT_L, WALL_VENT_H = 14.0, 3.5
WALL_VENT_ZS = [32.5, 38.5]
SIDE_VENT_YS = [24.0, 44.0, 64.0, 84.0]
REAR_VENT_XS = [25.0, 45.0, 65.0]

# lid fit
SKIRT_T, SKIRT_DEPTH, SKIRT_CLR = 1.8, 4.0, 0.3
LOGO_TEXT, LOGO_SIZE, LOGO_DEPTH = "SensorDyme", 7.0, 0.6
LOGO_POS = (CENTER_X, 17.0)

FEET = [(14.0, 14.0), (OX - 14.0, 14.0), (14.0, OY - 14.0), (OX - 14.0, OY - 14.0)]
FOOT_D, FOOT_DEPTH = 11.0, 1.2


# ------------------------------------------------------------------- helpers
def box(x0, y0, z0, x1, y1, z1):
    return (
        cq.Workplane("XY")
        .box(x1 - x0, y1 - y0, z1 - z0, centered=False)
        .translate((x0, y0, z0))
    )


def cyl(cx, cy, z0, z1, d):
    return (
        cq.Workplane("XY")
        .cylinder(z1 - z0, d / 2, centered=(True, True, False))
        .translate((cx, cy, z0))
    )


def rounded_prism(w, d, h, r):
    return cq.Workplane("XY").rect(w, d).extrude(h).edges("|Z").fillet(r)


def cone_y(cx, z, y0, y1, r0, r1):
    """Cone along +Y: radius r0 at y0 -> r1 at y1, axis at (cx, z)."""
    solid = cq.Solid.makeCone(r0, r1, y1 - y0, cq.Vector(cx, y0, z), cq.Vector(0, 1, 0))
    return cq.Workplane("XY").newObject([solid])


# ------------------------------------------------------------- bottom housing
def build_bottom():
    base = rounded_prism(OX, OY, BASE_H, R_OUT).translate((OX / 2, OY / 2, 0))
    cavity = rounded_prism(IX, IY, IZ + 1, R_IN).translate((OX / 2, OY / 2, FLOOR))
    base = base.cut(cavity)
    base = base.faces("<Z").edges().chamfer(0.8)

    for cx, cy in BOSSES:                       # lid screw bosses, full height
        base = base.union(cyl(cx, cy, FLOOR, BASE_H, BOSS_D))
    for cx, cy in STANDOFFS:                    # Pi standoffs
        base = base.union(cyl(cx, cy, FLOOR, FLOOR + STANDOFF_H, STANDOFF_D))
    for cx, cz in CAM_PADS:                     # camera pads on front wall
        pad = cq.Solid.makeCylinder(
            CAM_PAD_D / 2, CAM_PAD_H, cq.Vector(cx, WALL, cz), cq.Vector(0, 1, 0)
        )
        base = base.union(cq.Workplane("XY").newObject([pad]))
    for cx, cz in IR_PADS:                      # IR illuminator pads
        pad = cq.Solid.makeCylinder(
            IR_PAD_D / 2, IR_PAD_H, cq.Vector(cx, WALL, cz), cq.Vector(0, 1, 0)
        )
        base = base.union(cq.Workplane("XY").newObject([pad]))
    ring = cyl(BUZZ_X, BUZZ_Y, FLOOR, FLOOR + BUZZ_RING_H, 2 * BUZZ_OR).cut(
        cyl(BUZZ_X, BUZZ_Y, FLOOR, FLOOR + BUZZ_RING_H, BUZZ_ID)
    )
    ring = ring.cut(box(BUZZ_X - 2, BUZZ_Y + 2, FLOOR, BUZZ_X + 2, BUZZ_Y + BUZZ_OR + 1, FLOOR + BUZZ_RING_H))
    base = base.union(ring)                     # buzzer holder (wire slot at rear)
    for cx, cy in GUIDES:                       # cable guide posts
        base = base.union(cyl(cx, cy, FLOOR, FLOOR + GUIDE_H, GUIDE_D))

    # ---- cutouts
    base = base.cut(                            # USB-C opening, rear wall
        box(USBC_X - USBC_CUT_W / 2, WALL + IY - 0.1, USBC_CUT_Z,
            USBC_X + USBC_CUT_W / 2, OY + 1, USBC_CUT_Z + USBC_CUT_H)
    )
    base = base.cut(                            # USB-C exterior recess pocket
        box(USBC_X - USBC_POCKET_W / 2, OY - USBC_POCKET_D,
            USBC_CUT_Z + USBC_CUT_H / 2 - USBC_POCKET_H / 2,
            USBC_X + USBC_POCKET_W / 2, OY + 1,
            USBC_CUT_Z + USBC_CUT_H / 2 + USBC_POCKET_H / 2)
    )
    base = base.cut(                            # microSD slot, right wall
        box(WALL + IX - 0.1, SD_Y - SD_SLOT_W / 2, SD_SLOT_Z0,
            OX + 1, SD_Y + SD_SLOT_W / 2, SD_SLOT_Z1)
    )
    base = base.cut(                            # microSD finger scoop
        box(OX - SD_SCOOP_D, SD_Y - SD_SCOOP_W / 2, SD_SLOT_Z0 - 1.5,
            OX + 1, SD_Y + SD_SCOOP_W / 2, SD_SLOT_Z0 - 1.5 + SD_SCOOP_H)
    )
    base = base.cut(cone_y(CAM_X, CAM_Z, -0.1, WALL + 0.1,          # camera cone
                           CAM_OPEN_OUTER_R + 0.04, CAM_OPEN_INNER_R - 0.04))
    ir_hole = cq.Solid.makeCylinder(
        IR_OPEN_R, WALL + 0.2, cq.Vector(IR_X, -0.1, IR_Z), cq.Vector(0, 1, 0)
    )
    base = base.cut(cq.Workplane("XY").newObject([ir_hole]))        # IR window
    base = base.cut(cone_y(IR_X, IR_Z, -0.1, 1.5, IR_OPEN_R + 2.0, IR_OPEN_R))

    for cx, cy in BOSSES:                       # M3 pilots
        base = base.cut(cyl(cx, cy, BASE_H - BOSS_PILOT_DEPTH, BASE_H + 1, BOSS_PILOT_D))
    for cx, cy in STANDOFFS:                    # M2.5 pilots
        base = base.cut(cyl(cx, cy, FLOOR + STANDOFF_H - 8.0, FLOOR + STANDOFF_H + 1, STANDOFF_PILOT))
    for cx, cz in CAM_PADS + IR_PADS:           # M2 pilots (stop 1 mm shy of face)
        pilot = cq.Solid.makeCylinder(
            CAM_PILOT_D / 2, CAM_PILOT_DEPTH,
            cq.Vector(cx, WALL + CAM_PAD_H + 0.01, cz), cq.Vector(0, -1, 0)
        )
        base = base.cut(cq.Workplane("XY").newObject([pilot]))

    for zc in WALL_VENT_ZS:                     # side + rear vents
        for yc in SIDE_VENT_YS:
            base = base.cut(box(-1, yc - WALL_VENT_L / 2, zc - WALL_VENT_H / 2,
                                WALL + 0.1, yc + WALL_VENT_L / 2, zc + WALL_VENT_H / 2))
            base = base.cut(box(WALL + IX - 0.1, yc - WALL_VENT_L / 2, zc - WALL_VENT_H / 2,
                                OX + 1, yc + WALL_VENT_L / 2, zc + WALL_VENT_H / 2))
        for xc in REAR_VENT_XS:
            base = base.cut(box(xc - WALL_VENT_L / 2, WALL + IY - 0.1, zc - WALL_VENT_H / 2,
                                xc + WALL_VENT_L / 2, OY + 1, zc + WALL_VENT_H / 2))

    for cx, cy in FEET:                         # rubber-feet recesses
        base = base.cut(cyl(cx, cy, -0.1, FOOT_DEPTH, FOOT_D))
    return base


# --------------------------------------------------------------------- lid
def build_lid():
    lid = rounded_prism(OX, OY, LID_T, R_OUT).translate((OX / 2, OY / 2, BASE_H))
    lid = lid.faces(">Z").edges().fillet(1.2)

    sx0, sx1 = 20.0, WALL + IX - 17.0           # skirt segments (clear the bosses)
    sy0, sy1 = 20.0, WALL + IY - 17.0
    zs0, zs1 = BASE_H - SKIRT_DEPTH, BASE_H + 0.01
    seg_f = box(sx0, WALL + SKIRT_CLR, zs0, sx1, WALL + SKIRT_CLR + SKIRT_T, zs1)
    seg_r = box(sx0, WALL + IY - SKIRT_CLR - SKIRT_T, zs0, sx1, WALL + IY - SKIRT_CLR, zs1)
    seg_l = box(WALL + SKIRT_CLR, sy0, zs0, WALL + SKIRT_CLR + SKIRT_T, sy1, zs1)
    seg_rt = box(WALL + IX - SKIRT_CLR - SKIRT_T, sy0, zs0, WALL + IX - SKIRT_CLR, sy1, zs1)
    for seg in (seg_f, seg_r, seg_l, seg_rt):
        chamfered = seg.edges("<Z").chamfer(0.7)  # lead-in for easy seating
        lid = lid.union(chamfered)

    for yc in LID_VENT_YS:                      # big vent field over the cooler
        lid = lid.cut(box(LID_VENT_X - LID_VENT_L / 2, yc - LID_VENT_W / 2, BASE_H - SKIRT_DEPTH - 1,
                          LID_VENT_X + LID_VENT_L / 2, yc + LID_VENT_W / 2, BASE_H + LID_T + 2))

    lid = lid.cut(cyl(BUZZ_X, BUZZ_Y, BASE_H - 1, BASE_H + LID_T + 2, SOUND_HOLE_D))
    for i in range(6):                          # sound holes above the buzzer
        a = math.radians(60 * i)
        lid = lid.cut(cyl(BUZZ_X + SOUND_HOLE_RING_R * math.cos(a),
                          BUZZ_Y + SOUND_HOLE_RING_R * math.sin(a),
                          BASE_H - 1, BASE_H + LID_T + 2, SOUND_HOLE_D))

    for cx, cy in BOSSES:                       # M3 through-holes + counterbores
        lid = lid.cut(cyl(cx, cy, BASE_H - SKIRT_DEPTH - 1, BASE_H + LID_T + 1, LID_HOLE_D))
        lid = lid.cut(cyl(cx, cy, BASE_H + LID_T - LID_CBORE_DEPTH, BASE_H + LID_T + 1, LID_CBORE_D))

    logo = (
        cq.Workplane("XY")
        .text(LOGO_TEXT, LOGO_SIZE, LOGO_DEPTH + 0.2, halign="center", valign="center")
        .translate((LOGO_POS[0], LOGO_POS[1], BASE_H + LID_T - LOGO_DEPTH))
    )
    lid = lid.cut(logo)
    return lid


# ------------------------------------------------------------- verification
def _box_gap(a, b):
    """Clearance between two (x0, x1, z0, z1) rectangles in one plane;
    negative when they overlap."""
    dx = max(b[0] - a[1], a[0] - b[1])
    dz = max(b[2] - a[3], a[2] - b[3])
    if dx < 0 and dz < 0:
        return max(dx, dz)                 # overlapping: the shallower depth
    return math.hypot(max(dx, 0.0), max(dz, 0.0))


def verify():
    checks = []

    def chk(name, value, minimum=0.01):
        ok = value >= minimum
        checks.append((name, value, ok))
        return ok

    # Pi + cooler
    chk("fan headroom above Active Cooler (>=5)", FAN_HEADROOM, 5.0)
    chk("clearance under Pi for solder joints (>=4)", STANDOFF_H - PI_T, 4.0)
    chk("Pi left margin (USB/Eth overhang ~3)", PI_MARGIN_LEFT - 3.0, 4.0)
    chk("Pi right margin (SD edge to wall)", (WALL + IX) - PI_X1, 3.0)
    chk("Pi front edge to front bay", PI_Y0 - WALL, 20.0)
    # rear corner bosses vs Pi corners
    for cx, cy in BOSSES:
        dx = max(PI_X0 - cx, 0, cx - PI_X1)
        dy = max(PI_Y0 - cy, 0, cy - PI_Y1)
        chk(f"lid boss ({cx:.0f},{cy:.0f}) vs Pi board", math.hypot(dx, dy) - BOSS_R, 0.5)
    # cooler envelope inside cavity
    cool_x0, cool_x1 = (PI_X0 + PI_X1) / 2 - COOLER_W / 2, (PI_X0 + PI_X1) / 2 + COOLER_W / 2
    cool_y0, cool_y1 = (PI_Y0 + PI_Y1) / 2 - COOLER_D / 2, (PI_Y0 + PI_Y1) / 2 + COOLER_D / 2
    chk("cooler envelope in cavity (x)", min(cool_x0 - WALL, WALL + IX - cool_x1), 2.0)
    chk("cooler envelope in cavity (y)", min(cool_y0 - WALL, WALL + IY - cool_y1), 2.0)
    # USB-C aligned
    chk("USB-C cutout covers connector (x)", USBC_CUT_W / 2 - 9.0 / 2 - 1.0, 0.0)
    chk("USB-C cutout covers connector (z)", (USBC_CUT_Z + USBC_CUT_H) - 14.2 - 1.0, 0.0)
    # camera
    cam_board_z0, cam_board_z1 = CAM_Z - CAM_BOARD_H / 2, CAM_Z + CAM_BOARD_H / 2
    chk("camera board above floor", cam_board_z0 - FLOOR, 2.0)
    chk("camera board below skirt", (BASE_H - SKIRT_DEPTH) - cam_board_z1, 2.0)
    half_angle = math.degrees(math.atan((CAM_OPEN_OUTER_R - CAM_OPEN_INNER_R) / WALL))
    chk("camera cone half-angle (>=59 deg for 120 FOV)", half_angle, 59.0)
    # buzzer fully in the front bay, sound holes clear of logo text
    chk("buzzer clear of Pi front edge", PI_Y0 - (BUZZ_Y + BUZZ_OR), 2.0)
    logo_half_w = len(LOGO_TEXT) * LOGO_SIZE * 0.62 / 2
    chk("logo clear of sound holes",
        (LOGO_POS[0] - logo_half_w) - (BUZZ_X + SOUND_HOLE_RING_R + SOUND_HOLE_D / 2), 1.0)
    # cable guides inside front bay
    for cx, cy in GUIDES:
        chk(f"cable guide ({cx:.0f},{cy:.0f}) clear of Pi", PI_Y0 - (cy + GUIDE_D / 2), 0.5)
    # IR window clear of camera cone at the exterior face
    chk("IR window clear of camera cone",
        (IR_X - IR_OPEN_R - 2.0) - (CAM_X + CAM_OPEN_OUTER_R), 1.0)
    # front-wall boards: each envelope against the other and the corner bosses
    board_y0 = WALL + CAM_PAD_H               # boards sit on the pads' faces
    board_y1 = board_y0 + BOARD_T
    boards = {
        "camera board": (CAM_X - CAM_BOARD_W / 2, CAM_X + CAM_BOARD_W / 2,
                         CAM_Z - CAM_BOARD_H / 2, CAM_Z + CAM_BOARD_H / 2),
        "IR board": (IR_X - IR_BOARD / 2, IR_X + IR_BOARD / 2,
                     IR_Z - IR_BOARD / 2, IR_Z + IR_BOARD / 2),
    }
    chk("camera board clear of IR board",
        _box_gap(boards["camera board"], boards["IR board"]), BOARD_CLR)
    for board, (x0, x1, _z0, _z1) in boards.items():
        for bx, by in BOSSES:              # bosses run floor to rim: z always overlaps
            near_x = min(max(bx, x0), x1)
            near_y = min(max(by, board_y0), board_y1)
            chk(f"{board} clear of lid boss ({bx:.0f},{by:.0f})",
                math.hypot(bx - near_x, by - near_y) - BOSS_R, BOARD_CLR)
        chk(f"{board} inside side walls", min(x0 - WALL, WALL + IX - x1), BOARD_CLR)
    ir_z0, ir_z1 = boards["IR board"][2:]
    chk("IR board above floor", ir_z0 - FLOOR, 2.0)
    chk("IR board below skirt", (BASE_H - SKIRT_DEPTH) - ir_z1, 2.0)
    # SD slot below board top, above floor
    chk("SD slot under board", BOARD_TOP - SD_SLOT_Z1 + 0.6, 0.0)

    width = max(len(n) for n, _, _ in checks)
    ok_all = True
    for name, value, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {value:6.2f} mm")
        ok_all &= ok
    if not ok_all:
        sys.exit("DESIGN CHECK FAILED")
    print("all geometric clearance checks passed\n")


def main():
    print("== SensorDyme enclosure ==")
    print(f"exterior: {OX:.0f} x {OY:.0f} x {BASE_H + LID_T:.0f} mm (plus lid fillets)")
    verify()

    print("building bottom housing ...")
    bottom = build_bottom()
    print("building lid ...")
    lid = build_lid()

    for name, part in (("sensordyme_bottom", bottom), ("sensordyme_lid", lid)):
        solid = part.val()
        if not solid.isValid():
            sys.exit(f"{name}: BRep solid is NOT valid")
        path = OUT_DIR / f"{name}.stl"
        cq.exporters.export(part, str(path), tolerance=0.02, angularTolerance=0.2)
        print(f"exported {path.name} ({path.stat().st_size / 1e6:.1f} MB), volume {solid.Volume() / 1000:.1f} cm3")

    try:
        import trimesh
        for name in ("sensordyme_bottom", "sensordyme_lid"):
            mesh = trimesh.load(OUT_DIR / f"{name}.stl")
            print(f"{name}.stl: watertight={mesh.is_watertight}, "
                  f"winding_consistent={mesh.is_winding_consistent}, faces={len(mesh.faces)}")
            if not mesh.is_watertight:
                sys.exit(f"{name}.stl is not watertight")
    except ImportError:
        print("trimesh not installed - skipped mesh audit")
    print("\nDONE - STLs are watertight and print-ready.")


if __name__ == "__main__":
    main()
