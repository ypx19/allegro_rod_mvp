# Reversed Allegro Hand Geometry Previews

- Purpose: orientation and bottom-constraint inspection only; no policy was loaded.
- Transform: rigid 180° rotation about world X, applied only at the `palm` subtree root.
- Additional transform: rigid world-Z translation of the same subtree to request 30.0 mm palm-to-rod-top clearance.
- Horizontal transform: rigid world-XY translation `[0.009915014317635812, 0.0013009577553006414]` m, computed from MuJoCo body positions to move the thumb root 10.0 mm closer to the rod axis.
- Reference thumb-root distance: `65.259613` mm.
- Pivot: `[0.0, -0.05, 0.0]` m (rod center / grasp region).
- Joint frames and all palm-relative child transforms are unchanged.
- World-axis indicator: red X, green Y, blue Z.

## A0 reversed — bottom revolute
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_A0_bottom_revolute.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_A0_bottom_revolute.xml`
- Marker: yellow center sphere plus four cardinal satellites
- Placement: palm-bottom `0.100000000` m; rod-top `0.070000000` m; clearance `30.000000` mm; applied ΔZ `0.052386000` m.
- Thumb root: before `65.259613` mm; after `55.259613` mm; change `-10.000000` mm.
- Static signed fingertip distances to rod (m): `[-0.00044419585092246756, -0.01191172360271733, -0.012085599984904742]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[0.008203185043652464, -0.00023242453513485847, -0.0006584927920159894]`; geometric contacts: 2/3; normal forces (N): `[0.0, 34.57603665827218, 132.93449130669717]`; force contacts: 2/3.

## C3 reversed — bottom point connect
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_C3_bottom_point_connect.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_C3_bottom_point_connect.xml`
- Marker: magenta center sphere plus four diagonal satellites
- Placement: palm-bottom `0.100000000` m; rod-top `0.070000000` m; clearance `30.000000` mm; applied ΔZ `0.052386000` m.
- Thumb root: before `65.259613` mm; after `55.259613` mm; change `-10.000000` mm.
- Static signed fingertip distances to rod (m): `[-0.00044419585092246756, -0.01191172360271733, -0.012085599984904742]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[-0.0006334966066495317, -0.0011156308803709075, -0.0009269441042907643]`; geometric contacts: 3/3; normal forces (N): `[8.46713949599406, 31.36088637719799, 24.105438962292435]`; force contacts: 3/3.

