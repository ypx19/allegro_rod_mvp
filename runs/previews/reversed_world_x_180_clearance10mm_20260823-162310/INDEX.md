# Reversed Allegro Hand Geometry Previews

- Purpose: orientation and bottom-constraint inspection only; no policy was loaded.
- Transform: rigid 180° rotation about world X, applied only at the `palm` subtree root.
- Additional transform: rigid world-Z translation of the same subtree to request 10.0 mm palm-to-rod-top clearance.
- Pivot: `[0.0, -0.05, 0.0]` m (rod center / grasp region).
- Joint frames and all palm-relative child transforms are unchanged.
- World-axis indicator: red X, green Y, blue Z.

## A0 reversed — bottom revolute
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_A0_bottom_revolute.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_A0_bottom_revolute.xml`
- Marker: yellow center sphere plus four cardinal satellites
- Placement: palm-bottom `0.080000000` m; rod-top `0.070000000` m; clearance `10.000000` mm; applied ΔZ `0.032386000` m.
- Static signed fingertip distances to rod (m): `[-0.010233714106680883, -0.0125861928648194, -0.0029995747324857314]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[0.015634357026486446, -2.4133376416820423e-05, -0.0004558668036236998]`; geometric contacts: 2/3; normal forces (N): `[0.0, 15.60499008772417, 51.35811302043165]`; force contacts: 2/3.

## C3 reversed — bottom point connect
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_C3_bottom_point_connect.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_C3_bottom_point_connect.xml`
- Marker: magenta center sphere plus four diagonal satellites
- Placement: palm-bottom `0.080000000` m; rod-top `0.070000000` m; clearance `10.000000` mm; applied ΔZ `0.032386000` m.
- Static signed fingertip distances to rod (m): `[-0.010233714106680883, -0.0125861928648194, -0.0029995747324857314]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[-0.0007016806128808204, -0.001048687279945188, -0.001456659771188246]`; geometric contacts: 3/3; normal forces (N): `[11.417926750621804, 29.637962876657866, 39.950536586348015]`; force contacts: 3/3.

