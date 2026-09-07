# Reversed Allegro Hand Geometry Previews

- Purpose: orientation and bottom-constraint inspection only; no policy was loaded.
- Transform: rigid 180° rotation about world X, applied only at the `palm` subtree root.
- Pivot: `[0.0, -0.05, 0.0]` m (rod center / grasp region).
- Joint frames and all palm-relative child transforms are unchanged.
- World-axis indicator: red X, green Y, blue Z.

## A0 reversed — bottom revolute
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_20260823-161638/reversed_A0_bottom_revolute.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_20260823-161638/reversed_A0_bottom_revolute.xml`
- Marker: yellow center sphere plus four cardinal satellites
- Static signed fingertip distances to rod (m): `[-0.010233714106680877, -0.0125861928648194, -0.002999574732485735]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[-0.000508980510458298, -0.00021613405513667278, -0.00046212165106849536]`; contacts: 3/3.

## C3 reversed — bottom point connect
- Video: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_20260823-161638/reversed_C3_bottom_point_connect.mp4`
- Preview MJCF: `/data/ypx/allegro_rod_mvp/runs/previews/reversed_world_x_180_20260823-161638/reversed_C3_bottom_point_connect.xml`
- Marker: magenta center sphere plus four diagonal satellites
- Static signed fingertip distances to rod (m): `[-0.010233714106680877, -0.0125861928648194, -0.002999574732485735]`; contacts: 3/3.
- After 0.2 s signed distances (m): `[-0.0005720029927294232, -0.0009453534648698574, -0.0008202521624100009]`; contacts: 3/3.

