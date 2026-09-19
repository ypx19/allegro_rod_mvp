# Palm translation compare

Interactive local page for the 52 mm palm-origin offset between
`configs/hand_poses/my_grasp.json` and `palm_down_screwdriver`.

This is **not** T00 success. Palm-down tip-connect at mass-scale 1 is a
verified parallel result (FIND-20260918-001 / EXP-20260918-001).

## Launch (required for the live sim)

```bash
.venv/bin/python scripts/palm_translation_compare_web.py --port 8768
```

Then open http://127.0.0.1:8768/

- Left live panel: `my_grasp` on the project Allegro tip-connect XML.
- Right live panel: `palm_down_screwdriver` XML (`screwdriver_palm_down_h0.015.xml`).
- Shared free camera: drag orbit, scroll zoom, Shift or right-drag pan, `R` reset.
- Gold sphere = my_grasp palm origin; teal = palm_down; capsule = the 52 mm offset.
- `my_grasp → palm_down` copies only translation (the isolating follow-up).

## Videos

| Setup | Default clip | Role |
|---|---|---|
| my_grasp | revolute s=1, 11,271°, seed 10000 | Best verified success with this pose (hinge, not T00) |
| my_grasp | T00 cinematic seed 6 | Tip-connect prefix; not a pass |
| palm_down | bounded 60 s seed 4000 | Best tip-connect success (~10.36 turns) |
| palm_down | original 20 s seed 7000 | Independent 8/8 20 s protocol |

Opening `index.html` as a file shows the layout but not the EGL simulation.
