# T00 300k A/B/C/D seed-6 collapse timelines

- Date: 2026-09-14
- Checkpoints: `20260914-1712-t00-ablation-{A,B,C,D}-continue300k-seed0` step 306432
- Seed: 6, deterministic PPO
- Physics: tip_connect bottom, s=400, μ=4, 25 Hz, tilt_terminate=1.2
- Videos: copied from `runs/20260914-1748-t00-ablation-videos-300k/`

## HTML
Open `index.html` (A/B/C/D tabs + **step slider**, still frames, per-frame metrics).
The MP4 is an optional “open clip” link, not the analysis UI.
Data: `timelines.json`. Frames: `{A,B,C,D}/frames/`.
Every frame shows ω_axial (rad/s and °/s), tilt (rad and °), tilt vel dθ/dt
(finite difference, dt=0.04 s), ω_perp separately, n_contact, per-tip and
total `_touch()` force, plus tip error, reward, and C/D gated rotation / wobble.

Inheritable copy: `docs/pages/t00-ablation-demos-300k/`.
Live (docs server 8767): http://127.0.0.1:8767/pages/t00-ablation-demos-300k/?v=scrub20260915

Does **not** overwrite `runs/20260914-t00-seed6-tilt-collapse/`.

## Seed-6 events (re-rolled traces)

| Cond | 2+ → ≤1 | n=0 | Tilt-kill | Loss→kill | Recontact | Term |
|---|---:|---:|---:|---:|---|---|
| A | 49 | 57 | 65 | 16 | no | axis_tilt |
| B | 43 | 54 | 61 | 18 | no | axis_tilt |
| C | 50 | 60 | 65 | 15 | no | axis_tilt |
| D | 53 | 57 | 62 | 9 | no | axis_tilt |

All four: 3-contact only at reset, then 2-contact hold, then 1, then 0, then axis_tilt.
