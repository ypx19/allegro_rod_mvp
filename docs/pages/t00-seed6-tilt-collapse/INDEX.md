# T00 seed 6 tilt-collapse timeline

- Date: 2026-09-14
- Source: `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0`
- Seed: 6, deterministic PPO
- Replay tilt-kill raised to 3.14 rad so the drop is logged past the eval gate

## HTML
Open `index.html` (scrubber). Data: `timeline.json`. Frames: `frames/`.

## Window 29–34 (the requested 5–6 steps)

Seed 6 does **not** go 3.7° → 86° in this window. That 86° / 34-step video is eval seed 10000.

| step | t (s) | tilt (°) | dtilt (°/s) | contacts | F index / mid / thumb (N) |
|---:|---:|---:|---:|---:|---|
| 29 | 1.16 | 3.69 | −4.6 | 2 | 0 / 12.67 / 6.82 |
| 30 | 1.20 | 3.68 | −0.1 | 1 | 0 / 0 / 12.95 |
| 31 | 1.24 | 4.30 | +15.4 | 1 | 0 / 0 / 5.30 |
| 32 | 1.28 | 5.71 | +35.3 | 1 | 0 / 0 / 5.25 |
| 33 | 1.32 | 7.80 | +52.3 | 1 | 0 / 0 / 5.16 |
| 34 | 1.36 | 11.09 | +82.2 | 0 | 0 / 0 / 0 |

Index is already gone at 29. Middle leaves at 30. Thumb leaves at 34. Tilt rate then runs away. Eval `θ>1.2 rad` first fires at **step 40**.
