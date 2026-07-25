# Run Comparison

## Compared Runs
- Corrected Stage 2 baseline: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Stronger absolute penalty: `20260723-1230-stage2-tipjoint-tiltw025-seed0`
- Local recovery shaping: `20260723-1500-stage2-tilt-recovery40-seed0`

## Experimental Difference
All runs keep the tip ball/universal joint active and external stabilizer at zero. The absolute-penalty run changes tilt weight 0.10→0.25. The recovery run keeps tilt weight 0.10 and adds clipped `40 * (previous_tilt - current_tilt)`.

## Evaluation Protocol
Twenty deterministic episodes using seeds 0–19, 12-second limit, Stage 2 mass/friction randomization, rotation scale 160, and the same metric implementation and success gate.

## Metric Comparison
| Metric | Baseline | Absolute 0.25 | Recovery 40 |
|---|---:|---:|---:|
| Mean rotation | 1.13° | 1.76° | 4.22° |
| Mean tip error | 4.03 mm | 4.61 mm | 4.73 mm |
| Mean final tilt | 41.32° | 41.62° | 41.40° |
| Success rate | 0.00 | 0.00 | 0.00 |
| Drop rate | 1.00 | 1.00 | 1.00 |
| Axis-tilt terminations | 20/20 | 20/20 | 20/20 |
| Added/changed tilt reward per step | — | -5.01 absolute | -0.283 recovery |

## Representative Videos
- Recovery failure, seed 0: `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_00_seed0_rot14deg.mp4`
- Recovery failure, seed 6: `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_01_seed6_rot11deg.mp4`
- Recovery failure, seed 13: `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_02_seed13_rot10deg.mp4`

## Statistical Caveats
Each candidate used one training seed. Evaluation uses the same 20 fixed seeds, so the within-protocol comparison is controlled, but multi-training-seed evidence is unavailable.

## Conclusion
The recovery reward produced a small rotation increase but no survival improvement. Neither stronger absolute deviation punishment nor one-step recovery shaping solves the loss of axis control after direct stabilizer removal.

## Recommendation
Run a curriculum control that holds the stabilizer at 0.12 while enabling Stage 2 randomization. This isolates randomization adaptation from assist removal before attempting another stabilizer fade.
