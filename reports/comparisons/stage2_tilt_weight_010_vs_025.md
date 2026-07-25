# Stage 2 Tilt-Weight Comparison

## Compared Runs
- Baseline: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Candidate: `20260723-1230-stage2-tipjoint-tiltw025-seed0`

## Experimental Difference
Only axis-tilt penalty weight changed: 0.10→0.25.

## Evaluation Protocol
Same 2x512 parent, 20 deterministic episodes, seeds 0–19, tip joint active, axis stabilizer 0, Stage 2 randomization active.

## Metric Comparison
| Metric | Baseline | Weight 0.25 |
|---|---:|---:|
| Mean rotation | 1.13° | 1.76° |
| Mean tip error | 4.03 mm | 4.61 mm |
| Mean final tilt | 41.32° | 41.62° |
| Success rate | 0.00 | 0.00 |
| Drop rate | 1.00 | 1.00 |
| Axis-tilt terminations | 20/20 | 20/20 |
| Weighted tilt reward/step | -2.03 | -5.01 |

## Conclusion
The stronger penalty was applied but did not improve tilt recovery. The result rejects the hypothesis that weight magnitude alone is the dominant Stage 2 limitation.

## Recommendation
Change the learning signal from a larger absolute-state penalty to an explicit recovery/progress signal before another long run.
