# Run Comparison

## Compared Runs
- Baseline: Stage 1, solref 0.05, stabilizer 0.25.
- Candidate A: `20260723-0110-stage1-tip-solref010-lowlr-seed0`.
- Candidate B: `20260723-0130-stage1-stab020-lowlr-seed0`.
- Rejected follow-up: `20260723-0200-stage1-stab018-lowlr-seed0`.

## Experimental Difference
One assist factor changed at each transition: first tip solref 0.05→0.10, then stabilizer 0.25→0.20, then 0.20→0.18.

## Evaluation Protocol
Deterministic policy, 20 episodes, seeds 0–19, 12 seconds per episode, unchanged metric implementation.

## Metric Comparison
| Condition | Rotation (deg) | Tip error (mm) | Success | Drop | Gate |
|---|---:|---:|---:|---:|:---:|
| solref .05 / stab .25 | 266.84 | 3.29 | 0.65 | 0.05 | pass |
| solref .10 / stab .25 | 223.96 | 11.89 | 0.55 | 0.05 | pass |
| solref .10 / stab .20 | 199.58 | 16.04 | 0.40 | 0.15 | pass |
| solref .10 / stab .18 | 177.18 | 20.06 | 0.25 | 0.30 | fail |

## Training Curves
See `assist_fade_progress.png` for the evaluation progression.

## Representative Videos
Two deterministic successes are stored in the adopted 0.20 run's `videos/` directory.

## Statistical Caveats
All training used seed 0. Evaluation covers 20 fixed seeds, but multi-training-seed validation remains required.

## Conclusion
Tip softening to 0.10 and stabilizer reduction to 0.20 are supported. Reduction to 0.18 is rejected.

## Recommendation
Add a training objective or curriculum mechanism that directly penalizes dependence on external stabilizer torque before attempting further assist removal.
