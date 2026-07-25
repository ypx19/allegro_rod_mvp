# Stage 2 Axis-Tilt Weight 0.25

## Question
Is Stage 2 failing because the axis-deviation penalty weight 0.10 is too small?

## Change from Baseline
Only `axis_tilt_penalty_weight` changed from 0.10 to 0.25. Tip joint remained active at solref 0.10, axis stabilizer remained exactly zero, Stage 2 randomization remained active, rotation scale remained 160, and the 2x512 policy and optimizer settings were unchanged.

## Result
Failed. Every checkpoint had 20/20 axis-tilt terminations. The best mean rotation was 1.89° at step 60600. Final metrics were 1.76° rotation, 4.61 mm tip error, 41.62° final tilt, success 0, and drop 1.0.

## Reward Evidence
The weighted tilt term changed from -2.03 to -5.01 per step, confirming that the requested weight increase was active. Rotation reward changed only from +0.71 to +0.90. Explained variance remained negative late in training and value loss remained in the thousands.

## Visual Evidence
- `videos/stage2_best_00_seed13_rot7deg.mp4`
- `videos/stage2_best_01_seed0_rot7deg.mp4`
- `videos/stage2_best_02_seed9_rot7deg.mp4`
- `plots/reward_components_comparison.png`

## Decision
Reject weight 0.25 as a standalone fix. Do not increase it again without changing the reward form or curriculum, because the stronger absolute penalty dominated value learning without producing active recovery.

## Next Step
Test a recovery-shaped term based on reduction in tilt (`previous_tilt - current_tilt`) or a short stabilizer-fade curriculum, while keeping the absolute tilt weight controlled.
