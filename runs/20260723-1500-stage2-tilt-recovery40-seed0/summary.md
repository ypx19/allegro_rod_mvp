# Stage 2 Local Tilt-Recovery Reward

## Question
Does a clipped local reward for reducing axis tilt teach active recovery after the external stabilizer is removed?

## Change from Baseline
Added only `clip(40 * (previous_tilt - current_tilt), -2, 2)`. The absolute tilt weight stayed 0.10. The tip ball/universal joint stayed active at solref 0.10, the external axis stabilizer stayed at zero, Stage 2 randomization stayed active, and rotation scale stayed 160.

## Result
Failed. All six checkpoints had 20/20 axis-tilt terminations. The final checkpoint had the highest mean rotation at 4.22°, with 4.73 mm tip error, 41.40° final tilt, success 0, and drop 1.0.

## Reward Evidence
The recovery term was active and averaged -0.283 per step at the final checkpoint. It improved mean rotation relative to the 1.13° corrected Stage 2 baseline and the 1.76° stronger-absolute-penalty run, but it did not improve survival or reduce the dominant termination mode.

## Visual Evidence
- `videos/stage2_best_00_seed0_rot14deg.mp4`
- `videos/stage2_best_01_seed6_rot11deg.mp4`
- `videos/stage2_best_02_seed13_rot10deg.mp4`
- `plots/stage2_reward_strategy_comparison.png`

The video exporter found no success or near-success and therefore preserved the three highest-rotation failures.

## Decision
Reject recovery scale 40 as a standalone Stage 2 fix. Do not extend this run: it missed both the standard gate and the predeclared diagnostic criterion.

## Next Step
Separate abrupt assist removal from Stage 2 randomization. Adapt the best stabilizer-0.12 policy under Stage 2 randomization while keeping stabilizer 0.12, then fade the stabilizer only if that control retains useful performance.
