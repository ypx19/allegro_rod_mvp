# EXP-20260824-001: DexScrew tilt-growth penalty at tip-connect s=100

## Experiment question
Does a default-preserving DexScrew tilt-growth penalty, scaled so a 0.05 rad
increase near 0.25 rad costs as much as `2.5 * ω` at `ω=1`, raise net-angle
success at bottom tip-connect `s=100` above the 0.50 gate?

## Change from baseline
Unique change versus the revolute→tip-connect transfer: `--axis-tilt-growth-scale 50`
on the DexScrew path. Recovery remains 0. Physics, observations (48-D), LR,
contact settings, grasp, and pose are otherwise the Phase C / T00 recipe at
`s=100`. Parent VecNormalize is reused.

Formula (predeclared):

`r_growth = -50 * clip(curr − prev, 0, 0.05) * w(curr)`

`w(θ) = clip((θ − 0.05) / (0.25 − 0.05), 0, 1)`

Existing quadratic state penalty is unchanged.

## Final metrics
| Split | Success | Rotation deg | Final tilt deg | Max tilt deg | Growth share | Penalty share | Rotation share | Terminations |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Zero-shot fixed | 0.0 | 182.31 | 79.25 | 79.25 | 0.041 | 0.321 | 0.243 | axis_tilt 10/10 |
| Zero-shot unseen | 0.0 | 169.09 | 80.57 | 80.57 | 0.040 | 0.318 | 0.228 | axis_tilt 10/10 |
| Trained fixed | 0.0 | 280.12 | 82.67 | 82.67 | 0.033 | 0.214 | 0.382 | axis_tilt 10/10 |
| Trained unseen | 0.0 | 264.34 | 80.05 | 80.05 | 0.035 | 0.205 | 0.419 | axis_tilt 10/10 |

Online training: success 0; episode length 22.7 → 40.3; return −84.3 → −0.74.
Final PPO `approx_kl=0.176`, `clip_fraction=0.642`, `std=0.709`. No NaNs.

## Best metrics
No checkpoint met the net-angle gate. Best visual/rotation failure is seed 10005
(399° net, 85° final tilt, 65 steps, `axis_tilt`).

## Artifact links
- Config: `config.yaml`
- Metrics: `metrics.csv`
- Plot: `plots/train_return_length_kl.png`
- Checkpoint: `checkpoints/final_model.zip`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- Evals: `eval_fixed.json`, `eval_unseen.json`, `eval_zeroshot_fixed.json`, `eval_zeroshot_unseen.json`
- Videos: `videos/tip_connect_best_00_seed10005_rot399deg_tilt85deg_steps65_axis_tilt.mp4`
- Comparison: `reports/comparisons/20260824-dexscrew-tilt-growth-s100.md`

## Important observations
Max episode tilt equals final tilt on every evaluated split: collapse remains
monotonic. Unlike recovery (0.2–0.4% of Σ|r|), growth does fire (3.3–4.1%).
Mean growth is −0.45/−0.55 versus tilt penalty −2.93/−3.17 and rotation
+5.22/+6.49. Rotation share rose while tilt stayed ~80–83°.

## Failure modes
Lateral axis-tilt termination (~80–83°) after ~40–65 steps. Tip error stays
<1 mm. Net angle often exceeds 180° (trained seed 10005 reached 399°) but the
0.25 rad tilt gate never holds.

## Conclusion
Reject as a Phase T gate solution. Growth shaping helped versus recovery only
in the narrow sense that the new term is on-policy; it did not restore balance.

## Recommended next step
Do not start other masses. Do not retune this growth scale as the next
one-factor test. Prefer a tilt assist, grasp change, or another dynamics
intervention; reward-only growth at g=50 is insufficient here.
