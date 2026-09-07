# EXP-20260823-018: DexScrew back-to-balance at tip-connect s=100

## Experiment question
Does a default-preserving DexScrew one-sided tilt-recovery term, scaled to compete
with `2.5 * ω` near the 0.25 rad gate, produce net-angle success after a short
fine-tune from the accepted revolute s=100 checkpoint?

## Change from baseline
Unique change versus the revolute→tip-connect transfer: `--axis-tilt-recovery-scale 50`
on the DexScrew path. Physics, observations (48-D), LR, contact settings, grasp, and
pose are otherwise the Phase C / T00 recipe at `s=100`.

Formula (predeclared):

`r_recovery = 50 * clip(prev_tilt - current_tilt, 0, 0.05) * 1[current_tilt > 0.05]`

Default remains 0. Existing tilt penalty is unchanged.

## Final metrics
| Split | Success | Rotation deg | Final tilt deg | Max tilt deg | Recovery share | Penalty share | Terminations |
|---|---:|---:|---:|---:|---:|---:|---|
| Zero-shot fixed | 0.0 | 182.31 | 79.25 | 79.25 | 0.00 | 0.335 | axis_tilt 10/10 |
| Zero-shot unseen | 0.0 | 169.09 | 80.57 | 80.57 | 0.00 | 0.332 | axis_tilt 10/10 |
| Trained fixed | 0.0 | 190.95 | 81.49 | 81.49 | 0.002 | 0.265 | axis_tilt 10/10 |
| Trained unseen | 0.0 | 223.07 | 82.58 | 82.58 | 0.004 | 0.254 | axis_tilt 10/10 |

Online training: success 0; episode length 22.7 → 41.0; return -67.3 → -27.3.
Final PPO `approx_kl=0.199`, `clip_fraction=0.645`, `std=0.712`. No NaNs.

## Best metrics
No checkpoint met the net-angle gate. Best visual/rotation failure is seed 10006
(333° net, 81° final tilt, 46 steps, `axis_tilt`).

## Artifact links
- Config: `config.yaml`
- Metrics: `metrics.csv`
- Plot: `plots/train_return_length_kl.png`
- Checkpoint: `checkpoints/final_model.zip`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- Evals: `eval_fixed.json`, `eval_unseen.json`, `eval_zeroshot_fixed.json`, `eval_zeroshot_unseen.json`
- Videos: `videos/tip_connect_best_00_seed10006_rot333deg_tilt81deg_steps46_axis_tilt.mp4`
- Comparison: `reports/comparisons/20260823-dexscrew-tilt-recovery-s100.md`

## Important observations
Max episode tilt equals final tilt on every evaluated episode: collapse is
monotonic, so the one-sided recovery term almost never fires (mean +0.03 to
+0.06 / step versus tilt penalty −3.3 to −3.8).

## Failure modes
Lateral axis-tilt termination (~80–83°) after ~40–60 steps. Tip error stays
<1 mm. Contact occupancy remains nonzero but success tilt <0.25 rad is never met.

## Conclusion
Reject as a Phase T gate solution. The recovery term did not restore balance.

## Recommended next step
Do not start other masses or a tip-connect curriculum. Test a signal that is
nonzero while tilt is *increasing* (growth penalty near 0.25 rad) or a tilt
assist/curriculum; recovery-only shaping has no on-policy samples here.
