# Run Comparison

## Compared Runs
- Zero-shot parent: accepted revolute `s=100` checkpoint
  `runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/final_model.zip`
  evaluated on bottom tip-connect `s=100` with recovery scale 0.
- Candidate: `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`
  65,536-step PPO fine-tune with DexScrew recovery scale 50.
- Related (not mass-matched): Phase T `s=400` low-LR retry
  `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0`.

## Experimental Difference
The only intentional training difference versus zero-shot transfer is the
predeclared DexScrew one-sided back-to-balance term:

`r_recovery = 50 * clip(prev − current, 0, 0.05) * 1[current > 0.05 rad]`

Existing quadratic tilt penalty, 48-D observations, saved palm pose, recovered
grasp, proportional full-vector friction at `s=100`, and C gait settings are
shared. Parent VecNormalize is loaded (not reset; no affine adapter).

## Evaluation Protocol
Deterministic net-angle success, 10 episodes each, fixed seeds 10000–10009 and
unseen seeds 20000–20009. Success requires unwrapped angle ≥ π, tilt < 0.25 rad,
tip/anchor error < 0.02 m, finite values, and no physical drop. Contact is
diagnostic.

## Metric Comparison
| Metric | Zero-shot fixed | Recovery fixed | Zero-shot unseen | Recovery unseen |
|---|---:|---:|---:|---:|
| Success rate | 0.00 | 0.00 | 0.00 | 0.00 |
| Rotation deg | 182.31 | 190.95 | 169.09 | 223.07 |
| Final tilt deg | 79.25 | 81.49 | 80.57 | 82.58 |
| Max tilt deg | 79.25 | 81.49 | 80.57 | 82.58 |
| Tip error m | 0.00067 | 0.00073 | 0.00079 | 0.00073 |
| Drop rate | 1.00 | 1.00 | 1.00 | 1.00 |
| axis_tilt terms | 10/10 | 10/10 | 10/10 | 10/10 |
| Recovery / Σ\|r\| | 0.000 | 0.002 | 0.000 | 0.004 |
| Penalty / Σ\|r\| | 0.335 | 0.265 | 0.332 | 0.254 |
| Rotation / Σ\|r\| | 0.253 | 0.282 | 0.238 | 0.332 |
| ≥2-contact frac | 0.886 | 0.743 | 0.885 | 0.801 |

Related `s=400` T00 low-LR: success 0.0/0.0, rotation 177.02/175.45, final tilt
81.40°. Same termination mode; not a matched mass.

## Training Curves
`runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/plots/train_return_length_kl.png`

Online length 22.7 → 41.0 steps; return −67.3 → −27.3; success stayed 0.
Final `approx_kl=0.199`, `clip_fraction=0.645`, policy `std=0.712`.

## Representative Videos
No success episodes exist. Decode-checked failures:

- Trained high-angle failure: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/tip_connect_best_00_seed10006_rot333deg_tilt81deg_steps46_axis_tilt.mp4`
- Trained second failure: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/tip_connect_best_01_seed10007_rot289deg_tilt80deg_steps56_axis_tilt.mp4`
- Zero-shot failure: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/zeroshot/tip_connect_best_00_seed10000_rot355deg_tilt74deg_steps63_axis_tilt.mp4`

## Success Cases
None. Net angle sometimes exceeds 180° (trained seed 10006 reached 333°), but
final tilt remains ~80° so the 0.25 rad gate fails.

## Failure Cases
All 40 evaluated episodes (20 zero-shot + 20 trained) terminate on `axis_tilt`.
Per-episode max tilt equals final tilt, indicating monotonic collapse rather
than recover-then-hold.

## Statistical Caveats
One training seed and a 65,536-step budget. Rotation increased on unseen seeds,
but tilt and success did not. This is not a multi-seed result.

## Conclusion
The recovery term did not help the task gate. It increased mean rotation slightly
and left lateral tilt at least as bad as zero-shot. Recovery’s on-policy mass is
~0.2–0.4% of |reward| because tilt almost never decreases.

## Recommendation
Reject this reward change for Phase T. Do not start other masses. Next
discriminating test should provide a signal while tilt is growing, or change the
dynamics/initialization rather than paying only for already-decreasing tilt.
