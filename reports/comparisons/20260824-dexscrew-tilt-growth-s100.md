# Run Comparison

## Compared Runs
- Zero-shot parent: accepted revolute `s=100` checkpoint
  `runs/20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0/checkpoints/final_model.zip`
  evaluated on bottom tip-connect `s=100` with growth scale 50 (physics-identical
  to recovery's growth=0 zero-shot; only logged reward components differ).
- Candidate: `20260824-0000-dexscrew-tilt-growth-s100-tip-seed0`
  65,536-step PPO fine-tune with DexScrew growth scale 50, recovery 0.
- Prior reward probe: `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`
  same parent, budget, pose/grasp, and gait flags; unique change was recovery
  scale 50 instead of growth scale 50.

## Experimental Difference
The only intentional training difference versus zero-shot transfer is the
predeclared DexScrew tilt-growth penalty:

`r_growth = -50 * clip(curr − prev, 0, 0.05) * w(curr)`

`w(θ) = clip((θ − 0.05) / 0.20, 0, 1)`

Recovery stays at default 0. Existing quadratic tilt penalty, 48-D
observations, saved palm pose, recovered grasp, proportional full-vector
friction at `s=100`, and C gait settings are shared. Parent VecNormalize is
loaded.

Versus the recovery run, the unique difference is which one-sided Δtilt term
is enabled (growth vs recovery), not scale family, budget, or parent.

## Evaluation Protocol
Deterministic net-angle success, 10 episodes each, fixed seeds 10000–10009 and
unseen seeds 20000–20009. Success requires unwrapped angle ≥ π, tilt < 0.25 rad,
tip/anchor error < 0.02 m, finite values, and no physical drop. Contact is
diagnostic.

## Metric Comparison
| Metric | Zero-shot fixed | Growth fixed | Recovery fixed | Zero-shot unseen | Growth unseen | Recovery unseen |
|---|---:|---:|---:|---:|---:|---:|
| Success rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Rotation deg | 182.31 | 280.12 | 190.95 | 169.09 | 264.34 | 223.07 |
| Final tilt deg | 79.25 | 82.67 | 81.49 | 80.57 | 80.05 | 82.58 |
| Max tilt deg | 79.25 | 82.67 | 81.49 | 80.57 | 80.05 | 82.58 |
| Tip error m | 0.00067 | 0.00077 | 0.00073 | 0.00079 | 0.00068 | 0.00073 |
| Drop rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| axis_tilt terms | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Growth / Σ\|r\| | 0.041 | 0.033 | 0.000 | 0.040 | 0.035 | 0.000 |
| Recovery / Σ\|r\| | 0.000 | 0.000 | 0.002 | 0.000 | 0.000 | 0.004 |
| Penalty / Σ\|r\| | 0.321 | 0.214 | 0.265 | 0.318 | 0.205 | 0.254 |
| Rotation / Σ\|r\| | 0.243 | 0.382 | 0.282 | 0.228 | 0.419 | 0.332 |
| ≥2-contact frac | 0.886 | 0.699 | 0.743 | 0.885 | 0.744 | 0.801 |

Zero-shot rotation/tilt match the recovery run's zero-shot, as expected for a
physics-identical parent.

## Training Curves
`runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/plots/train_return_length_kl.png`

Online length 22.7 → 40.3 steps; return −84.3 → −0.74; success stayed 0.
Final `approx_kl=0.176`, `clip_fraction=0.642`, policy `std=0.709`.
Recovery run: length 22.7 → 41.0, return −67.3 → −27.3, `approx_kl=0.199`,
`clip_fraction=0.645`.

## Representative Videos
No success episodes exist. Decode-checked failures:

- Trained high-angle failure: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/videos/tip_connect_best_00_seed10005_rot399deg_tilt85deg_steps65_axis_tilt.mp4`
- Trained second failure: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/videos/tip_connect_best_01_seed10003_rot308deg_tilt86deg_steps67_axis_tilt.mp4`
- Zero-shot failure: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/videos/zeroshot/tip_connect_best_00_seed10000_rot355deg_tilt74deg_steps63_axis_tilt.mp4`
- Recovery high-angle failure: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/tip_connect_best_00_seed10006_rot333deg_tilt81deg_steps46_axis_tilt.mp4`

## Success Cases
None. Net angle often exceeds 180° (trained seed 10005 reached 399°), but
final tilt remains ~80–85° so the 0.25 rad gate fails.

## Failure Cases
All 40 growth-run episodes (20 zero-shot + 20 trained) terminate on `axis_tilt`.
Per-episode max tilt equals final tilt, indicating monotonic collapse. Growth
is nonzero while increasing (mean −0.45 to −0.55 / step), unlike recovery
(+0.03 to +0.06). That on-policy mass did not stop the fall.

## Statistical Caveats
One training seed and a 65,536-step budget. Rotation increased more than in
the recovery run, but tilt and success did not. This is not a multi-seed result.

## Conclusion
Growth shaping helped versus recovery only as a credit-assignment check: the
new term occupies ~3–4% of |reward| instead of ~0.2–0.4%. It did not help the
task gate. Final/max tilt stayed ~80°, and rotation share rose (38–42%) while
growth stayed an order of magnitude smaller than rotation.

## Recommendation
Reject this reward change for Phase T. Do not start other masses. Do not retune
g as the next one-factor test. Prefer a tilt assist, grasp change, or other
dynamics/initialization change rather than another Δtilt-only scale.
