# Run Comparison

## Compared Runs
- Published T00 (transfer, A flags, not retrained): `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0` smoke in `runs/20260914-t00-ablation-eval-smoke/`
- A: `runs/20260914-1638-t00-ablation-A-hist1-rewbase-seed0` (48-D, baseline reward, from scratch)
- B: `runs/20260914-1638-t00-ablation-B-hist4-rewbase-seed0` (192-D stack, baseline reward)
- C: `runs/20260914-1638-t00-ablation-C-hist1-rewsup-seed0` (48-D, support-aware reward)
- D: `runs/20260914-1638-t00-ablation-D-hist4-rewsup-seed0` (192-D stack, support-aware reward)

## Experimental Difference
Only `obs_history_len` ∈ {1,4} and `support_aware_reward_enabled` ∈ {false,true}.
T00 physics, grasp, s=400, μ=4, tilt kill 1.2, PPO `[512,256,128]`, 32 envs,
LR `3e-5`, 100k steps, seed 0, eval seeds 10000/20000 are shared.
A–D are from-scratch T00 policies, not revolute transfers.

## Evaluation Protocol
Net-angle success, 10 fixed + 10 unseen, tilt terminate 1.2 rad.
Collapse metrics: re-contact within 10 steps, post-loss ω_perp over 5 steps,
axis_tilt preceded by support loss in lookback 10.

## Metric Comparison
Fixed / unseen.

| Metric | T00 transfer | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Success | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Rotation (deg) | 177 / 176 | 305 / 303 | 297 / 295 | 275 / 276 | 327 / 329 |
| Episode length | ~35 | 54.0 / 54.4 | 50.0 / 50.1 | 36.5 / 36.6 | 49.7 / 49.0 |
| Recontact rate | 0.00 | 0.17 / 0.09 | 0.09 / 0.09 | 0.00 / 0.00 | 0.00 / 0.00 |
| 1-contact duration | — | 1.70 / 1.80 | 0.60 / 1.20 | 0.80 / 0.60 | 2.00 / 2.20 |
| ≥2-contact frac | — | 0.78 / 0.77 | 0.81 / 0.80 | 0.80 / 0.81 | 0.78 / 0.79 |
| Δω_perp after loss | — | 2.18 / 2.00 | 2.87 / 2.49 | **7.87 / 8.22** | 2.45 / 3.17 |
| axis_tilt preceded by loss | 1.00 (seed 6) | 0.40 / 0.20 | 0.80 / 0.70 | **1.00 / 1.00** | 0.50 / 0.50 |
| Final tilt (deg) | ~80 | 77 / 81 | 79 / 76 | 82 / 80 | 81 / 82 |
| Tip error (mm) | 1.8 | 1.7 / 1.8 | 1.7 / 1.6 | 1.9 / 1.9 | 1.8 / 1.9 |
| Terminations | all axis_tilt | 10/10 axis_tilt | 10/10 | 10/10 | 10/10 |

## Training Curves
Online `ep_len_mean` at 100k: A ~54, B similar, C shorter, D ~55 before eval.
KL remained small (~0.005–0.01). No NaN.

## Representative Videos
None exported; diagnostic plots:
- A seed 6: `runs/20260914-1638-t00-ablation-A-hist1-rewbase-seed0/traces/fixed/trace_seed6.png`
- C seed 6: `runs/20260914-1638-t00-ablation-C-hist1-rewsup-seed0/traces/fixed/trace_seed6.png`
- D seed 6: `runs/20260914-1638-t00-ablation-D-hist4-rewsup-seed0/traces/fixed/trace_seed6.png`

## Success Cases
None. Net-angle success is 0 on all 80 eval episodes.

## Failure Cases
Every episode still dies on `axis_tilt` near 80°. Support-loss events occur in
100% of A–D episodes. Re-contact within 400 ms is rare (A/B ~10–17%) or absent (C/D).

## Statistical Caveats
Single training seed. A–D are from-scratch, so A is not the published transfer
T00 (177°, length ~35). Relative A vs B vs C vs D is the valid 2×2.

## Conclusion
H1 is not supported: 4-frame history (B) does not raise re-contact or length vs A.
H2 is rejected in this form: support-aware reward (C) **worsens** post-loss ω_perp
(~8 vs ~2) and shortens episodes to ~37 steps; D does not recover A.
Do not retune λ next.

## Recommendation
Inspect grasp geometry / contact mechanics / whether ≥2-contact recovery is
physically feasible under this tip-connect grasp. Do not start T01.
