# Stage 0 Rolling Contact Gate

## Question
Can a rolling 20-step accumulated contact-reward threshold of +5 force discovery of simultaneous three-finger contact when the discrete contact reward is `0:-10, 1:-1, 2:+0.1, 3:+30`?

## Change from Baseline
Relative to `20260724-2200-stage0-contact30-seed0`, only the rolling contact gate was added. A trajectory terminates with `contact_support` when a full 20-step window sums below +5, and the same gate is required for success.

## Result
Failed. All five evaluated checkpoints had 0% three-contact occupancy and 0% success over fixed seeds 0–19. The best mean rotation was 149.93° at 20k steps, with 17/20 `contact_support` terminations and 3/20 `axis_tilt` terminations.

## Key Metrics

| Checkpoint | Rotation | Tip error | 3-contact fraction | Contact-gate terminations | Success |
|---|---:|---:|---:|---:|---:|
| 5k | 67.96° | 1.79 mm | 0.000 | 19/20 | 0.00 |
| 10k | 84.79° | 1.42 mm | 0.000 | 1/20 | 0.00 |
| 15k | 120.90° | 1.00 mm | 0.000 | 17/20 | 0.00 |
| 20k | 149.93° | 0.63 mm | 0.000 | 17/20 | 0.00 |
| 25k | 140.33° | 0.78 mm | 0.000 | 15/20 | 0.00 |

Finger2 contact occupancy was 0% at every checkpoint. The gate therefore enforced its condition but did not provide a learnable route to satisfy it from the current reset distribution.

## Artifacts
- Metrics: `metrics.csv`
- Plot: `plots/contact_gate_evaluation.png`
- Representative failure: `videos/stage0_best_00_seed17_rot167deg.mp4`
- Evaluation JSON: `checkpoints/*_eval.json`
- Training log: `logs/train.log`

## Conclusion
Reject the hard gate as the first Stage 0 curriculum condition. The failure type is exploration failure/curriculum mismatch, not a failure of the termination implementation.

## Recommended Next Step
Start Stage 0 from the already verified settled three-contact reset distribution, keep the same reward table, and delay activation of the rolling gate until the reset grasp has been maintained for an initial grace period. This changes the initialization before tuning the threshold further.
