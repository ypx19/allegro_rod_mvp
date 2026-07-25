# Metrics Reference

## Gate used by `scripts/eval_policy.py`
Passed iff all hold on the eval set mean/rate:
- `axis_rotation_deg_mean > 180`
- `tip_error_m_mean < 0.02`
- `drop_rate <= 0.15`

Episode `is_success` (env info) additionally requires:
- `unwrapped_angle > π`
- `tip_error < 0.02`
- `axis_tilt < 0.25` rad
- not dropped / not unstable

## Latest hanging-tip curriculum snapshot (2026-07-23)

| Stage | Checkpoint | success_rate | rot_deg_mean | tip_err_m | drop_rate | passed |
|------:|---|---:|---:|---:|---:|:---:|
| 0 | `checkpoints/stage0/final_model.zip` | 0.95 | 527.6 | 0.00058 | 0.05 | yes |
| 1 | `checkpoints/stage1/final_model.zip` | 0.00 | -38.6 | 0.017 | 0.60 | no |
| 2 | `checkpoints/stage2/final_model.zip` | 0.00 | -52.5 | 0.008 | 0.00 | no |

## Training signals to watch
- `rollout/ep_rew_mean`, `success_rate`
- `train/value_loss`, `entropy_loss`, policy `std` (late explosion correlates with poor Stage 1/2)
- Env info: `axis_tilt_deg`, `unstable`, `contact_count`

## Axis Stabilizer Torque
Definition: Euclidean norm of the externally applied orientation-stabilizer torque.
Unit: MuJoCo torque units.
Aggregation: per-step mean and per-episode maximum.
Stage 2 requirement: exactly zero.

## Reward Components
Evaluation records per-step means for rotation, tip error, raw and weighted axis-tilt penalty, lateral angular velocity, contact, proximity, force, and action-rate terms. Raw and weighted axis-tilt values must both be retained when changing its weight.

## Termination Reason
Operational categories: `axis_tilt`, `tip_error`, `rod_height`, `nonfinite_reward`, `unstable`, or `none` for time truncation.
