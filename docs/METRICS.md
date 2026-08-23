# Metrics Reference

## Gate used by `scripts/eval_policy.py`

### `reward_style=stage` (legacy curriculum)
Passed iff all hold on the eval set mean/rate:
- `axis_rotation_deg_mean > 180`
- `tip_error_m_mean < 0.02`
- `drop_rate <= 0.15`

Episode `is_success` requires:
- `unwrapped_angle > π`
- `tip_error < 0.02`
- `axis_tilt < 0.25` rad (skipped for `physics_mode=revolute`)
- not dropped / not unstable
- when configured, the rolling contact-support gate is satisfied

### `reward_style=dexscrew`
Reward uses **axial ω** (`clip(ω)·scale`); unwrapped angle is logging/metric only.
On `physics_mode=tip_connect`, **axis tilt is punished** by default (`dexscrew_tilt_scale=1.0`, σ≈0.15 rad); revolute keeps tilt_scale=0.

Episode `is_success` requires:
- sustain `axial_omega > omega_success_threshold` (default **0.5 rad/s**) for **`omega_success_hold_seconds`** consecutive seconds (default **10.0 s** → 250 steps at 25 Hz; episode default 20 s)
- `tip_error < 0.02`
- `axis_tilt < 0.25` rad (skipped for revolute)
- not dropped; contact gate if configured

Eval `passed` iff:
- `success_rate >= 0.5`
- `tip_error_m_mean < 0.02`
- `drop_rate <= 0.15`

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

## Parallel training stack (EXP-infra, 2026-08-02)
Definition:
DexScrew-style track trainer `scripts/train_parallel.py` uses `SubprocVecEnv`, CUDA policy device, `net_arch` `[512,256,128]`, and `VecNormalize` (obs + reward, `clip_obs=10`).
Unit:
Env steps are total across workers (`num_envs * env.step` calls).
Aggregation:
`runs/<run_id>/metrics.csv` logs per-rollout `step`, `wall_time`, episode return/length when available, and SB3 `train_*` scalars. Save `checkpoints/vecnormalize.pkl` with every final model.
Success threshold (infra):
Finite rewards/losses; `num_envs ≥ 8`; checkpoint + VecNormalize reload.
Reference runs:
- EXP-20260802-001 / `20260802-0217-exp-infra-subproc8-cuda-seed0` (8 envs, 2e5 steps, ~939 fps; infra passed, Stage 0 success_rate=0).
- EXP-20260802-002 / `20260802-0220-exp-infra-subproc64-1e9-seed0` (64 envs, 1e9 steps, running; see FIND-20260802-001).
Code modification:
New file only: `scripts/train_parallel.py`. Legacy `scripts/train.py` defaults unchanged.

## Axis Stabilizer Torque
Definition: Euclidean norm of the externally applied orientation-stabilizer torque.
Unit: MuJoCo torque units.
Aggregation: per-step mean and per-episode maximum.
Stage 2 requirement: exactly zero.

## Reward Components
Evaluation records per-step means for rotation, tip error, raw and weighted axis-tilt penalty, lateral angular velocity, contact, proximity, force, and action-rate terms. Raw and weighted axis-tilt values must both be retained when changing its weight.

## Fingertip Contact Count and Discrete Contact Reward
Definition:
Each fingertip is in contact when its summed normal contact force against the rod exceeds 0.05 N. Contact count is the number of contacting fingertips, from 0 to 3.
Unit:
Count; per-step contact reward is unitless.
Aggregation:
Report the fraction of evaluation steps at each contact count and the per-finger contact-step fraction.
Evaluation frequency:
Every checkpoint evaluation.
Success threshold:
No standalone threshold. For EXP-20260724-003, diagnostic support requires a nonzero three-contact fraction and fewer axis-tilt terminations without sacrificing rotation.
Edge cases:
Numerically unstable steps report zero contacts. Multiple contacts on one fingertip are summed but count as one fingertip.
Implementation:
`RodRotationEnv._touch`, `RodRotationEnv._contact_reward`, and `scripts/eval_policy.py`.
Discrete EXP-20260724-003 mapping:
0 → -10.0, 1 → -1.0, 2 → +0.1, 3 → +10.0. EXP-20260724-009 and later contact experiments use +30.0 for three contacts. The legacy linear mapping remains available for reproduction.

Allegro bottom-tip curriculum note (2026-08-23):
`three_contact_required=True` changes the rolling gate sample to a binary
three-contact indicator, independent of the configured bonus magnitude. The
smoke used a +3.0 three-contact bonus and exposed a static-grasp optimum
(`DBG-20260823-001`). The next controlled experiment should use +0.3 while
retaining the 25-step window, threshold 18 (72%), and no rotation credit below
three contacts.

## Rolling Contact-Support Gate
Definition:
The sum of raw discrete contact rewards over the most recent configured number of steps.
Unit:
Reward units.
Aggregation:
Per-step rolling sum; evaluation reports termination reason counts and three-contact step occupancy.
Success threshold:
For EXP-20260724-010, a full 20-step window must sum to at least +5. The condition is required for both episode continuation and success.
Edge cases:
The gate is not ready before the first full window. With the +30/+0.1/-1/-10 mapping, a +5 threshold cannot be passed by only zero-, one-, or two-contact states.
Implementation:
`RodRotationEnv._contact_gate_status` and `RodRotationEnv.step`.

## Termination Reason
Operational categories: `axis_tilt`, `tip_error`, `rod_height`, `contact_support`, `nonfinite_reward`, `unstable`, or `none` for time truncation.
