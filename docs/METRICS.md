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

## Two-Phase Mass/Friction Curriculum Gate (legacy declaration 2026-08-23)
Applies to `scripts/run_two_phase_force_curriculum.py`, using deterministic fixed
seed 10000 and unseen seed 20000 evaluations at every accepted stage.

A stage is accepted only when both seed sets satisfy all of:
- sustained-omega success rate >= 0.50;
- mean unwrapped axis rotation > 180 degrees;
- three-tip contact occupancy >= 0.72;
- violation/early-termination rate <= 0.15;
- bottom tip-connect endpoint error < 0.02 m;
- mean rotation reward is positive and at least as large as the mean +0.3
  three-contact reward.

Phase T may start only after Phase R passes this gate at `s=1`. Return is not a
gate. The training contact bonus is +0.3 while the existing binary 25-step /
18-step support window and no-rotation-credit-without-three-contacts rule remain
unchanged.

For the later user-authorized gait curriculum, the explicit `net_angle` section
below supersedes these task-gate bullets without rewriting historical results.
Both fixed and unseen success rates must be >=0.50 under the per-episode
net-angle formula. Contact occupancy/window, omega hold, and contact termination
are diagnostic only. Phase ordering remains unchanged.

## Fingertip Normal Pressing Force
Definition:
For each fingertip/rod contact, `mj_contactForce(...)[0]` in the MuJoCo contact
frame, summed over simultaneous contacts on that fingertip. This is measured
normal contact force and is explicitly not actuator torque, actuator force, or
control effort.

Unit:
Newtons.

Aggregation:
Per-tip and total-across-tips mean, median, and p95 over every deterministic
evaluation step remain visible. Historical EXP-015/016 additionally reported
the median total normal force over
timesteps satisfying `axial_omega > 0.5 rad/s AND contact_count >= 2`.
Eligible and excluded timestep fractions are both reported so poor-contact and
nonrotating intervals are not hidden.

Evaluation frequency:
At every mass/friction calibration trial, including rejected trials.

Interpretation:
This is the learned policy's actual contact force, not the minimum normal force
required by the physics. EXP-015/016 conditioned on only about 10–18% of steps,
while the policy's gait, contact set, speed, and action sequence changed between
stages. Therefore this metric is diagnostic only and must not gate corrected
curriculum progression. Its historical values remain valid raw measurements.

Edge cases:
All-step metrics retain zero-contact steps as 0 N. Conditioned metrics exclude
them by definition but report the excluded fraction. Sparse eligibility prevents
this statistic from estimating required force.

Implementation:
`RodRotationEnv._touch`, `scripts/eval_policy.py`, and
`scripts/run_two_phase_force_curriculum.py`.

## Explicit Rod/Fingertip Friction Scaling
Definition:
`contact_friction_scale` multiplies only the first MuJoCo friction component
(sliding coefficient) by default, preserving historical behavior.
`contact_friction_scaling_mode=full_vector` instead multiplies sliding,
torsional, and rolling inputs on `rod_geom` and `tip0`/`tip1`/`tip2`.

All four geoms currently have equal priority 0. MuJoCo therefore combines each
input component with an element-wise maximum. Scaling both sides is required;
the effective five-value contact vector is
`[slide, slide, torsion, rolling, rolling]`. Evaluations record both per-geom
inputs and effective rod/pad pair vectors.

Corrected proportional schedule:
For a fixed angular trajectory, inertia and required tangential torque scale
approximately with rod mass scale `s`. Since friction capacity is
`tau ~= mu * N * r`, holding required normal force `N` constant motivates
`friction_scale(s) = 4*s/400`. No clipping or policy-force adaptation is used.
The full per-geom vector is `[1.8,0.05,0.001] * friction_scale(s)`.

## Controlled Required-Force Validation
The corrected validation uses no learned policy. It asks whether a stationary
angular trajectory can reject a deterministic axial torque
`tau(s)=5*s/400 N m` under fixed preload controls. Friction uses the full
proportional schedule, rod damping/armature/frictionloss scale by `s/400`, and
stabilizer torque is zero.

The preload sweep is exactly `[0,0.25,0.5,0.75,1,1.5,2,3]` at `s=400,25,1`.
Tracking passes when maximum angle error is below 5 degrees, maximum axial speed
below 0.2 rad/s, at least two contacts exist for 90% of steps, and values remain
finite. The first passing measured median total normal force estimates required
preload.

Analytical comparison:
`N_min ~= tau / (mu_slide*r + mu_torsion)`, with `r=0.01 m`.
This assumes full simultaneous friction capacity and ignores contact-force
distribution, elliptic friction coupling, compliance, actuator saturation, and
discrete preload resolution, so it is an approximation rather than an exact
invariance proof.

## Axial Slip Proxy
Definition:
For contacting fingertips, subtract rod center-of-mass velocity from the mean
fingertip linear velocity, project onto the rod axis, and take the absolute value.
No-contact steps report zero.

Unit:
Meters per second.

Aggregation:
Mean, median, p95, and maximum over deterministic evaluation steps. This is a
kinematic proxy, not a direct Coulomb stick/slip solver state.

Implementation:
`RodRotationEnv._axial_slip_proxy` and `scripts/eval_policy.py`.

## Companion Grasp Reset Robustness
Definition:
A configured Allegro reset is robust for one physics/mass condition only when
every declared fixed seed completes the full audit horizon with finite
observations, all three fingertip normal forces above 0.05 N at every step, no
`contact_support` or other early termination, no rod collision involving a
non-fingertip hand geom, and finite constraint error.

Unit:
Pass count over total seeds; three-contact occupancy is a fraction; per-tip
normal force is N; revolute endpoint and tip-connect anchor error is m.

Aggregation:
Report pass count, minimum per-seed three-contact occupancy, median-over-seeds
per-tip median force, maximum constraint error, and maximum non-tip rod-contact
count separately for each physics mode and mass endpoint.

Evaluation frequency:
Before policy training for a new palm/grasp pair and after changing reset qpos,
grasp preload, ramp, hold, noise, contact physics, or endpoint mass.

Success threshold:
10/10 seeds pass a 100-step zero-action audit at both `s=400` and `s=1` for a
preset claimed to support a phase. A preset is not called shared across physics
modes unless all four endpoint conditions pass.

Edge cases:
Policy performance is not inferred from this metric. A reset may pass while a
learned action sequence later loses support. Contacts caused by deep
interpenetration or non-fingertip hand geoms invalidate the candidate.

Implementation:
`scripts/evaluate_hand_grasp_reset.py` and
`tests.test_contact_detection.ContactDetectionTest.test_saved_pose_revolute_grasp_is_robust_at_mass_endpoints`.

## Rotation Reward Contact-Credit Policy
Definition:
`rotation_requires_three_contacts=true` sets only the rotation reward component
to zero when fewer than three fingertip forces exceed 0.05 N. It does not alter
measured angle, contact reward, contact occupancy, or rolling-gate state.
`false` preserves the genuine signed rotation component on those steps.

Default:
`true`, preserving the strict curriculum behavior.

Independent termination control:
`contact_support_termination_enabled=true` terminates when the configured rolling
contact gate fails. Setting it false leaves the 25-step/18-hit gate calculation,
contact occupancy, and contact reward active, but does not terminate because of
that gate. Such runs still fail the original task-quality criterion when
three-contact occupancy is below 0.72.

Implementation:
`RodRotationEnv._rotation_reward_credit`, `RodRotationEnv.step`,
`scripts/train_parallel.py`, `scripts/eval_policy.py`, and
`scripts/export_success_videos.py`.

## Scaled Contact Reward and Finger-Gait Metrics
Contact reward scaling:
`contact_reward_scale` multiplies the complete contact reward component after the
raw contact-count ladder is computed. Default 1.0 preserves previous behavior.
EXP-20260823-012 uses 0.10, mapping raw counts 0/1/2/3
`[-10,-1,+0.1,+0.3]` to `[-1,-0.1,+0.01,+0.03]`.

Leave-return event:
For each fingertip, one event is counted when contact transitions present→absent
and later absent→present. Consecutive absent steps count as one leave phase.

Contact fractions:
Report global fractions with at least one and at least two contacts, global
three-contact occupancy, and minimum 25-step three-contact occupancy per episode.

Complete unsupported duration:
Longest consecutive zero-contact interval, reported in steps and seconds.

Rotation cycles:
Net completed cycles are `floor(abs(final unwrapped angle) / 2π)`. Also report
cumulative absolute per-step rotation to distinguish repeated motion from net
angle. A repeated-cycle episode has at least two completed net cycles.

Gait safety:
Report per-tip longest contact-loss duration, rod tip/constraint error maximum,
normal-force p95, numerical-instability episodes, omega fraction above threshold,
and the existing ten-second sustained-omega metric.

Implementation:
`scripts/eval_policy.py` and `RodRotationEnv._scaled_contact_reward`.

## Explicit Success Modes (declared 2026-08-23)
`success_mode=omega_hold` is the default and preserves historical behavior:
DexScrew success requires the configured continuous omega hold plus legacy
tip/tilt/contact/drop checks.

`success_mode=net_angle` is the gait evaluation mode. Per-episode success is:

`finite_stable AND NOT physical_drop AND net_unwrapped_angle_rad >= pi AND
axis_tilt_rad < 0.25 AND tip_error_m < 0.02`.

The threshold is positive `pi rad = 180 deg`. Negative rotation fails. Cumulative
absolute rotation is not used, so oscillation with zero net angle fails.
Contact-window satisfaction, uninterrupted omega, and contact-support termination
are not success requirements. Contact quality remains separately reported.

Evaluations report three fields without rewriting history:
- `success_rate`: rate for the selected explicit mode;
- `legacy_omega_success_rate`;
- `net_angle_success_rate`.

Physical drop includes rod-height, physical tip-error termination, axis-tilt
termination where applicable, and nonfinite/unstable simulation. Evaluation of
gait checkpoints disables contact-support termination explicitly while retaining
all contact diagnostics.

## DexScrew Axis-Tilt Recovery (declared 2026-08-23)
Definition:
One-sided back-to-balance term on `reward_style=dexscrew`:

`r_recovery = scale * clip(prev_tilt - current_tilt, 0, 0.05) * 1[current_tilt > 0.05]`

Units:
Tilt is radians. The reward is unitless. `scale` is `--axis-tilt-recovery-scale`
(default 0). Deadzone and clip are 0.05 rad. The existing quadratic tilt penalty
`r_tilt = -clip((θ/0.15)^2, 0, 25)` with `dexscrew_tilt_scale=1.0` on tip-connect
is unchanged.

Default:
`scale=0`, so historical DexScrew totals are unchanged. The stage-path recovery
formula `clip((prev-curr)*scale, -2, 2)` is unmodified and still uses the same
CLI flag.

Experiment scale (EXP-20260823-018 only):
`scale=50`. A 0.05 rad decrease above the deadzone yields +2.5, equal to
`2.5 * ω` at `ω=1 rad/s`. Holding upright (`current <= 0.05`) yields 0.

Predeclared per-step values at tilt-penalty scale 1.0:

| Current tilt (rad) | Penalty | Recovery at Δθ=+0.05 (max credit) | Recovery at Δθ=+0.01 |
|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 (deadzone) | 0.00 |
| 0.15 | -1.00 | +2.50 | +0.50 |
| 0.25 | -2.78 | +2.50 | +0.50 |
| 0.50 | -11.11 | +2.50 | +0.50 |
| 0.75 | -25.00 (saturated) | +2.50 | +0.50 |

Aggregation:
Per-step mean `reward_axis_tilt_recovery` and its share of Σ|reward components|.
Evaluation also reports mean final tilt and mean/max of per-episode maximum tilt
(`axis_tilt_deg_max_mean`, `axis_tilt_deg_max_max`).

Implementation:
`allegro_rod_mvp/rewards_dexscrew.py`, `RodRotationEnv.step`, and
`scripts/eval_policy.py`.

## DexScrew Axis-Tilt Growth Penalty (declared 2026-08-23)
Definition:
One-sided penalty on `reward_style=dexscrew` while tilt is increasing:

`r_growth = -g * clip(current_tilt - previous_tilt, 0, 0.05) * w(current_tilt)`

`w(θ) = clip((θ - 0.05) / (0.25 - 0.05), 0, 1)`

Units:
Tilt is radians. The reward is unitless. `g` is `--axis-tilt-growth-scale`
(default 0). Deadzone 0.05 rad, gate 0.25 rad, clip 0.05 rad. Weight is
evaluated at current tilt so a jump through the success gate is not free.
The existing quadratic state penalty
`r_tilt = -clip((θ/0.15)^2, 0, 25)` with `dexscrew_tilt_scale=1.0` on
tip-connect is unchanged. Recovery remains a separate default-off term.

Default:
`g=0`, so historical DexScrew totals are unchanged. Stage-path rewards ignore
this term.

Experiment scale (EXP-20260823-019 only):
`g=50`. A 0.05 rad increase at or above 0.25 rad yields −2.5, equal to
`2.5 * ω` at `ω=1 rad/s`. A 0.01 rad creep at the same tilt yields −0.50,
so rotation can still dominate slow drift and the policy is not forced to
freeze. Below 0.05 rad, `w=0`.

Predeclared per-step values at growth scale 50 and tilt-penalty scale 1.0
(rows are current tilt after the increase):

| Current tilt (rad) | w(θ) | State penalty | Growth Δθ=+0.01 | Growth Δθ=+0.05 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.15 | 0.50 | -1.00 | -0.25 | -1.25 |
| 0.25 | 1.00 | -2.78 | -0.50 | -2.50 |
| 0.50 | 1.00 | -11.11 | -0.50 | -2.50 |
| 0.75 | 1.00 | -25.00 (saturated) | -0.50 | -2.50 |

Aggregation:
Per-step mean `reward_axis_tilt_growth` and its share of Σ|reward components|.
Also report mean final tilt, mean/max of per-episode maximum tilt, and whether
max tilt equals final tilt (monotonic collapse).

Implementation:
`allegro_rod_mvp/rewards_dexscrew.py`, `RodRotationEnv.step`,
`scripts/train_parallel.py`, `scripts/eval_policy.py`.

## Observation Frame Stack
Definition:
`obs_stack_t = [obs_t, obs_{t-1}, ..., obs_{t-H+1}]` with newest frame first.
On reset, missing frames repeat the initial observation. `H=1` is the
historical 48-D T00 observation. `H=4` is 192-D (~160 ms at 25 Hz).
Unit:
Dimensionless concatenated observation.
Aggregation:
Not averaged; the policy input dimension is `48 * H` for the current Allegro
layout (`privileged_obs=False`).
Implementation:
`allegro_rod_mvp/obs_history.py`, `RodRotationEnv`, flag `obs_history_len`.

## Support-Gated Rotation Reward
Definition:
Existing axial rotation reward multiplied by
`scale_0` if `n_contact==0`, `scale_1` if `n_contact==1`, `scale_2plus` if
`n_contact>=2`. Defaults `{0.0, 0.1, 1.0}`. Disabled unless
`support_aware_reward_enabled`.
Unit:
Same as the DexScrew rotation term (`rotate_scale * clip(ω_axial, -4, 4)`).
Logged names:
`reward_rotation_before_support_gate`, `reward_rotation` (after gate),
`reward_support_gating_effect` (after − before).
Implementation:
`allegro_rod_mvp/support_aware_reward.py`, `RodRotationEnv.step`.

## Low-Support Lateral Wobble Penalty
Definition:
`ω_perp = ω - (ω · axis_hat) axis_hat`. When `n_contact < 2`,
`r = -λ ||ω_perp||^2`. When `n_contact >= 2` or the support-aware flag is off,
the term is 0. Default `λ=0.5`.
Unit:
Reward units; `||ω_perp||` in rad/s. This is the same quantity already logged
as `lateral_omega` / `omega_perp_norm`.
Logged name:
`reward_low_support_wobble`.
Implementation:
`allegro_rod_mvp/support_aware_reward.py`. Contact force remains
`mj_contactForce(...)[0]` vs rod, threshold 0.05 N.

## Support-Loss and Re-contact
Definition:
Support loss: `n_contact(t-1) >= 2` and `n_contact(t) <= 1`.
Successful re-contact: return to `n_contact >= 2` within
`recontact_window_steps` (default 10 = 400 ms at 25 Hz).
`recontact_success_rate = recovered_events / support_loss_events`.
Also log mean/max 1-contact run length, time in 0/1/≥2 contact, and
fraction of episodes with at least one support-loss event.
Implementation:
`allegro_rod_mvp/support_collapse_metrics.py`, `scripts/eval_policy.py`
key `support_collapse`.

## Post-Support-Loss Lateral Instability
Definition:
At each support-loss time `t`, record `||ω_perp||_t`,
`max_{k=0..5} ||ω_perp||_{t+k}`, and
`Δ = max(ω_perp[t:t+5]) - ω_perp[t-1]`.
Unit:
rad/s.
Implementation:
`episode_support_collapse_metrics`.

## Tilt Excursion and Recovery (diagnostic, not termination)
Definition:
An excursion starts at an upward crossing of 0.25 rad and recovers if tilt
later falls below 0.20 rad before physical termination. 0.25 rad is **not**
a kill; `tilt_terminate_rad` remains 1.2 on T00.
Logged:
max/final tilt, time above 0.25 rad, upward crossings, excursion count,
recovery rate, mean time-to-recover, mean peak tilt of recovered excursions.

## Axis Tilt Angle
Definition:
Angle between the rod long axis and the world vertical target axis,
`arccos(|â · ẑ|)`.
Unit:
Radians in code; degrees also shown on analysis pages.
Implementation:
`RodRotationEnv` info `axis_tilt_rad` / `axis_tilt_deg`.

## Tilt Velocity dθ/dt (finite difference)
Definition:
`(tilt[t] - tilt[t-1]) / dt` with `dt = 1 / policy_hz` (0.04 s at 25 Hz).
Step 0 is defined as 0. This is the rate of change of the **tilt angle
scalar**, not a component of body ω.
Unit:
rad/s and deg/s.
Does not apply to:
`ω_perp` / `lateral_omega` (those are `||ω − (ω·â)â||`).
Implementation:
`scripts/export_t00_ablation_timelines.py` `_annotate_velocities`.
Logged names: `tilt_vel_rad_s`, `tilt_vel_deg_s`.

## Axial Rotation Velocity ω_axial
Definition:
`ω_axial = −ω · â`, same sign convention as unwrapped axial progress.
Unit:
rad/s; analysis pages also show deg/s.
Implementation:
`RodRotationEnv._axial_omega`; info `axial_omega`.

## Lateral Angular Rate ω_perp
Definition:
`||ω − (ω·â)â||`. This is the magnitude of angular velocity orthogonal
to the rod axis. It is **not** d(tilt)/dt.
Unit:
rad/s.
Implementation:
env info `omega_perp_norm` / `lateral_omega`.

## Axis-Tilt Death Preceded by Support Loss
Definition:
For every `termination_reason=axis_tilt` episode, a support-loss event in the
last `collapse_lookback_steps` (default 10) counts as preceded-by-collapse.
Also mean delay from that event to termination and to the post-loss ω_perp
spike.
Logged name:
`fraction_axis_tilt_deaths_preceded_by_support_loss`.
This is the primary T00 mechanism metric for EXP-20260914-001.
Diagnostic traces: `runs/<run_id>/traces/{fixed,unseen}/trace_seed<seed>.{csv,json,png}`.
