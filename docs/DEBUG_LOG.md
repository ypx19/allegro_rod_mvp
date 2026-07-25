# Debug Log

## DBG-20260724-002: Corrected finger contact remains outside reset exploration basin
- Date: 2026-07-24
- Status: investigating
- Related runs: `20260724-2045-finger2-spatial-dof`, `20260724-2100-spatial-finger2-retrain-seed0`
- Related files: `allegro_rod_mvp/env.py`, `models/three_finger_rod.xml`
- Severity: high
- First observed: EXP-20260724-006

### Symptom
Although three-contact grasps are now geometrically and dynamically reachable, corrected-geometry retraining never produces a two- or three-contact evaluation step. Only the middle fingertip contacts.

### Expected Behavior
The discrete reward should become observable after spatial finger 2 learns to approach the rod, leading to some nonzero multi-contact occupancy.

### Evidence
- EXP-005 found 87 geometric and three dynamically settled three-contact candidates.
- EXP-006: every checkpoint has 0% two-/three-contact occupancy.
- All checkpoints retain 20/20 axis-tilt terminations.
- Mean contact reward remains approximately -8 per step.
- Force penalty remains zero, confirming the policy does not touch with additional fingers.

### Hypotheses
1. The legacy reset grasp is outside the three-contact exploration basin.
2. The discontinuous reward gives no directional signal for approaching with finger 2.
3. The old parent policy strongly preserves its pre-geometry action pattern.

### Root Cause
Not yet confirmed. The immediate measured mechanism is failure to visit multi-contact states.

### Resolution
None. EXP-20260724-007 will test reset initialization as a single factor before changing reward smoothness or parent policy.

### Prevention
For newly reachable task states, validate that reset/exploration distributions actually visit the state before relying on sparse bonuses.

### Lessons Learned
Making a state reachable does not make a discontinuous reward learnable when the policy never enters that state.

---

## DBG-20260724-001: Finger 2 motion plane cannot intersect the rod
- Date: 2026-07-24
- Status: resolved
- Related runs: `20260724-1718-stage2-discrete-contact-seed0`, `20260724-1730-contact-reachability`, `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Related files: `allegro_rod_mvp/env.py`, `models/three_finger_rod.xml`, `scripts/eval_policy.py`
- Severity: high
- First observed: per-finger diagnostic before EXP-20260724-003

### Symptom
Across the failed Stage 2 baseline, every discrete-contact checkpoint, and a stable stabilizer-0.10 control, evaluation recorded only fingertip 2 contact. No step recorded two or three contacting fingertips.

### Expected Behavior
The three-finger manipulation task and its proposed reward require each fingertip—and ideally all three simultaneously—to be able to contact the rod.

### Reproduction
```bash
python scripts/eval_policy.py runs/20260723-1045-capacity512-stab012-denseckpt-seed0/checkpoints/ppo_rod_84200_steps.zip --stage 2 --tip-connect --tip-connect-solref 0.10 --axis-stabilizer-scale 0.10 --axis-tilt-penalty-weight 0.10 --rotation-reward-scale 160
```

### Evidence
- Stabilizer-0 baseline: 80.47% zero contact, 19.53% fingertip 2 only.
- Stabilizer-0.10 stable control: 72.82% zero contact, 27.18% fingertip 2 only, 176.55° rotation, zero drops.
- EXP-20260724-003: every checkpoint had 0% two-/three-contact steps.
- Plot: `runs/20260724-1718-stage2-discrete-contact-seed0/plots/contact_reward_evaluation.png`.

### Hypotheses
1. The current grasp/kinematics cannot bring all three tip geoms to the rod simultaneously.
2. Collision filtering or contact geometry prevents fingers 1 and 3 from contacting.
3. The force-based detector or geometry IDs miss valid contacts.
4. The policy never explores the coordinated configuration despite it being reachable.

### Investigation
- Added per-finger contact logging and full contact-count step distributions.
- Re-evaluated unstable Stage 2 and stable stabilizer-0.10 policies.
- Confirmed the steep reward alone does not produce multi-finger contact.
- Ran 60,000 bounded configurations across three seeds using exact signed MuJoCo geometry distance.
- Found 4,166 two-contact configurations and zero three-contact configurations.
- Verified fingers 0 and 1 start with simultaneous 32–54 N contact, so force detection works.
- Finger 2 remained at least 66.66–68.27 mm from the rod surface.

### Root Cause
Confirmed geometry error. Finger 2 is based at world `y=+0.04 m` and its `euler="1.5708 0 0"` orientation makes the planar chain move in XZ at fixed Y. The rod is centered near `y=-0.05 m`. The 90 mm plane separation exceeds the 24 mm combined collision radii, so finger 2 cannot contact the rod at any joint angles.

### Resolution
Changed `f2_j0` from local Z to local X, creating a nonparallel spatial axis before the two distal Z flexion axes. No other geometry, actuator, solver, observation, or reward changed.

### Verification
- Search results: `runs/20260724-1730-contact-reachability/reachability.json`
- Metrics: `runs/20260724-1730-contact-reachability/metrics.csv`
- Visual evidence: `runs/20260724-1730-contact-reachability/images/contact_reachability_comparison.png`
- Repeated over reset seeds 0, 1, and 2.
- Corrected run: `runs/20260724-2045-finger2-spatial-dof/`.
- Finger 2 reached -14.96 to -15.21 mm signed distance.
- Dynamic replay found settled three-contact force on all three seeds.

### Prevention
Do not define future success rewards around simultaneous three-finger contact until reachability and detection are verified.

### Lessons Learned
A reward cannot teach a target state excluded by the model's kinematics. Validate geometric reachability before reward design.

---

## DBG-20260723-005: Stage 2 cannot recover axis tilt with absolute penalty scaling
- Date: 2026-07-23
- Status: investigating
- Related runs: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`, `20260723-1230-stage2-tipjoint-tiltw025-seed0`, `20260723-1500-stage2-tilt-recovery40-seed0`
- Related files: `allegro_rod_mvp/env.py`, `scripts/train.py`, `scripts/eval_policy.py`
- Severity: high
- First observed: corrected Stage 2 baseline

### Symptom
With the endpoint ball/universal joint active and external axis stabilizer off, the rod tilts past 0.7 rad and terminates in every evaluated episode.

### Expected Behavior
The policy should use fingertip actions to keep the rod axis near vertical while accumulating positive axial rotation.

### Reproduction
```bash
python scripts/eval_policy.py runs/20260723-1230-stage2-tipjoint-tiltw025-seed0/checkpoints/ppo_rod_60600_steps.zip --stage 2 --tip-connect --tip-connect-solref 0.10 --axis-stabilizer-scale 0 --axis-tilt-penalty-weight 0.25 --rotation-reward-scale 160
```

### Evidence
- Baseline weight 0.10: 20/20 axis-tilt terminations.
- Weight 0.25 after 25k training: 20/20 axis-tilt terminations at every checkpoint.
- Weighted tilt term changed from -2.03 to -5.01 per step.
- Late explained variance remained negative; value loss remained approximately 3,000–4,000.
- Videos: `runs/20260723-1230-stage2-tipjoint-tiltw025-seed0/videos/`.

### Hypotheses
1. Absolute tilt penalty does not provide a sufficiently local recovery direction.
2. Episodes become too short for PPO to discover recovery after assist removal.
3. The policy needs an explicit tilt-rate/recovery observation or reward.
4. (2026-07-24, unverified) The policy rotates the rod using effectively one finger; single-finger contact pushes the rod off-axis and drives the tilt. A reward requiring all three fingertips to maintain positive contact simultaneously could force coordinated manipulation and reduce tilt. Verify the single-finger premise via per-fingertip contact logging on existing checkpoints before training. See EXP-20260724-001.
5. (2026-07-24, unverified) Stage 1 axis-stabilizer pretraining may induce dependence on external orientation torque that does not transfer to stabilizer-free Stage 2; it is unproven whether Stage 0/1 pretraining benefits Stage 2 at all. See EXP-20260724-002.

### Investigation
- Increased only the tilt weight 0.10→0.25.
- Verified the weighted component numerically.
- Evaluated every 5k checkpoint under the same fixed seeds.
- Added a clipped local recovery term, `40 * (previous_tilt - current_tilt)`, while restoring the absolute tilt weight to 0.10.
- Verified four deterministic reward-helper tests, an environment check, and a 2048-step Stage 2 training smoke test.
- The recovery run improved mean rotation to 4.22° but still produced 20/20 axis-tilt terminations at every checkpoint.
- The unmodified stabilizer-0.12 checkpoint transferred to Stage 2 randomization at 178.51° with drop 0.05, ruling out randomization as the main cause.
- A fixed-policy sweep localized the assist cliff between stabilizer 0.10 (176.55°, drop 0) and 0.08 (132.42°, drop 0.40).

### Root Cause
The immediate mechanism is a nonlinear assist-removal cliff between stabilizer 0.10 and 0.08. The deeper cause of the policy's inability to generate the required lateral correction without assist remains unknown.

### Resolution
None yet; weight 0.25 and recovery scale 40 were rejected. The next control separates Stage 2 randomization adaptation from stabilizer removal.

### Verification
Each negative result was repeated across six checkpoints and 20 deterministic episodes per checkpoint.

### Prevention
Continue logging raw and weighted reward components and termination reasons. Do not infer recovery from total return alone.

### Lessons Learned
Making an absolute state penalty larger can worsen value scaling without teaching the action sequence that reverses the state error.

---

## DBG-20260723-004: Resume-time learning-rate override was ineffective
- Date: 2026-07-23
- Status: resolved
- Related runs: `20260723-0100-stage1-tip-solref010-seed0`
- Related files: `scripts/train.py`
- Severity: high
- First observed: Planned `learning_rate=1e-5` run logged `learning_rate=0.0003`.

### Symptom
`PPO.load()` followed by assigning `model.learning_rate` did not change the optimizer or SB3 learning-rate schedule.

### Expected Behavior
The resumed run must log and use the CLI-provided learning rate.

### Root Cause
Stable-Baselines3 stores a learning-rate schedule and optimizer parameter-group values separately from the public attribute.

### Resolution
Update `model.learning_rate`, `model.lr_schedule`, and every optimizer parameter group when resuming.

### Verification
A resumed 1024-step smoke test must log `learning_rate = 1e-05` before the formal rerun.

### Prevention
Treat the SB3 training log as the resolved configuration check for every resume-time override.

### Lessons Learned
Assigning a restored model's public hyperparameter attribute is not sufficient when the framework caches schedules or optimizer state.

---

## DBG-20260723-001: MuJoCo free-joint NaN (DOF 12)
- Date: 2026-07-23
- Status: mitigated
- Related runs: pre-hanging curriculum; stiff `solref=0.004` + `kp=40`
- Related files: `models/three_finger_rod.xml`, `allegro_rod_mvp/env.py`
- Severity: high
- First observed: Stage 1 eval / train logs (`QACC`/`QVEL` at DOF 12)

### Symptom
Repeated MuJoCo warnings: `Nan, Inf or huge value in QACC/QVEL at DOF 12` (rod free joint). Training/eval continued with corrupted dynamics.

### Expected Behavior
Finite qvel/qacc; unstable episodes should terminate cleanly without poisoning PPO batches.

### Reproduction
```bash
python scripts/eval_policy.py checkpoints/stage1/final_model.zip --stage 1
# or aggressive random/bang-bang actions with solref=0.004, kp=40
```

### Evidence
- Logs: `checkpoints/stage1/train.log`, `checkpoints/curriculum_driver.log`
- Frequency: frequent under stiff contacts + high actuator gains
- Device: Mac arm64, mujoco 3.10, CPU PPO

### Hypotheses
1. Contact/`connect` solref too stiff vs timestep.
2. High position-actuator kp injects large contact impulses.
3. Missing free-joint damping.

### Investigation
- Mapped DOF 12 to `rod_free`.
- Softened geom/`connect` solref `0.004→0.008`, kp `40→28`, added free-joint damping, raised solver iterations.
- Added finite/huge qvel-qacc guards → terminate with `-15` and `unstable=True`.

### Root Cause
Over-stiff contacts and tip equality combined with strong actuators excited the free-joint solver.

### Resolution
XML softening + damping + episode-level instability termination.

### Verification
`check_env` passes; stress rollouts terminate as `unstable` instead of continuing with NaNs. Residual rare warnings may still print from MuJoCo before the guard fires.

### Prevention
Keep solref ≥ ~0.008 for this mesh-free scene; retain instability guard in `env.step`.

### Lessons Learned
Stiffer tip anchoring is not free: equality solref and contact solref must be co-tuned with actuator gains.

---

## DBG-20260723-002: Stage 0 axis collapse / 100% drop (bottom tip anchor)
- Date: 2026-07-23
- Status: resolved
- Related runs: curriculum after axis-tilt reward + `solref=0.004/0.008` with tip at bottom
- Related files: `models/three_finger_rod.xml`, `allegro_rod_mvp/env.py`
- Severity: critical
- First observed: Stage 0 eval `drop_rate=1.0`, `axis_rotation_deg_mean≈0`

### Symptom
After adding axis-tilt penalty, Stage 0 episodes almost always terminated with large `axis_tilt`; mean rotation ~0°; tip error remained tiny.

### Expected Behavior
Stage 0 tip-anchored rod should allow axial spin with bounded tilt.

### Reproduction
```bash
python scripts/eval_policy.py checkpoints/stage0/final_model.zip --stage 0
# Instrumented: 18/20 terminations were axis_tilt > 0.7 rad
```

### Evidence
- Eval JSON: rotation ≈ -1° to -3°, drop_rate 0.9–1.0
- Zero-action rollouts accumulated tilt under gravity with tip at bottom

### Hypotheses
1. Tip `<connect>` at bottom creates an inverted pendulum.
2. Axis reward/termination too strict relative to unstable physics.
3. Policy actively tilts; gravity alone insufficient.

### Investigation
- Confirmed tip site at world z≈-0.07 (bottom) with COM above tip.
- Pure hang test with tip moved to top restored from 30° tilt to ~1.4°.
- Bottom tip: gravity drives axis fall → tilt terminations dominate.

### Root Cause
Bottom tip anchor is mechanically unstable (inverted pendulum). Axis objectives fought physics.

### Resolution
1. Move tip/`connect` anchor to top (hanging pendulum).
2. Stage-0-only vertical axis stabilizer torque (spring+damper on lateral orientation).
3. Fix axial sign so natural rolling accumulates as +rotation.
4. Slightly relax success tilt threshold to 0.25 rad.

### Verification
Open-loop hanging+stabilizer: ~+293°, `is_success=True`. Trained Stage 0: success_rate 0.95, mean rotation ~528°.

### Prevention
Document tip must hang from top when using point `connect` under gravity. Prefer Stage-0 stabilizer or true axial hinge if axis must stay fixed.

### Lessons Learned
“Tip fixed” ≠ “axis fixed”. A spherical point constraint leaves tilt free; gravity chooses the stable hanging pose only if the tip is above the COM.

---

## DBG-20260723-003: Stage 1/2 transfer failure (negative rotation)
- Date: 2026-07-23
- Status: investigating
- Related runs: `20260723-0010-hanging-tip-stabilizer-curriculum`, `20260723-0026-stage1-softstab-seed0` (rejected), `20260723-0034-stage1-soft-tip-seed0` (running)
...

### Investigation
- Curriculum Stage 1 full budget failed (negative rotation, high drop).
- EXP-002: Stage1 xfrc stabilizer 0.5 without tip connect → immediate instability (ep_len≈5, DOF12 NaN) → rejected.
- EXP-003: Stage1 soft tip connect (solref 0.05), stabilizer 0 → running.

### Root Cause
Unknown (transfer gap confirmed; xfrc-without-tip ruled out as assist).
## DBG-20260724-002: Hard contact gate blocks Stage 0 exploration
- Date: 2026-07-24
- Status: investigating
- Related runs: `20260724-2200-stage0-contact30-seed0`, `20260724-2230-stage0-contact-gate-seed0`
- Related files: `allegro_rod_mvp/env.py`
- Severity: high
- First observed: all gated Stage 0 checkpoints had 0% three-contact occupancy

### Symptom
Fresh PPO trajectories usually terminate at the first failed 20-step contact window. Finger2 has 0% contact occupancy, and every evaluated checkpoint has 0% success.

### Expected Behavior
The policy should discover a three-contact step and accumulate at least +5 contact reward per full 20-step window.

### Reproduction
```bash
python scripts/eval_policy.py runs/20260724-2230-stage0-contact-gate-seed0/checkpoints/ppo_rod_20000_steps.zip --stage 0 --episodes 20 --seed 0 --axis-stabilizer-scale 1.0 --contact-reward-mode discrete --three-contact-reward 30 --contact-window-steps 20 --contact-window-threshold 5
```

### Evidence
- 20k checkpoint: 149.93° rotation, 0.63 mm tip error, success 0/20.
- Terminations: 17/20 `contact_support`, 3/20 `axis_tilt`.
- Finger2 and three-contact occupancy: both 0%.
- Representative video: `runs/20260724-2230-stage0-contact-gate-seed0/videos/stage0_best_00_seed17_rot167deg.mp4`.

### Hypotheses
1. The current reset distribution places finger2 outside a readily discoverable contact basin.
2. A 20-step deadline is too short for exploration from that reset.
3. A settled three-contact reset plus grace period will expose the policy to the desired state.

### Investigation
- Verified deterministically that an unsupported rollout terminates at step 20 with `termination_reason=contact_support`.
- Evaluated every 5k checkpoint on fixed seeds 0–19.
- Confirmed zero three-contact and finger2 occupancy at every checkpoint.

### Root Cause
Not yet confirmed. Current evidence indicates exploration failure/curriculum mismatch rather than incorrect gate logic.

### Resolution
None yet. Preserve the failed hard-gate run and test contact-friendly initialization next.

### Verification
The gate implementation has unit tests and a deterministic 20-step smoke test. A behavioral resolution remains unverified.

### Prevention
Before hard-gating a sparse behavior, verify that the reset distribution or a staged curriculum exposes the policy to successful examples.

### Lessons Learned
A logically correct threshold can destroy the exploration horizon when its passing state is absent from the reset distribution.

---
