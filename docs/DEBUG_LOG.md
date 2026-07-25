# Debug Log

## DBG-20260724-001: Multi-finger contact target may be unreachable or mismeasured
- Date: 2026-07-24
- Status: investigating
- Related runs: `20260724-1718-stage2-discrete-contact-seed0`, `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
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

### Root Cause
Unknown.

### Resolution
None. EXP-20260724-004 is planned as a bounded mechanical/contact-sensor reachability test.

### Verification
Pending controlled configurations and visual evidence.

### Prevention
Do not define future success rewards around simultaneous three-finger contact until reachability and detection are verified.

### Lessons Learned
A large reward cannot teach a target state that may be unreachable or absent from the policy's explored state distribution.

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
