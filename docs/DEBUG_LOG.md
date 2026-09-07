# Debug Log

## DBG-20260802-001: PPO action std explosion → NaN on long Stage 0 parallel run
- Date: 2026-08-02
- Status: open
- Related runs: `20260802-0220-exp-infra-subproc64-1e9-seed0` (EXP-20260802-002)
- Related files: `scripts/train_parallel.py`, SB3 `PPO` / `ActorCriticPolicy` (`log_std`)
- Severity: high
- First observed: EXP-20260802-002 crash at ~3.17e7 env-steps

### Symptom
Long 1e9-budget Stage 0 job crashed in `PPO.train()` with `ValueError`: Gaussian action `loc` is all-NaN. Online logs show policy `std` growing without bound (≈1 → 32 at 5e6 → 6e4 at 1e7 → 1e18 at crash). After ~5e6 steps, `success_rate` fell from ~0.93 to 0 and return collapsed.

### Expected Behavior
Extended training should keep finite parameters; success either improve or plateau without NaNs. Checkpoints past peak quality should remain loadable for eval.

### Reproduction
```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/train_parallel.py \
  --stage 0 --num-envs 64 --device cuda --net-arch 512,256,128 \
  --n-steps 128 --batch-size 512 --steps 1000000000 --seed 0 \
  --checkpoint-freq 5000000 \
  --run-id 20260802-0220-exp-infra-subproc64-1e9-seed0
```
(Reproduced once on this seed/config; expect failure after O(1e7) steps when `std` ≫ 1e3.)

### Evidence
- Console: `runs/20260802-0220-exp-infra-subproc64-1e9-seed0/logs/console.log`
- Metrics: `metrics.csv` through step 31719424
- Checkpoints: 5e6 (pre-collapse) through 3e7 (post-collapse); no `final_model` / `vecnormalize.pkl`

### Hypotheses
1. Default `ent_coef=0.01` encourages unbounded `log_std` growth on this task/horizon.
2. VecNormalize reward scaling + large parallel batch amplifies unstable policy updates (`approx_kl` ≫ clip).
3. Missing `log_std` clip / target-KL early stopping allows irreversible divergence after a good policy is found.

### Investigation
- Check performed: milestone scrape of console `std` / `success_rate` / `ep_rew_mean`.
- Result: clear success peak ~5e6 then monotonic std blow-up.
- Interpretation: failure mode is optimization instability, not “Stage 0 unlearnable.”

### Root Cause
Unknown (not yet ablated). Leading mechanism: unconstrained Gaussian `std` growth under continued PPO updates.

### Resolution
None yet. Mitigations to try: `ent_coef=0`, clip `log_std`, reduce LR, stop/select at best checkpoint, save VecNormalize every checkpoint.

### Verification
Pending mitigation run.

### Prevention
Save VecNormalize alongside every CheckpointCallback dump; log/alert when `std` exceeds a threshold; prefer `ent_coef=0` for long DexScrew-style runs (plan already suggested migrating toward it).

### Lessons Learned
Longer budgets can find Stage 0 success and then destroy it. Always keep dense mid-run checkpoints and stabilize entropy/`log_std` before multi-day jobs.

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

## DBG-20260823-001: Three-contact bonus creates a static-grasp optimum
- Date: 2026-08-23
- Status: investigating
- Related runs: `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Related files: `allegro_rod_mvp/env.py`, `scripts/run_allegro_tip_bottom_curriculum.py`
- Severity: high
- First observed: final nominal-mass curriculum smoke evaluation

### Symptom
The final policy maintains three fingertip contacts for every evaluation step and survives the full episode, but mean rotation is only 40.11° and sustained-ω success is 0%.

### Expected Behavior
After preserving support, the policy should sustain axial angular velocity above 0.5 rad/s for 10 seconds.

### Reproduction
```bash
.venv/bin/python scripts/run_allegro_tip_bottom_curriculum.py \
  --curriculum-id 20260823-0405-allegro-tip-bottom-smoke-seed0 \
  --start-scale 10 --smoke --num-envs 4 --device cuda --seed 0
```

### Evidence
- Final three-contact occupancy: 1.000.
- Final drop rate: 0.00.
- Final tip error: 1.40 mm.
- Final mean rotation reward: +0.085/step.
- Final mean contact reward: +3.000/step.
- Final rotation: 40.11°; success: 0/5.

### Hypotheses
1. The +3 contact bonus dominates the angular-velocity reward and makes a static grasp locally optimal.
2. The hard support and no-rotation-credit gates are sufficient to preserve coordination without such a large positive contact bonus.
3. The 20k-per-stage smoke budget is too short to learn rotation, but extending it with the same reward ratio would reinforce the static optimum.

### Investigation
- Verified contact reachability and reset stability independently before training.
- Executed every constraint/stabilizer/mass transition.
- Compared individual reward components rather than total return.
- Observed the same low-ω behavior despite excellent contact, endpoint, and survival metrics.

### Root Cause
The immediate reward imbalance is confirmed: contact contributes approximately 35 times more positive reward than rotation in the final evaluation. Whether reducing it alone is sufficient to learn sustained rotation is not yet confirmed.

### Resolution
Planned controlled change: reduce only `three_contact_reward` from 3.0 to 0.3. Keep the 25-step/18-step support gate and `three_contact_required` rotation gate unchanged.

### Verification
Pending a fresh A0 revolute run with dense checkpoint evaluation.

### Prevention
Require reward-component share checks before promoting future shaping changes. A support bonus must not exceed the task-progress signal once support is already reliably initialized.

### Lessons Learned
A contact-friendly reset can solve exploration while an oversized continuing contact bonus still prevents task progress.

---

## DBG-20260823-002: Web pose fields and sliders did not target their rows
- Date: 2026-08-23
- Status: resolved
- Related runs: `20260823-hand-pose-web-validation`
- Related files: `scripts/edit_hand_pose_web.py`, `tests/test_hand_pose_web.py`
- Severity: high
- First observed: live revolute editor at `127.0.0.1:33835`

### Symptom
Typing Y or Z did not change `/api/state`, and the translation/rotation sliders appeared nonfunctional. Decimal and negative text editing was unreliable.

### Expected Behavior
Each X/Y/Z and roll/pitch/yaw field and slider must independently update the palm pose, diagnostics, and MuJoCo render. Text fields must allow normal intermediate editing and commit on Enter, change, or blur.

### Reproduction
```bash
MUJOCO_GL=egl .venv/bin/python scripts/edit_hand_pose_web.py \
  --physics revolute --output configs/hand_poses/my_grasp.json \
  --host 127.0.0.1 --port 0
```

### Evidence
- Headless Chromium found only 2 unique range IDs and 2 unique numeric IDs across 6 pose rows.
- Generated X/Y/Z slider bounds were respectively `0..-500`, `1..-500`, and `2..-500`; all had `min > max`.
- Typing `-12.5` into the visible Y field left `/api/state.translation_mm` at `[0, 0, 0]`.
- No JavaScript console exception occurred; this was incorrect DOM construction and event targeting.

### Hypotheses
1. Pose definition tuples were decoded with the wrong field positions.
2. Repeated IDs made every axis listener target the first element for each vector.
3. Keystroke-time numeric conversion rejected normal intermediate text states.
4. Shared debounce state and unordered async responses could lose or visually revert newer edits.

### Investigation
- Compared `poseDefs` (six fields including an index) with `row()` (five-field destructuring).
- Inspected the rendered DOM and confirmed duplicate IDs, malformed labels, and reversed range bounds.
- Reproduced the unchanged backend state in Chromium.
- Exercised all pose and camera controls after the fix and compared PNG SHA-256 values before and after each mutation.
- Artificially delayed an older pose response until after a newer one to verify stale-response suppression.

### Root Cause
`row()` interpreted each six-element pose definition as a five-element camera definition. The axis index became the range minimum, `-500/-360` became the maximum, and the real maximum became the displayed unit. IDs used only the vector key, so X/Y/Z shared one ID and roll/pitch/yaw shared another. Separately, number fields posted on every `input` event, converting blank/sign/partial-decimal states before they were valid commits.

### Resolution
- Use explicit control-definition objects and axis-qualified unique IDs.
- Keep numeric text as editable decimal text; validate and commit only on Enter, change, or blur.
- Keep paired slider/text values synchronized through numeric drafts.
- Use independent pose/camera debounce timers and a mutation sequence that ignores stale responses.
- Use unrestricted range values plus explicit fine/coarse keyboard increments so decimal text remains representable.
- Surface network, HTTP, malformed-response, validation, and render failures in the status panel.
- Cancel pending mutations before reset, load, and settle actions.
- Validate and flush the complete pose draft before save/overwrite so a pending slider debounce cannot save the previous pose.

### Verification
- `py_compile`: passed for editor and tests.
- Focused backend/API/DOM/Chromium suite: 6/6 passed in 7 seconds.
- Real Chromium covered positive, negative, decimal, zero, blank, sign, and partial-decimal edits; Enter/change/blur; all 6 pose sliders; all 6 camera sliders; fine/coarse; delayed stale response; save/load/reset; and visible API errors.
- Every pose slider changed `/api/state` and the rendered PNG hash. Every camera slider changed camera state and the rendered PNG hash.
- Revolute and bottom point-connect rendering passed in the focused suite.
- Browser save/load/overwrite-safe behavior used only an isolated temporary pose directory; `configs/hand_poses/my_grasp.json` was not written.

### Prevention
Retain the DOM contract test for unique IDs and valid bounds plus the opt-in Playwright interaction test (`HAND_POSE_BROWSER_TESTS=1`). Backend-only endpoint tests are insufficient for generated controls.

### Lessons Learned
For generated Web controls, validate the rendered DOM and actual browser events. A healthy API cannot detect duplicate IDs, invalid HTML range constraints, or hostile keystroke-time validation.

---

## DBG-20260823-003: Saved pose does not enter three-tip grasp basin
- Date: 2026-08-23
- Status: mitigated
- Related runs: `20260823-1730-two-phase-force-pose-smoke-seed0`, `20260823-1740-pose-revolute-s1-smoke-seed0`, `20260823-1740-pose-tip-s400-smoke-seed0`, `20260823-1740-pose-tip-s1-smoke-seed0`
- Related files: `configs/hand_poses/my_grasp.json`, `allegro_rod_mvp/env.py`
- Severity: critical
- First observed: mandatory pre-training endpoint smoke with the newly saved pose

### Symptom
The pose file is schema-valid and applies to both model variants, but the settled
policy state has no index-fingertip contact. The strict `s=400` revolute smoke
terminates every fixed and unseen evaluation episode on the three-contact gate.
Bottom tip-connect at `s=400` also terminates on rod height after one policy step.

### Expected Behavior
The saved pose should place the existing reset/grasp joint trajectory inside a
stable three-tip contact basin in both physics modes before PPO training.

### Reproduction
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_two_phase_force_curriculum.py \
  --curriculum-id 20260823-1730-two-phase-force-pose-smoke-seed0 \
  --hand-pose-config configs/hand_poses/my_grasp.json --smoke --device cuda:0
```

### Evidence
- Pose SHA-256: `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`.
- Fixed seeds 10000–10001 at revolute `s=400`: success 0, three-tip occupancy 0,
  drop/violation rate 1.0, index force p95 0 N, total normal-force p95 49.78 N.
- Unseen seeds 20000–20001: success 0, three-tip occupancy 0, drop rate 1.0,
  index force p95 0 N, total p95 48.75 N.
- The apparent 224–229 degree hinge displacement is not task success: it occurs
  during unsupported dynamics, receives zero rotation reward because the support
  gate is active, and ends at 25 steps.
- Representative failure:
  `runs/20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0/videos/revolute_best_00_seed10000_rot222deg_tilt0deg_steps25_contact_support.mp4`
  (26/26 frames decoded, 640x480, 25 FPS).

### Hypotheses
1. The saved rigid palm transform is incompatible with the unchanged Allegro reset/grasp joint vectors.
2. A pose-specific reset joint configuration is required.
3. The world-axis rotation/translation moved the index pad outside the rod's contact plane.

### Investigation
- Validated JSON schema, normalized quaternion, compatibility declarations, and
  identical file hash in both physics variants.
- Confirmed mass and all three inertia components scale exactly by `s`.
- Confirmed explicit friction changes only sliding friction on the rod and all
  three pads; torsional/rolling friction and other dynamics remain fixed.
- Ran short PPO and checkpoint reload smokes at both mass endpoints and both
  physics modes. Plumbing passed, but behavior gates failed.

### Root Cause
The saved palm transform was combined with Allegro reset/grasp vectors fitted to
the old palm pose. A bounded per-finger signed-distance search found a valid
pose-specific 12-DoF vector without changing the palm.

### Resolution
Added a separately validated companion grasp config and optional environment,
training, evaluation, curriculum, and video-export plumbing. The user's root pose
was not edited. The mitigation is complete for revolute endpoint masses, but the
same vector is not robust in heavy bottom tip-connect.

### Verification
- Full headless regression suite: 27 passed, 1 optional browser test skipped.
- Every short PPO run saved and reloaded its model and VecNormalize state.
- The failed strict curriculum stopped at Phase R, `s=400`, before any lower-mass
  stage or tip-connect curriculum stage.
- `20260823-1740-my-grasp-shared-reset-audit-seed0`: revolute `s=400` and `s=1`
  each pass 10/10 seeds for 100/100 three-contact steps, with zero non-tip rod
  collisions. Tip-connect `s=1` also passes 10/10; tip-connect `s=400` passes
  0/10 and has minimum occupancy 0.765.
- Regression suite: 6/6 focused contact/reset tests pass.

### Prevention
The new strict runner gates every stage on fixed and unseen deterministic task
metrics and never enters Phase T unless revolute `s=1` passes. Pose-specific
contact audit must precede a performance-budget run.

### Lessons Learned
A schema-valid rigid palm pose can preserve kinematics while invalidating the
joint-space grasp reset. High unwrapped hinge motion without support credit must
not be interpreted as learned rotation.

---

## DBG-20260823-004: Supported s400 policy fails sustained rotation gate
- Date: 2026-08-23
- Status: investigating
- Related runs: `20260823-1810-two-phase-grasp-recovery-seed0`, `20260823-1820-two-phase-grasp-recovery-cont-seed0`, `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Related files: `configs/hand_grasps/my_grasp_revolute_shared.json`, `scripts/run_two_phase_force_curriculum.py`
- Severity: high
- First observed: strict Phase R evaluation after the reset regression passed

### Symptom
The recovered reset is robust under zero action, but PPO actions eventually
violate the rolling support gate and never maintain axial omega above 0.5 rad/s
for the required 10 seconds.

### Expected Behavior
At `s=400`, fixed and unseen deterministic sets should have success >=0.5, mean
rotation >180°, and early-termination rate <=0.15 before a pressing-force
reference is accepted.

### Reproduction
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_two_phase_force_curriculum.py \
  --curriculum-id 20260823-1830-two-phase-grasp-recovery-cont2-seed0 \
  --hand-pose-config configs/hand_poses/my_grasp.json \
  --revolute-grasp-config configs/hand_grasps/my_grasp_revolute_shared.json \
  --initial-revolute-checkpoint runs/20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0/checkpoints/final_model.zip \
  --steps-per-stage 100000 --eval-episodes 10 --device cuda:0
```

### Evidence
- 100k: fixed/unseen rotation 179.12°/180.49°, violations 1.0/1.0.
- 200k: rotation 200.56°/192.06°, violations 0.20/0.50, maximum omega hold
  1.092/1.080 s, force p95 163.98/168.72 N.
- 300k: rotation 183.18°/188.97°, violations 1.0/0.9, maximum omega hold below
  0.95 s, force p95 231.39/233.83 N.
- Representative non-dropped 200k rollout: 211.2°, 500 steps, but no sustained
  omega success. The 501-frame H.264 video decodes at 640x480 and 25 FPS.

### Hypotheses
1. PPO increases preload/action magnitude to preserve motion, causing force and
   support instability.
2. The current reward optimizes total angle but not ten-second speed persistence.
3. The declared success threshold may be stricter than the behavior represented
   by total-angle metrics; this must be tested as an evaluation ablation, not
   silently changed.

### Investigation
- Extended one factor only: training budget, in two 100k increments from the same
  policy lineage.
- Rotation and violations improved at 200k, then regressed at 300k while force
  p95 increased by about 40%.
- Matched ablation A disabled only the fewer-than-three-contact rotation-credit
  rule. Rotation increased to 294.01°/277.39° fixed/unseen, but all episodes
  still terminated on contact support and omega hold remained near one second.
- Separately identified ablation B additionally disabled support termination.
  Full episodes reached 374.04°/379.20°, but three-contact occupancy collapsed to
  0.288/0.335 and sustained-omega success remained zero.
- Finger-gait ablation C scaled the full contact ladder to 0.10. Rotation rose to
  533.97°/546.98° and every finger showed leave-return events, but >=2-contact
  fraction fell to 0.604/0.578, longest zero-contact intervals reached
  1.76/1.68 s, and sustained-omega success remained zero.

### Root Cause
The immediate mechanism is now isolated: hard reward gating suppresses transient
rotation, while removing it exposes an under-supported rotation strategy.
Weakening contact pressure further creates disengagement/recontact but not
coordinated continuous gaiting. The cause of missing sustained supported rotation
remains unresolved.

### Resolution
No accepted resolution. Keep the original task gate. Do not accept a force
reference, anneal mass, or enter Phase T.

### Verification
Fixed and unseen ten-episode evaluations were preserved for every checkpoint
lineage stage. The strict runner stopped at revolute `s=400` each time.

### Prevention
Keep reset robustness tests separate from policy evaluation and retain per-stage
force statistics so increasing preload cannot masquerade as curriculum progress.

### Lessons Learned
A robust zero-action grasp does not ensure a learned policy will preserve support.
Checkpoint selection must jointly consider task progress, termination rate, and
contact force rather than total rotation alone.

---

## DBG-20260823-005: s25 policy-force statistic was an invalid stage gate
- Date: 2026-08-23
- Status: resolved
- Related runs: `20260823-1855-net-angle-C-force-curriculum-seed0`
- Related files: `scripts/run_two_phase_force_curriculum.py`, `scripts/eval_policy.py`
- Severity: high
- First observed: Phase R transition from accepted `s=50` to `s=25`

### Symptom
All fixed and unseen `s=25` policies pass the net-angle task gate, but their
conditioned pressing force is below the accepted `s=400` tolerance band.

### Expected Behavior
The fixed-set median total normal force on `axial_omega >0.5 rad/s` and
`contact_count >=2` steps should stay in `[78.447,117.670] N`.

### Reproduction
```bash
.venv/bin/python scripts/run_two_phase_force_curriculum.py \
  --curriculum-id 20260823-1855-net-angle-C-force-curriculum-seed0 \
  --hand-pose-config configs/hand_poses/my_grasp.json \
  --revolute-grasp-config configs/hand_grasps/my_grasp_revolute_shared.json \
  --initial-revolute-checkpoint runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/final_model.zip \
  --steps-per-stage 100000 --eval-episodes 10 --device cuda:1
```

### Evidence
- Reference: 98.058 N; eligibility 18.16%.
- `s=25` physical initial `mu=0.25`: 65.903 N.
- Adaptive trials `mu=0.168020`, `0.100933`, `0.10`: 58.906, 54.900,
  53.386 N.
- Every trial has fixed+unseen net-angle success 1.0, zero drop, negligible tip
  error, and finite rollouts.
- All failed trials, checkpoints, configs, metrics, and logs are preserved.

### Hypotheses
1. The fixed-policy torque/force relation is not predictive after PPO adapts at
   each friction value.
2. At this mass, the reward permits faster low-force spinning and does not
   regulate pressing force.
3. The safe lower friction bound makes the desired force unreachable along the
   physically derived decreasing-mu direction.

### Investigation
The initial estimate followed `mu(s)=4*s/400`. Force matching passed directly at
`s=200` and `s=100`; `s=50` passed after one bounded adjustment. At `s=25`,
three adjustments monotonically reduced both `mu` and measured force until the
safe bound was reached.

A separate predeclared diagnostic then extended only the friction bound to scales
0.05 and 0.025, independently retraining each from the accepted `s=50` parent.
Fixed conditioned force was 54.669 and 49.427 N respectively. Relative to
53.386 N at scale 0.10, the response rises slightly and then falls, so it is not
monotone in the physically expected direction. Fixed >=2-contact fraction falls
to 0.260 and 0.157.

### Root Cause
Measurement-validity error. The statistic measured the learned policy's actual
normal force only on about 10–18% of conditioned timesteps. The policy changed
its gait, speed, and contact geometry after every retraining, so this was not an
estimate of minimum required pressing force.

### Resolution
All historical runs and raw values remain unchanged. Policy force is now
diagnostic only and no longer blocks stages. Corrected progression uses the
per-episode net-angle gate and a separate no-policy controlled preload sweep.

### Verification
State status is `force_saturated_R_s25`. The accepted `s=50` checkpoint and all
four rejected `s=25` lineages load successfully. Fixed and unseen evaluations
were saved separately. Both new low-friction runs also load, pass net-angle
success 1.0/1.0, and have zero numerical instability, but fail the force gate.

### Prevention
Do not infer required force from learned-policy force. Validate required force
with fixed trajectory, torque, geometry, and preload controls.

### Lessons Learned
Actual policy force and minimum physically required force are different metrics.

---

## DBG-20260823-006: Tip-connect s400 transfer tilts before net-angle gate
- Date: 2026-08-23
- Status: open
- Related runs: `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-seed0`, `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-ext1-seed0`, `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0`, `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`, `20260824-0000-dexscrew-tilt-growth-s100-tip-seed0`
- Severity: high

### Symptom
Every fixed and unseen episode terminates on `axis_tilt` near but below the
180-degree net-angle target. Phase T cannot enter `s=200`.

### Evidence
The low-learning-rate retry reaches 177.02/175.45 degrees fixed/unseen, but
success is 0/0, drop rate is 1/1, and all 20 episodes terminate on `axis_tilt`.
Tip error stays below 4.04 mm and no numerical instability occurs. A
representative failure reaches 179 degrees and 86 degrees tilt.

### Investigation
The point-connect equality itself has effectively zero axial generalized
resistance when tested at 1 rad/s with contacts disabled
(`3.28e-7`, `-1.65e-8`, and `-1.97e-11 N m` at `s=400,25,1`). Rod damping,
armature, and frictionloss are proportional. The heavy reset has no zero-action
physical drops over ten seeds, but continuous 3-tip occupancy is only 0.39 at
worst. One same-hyperparameter extension and one predeclared 10x lower-learning-
rate retry both fail.

### Follow-up (EXP-20260823-018, s=100 recovery)
A one-sided DexScrew back-to-balance term at scale 50, fine-tuned 65,536 steps
from the accepted revolute `s=100` checkpoint, still yields success 0.0/0.0.
Final/max tilt remains ~80–83°, and recovery’s share of Σ|r| is 0.2–0.4%
because per-episode max tilt equals final tilt (monotonic collapse). The same
`axis_tilt` termination appears at `s=100` as at `s=400`.

### Follow-up (EXP-20260824-001, s=100 growth)
A one-sided DexScrew tilt-growth penalty at `g=50` (recovery left at 0),
fine-tuned 65,536 steps from the same revolute `s=100` parent, still yields
success 0.0/0.0. Growth does fire (3.3–4.1% of Σ|r|, mean −0.45/−0.55) unlike
recovery, but final/max tilt remains ~80–83° and rotation share rises to
38–42%. Collapse stays monotonic.

### Root Cause
The immediate blocker is uncontrolled lateral tilt after transferring the
revolute policy to the free-rotation point-connect dynamics. Whether improved
tip-connect grasp geometry or a dedicated tilt curriculum is sufficient remains
unconfirmed. Recovery-only shaping does not create on-policy decrease events.
Growth-only shaping at `g=50` is on-policy but does not hold the 0.25 rad gate.

### Resolution
None. Phase T remains blocked. Lower masses were not started after the `s=100`
recovery and growth experiments also missed the 0.50 success gate.

### Verification
Fixed and unseen ten-episode metrics and a decode-checked failure video are
preserved in the low-learning-rate retry directory and in
`runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/` and
`runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/`.
