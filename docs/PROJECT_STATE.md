# Project State

## Current Objective
Train the 12-DoF Allegro index/middle/thumb model to rotate the rod in the bottom-tip point-connect task while maintaining three fingertip contacts.

## Current Best Result
The end-to-end smoke reaches the nominal-mass, zero-stabilizer, retained point-connect stage with 100% three-contact occupancy and zero drops, but only 40.11° mean rotation and 0% sustained-ω success over 5 deterministic episodes.

## Best Checkpoint
`runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C3-mass-1-s1-retry0/checkpoints/final_model.zip`

## Active Configuration
- Hand: Allegro V3-derived index + middle + thumb, 4 joints each.
- Curriculum: revolute → soft/assisted point connect → hard point connect → zero stabilizer → mass scales 4, 2, 1.
- Final physics: bottom point-connect retained; mass scale 1; stabilizer 0.
- Support: 25-step window, at least 18 three-contact steps; no rotation credit below three contacts.
- Current tested contact bonus: +3.0 per three-contact step.

## What Is Working
- Both new MJCF variants compile with 12 actuators and matched 48-D observations.
- Joint frames, axes, ranges, and palm-relative mounts match the official Allegro V3 index/middle/thumb model.
- Bottom reset produces three fingertip contacts on 50/50 fixed seeds in revolute and point-connect checks.
- Unit tests, Gymnasium environment check, checkpoint save/load, VecNormalize transfer, and every curriculum transition pass plumbing checks.
- Final smoke keeps the endpoint error at 1.40 mm, all three contacts for 100% of steps, zero drops, and zero external stabilizer torque.
- `scripts/edit_hand_pose_web.py` is the primary pose editor: it serves a headless EGL-rendered local Web UI with palm/camera controls, geometry/contact diagnostics, safe load/save, and explicit overwrite confirmation.
- The Web UI and optional legacy keyboard editor both edit only the world-parented Allegro palm root and save the same versioned validated JSON consumed by training, evaluation, teleoperation, curriculum, and video export.
- Pose-enabled revolute and point-connect models preserve identical 12-D actions, 48-D observations, child transforms, and joint axes; run artifacts snapshot pose path, SHA-256, and content.

## Known Problems
- The policy exploits the +3/step contact bonus by holding the rod nearly static.
- In final smoke evaluation, mean rotation reward is +0.085/step versus +3.0/step contact reward.
- Sustained angular-speed success remains 0%; maximum hold is 0 s in the selected final checkpoint.
- Early smoke stages often terminate on `contact_support`; 20k steps is not a performance budget.

## Current Hypotheses
1. The three-contact bonus is too large relative to the angular-velocity reward and creates a static-grasp optimum.
2. Reducing only the three-contact bonus to +0.3 while retaining the hard support gate and rotation gate should preserve contact but make rotation the dominant positive signal.
3. A longer A0 revolute run is required before evaluating transfer quality.

## Most Recent Experiment
`20260823-0405-allegro-tip-bottom-smoke-seed0`: all eight curriculum stages executed. Final: rotation 40.11°, tip error 1.40 mm, three-contact occupancy 1.0, drop rate 0, success rate 0.
Deterministic selected-checkpoint videos for all eight stages, with exact VecNormalize/config reconstruction and per-frame metrics overlays, are indexed at `runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/INDEX.md`.

## Pending Geometry Preview
The latest 180° world-X preview is indexed at `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/INDEX.md`. Relative to the 10 mm-clearance preview, the rigid hand subtree moves another +20 mm in Z and `[+9.915, +1.301]` mm in XY. Palm clearance is 30 mm and thumb-root radial distance falls from 65.260 to 55.260 mm. After 0.2 s, C3 retains 3/3 contacts but A0 retains only 2/3; no joints were retuned. The orientation has not been accepted for training or made the default.

## Next Recommended Experiment
Run a controlled A0 revolute experiment changing only `three_contact_reward` from 3.0 to 0.3, with dense checkpoint evaluation. Proceed to point-connect transfer only if rotation/ω improves while three-contact occupancy remains at least 0.72.

## Blocked Items
No correctness blocker. Full training is intentionally paused at the reward-scale decision; the smoke result does not establish task success. The reversed hand orientation is also pending visual confirmation and must not replace the validated default before approval.

## Important Commands
```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/check_contact_reachability.py --samples 100 --seeds 3 --hand-model allegro --physics tip_connect --tip-anchor bottom --out-dir runs/<run_id>
.venv/bin/python scripts/run_allegro_tip_bottom_curriculum.py --start-scale 10 --num-envs 32 --device cuda --seed 0
.venv/bin/python scripts/edit_hand_pose_web.py --physics revolute --output configs/hand_poses/<name>.json
.venv/bin/python scripts/train_parallel.py --physics revolute --tip-anchor bottom --hand-pose-config configs/hand_poses/<name>.json --run-id <run_id>
```
