# Project State

## Current Objective
Resolve the heavy bottom-tip-connect lateral-tilt failure after the corrected
proportional-physics revolute curriculum reached `s=1`.

## Current Best Result
The saved-pose revolute reset is now robust at both endpoint masses: 10/10 seeds
retain 3/3 contacts for 100 steps at `s=400` and `s=1`, with no non-tip rod
collisions. The best saved-pose policy is the 200k-step continuation: fixed/unseen
rotation 200.56°/192.06°, but violation rates 0.20/0.50 and sustained-omega
success 0. Phase R remains unaccepted at `s=400`.
Ablation A increases angle to 294.01°/277.39° but terminates every episode on
support. Ablation B reaches 374.04°/379.20° only while three-contact occupancy
collapses to 0.288/0.335. Neither is a better task checkpoint.
The corrected full-vector proportional curriculum passes every revolute stage
from `s=400` to `s=1` with fixed/unseen net-angle success 1.0/1.0 and no
numerical instability. At `s=1`, rotation is 11,384.68°/11,337.89°, but >=2-tip
contact fraction is only 0.0146/0.0180, so support collapse remains explicit.
Phase T starts at `s=400` but fails: the controlled low-LR retry achieves only
177.02°/175.45°, and all 20 episodes terminate on axis tilt.
A DexScrew back-to-balance fine-tune at tip-connect `s=100` (EXP-20260823-018)
also fails: success 0.0/0.0, rotation 190.95/223.07°, final tilt 81.49/82.58°.
A DexScrew tilt-growth fine-tune at the same mass (EXP-20260824-001) also
fails: success 0.0/0.0, rotation 280.12/264.34°, final tilt 82.67/80.05°.

## Best Checkpoint
`runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/checkpoints/final_model.zip`

## Active Configuration
- Hand: Allegro V3-derived index + middle + thumb, 4 joints each.
- Curriculum: revolute → soft/assisted point connect → hard point connect → zero stabilizer → mass scales 4, 2, 1.
- Final physics: bottom point-connect retained; mass scale 1; stabilizer 0.
- Gait transfer: rotation credit below three contacts enabled, contact-support
  termination disabled, discrete contact ladder scaled by 0.10.
- Success: positive net unwrapped angle >=pi, tilt <0.25 rad, tip error <0.02 m,
  finite/stable episode, and no physical drop. Contact remains diagnostic.
- Runner uses +0.3 raw three-contact reward, explicit full-vector rod+pad friction,
  fixed solref, and schedule `400,200,100,50,25,12.5,6.25,3.125,1.5625,1`.
- Required pose: `configs/hand_poses/my_grasp.json`, SHA-256
  `2d8ac7f17a6693855543395d52022524c1a2956422915ee2962544019f6e7c9f`.
- Revolute reset companion:
  `configs/hand_grasps/my_grasp_revolute_shared.json`, SHA-256
  `fdd9ead60842eca3167f98ecda3c496ea752cff3fa9e053de47a4544e061b03a`.
- DexScrew tilt recovery reuses `--axis-tilt-recovery-scale` (default 0).
  Experiment scale 50 is not the new default.
- DexScrew tilt-growth uses `--axis-tilt-growth-scale` (default 0).
  Experiment scale 50 is not the new default.

## What Is Working
- Both new MJCF variants compile with 12 actuators and matched 48-D observations.
- Joint frames, axes, ranges, and palm-relative mounts match the official Allegro V3 index/middle/thumb model.
- Bottom reset produces three fingertip contacts on 50/50 fixed seeds in revolute and point-connect checks.
- Unit tests, Gymnasium environment check, checkpoint save/load, VecNormalize transfer, and every curriculum transition pass plumbing checks.
- Final smoke keeps the endpoint error at 1.40 mm, all three contacts for 100% of steps, zero drops, and zero external stabilizer torque.
- `scripts/edit_hand_pose_web.py` is the primary pose editor: it serves a headless EGL-rendered local Web UI with palm/camera controls, geometry/contact diagnostics, safe load/save, and explicit overwrite confirmation.
- The Web editor's six pose fields/sliders and six camera sliders have real-Chromium coverage, including decimal/negative/zero editing, fine/coarse increments, stale-response suppression, render changes, and reset/load/save errors.
- The Web UI and optional legacy keyboard editor both edit only the world-parented Allegro palm root and save the same versioned validated JSON consumed by training, evaluation, teleoperation, curriculum, and video export.
- Pose-enabled revolute and point-connect models preserve identical 12-D actions, 48-D observations, child transforms, and joint axes; run artifacts snapshot pose path, SHA-256, and content.
- Companion grasp configs are schema/hash/joint-limit validated and snapshot the
  exact reset/preload vectors. The revolute companion passes 20/20 endpoint-mass
  reset cases across seeds 0–9.

## Known Problems
- The policy exploits the +3/step contact bonus by holding the rod nearly static.
- In final smoke evaluation, mean rotation reward is +0.085/step versus +3.0/step contact reward.
- Sustained angular-speed success remains 0%; maximum hold is 0 s in the selected final checkpoint.
- Early smoke stages often terminate on `contact_support`; 20k steps is not a performance budget.
- The recovered revolute reset is not universal: bottom tip-connect `s=400`
  passes 0/10 seeds and has minimum 3-tip occupancy 0.765. At `s=1` it passes
  10/10.
- PPO policy actions can still destroy an otherwise robust reset. After 200k
  steps the best policy exceeds 180° but violates support in 20% fixed and 50%
  unseen episodes; a further 100k regresses to 100%/90% violations.
- Ungating rotation credit below three contacts increases transient angle but does
  not produce ten-second sustained omega. Removing support termination creates an
  unsupported high-angle policy rather than solving contact retention.
- Historical conditioned-force matching was a measurement-validity error: it
  sampled the policy's actual force on only about 10–18% of steps, not minimum
  required force. Raw measurements remain preserved but no longer gate stages.
- The controlled preload sweep does not prove exact invariance: no multiplier
  through 3 passes at `s=400` or `s=25`; `s=1` first passes at 1.5 (9.314 N).
- Tip-connect `s=400` fails through one standard extension and one predeclared
  low-LR retry; all fixed/unseen episodes terminate on lateral axis tilt.
- Tip-connect `s=100` zero-shot, recovery fine-tune, and growth fine-tune all
  terminate 20/20 on lateral tilt (~80°). Recovery almost never fires;
  growth does fire (~3–4% of |reward|) but still misses the 0.25 rad gate.

## Current Hypotheses
1. The three-contact bonus is too large relative to the angular-velocity reward and creates a static-grasp optimum.
2. Reducing only the three-contact bonus to +0.3 while retaining the hard support gate and rotation gate should preserve contact but make rotation the dominant positive signal.
3. A longer A0 revolute run is required before evaluating transfer quality.
4. The remaining Phase R failure is policy optimization/contact retention, not
   reset reachability. Continuing identical PPO after 200k is unlikely to resolve
   it because the 300k checkpoint regressed while normal-force p95 rose.
5. A gait-phase observation or explicit leave-then-return objective may be needed
   to coordinate regrasp timing; contact reward scale alone does not provide phase.
6. Further weakening contact reward is not supported: scale 0.10 worsened B's
   occupancy without producing sustained omega.
7. DexScrew one-sided tilt recovery cannot help if on-policy tilt never decreases.
   Growth at `g=50` does fire during that collapse (~4% of |reward|) and still
   fails the 0.25 rad gate; a tilt assist or grasp change is more plausible
   than another Δtilt-only scale.

## Most Recent Experiment
`EXP-20260824-001` adds DexScrew tilt-growth (scale 50, default 0; recovery 0)
and fine-tunes 65,536 steps from revolute `s=100` into bottom tip-connect
`s=100`. Fixed/unseen net-angle success remains 0.0/0.0.

## Most Recent Debugging Session
`DBG-20260823-002` resolved the Web pose-control failure. The cause was malformed pose-row tuple decoding plus duplicate DOM IDs; Chromium verification now covers every field and slider, stale responses, rendered-image changes, and safe file actions. No saved user pose was overwritten.

## Pending Geometry Preview
The latest 180° world-X preview is indexed at `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/INDEX.md`. Relative to the 10 mm-clearance preview, the rigid hand subtree moves another +20 mm in Z and `[+9.915, +1.301]` mm in XY. Palm clearance is 30 mm and thumb-root radial distance falls from 65.260 to 55.260 mm. After 0.2 s, C3 retains 3/3 contacts but A0 retains only 2/3; no joints were retuned. The orientation has not been accepted for training or made the default.

## Next Recommended Experiment
Do not start other tip-connect masses. Do not retune growth or recovery scale.
Test a tilt assist or a dedicated tip-connect grasp change. Both one-sided
Δtilt terms at scale 50 are rejected.

## Blocked Items
Phase R is complete. Phase T is blocked at `s=400` by `DBG-20260823-006`;
the `s=100` recovery and growth probes also missed the 0.50 gate, so the
curriculum stays paused.

## Status Deck
Open `reports/decks/20260902-experiment-status.html` in a browser
(arrow keys / space). Snapshot date 2026-09-02.

## Important Commands
```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_env.py
.venv/bin/python scripts/check_contact_reachability.py --samples 100 --seeds 3 --hand-model allegro --physics tip_connect --tip-anchor bottom --out-dir runs/<run_id>
.venv/bin/python scripts/run_allegro_tip_bottom_curriculum.py --start-scale 10 --num-envs 32 --device cuda --seed 0
.venv/bin/python scripts/edit_hand_pose_web.py --physics revolute --output configs/hand_poses/<name>.json
HAND_POSE_BROWSER_TESTS=1 MUJOCO_GL=egl .venv/bin/python -m unittest tests.test_hand_pose_web.HandPoseWebBrowserTest -v
.venv/bin/python scripts/train_parallel.py --physics revolute --tip-anchor bottom --hand-pose-config configs/hand_poses/<name>.json --run-id <run_id>
.venv/bin/python scripts/run_two_phase_force_curriculum.py --hand-pose-config configs/hand_poses/my_grasp.json --device cuda:1
.venv/bin/python scripts/evaluate_hand_grasp_reset.py --hand-pose-config configs/hand_poses/my_grasp.json --hand-grasp-config configs/hand_grasps/my_grasp_revolute_shared.json --out-dir runs/<run_id>
```
