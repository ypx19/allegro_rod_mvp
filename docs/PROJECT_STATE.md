# Project State

## Current Objective
Free tip with tip-stop 0.5 cm: tip-penalty ×5 (EXP-010, 900k) did not clear
the tip gate. Best overall remains tip-connect EXP-005. Next free-tip lever
should not be tip-weight alone (try soft tip) or pause free tip.

## Current Best Result
Live s=1 transfer (EXP-20260918-005): palm-down translation + default joints,
tracker reward, 0.35 rad kill, palm-down PPO, 300k — **20/20 full 20 s**,
drop 0, tilt ~2°, tip <1 mm, net-angle success 0.80/0.50.
Checkpoint:
`runs/20260918-1528-s1-palm-down-ppo-tilt035-seed0/checkpoints/final_model.zip`
Verified parallel track (EXP-20260918-001): bundled palm-down tip-connect at
mass-scale 1 (different XML/PPO path) also stable; see FIND-20260918-001.
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
A T00 2×2 of frame stacking × support-aware reward (EXP-20260914-001) also
fails: 80/80 `axis_tilt`. History does not raise re-contact; the support-aware
reward shortens episodes and increases post-loss ω_perp.

## Best Checkpoint
s=1 tip-connect screwdriver (live stack):
`runs/20260918-1528-s1-palm-down-ppo-tilt035-seed0/checkpoints/final_model.zip`
with `vecnormalize.pkl`
Palm-down tip-connect bundle (verified, mass-scale 1 only):
`palm_down_screwdriver/checkpoint/policy.zip` with `normalize.pkl`
Phase T / revolute curriculum (unchanged, not the tip-connect best):
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
- Palm-down tip-connect PPO (300k, CPU) is a verified fixed-tip screwdriver:
  ~3.28 turns / 20 s, ~10.35 turns / 60 s bounded, tilt <3°, tip <1 mm,
  never zero fingertip contacts. Videos:
  `docs/media/palm-down-screwdriver-20s-seed7000.mp4`,
  `docs/media/palm-down-screwdriver-60s-seed4000.mp4`.
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
- Palm-down 300k policy leaves the 20 s training observation range after ~3
  turns: unbounded 60 s keeps tilt/tip but fails speed continuity unless
  observation 37 (cumulative turns) is clipped to 3.

## Current Hypotheses
1. The T00 axis-tilt death is a support-collapse sequence: 2-finger support →
   active release to 1 finger → `||ω_perp||` rise → tilt growth → no rapid
   re-contact. Seed 6 replay is the frame-level example.
2. H1 has an early 100k online-length signal (FIND-20260914-002) that
   **narrows by 300k** (FIND-20260914-003: C longest; B≈A; D≈A). Eval
   collapse metrics still reject hist4 as a T00 fix.
3. H2 as gated-rotation + λ=0.5 wobble is rejected on 100k eval: C got
   shorter and more violent after support loss. At 300k online, C is
   longest but that is not re-contact.
4. At **s=1**, palm-down tracker + 0.35 kill + palm-down PPO reproduces
   stable tip-connect screwdriving on our_hand (FIND-20260918-006).
   FIND-20260918-005 showed s=1 alone with T00 PPO is not enough.
   FIND-20260918-002/003 rejected pose and tracker+0.35 at **s=400**.
   Whether palm-down PPO transfers up the mass curriculum is untested.
5. **Important (FIND-20260918-008 / DBG-20260918-002):** success gaits are
   multi-link wedges while the policy only observes tip forces. Proximal
   contact is learned indirectly via proprioception + rod dynamics. Position
   servos + contact constraints jam rather than “double-apply” torque; high
   proximal Fn is expected and is a mass-up / hardware-transfer risk. Future
   options: all-link contact diagnostics, non-tip force features/penalties,
   actuator-saturation gates — do not silently redefine tip `contact_count`.
6. FIND-20260918-009: tip solref **0.004** (palm-down literal) on our_hand
   raises spin but reintroduces `axis_tilt` deaths; keep **0.008**.

## Most Recent Experiment
`EXP-20260918-010` / `20260918-1829-s1-freetip-tipscale1-900k-seed0`:
tip-scale 1.0 (5×) +900k from EXP-009. Length 298/244, tip ~7.5/5.9 mm,
tip_error×10, success 0. **Rejected** as free-tip fix.

## Next Recommended Experiment
Soft-tip solref fade from EXP-009/010 stack, or pause free tip and mass-up
tip-connect EXP-005. Avoid further tip-weight-only or budget-only continues.

## Most Recent Debugging Session
`DBG-20260915-001` (resolved): 300k A–D analysis page no longer uses an MP4
as the UI. Seed-6-style step slider + stills + per-frame ω_axial / tilt /
dθ/dt / n_contact / fingertip force. Live
http://127.0.0.1:8767/pages/t00-ablation-demos-300k/?v=scrub20260915
Parent issue `DBG-20260823-006` remains open (T00 tilt death).

## Blocked Items
Phase R is complete. Phase T is blocked at `s=400` by `DBG-20260823-006`;
the `s=100` recovery and growth probes also missed the 0.50 gate, so the
curriculum stays paused.

## Project-page fragments to inherit
Four standalone HTML pages are ready to copy into `docs/demo.html` later
(see `docs/pages/README.md`):
1. `docs/pages/t00-ablation-history.html` — T00 2×2 curves + early history finding.
2. `docs/pages/t00-seed6-tilt-collapse/index.html` — seed-6 contact/tilt timeline
   (canonical original: `runs/20260914-t00-seed6-tilt-collapse/index.html`).
3. `docs/pages/t00-ablation-demos-300k/index.html` — 300k A/B/C/D seed-6
   **frame-slider** analysis (canonical traces: `runs/20260914-1805-t00-ablation-demos-300k/`).
4. `docs/pages/palm-translation-compare/index.html` — my_grasp vs palm_down
   translation (live: `scripts/palm_translation_compare_web.py --port 8768`).

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
.venv/bin/python scripts/policy_viz_board.py --model runs/<run>/checkpoints/final_model.zip --physics tip_connect --tip-anchor bottom --rod-mass-scale 400 --hand-pose-config configs/hand_poses/my_grasp.json --port 8770
.venv/bin/python scripts/palm_translation_compare_web.py --port 8768
HAND_POSE_BROWSER_TESTS=1 MUJOCO_GL=egl .venv/bin/python -m unittest tests.test_hand_pose_web.HandPoseWebBrowserTest -v
.venv/bin/python scripts/run_t00_support_ablation.py --conditions A,B,C,D --device cuda:0 --seed 0
.venv/bin/python -m unittest tests.test_t00_support_ablation -v
.venv/bin/python scripts/run_two_phase_force_curriculum.py --hand-pose-config configs/hand_poses/my_grasp.json --device cuda:1
.venv/bin/python scripts/evaluate_hand_grasp_reset.py --hand-pose-config configs/hand_poses/my_grasp.json --hand-grasp-config configs/hand_grasps/my_grasp_revolute_shared.json --out-dir runs/<run_id>
# Palm-down screwdriver bundle (must use bundled env via PYTHONPATH)
cd palm_down_screwdriver && PYTHONPATH=. MUJOCO_GL=egl ../.venv/bin/python test_final_alignment.py
PYTHONPATH=palm_down_screwdriver MUJOCO_GL=egl .venv/bin/python palm_down_screwdriver/render_candidate.py --module experiment_palm_down_bounded --checkpoint palm_down_screwdriver/checkpoint/policy.zip --norm palm_down_screwdriver/checkpoint/normalize.pkl --out runs/<run_id>/demo --seed 4000 --seconds 60
```
