# Experiment Log

## EXP-20260724-002: Does Stage 0/1 pretraining help or hurt Stage 2? (idea, unverified)
- Run ID: TBD
- Date: 2026-07-24
- Status: idea / planned
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0` (pretrained) vs a from-scratch Stage 2 control
- Random seed: 0 (multi-seed later)
- Device: CPU

### Question
Is a policy pretrained under the Stage 1 axis stabilizer actually beneficial for Stage 2 (which has no stabilizer), or does stabilizer-era pretraining induce a dependence that harms unassisted Stage 2?

### Hypothesis (falsifiable, not yet tested)
The Stage 1 stabilizer may teach the policy to rely on external orientation torque, so transferring that policy into stabilizer-free Stage 2 could be worse than, or no better than, training Stage 2 more directly. This is only an assumption; it has NOT been verified.

### Change from Baseline
Compare, under identical Stage 2 config (tip joint on, stabilizer 0, same rewards/network/eval):
1. resume from the stabilizer-faded Stage 1 parent (current approach); vs
2. a control that reaches Stage 2 with less/zero stabilizer exposure (e.g. shorter stabilizer schedule, or from-scratch Stage 2 with tip joint).

### Success Criteria
Define before interpreting: a meaningful difference in Stage 2 axis-tilt termination rate and mean rotation between pretrained vs control on seeds 0–19. If pretrained ≈ control, stabilizer dependence is not the main issue.

### Note
Deferred. Record now so the assumption is not silently assumed true. Revisit after EXP-20260724-001.

## EXP-20260724-001: Three-finger simultaneous-contact reward (idea, planned)
- Run ID: TBD
- Date: 2026-07-24
- Status: idea / planned
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0` (near-gate 2x512 parent)
- Random seed: 0 (multi-seed later)
- Device: CPU

### Question
Does the policy currently rotate the rod using effectively one finger, and does that single-finger contact push the rod off-axis (induce tilt), causing the Stage 2 axis-tilt terminations?

### Hypothesis (falsifiable, not yet tested)
Requiring all three fingertips to maintain positive contact with the rod will force coordinated multi-finger manipulation, reduce the lateral push that tilts the rod, and lower axis-tilt terminations while preserving positive axial rotation.

### Change from Baseline
Add a reward term that rewards simultaneous positive contact signal on all three fingertips (or penalizes fewer than 3 in contact). Keep tip joint, stabilizer schedule, rotation/tilt weights, network, and optimizer otherwise fixed. Change only this one term.

### Pre-experiment diagnostic (do first, cheap)
Before training, confirm the premise on existing checkpoints: log per-fingertip contact state during Stage 1/Stage 2 eval and measure how often only one finger is in contact, and whether single-finger contact correlates with tilt onset. If the premise is false, revise the idea before spending training.

### Success Criteria
Define before interpreting. Diagnostic: fewer than 20/20 axis-tilt terminations under Stage 2 with stabilizer 0. Gate: rotation >180°, tip error <0.02 m, drop ≤0.15 on seeds 0–19. Watch for reward hacking (fingers just touching without producing rotation).

### Risks / watch-outs
- Contact bonus could dominate and cause the policy to grip without rotating (log raw + weighted component and rotation together).
- Contact signal definition and threshold must be documented in METRICS.md.
- Must be tested as a single-factor change per AGENTS.md §2.2.

## EXP-20260723-016: Stage 2 short adaptation at stabilizer 0.10
- Run ID: `20260723-1700-stage2-stab010-shortadapt-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0`
- Random seed: 0
- Device: CPU

### Question
Can a bounded 5k adaptation cross the rotation gate at the last stable side of the measured assist cliff?

### Hypothesis
Reducing stabilizer only from 0.12 to 0.10 and selecting every 1k steps can recover the 3.45° needed to pass without inducing drop.

### Change from Baseline
Only external stabilizer scale changes 0.12→0.10 under Stage 2. Tip joint, rewards, network, optimizer settings, and evaluation remain fixed.

### Success Criteria
At least one checkpoint passes rotation >180°, tip error <0.02 m, and drop ≤0.15 over seeds 0–19.

## EXP-20260723-015: Stage 2 randomization adaptation at stabilizer 0.12
- Run ID: `20260723-1600-stage2-rand-adapt-stab012-seed0`
- Date: 2026-07-23
- Status: completed (failed training; diagnostic control supported)
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0`
- Random seed: 0
- Device: CPU

### Question
Can the near-gate stabilizer-0.12 policy adapt to Stage 2 mass/friction randomization when assist strength is held fixed?

### Hypothesis
If direct stabilizer removal is the primary curriculum cliff, the policy should retain useful rotation and survival when only Stage 2 randomization is introduced.

### Change from Baseline
Only curriculum stage/randomization changes from Stage 1 to Stage 2. Tip joint, stabilizer 0.12, absolute tilt weight 0.10, recovery scale 0, rotation scale 160, network, and optimizer settings remain fixed.

### Success Criteria
Standard fixed-seed Stage 2 gate: rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
Before training, the parent transferred at 178.51°, 8.25 mm, and drop 0.05. The first 3k checkpoint fell to 173.79° and the final checkpoint to 66.70°, while drop stayed ≤0.05.

### Interpretation
Stage 2 randomization itself is not the dominant failure. Continued training at fixed stabilizer 0.12 caused conservative loss of rotation.

### Additional Diagnostic
A fixed-policy stabilizer sweep localized the cliff: scale 0.10 retained 176.55° with drop 0, while 0.08 fell to 132.42° with drop 0.40. Scale 0.02 and 0 produced 20/20 tilt terminations.

### Decision
reject the adapted checkpoints; retain the unmodified parent

### Next Step
Short, densely checkpointed Stage 2 adaptation at stabilizer 0.10.

## EXP-20260723-014: Stage 2 local tilt-recovery shaping
- Run ID: `20260723-1500-stage2-tilt-recovery40-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Random seed: 0
- Device: CPU
- Duration: ~15 seconds training
- Checkpoint: `runs/20260723-1500-stage2-tilt-recovery40-seed0/checkpoints/final_model.zip`

### Question
Does a local reward for reducing tilt teach active recovery where a larger absolute penalty failed?

### Hypothesis
Adding clipped `40 * (previous_tilt - current_tilt)` with range [-2, 2] will reduce axis-tilt terminations while preserving the existing absolute tilt weight and rotation incentive.

### Change from Baseline
Only the tilt-recovery shaping term is added. Absolute tilt weight remains 0.10.

### Success Criteria
Standard gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support: fewer than 20/20 axis-tilt terminations.

### Result
All six checkpoints had 20/20 axis-tilt terminations. The final checkpoint was best by mean rotation: 4.22°, 4.73 mm tip error, 41.40° final tilt, success 0, and drop 1.0. The recovery component was active at -0.283 per step.

### Key Metrics
| Metric | Baseline | Recovery 40 | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 4.22° | +3.09° |
| Tip error | 4.03 mm | 4.73 mm | +0.70 mm |
| Final tilt | 41.32° | 41.40° | +0.08° |
| Drop rate | 1.00 | 1.00 | 0 |
| Axis-tilt terminations | 20/20 | 20/20 | 0 |

### Visual Evidence
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_00_seed0_rot14deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_01_seed6_rot11deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_02_seed13_rot10deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/plots/stage2_reward_strategy_comparison.png`

### Interpretation
The small rotation increase is measurable, but the experiment fails its diagnostic criterion because survival did not change. Direct removal of the stabilizer remains the dominant failure.

### Decision
reject

### Next Step
Hold stabilizer at 0.12 while enabling Stage 2 randomization to isolate curriculum mismatch from randomization adaptation.

## EXP-20260723-013: Stage 2 stronger axis-deviation penalty
- Run ID: `20260723-1230-stage2-tipjoint-tiltw025-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Random seed: 0
- Device: CPU

### Question
Is Stage 2 failing because axis deviation is under-penalized?

### Hypothesis
Increasing axis-tilt penalty weight from 0.10 to 0.25 will reduce axis-tilt terminations and raise survival without eliminating positive axial rotation.

### Change from Baseline
Only `axis_tilt_penalty_weight`: 0.10→0.25.

### Success Criteria
On 20 deterministic episodes, rotation >180°, tip error <0.02 m, and drop ≤0.15. A diagnostic improvement requires fewer than 20 axis-tilt terminations without numerical instability.

### Result
All six evaluated checkpoints had 20/20 axis-tilt terminations. Best rotation was 1.89°; final was 1.76°, 4.61 mm tip error, 41.62° tilt, success 0, drop 1.0. No NaN/Inf was observed.

### Key Metrics
| Metric | Baseline | Weight 0.25 | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 1.76° | +0.63° |
| Tip error | 4.03 mm | 4.61 mm | +0.58 mm |
| Final tilt | 41.32° | 41.62° | +0.30° |
| Drop rate | 1.00 | 1.00 | 0 |
| Weighted tilt term | -2.03 | -5.01 | -2.98 |

### Interpretation
Measured fact: the increased penalty was active. Measured fact: it did not reduce tilt terminations. The evidence contradicts the hypothesis that insufficient absolute tilt weight is the primary cause.

### Decision
reject

### Next Step
Use a recovery-shaped tilt signal or curriculum rather than increasing the same absolute penalty again.

## EXP-20260723-012: Corrected Stage 2 baseline
- Run ID: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Status: completed (failed)

### Result
With tip joint active and stabilizer off: rotation 1.13°, tip error 4.03 mm, final tilt 41.32°, success 0, drop 1.0; 20/20 ended on axis tilt.

### Decision
Increase the axis-deviation penalty in a controlled experiment.

## EXP-20260723-011: 2x512 assist fade
- Supporting runs: `20260723-0820-capacity512-stab018-tiltw010-rot160-seed0`, `20260723-0900-capacity512-stab015-tiltw010-rot160-seed0`, `20260723-1000-capacity512-stab012-tiltw010-rot160-seed0`
- Status: completed

### Result
Stabilizer 0.18 passed at 229.57°/14.08 mm/drop 0.10. Stabilizer 0.15 passed at 200.00°/12.15 mm/drop 0.10. Stabilizer 0.12 narrowly failed; dense selection peaked at 179.19°/8.13 mm/drop 0.05.

### Decision
Adopt the stabilizer-0.15 checkpoint as the Stage 2 parent.

## EXP-20260723-010: Model-capacity ablation
- Candidate: `20260723-0610-capacity512-stab018-seed0`
- Matched control: `20260723-0611-capacity256-resetopt-stab018-seed0`
- Status: completed

### Result
Function-preserving expansion increased parameters 159,251→580,627 with maximum initial action error 1.79e-7. After matched 25k training, 512 reached 126.39° versus 37.79° for 256, both with drop 0.05.

### Interpretation
Capacity is a secondary limitation, not a sufficient fix. Reward balance was also required.

## EXP-20260723-009: Reward-balance bracket
- Supporting runs: `20260723-0300-stage1-stab018-tiltw010-seed0`, `20260723-0340-stage1-stab018-tiltw010-rot64-seed0`, `20260723-0410-stage1-stab018-tiltw010-rot256-seed0`, `20260723-0450-stage1-stab018-tiltw010-rot128-seed0`
- Status: completed

### Result
Tilt weight 0.10 with rotation 16 collapsed to inactivity; rotation 64 remained conservative; rotation 256 was unstable; rotation 128 approached the gate. The 512 model with rotation 160 later passed.

### Decision
Use tilt weight 0.10 / rotation scale 160 for assist fade; revisit tilt weight specifically for Stage 2.

## EXP-20260723-007: Stabilizer fade 0.20 to 0.18
- Run ID: `20260723-0200-stage1-stab018-lowlr-seed0`
- Date: 2026-07-23
- Status: completed (failed gate)
- Parent or baseline run: EXP-20260723-006
- Random seed: 0
- Device: CPU
- Duration: ~20 seconds training
- Checkpoint: `runs/20260723-0200-stage1-stab018-lowlr-seed0/checkpoints/ppo_rod_326480_steps.zip`

### Question
Can low-LR fine-tuning preserve the gate when stabilizer scale changes 0.20→0.18?

### Hypothesis
The small reduction can be adapted within 25k steps.

### Success Criteria
Rotation >180°, tip error <0.02 m, drop ≤0.15 over seeds 0–19.

### Result
Rejected. Best checkpoint: 177.18°, 20.06 mm, success 0.25, drop 0.30.

### Decision
investigate a new reward/curriculum limitation.

## EXP-20260723-006: Stabilizer fade 0.25 to 0.20
- Run ID: `20260723-0130-stage1-stab020-lowlr-seed0`
- Date: 2026-07-23
- Status: completed
- Parent or baseline run: EXP-20260723-005R
- Random seed: 0
- Device: CPU
- Checkpoint: `runs/20260723-0130-stage1-stab020-lowlr-seed0/checkpoints/ppo_rod_301480_steps.zip`

### Question
Can low-LR fine-tuning reduce the stabilizer to 0.20?

### Success Criteria
Rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
Periodic checkpoint passed: 199.58°, 16.04 mm, success 0.40, drop 0.15. Final checkpoint failed and was rejected.

### Decision
adopt periodic checkpoint.

## EXP-20260723-005R: Tip solref fade 0.05 to 0.10 with verified low LR
- Run ID: `20260723-0110-stage1-tip-solref010-lowlr-seed0`
- Date: 2026-07-23
- Status: completed
- Parent or baseline run: `20260723-0040-stage1-soft-tip-stab025-seed0`
- Random seed: 0
- Device: CPU
- Checkpoint: `runs/20260723-0110-stage1-tip-solref010-lowlr-seed0/checkpoints/final_model.zip`

### Result
Passed: rotation 223.96°, tip error 11.89 mm, success 0.55, drop 0.05.

### Decision
adopt.

## EXP-20260723-005: Stage 1 tip solref 0.10 low-LR fine-tune (invalid)
- Run ID: `20260723-0100-stage1-tip-solref010-seed0`
- Date: 2026-07-23
- Status: failed
- Parent or baseline run: `20260723-0040-stage1-soft-tip-stab025-seed0`

### Question
Can a low-learning-rate fine-tune preserve the Stage 0 policy while weakening the tip constraint?

### Change from Baseline
Planned: tip solref `0.05→0.10`, LR `3e-4→1e-5`, entropy coefficient `0.01→0`.

### Result
Invalid experiment. The log showed LR remained `3e-4`; rollout success fell from 0.97 to 0.41.

### Interpretation
This run does not test the stated hypothesis because the LR override was not applied (DBG-20260723-004).

### Decision
reject

### Next Step
Verify the LR fix with a 1024-step smoke test, then rerun under a new run ID.

## EXP-20260723-003: Stage 1 soft tip-connect fade
- Run ID: `20260723-0034-stage1-soft-tip-seed0`
- Date: 2026-07-23
- Status: running
- Parent or baseline run: EXP-20260723-001 Stage 1 (connect off)
- Checkpoint: `runs/20260723-0034-stage1-soft-tip-seed0/checkpoints/`
### Question
Does keeping tip `<connect>` active but softer (`solref` 0.05) enable Stage 1 transfer from Stage 0?
### Hypothesis
A softer tip spring preserves tip locality while requiring more finger work than Stage 0; eval gate beats connect-off baseline (success 0, rot -38.6°).
### Change from Baseline
Only Stage1 equality: remains **active** with `solref[0]=0.05` (was disabled). Axis stabilizer stays 0.
### Success Criteria
`eval_policy` Stage 1 gate.
### Result
Pending.
### Decision
Pending.

## EXP-20260723-002: Stage 1 with 50% axis stabilizer
- Run ID: `20260723-0026-stage1-softstab-seed0`
- Date: 2026-07-23
- Status: failed / rejected
- Parent or baseline run: EXP-20260723-001 Stage 1 (stabilizer 0)
### Question
Does half-strength vertical axis stabilizer (tip connect off) restore Stage 1 transfer?
### Hypothesis
scale=0.5 fixes abrupt assist removal.
### Result
Rejected. `ep_len_mean≈5`, `ep_rew_mean≈-14`, frequent DOF-12 NaNs. See `runs/20260723-0026-stage1-softstab-seed0/summary.md`.
### Interpretation
xfrc orientation spring without firm tip constraint is unstable in this scene.
### Decision
reject
### Next Step
EXP-20260723-003 soft tip-connect fade.

## EXP-20260723-001: Hanging tip + Stage0 axis stabilizer curriculum
- Run ID: `20260723-0010-hanging-tip-stabilizer-curriculum`
- Date: 2026-07-23
- Status: completed (Stage0 pass; Stage1/2 failed within budget)
- Checkpoint: `checkpoints/stage0/final_model.zip` (best)
### Question
Can Stage 0 succeed with a stable tip anchor and axis objective, and does that transfer to Stage 1/2?
### Hypothesis
Hanging tip + Stage-0 stabilizer enables Stage 0; abrupt Stage 1 removal still transfers via tip/tilt rewards.
### Result
Stage 0 **pass** (success 0.95, rot ~528°). Stage 1/2 **fail** (negative mean rotation). Artifacts under `runs/20260723-0010-hanging-tip-stabilizer-curriculum/` and `videos/stage0_hanging/`.
### Decision
revise Stage 1 assist fade (see EXP-002/003).

## EXP-20260722-001: Bottom-tip MVP curriculum (pre-hanging)
- Run ID: `20260722-bottom-tip-mvp-attempts`
- Status: failed / superseded
### Result
Bottom tip + axis objective → inverted pendulum collapse (DBG-002). Superseded by hanging tip.
### Decision
reject
## EXP-20260724-003: Stage 2 steep discrete contact reward
- Run ID: `20260724-1718-stage2-discrete-contact-seed0`
- Date: 2026-07-24
- Status: planned
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Git commit: `8e262a006c7a427034cdcc3a5715321d4400e326` (preserved baseline)
- Git branch: `main`
- Random seed: 0
- Device: CPU

### Question
Can a steep simultaneous-contact reward make the policy coordinate all three fingertips and prevent the Stage 2 axis-tilt collapse?

### Hypothesis
Replacing only the linear contact bonus with the discrete ladder `0→-10, 1→-1, 2→0.1, 3→10` will produce nonzero three-contact occupancy and fewer than 20/20 axis-tilt terminations.

### Change from Baseline
Only contact reward mapping changes. Tip joint, stabilizer 0, Stage 2 randomization, other rewards, parent checkpoint, network, optimizer, and evaluation seeds remain fixed.

### Configuration
- Algorithm: PPO
- Environment: Stage 2, tip connect solref 0.10, stabilizer 0
- Reward terms: rotation 160, tilt penalty weight 0.10, discrete contact reward
- Observation space: 48-dimensional baseline observation
- Action space: 9 normalized joint commands
- Network: policy/value 2x512
- Optimizer: Adam, resumed optimizer
- Learning rate: 1e-5
- Batch size: 128
- Horizon: 1024
- Number of environments: 1
- Training steps: 25,000
- Domain randomization: Stage 2 mass/friction
- Curriculum stage: 2
- Evaluation protocol: deterministic seeds 0–19 every 5k

### Success Criteria
Gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support: nonzero three-contact step fraction and fewer than 20/20 axis-tilt terminations. Reject reward hacking if contact improves while rotation collapses.

### Pre-experiment Diagnostic
The baseline has 0 contacts on 80.47% of steps, 1 contact on 19.53%, and never 2 or 3. Only fingertip 2 contacts (19.53% of steps); all 20 episodes end on axis tilt. This supports the single-finger/lost-contact premise.

### Result
All five periodic checkpoints and the final checkpoint had 20/20 axis-tilt terminations, zero two-contact steps, and zero three-contact steps. Final rotation was 0.92°, tip error 4.85 mm, drop 1.0, with the contact component averaging -8.147 per step. The best checkpoint by rotation reached only 1.53°.

### Key Metrics
| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 0.92° | -0.21° |
| Tip error | 4.03 mm | 4.85 mm | +0.82 mm |
| Drop rate | 1.00 | 1.00 | 0 |
| Three-contact step fraction | 0 | 0 | 0 |
| One-contact step fraction | 0.1953 | 0.2027 | +0.0074 |
| Mean contact reward | +0.050 | -8.147 | -8.197 |

### Visual Evidence
- `runs/20260724-1718-stage2-discrete-contact-seed0/plots/contact_reward_evaluation.png`
- `runs/20260724-1718-stage2-discrete-contact-seed0/videos/stage2_best_00_seed0_rot8deg.mp4`

### Interpretation
Measured fact: the steep reward was active and dominated the reward scale. Measured fact: it did not create any multi-finger contact or reduce tilt terminations. A stabilizer-0.10 control also produced no multi-finger contact despite 176.55° rotation and zero drops, so three-contact reachability/detection is not yet validated.

### Decision
reject

### Next Step
Run EXP-20260724-004 to validate per-finger and simultaneous-contact mechanical reachability before attempting another contact-reward training run.

## EXP-20260724-004: Three-fingertip contact reachability validation
- Run ID: `20260724-1730-contact-reachability`
- Date: 2026-07-24
- Status: completed (failed reachability; root cause confirmed)
- Parent or baseline run: `20260724-1718-stage2-discrete-contact-seed0`
- Git commit: `46fa9b8` at experiment start
- Git branch: `main`
- Random seed: deterministic sweep plus fixed random-search seeds
- Device: CPU

### Question
Can the current hand geometry, joint limits, and contact detector produce valid contact for each fingertip and all three simultaneously?

### Hypothesis
At least one collision-free configuration within actuator limits should register all three fingertips above 0.05 N while keeping tip error below 0.02 m and axis tilt below 0.25 rad.

### Change from Baseline
No learning and no reward change. Search controlled joint configurations/actions around the initial grasp and record per-finger force, fingertip-to-rod distance, pose, and rendered evidence.

### Success Criteria
Find and reproduce at least one configuration for each individual fingertip and at least one simultaneous three-contact configuration across three resets. If none is found after a documented bounded search, inspect collision groups/contact-force measurement and revise the grasp or target from three contacts to a mechanically supported requirement.

### Planned Evidence
- Machine-readable search results with qpos, forces, distances, and pose errors
- Images/video of best one-, two-, and three-contact candidates
- Regression test for per-finger contact detection if a detector issue is found
- Explicit go/no-go decision for another contact-reward training experiment

### Result
Across 60,000 configurations over seeds 0–2, fingers 0 and 1 each reached approximately -24 mm signed distance, but finger 2 remained 66.66–68.27 mm from the rod surface. The search found 4,166 two-contact states and zero three-contact states. Normal resets registered simultaneous 32–54 N force on fingers 0 and 1, confirming the detector works.

### Key Metrics
| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Finger 0 min distance | -23.99 mm | -23.99 mm | -23.99 mm |
| Finger 1 min distance | -24.00 mm | -24.00 mm | -24.00 mm |
| Finger 2 min distance | +66.66 mm | +68.27 mm | +67.92 mm |
| Two-contact samples | 1,437 | 1,383 | 1,346 |
| Three-contact samples | 0 | 0 | 0 |

### Visual Evidence
- `runs/20260724-1730-contact-reachability/images/contact_reachability_comparison.png`

### Interpretation
The hypothesis is rejected: three-finger contact is mechanically impossible in the current model. Finger 2 moves in an XZ plane fixed at world Y=+0.04 m while the rod lies near Y=-0.05 m. The resulting 90 mm plane separation is larger than the combined 24 mm collision radii.

### Decision
Investigate and correct the model geometry. Do not train another three-contact reward until the same test produces reproducible three-contact force.

### Next Step
EXP-20260724-005: change only finger-2 placement/orientation so its motion plane intersects the rod, then rerun the identical no-learning reachability test.

## EXP-20260724-005: Correct finger-2 contact geometry
- Run ID: `20260724-2045-finger2-spatial-dof`
- Date: 2026-07-24
- Status: completed (adopt)
- Parent or baseline run: `20260724-1730-contact-reachability`
- Device: CPU

### Question
Does aligning finger 2's motion plane with the rod make three-fingertip contact reproducibly reachable without introducing instability?

### Hypothesis
Changing only finger 2's base placement/orientation will reduce its minimum signed distance from at least +66 mm to ≤0 and produce a dynamically settled three-contact configuration on all three fixed reset seeds.

### Change from Baseline
Geometry only: adjust finger 2's base Y coordinate and/or orientation. Do not change rewards, observations, actuators, solver settings, rod geometry, or other fingers.

### Success Criteria
1. Finger 2 individual minimum signed distance ≤0 on seeds 0–2.
2. At least one three-contact configuration settles above 0.05 N on every fingertip across three resets.
3. Environment checks remain finite and stable.
4. Rendered evidence confirms genuine fingertip-rod contact rather than interpenetration or another collision artifact.

### Next Step if Successful
Establish a corrected-geometry baseline before deciding whether gradual per-finger contact shaping is still needed.

### Change Implemented
Changed only `f2_j0` axis from local Z (`0 0 1`) to local X (`1 0 0`). The two distal axes remain local Z, creating one nonparallel abduction/adduction axis followed by two flexion axes.

### Result
Finger 2 minimum signed distance improved from +66.66–68.27 mm to -14.96–15.21 mm. The 60,000-sample search found 87 geometric three-contact configurations. Exhaustive dynamic replay of those candidates found a settled three-contact grasp for every seed.

### Key Metrics
| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Three-contact samples | 28 | 33 | 26 |
| Settled contact count | 3 | 3 | 3 |
| Finger 0 force | 33.52 N | 43.58 N | 240.06 N |
| Finger 1 force | 37.17 N | 38.62 N | 228.06 N |
| Finger 2 force | 9.50 N | 7.35 N | 8.39 N |

### Visual Evidence
- `runs/20260724-2045-finger2-spatial-dof/images/spatial_finger2_three_contact.png`

### Interpretation
The geometry correction succeeds. Seed 2 has excessive force, so the result is reachability evidence rather than an acceptable initialization. The moderate-force seed-0 candidate is suitable as a reference.

### Decision
adopt

### Next Step
Run EXP-20260724-006 as a separate corrected-geometry training experiment, tracking contact occupancy, per-finger forces, excessive-force penalty, rotation, and tilt terminations.

## EXP-20260724-006: Corrected-geometry discrete-contact retraining
- Run ID: `20260724-2100-spatial-finger2-retrain-seed0`
- Date: 2026-07-24
- Status: completed (failed)
- Parent checkpoint: `runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip`
- Geometry baseline: `20260724-2045-finger2-spatial-dof`
- Random seed: 0
- Device: CPU

### Question
With finger 2 now spatially capable of contact, can the discrete contact reward produce sustained three-finger contact and reduce Stage 2 axis-tilt terminations?

### Hypothesis
The corrected geometry will yield nonzero three-contact occupancy during training/evaluation, unlike EXP-20260724-003, and reduce axis-tilt terminations below 20/20 while retaining positive rotation.

### Change from EXP-20260724-003
Only the adopted finger-2 first-joint axis differs. Reward mapping, parent checkpoint, Stage 2 environment, tip joint, stabilizer 0, randomization, network, optimizer, learning rate, checkpoint frequency, and fixed evaluation seeds remain unchanged.

### Success Criteria
Primary gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support requires nonzero three-contact step fraction and fewer than 20/20 axis-tilt terminations. Reject force exploitation if excessive-force penalty or fingertip forces dominate while rotation does not improve.

### Result
Every periodic checkpoint and the final checkpoint retained 0% two-/three-contact occupancy and 20/20 axis-tilt terminations. Final rotation was 1.44°, and the best checkpoint reached 1.53°. The contact reward averaged approximately -8.1 to -8.3 per step; force penalty remained zero because the policy never reached additional contacts.

### Key Metrics
| Metric | Zero-shot | Best | Final |
|---|---:|---:|---:|
| Rotation | 1.26° | 1.53° | 1.44° |
| Tip error | 4.55 mm | 4.11 mm | 3.98 mm |
| Drop rate | 1.00 | 1.00 | 1.00 |
| Three-contact fraction | 0 | 0 | 0 |
| Axis-tilt terminations | 20/20 | 20/20 | 20/20 |

### Visual Evidence
- `runs/20260724-2100-spatial-finger2-retrain-seed0/plots/contact_occupancy.png`
- `runs/20260724-2100-spatial-finger2-retrain-seed0/videos/stage2_best_00_seed0_rot10deg.mp4`

### Interpretation
Measured fact: finger 2 is reachable after EXP-005. Measured fact: PPO never visits any multi-contact state from the legacy reset distribution during fixed-seed evaluation. The evidence suggests an initialization/exploration failure rather than remaining geometric impossibility.

### Decision
reject checkpoints; retain corrected geometry

### Next Step
EXP-20260724-007: initialize from the verified moderate-force three-contact grasp as the only change, validate reset robustness first, then run a bounded training control if the smoke gate passes.

## EXP-20260724-007: Three-contact reset initialization
- Run ID: `planned-20260724-three-contact-reset`
- Date: 2026-07-24
- Status: planned
- Parent: `20260724-2100-spatial-finger2-retrain-seed0`
- Device: CPU

### Question
Is the corrected policy failing because the reset distribution starts outside the reachable three-contact basin?

### Hypothesis
Replacing only `_GRASP_QPOS` with the verified moderate-force seed-0 three-contact configuration will produce three-contact resets across randomized rod phases and make the discrete reward observable early enough for PPO to preserve contact.

### Change from Baseline
Reset joint configuration only. Keep corrected geometry, reward, observation, action space, Stage 2 dynamics, parent checkpoint, optimizer, and evaluation unchanged.

### Pre-training Gate
Across seeds 0–19 after settling: finite dynamics, all three forces >0.05 N in at least 90% of resets, median force below 50 N per finger, tip error <0.02 m, and no immediate tilt termination.

### Training Success Criteria
At least one checkpoint must show nonzero three-contact evaluation occupancy and fewer than 20/20 tilt terminations. Full task gate remains rotation >180°, tip error <0.02 m, drop ≤0.15.
## EXP-20260724-008: Stage 1 training with stabilizer 1.0
- Run ID: `20260724-2130-spatial-stage1-stabilizer1-train-seed0`
- Date: 2026-07-24
- Status: completed
- Parent checkpoint: `runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip`
- Geometry commit: `080e367`
- Random seed: 0
- Device: CPU

### Question
Can Stage 1 PPO training with stabilizer 1.0 preserve rotation and endpoint stability while using the spatial finger and discrete contact reward?

### Hypothesis
Strong axis assistance will prevent tilt collapse and allow a post-training checkpoint to pass rotation >180°, tip error <0.02 m, and drop ≤0.15.

### Change from Parent
Stage 1 stabilizer is 1.0, corrected spatial finger 2 is active, and the discrete contact reward is used. Tip solref 0.10, network, optimizer, and fixed evaluation seeds remain controlled.

### Success Criteria
At least one post-training checkpoint passes rotation >180°, tip error <0.02 m, and drop ≤0.15 on deterministic seeds 0–19.

### Result
Three of six post-training checkpoints passed. The selected 65,600-step checkpoint reached 269.30° rotation, 18.86 mm tip error, 0.90 success rate, and 0.05 drop. Later checkpoints rotated farther but failed the endpoint-error gate.

### Key Metrics
| Checkpoint | Rotation | Tip error | Success | Drop | Passed |
|---|---:|---:|---:|---:|---|
| Pre-training | 232.59° | 13.85 mm | 0.95 | 0.05 | yes |
| 55,600 | 217.90° | 14.8 mm | 0.80 | 0.05 | yes |
| 60,600 | 245.30° | 17.3 mm | 0.75 | 0.05 | yes |
| 65,600 | 269.30° | 18.86 mm | 0.90 | 0.05 | yes |
| Final | 317.90° | 22.6 mm | 0.65 | 0.05 | no |

### Visual Evidence
- `runs/20260724-2130-spatial-stage1-stabilizer1-train-seed0/videos/stage1_success_00_seed0_rot239deg.mp4`

### Interpretation
The stabilizer prevents tilt collapse and supports strong assisted rotation. Training trades endpoint accuracy for rotation after 65,600 steps. No checkpoint produces three-contact occupancy, so the contact-coordination objective remains unmet.

### Decision
Adopt the 65,600-step checkpoint for assisted Stage 1 demonstrations; reject later checkpoints.

### Next Step
Use the verified three-contact reset initialization if the next objective remains coordinated contact. Treat this checkpoint as assisted and do not compare it directly with stabilizer-free Stage 2.
## EXP-20260724-009: Stage 0 simultaneous-contact reward +30
- Run ID: `20260724-2200-stage0-contact30-seed0`
- Date: 2026-07-24
- Status: completed (failed)
- Parent: none; fresh PPO
- Random seed: 0
- Device: CPU

### Question
Does increasing only the three-contact reward from +10 to +30 cause a fresh Stage 0 policy to discover simultaneous three-finger contact?

### Success Criteria
Nonzero three-contact occupancy and the standard task gate: rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
No evaluated checkpoint produced a three-contact step. The best success checkpoint reached 107.9° rotation, success 0.15, and drop 0.85. The 15k checkpoint reached 226.0° but dropped in all episodes.

### Decision
reject; activate the predefined 20-step rolling contact gate in EXP-20260724-010

### Next Step
Terminate when the rolling 20-step accumulated contact reward is below +5, and require the same gate for success.
