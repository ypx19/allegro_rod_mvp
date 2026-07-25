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
- Status: planned
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
