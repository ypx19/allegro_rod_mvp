# Findings

## FIND-20260824-001: DexScrew tilt-growth penalty fires during monotonic collapse but does not restore the 0.25 rad gate
- Confidence: medium
- Supporting runs: `20260824-0000-dexscrew-tilt-growth-s100-tip-seed0`; recovery sibling `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`; zero-shot parent `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Related debug issues: `DBG-20260823-006`, FIND-20260823-002, FIND-20260723-011
- Applies to: DexScrew `reward_style` on bottom tip-connect after revolute transfer, growth `g=50`, deadzone 0.05 rad, gate 0.25 rad, clip 0.05 rad, 65,536-step fine-tune
- Does not apply to: larger growth scales; two-sided Δtilt; tilt assists; longer budgets; other masses

### Finding
A one-sided tilt-growth penalty
`r_growth = -50 * clip(curr−prev, 0, 0.05) * w(curr)` with
`w=clip((θ−0.05)/0.20, 0, 1)` occupies ~3–4% of Σ|reward| during monotonic
tip-connect collapse, versus ~0.2–0.4% for recovery-only shaping. Final/max
tilt still saturates near 80°, and net-angle success stays 0. After the
fine-tune, rotation share rises (38–42%) while mean growth remains ~−0.5
versus rotation +5.2 to +6.5.

### Evidence
EXP-20260824-001: 65,536-step fine-tune from accepted revolute `s=100`.
Fixed/unseen success 0.0/0.0. Rotation 280.12°/264.34° versus zero-shot
182.31°/169.09° and recovery 190.95°/223.07°. Decode-checked failures reach
399° rotation with 85° final tilt.

### Implication
Do not further increase this growth scale as the next one-factor test, and do
not start other tip-connect masses. Growth solved the “term never fires”
problem left by recovery, but not the lateral-tilt failure.

### Caveats
One training seed and a short fine-tune. This does not rule out a tilt assist,
grasp change, or a much stronger state penalty.

## FIND-20260823-002: DexScrew one-sided tilt recovery does not fire during monotonic tip-connect collapse
- Confidence: medium
- Supporting runs: `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`; zero-shot parent `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Related debug issues: `DBG-20260823-006`, FIND-20260723-011
- Applies to: DexScrew `reward_style` on bottom tip-connect after revolute transfer, recovery `scale=50`, deadzone/clip 0.05 rad
- Does not apply to: two-sided stage-path recovery; longer budgets; other masses
  (tilt-growth at `g=50` was tested in FIND-20260824-001 and also missed the gate)

### Finding
A one-sided back-to-balance term that pays only when tilt decreases has ~0.2–0.4%
reward mass when every evaluated episode’s max tilt equals its final tilt. The
policy still terminates on ~80° lateral tilt; net-angle success stays 0.

### Evidence
EXP-20260823-018: 65,536-step fine-tune from accepted revolute `s=100`. Fixed/unseen
success 0.0/0.0 versus zero-shot 0.0/0.0. Mean recovery +0.032/+0.056 per step
versus tilt penalty −3.83/−3.34. Decode-checked failures reach 333° rotation with
81° final tilt.

### Implication
Do not further increase this recovery scale as the next one-factor test. Prefer a
signal that is nonzero while tilt is growing, or change grasp/assist dynamics.

### Caveats
One training seed and a short fine-tune. This does not rule out two-sided
recovery or a tilt curriculum.

## FIND-20260823-001: Allegro-aligned wrap reset removes the sparse three-contact initialization failure
- Confidence: high
- Supporting runs: `20260823-geometry-reachability-allegro-tip-bottom-v2`, `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Related debug issues: `DBG-20260724-002`, `DBG-20260823-001`
- Applies to: 12-DoF Allegro index/middle/thumb primitive model with ramped bottom-tip reset
- Does not apply to: the legacy 9-DoF surrogate or arbitrary rod/palm placements

### Finding
Using official Allegro joint frames with an optimized shallow-to-preload reset makes simultaneous three-fingertip support reproducible instead of exploration-sparse.

### Evidence
Both revolute and point-connect reset audits produced three contacts on 50/50 fixed seeds. The final curriculum smoke maintained three-contact occupancy of 1.0 at nominal mass and zero stabilizer torque.

### Implication
Future training should not increase contact reward to discover the third finger. Contact initialization is now solved; optimize rotation reward balance and curriculum duration instead.

### Caveats
The middle primitive pad radius is 2 mm larger than the Menagerie collision primitive so the adjacent distal box does not occlude pad contact. Revolute reset has occasional high transient index force and still needs visual artifact review.

## FIND-20260802-002: Unconstrained long PPO blows up `std` after Stage 0 success peak
- Confidence: high
- Supporting runs: `20260802-0220-exp-infra-subproc64-1e9-seed0` (EXP-20260802-002)
- Related debug issues: `DBG-20260802-001`
- Applies to: long SB3 PPO runs with default `ent_coef=0.01` and unclipped `log_std` on this Stage 0 stack
- Does not apply to: claiming Stage 0 cannot reach ~0.9 success; Arm A reward changes

### Finding
Scaling to 1e9 steps **did** produce high online Stage 0 success (~0.88 at 1e6, ~0.93 at 5e6), then policy `std` exploded (→1e18) and training crashed with NaN action means at ~3.17e7 steps. Pure “train longer” without entropy/`log_std` control is unsafe.

### Evidence
Console milestones in EXP-20260802-002; crash traceback in `logs/console.log`; best ckpt `ppo_rod_5000000_steps.zip`.

### Implication
Use dense checkpoints, early-stop/select peak policies, set `ent_coef→0` and/or clip `log_std`, and save VecNormalize every checkpoint before multi-day jobs. Then resume Arm A.

### Caveats
Single seed; online success not yet confirmed with `eval_policy.py`. Peak may shift with different hyperparams.

## FIND-20260802-001: Parallel CUDA stack works; 2e5 Stage 0 steps insufficient for success under new net
- Confidence: medium (updated by FIND-20260802-002)
- Supporting runs: `20260802-0216-exp-infra-plumbing-check-seed0`, `20260802-0217-exp-infra-subproc8-cuda-seed0` (EXP-20260802-001); follow-up `20260802-0220-exp-infra-subproc64-1e9-seed0` (EXP-20260802-002, failed after peak)
- Related debug issues: `DBG-20260802-001` (post-peak divergence)
- Applies to: DexScrew-style MuJoCo+SB3 parallel training on Stage 0 with `[512,256,128]` + VecNormalize
- Does not apply to: claiming Stage 0 is unlearnable; Arm A revolute/ω reward; legacy CPU `train.py` Stage 0 quality

### Finding
`scripts/train_parallel.py` (SubprocVecEnv + CUDA + 3-layer MLP + VecNormalize) trains and reloads. Under unchanged Stage 0 reward, **2e5** total env-steps left **`success_rate=0`**, but longer training later reached ~0.9 online success by ~5e6 (then diverged — FIND-20260802-002).

### Evidence
EXP-20260802-001 metrics/TB/load smoke; EXP-20260802-002 console milestones.

### Implication
Adopt the parallel stack. Treat short smokes as infra gates only. Prefer ~5e6–1e7 budgets with std control over unconstrained 1e9.

### Caveats
Single seed; formal 20-episode eval of the 5e6 ckpt still pending.

## FIND-20260730-001: DexScrew “good rotation” is revolute-constrained sim + real BC, not free-object RL
- Confidence: high
- Supporting runs: literature / code review of `references/dexscrew` (arXiv 2512.02011)
- Related debug issues: Stage 2 stabilizer cliff (PROJECT_STATE)
- Applies to: curriculum design for tip-connect / axis assists; interpreting external screwdriving RL claims
- Does not apply to: claiming our Stage 1 degrees equal their real progress ratios

### Finding
DexScrew trains sim rotation on a **fixed base + revolute joint** (threads omitted), distills a proprio student, then finishes screwdriving with **real tactile + history behavior cloning** (~95% progress). Direct sim2real of their rotation policy is only ~42% progress and never completes. Their published success does not imply free-object Stage 2 PPO should work.

### Evidence
Paper §§III–IV and configs in `configs/task/XHandHoraScrewDriver.yaml`; reward on nut DOF ω in `dexscrew/tasks/xhand_hora.py`. Full write-up: `reports/comparisons/dexscrew_vs_allegro_rod_mvp.md`.

### Implication
Treat tip-connect / hinge-style constraints as Stage-A skill learning (closer to DexScrew), not a temporary crutch to delete before gaits are solid. Optional ablations: true revolute rod; pose-diff/work penalties.

### Caveats
Different hand, simulator, and task; comparison is methodological, not a matched eval.

## FIND-20260724-002: Geometry reachability alone does not overcome sparse contact exploration
- Confidence: medium
- Supporting runs: `20260724-2045-finger2-spatial-dof`, `20260724-2100-spatial-finger2-retrain-seed0`
- Related debug issues: `DBG-20260724-002`
- Applies to: corrected spatial finger geometry with legacy reset and discrete -10/-1/0.1/10 contact reward
- Does not apply to: future three-contact initialization or dense approach shaping

### Finding
After finger 2 became geometrically reachable, a matched 25k retraining run still never produced a multi-contact evaluation step. Reachability was necessary but insufficient because the legacy reset and policy did not enter the three-contact basin.

### Evidence
EXP-005 dynamically verified three-contact grasps. EXP-006 evaluated zero two-/three-contact occupancy at all checkpoints, with 20/20 tilt terminations and best rotation 1.53°.

### Implication
Test reset/exploration coverage before further reward scaling. The next discriminating factor is three-contact initialization, not a larger contact bonus.

### Caveats
One training seed and a 25k adaptation budget. This does not prove that longer or from-scratch training can never discover contact.

## FIND-20260724-001: Finger 2 is kinematically excluded from rod contact
- Confidence: high
- Supporting runs: `20260724-1730-contact-reachability`
- Related debug issues: `DBG-20260724-001`
- Applies to: `models/three_finger_rod.xml` at commit `46fa9b8`
- Does not apply to: future geometry after EXP-20260724-005

### Finding
Finger 2 cannot contact the rod under the current model at any joint configuration because its planar chain remains at world Y=+0.04 m while the rod lies near Y=-0.05 m. The minimum observed surface gap is 66.66 mm, consistent with the 90 mm plane separation minus 24 mm combined collision radii.

### Evidence
A 60,000-configuration, three-seed bounded search found 4,166 two-contact configurations and no three-contact configuration. Fingers 0 and 1 each achieved approximately -24 mm signed distance and registered simultaneous 32–54 N reset forces; finger 2 remained at least 66.66–68.27 mm away.

### Implication
Do not use a three-contact reward or interpret its failure as an RL exploration problem until finger-2 geometry is corrected and reachability is reverified.

### Caveats
The conclusion is specific to the current simplified planar hand model and rod placement. Geometry changes require rerunning the reachability protocol.

## F1 — Stage 0 needs a stable tip constraint under gravity
A MuJoCo `<connect>` tip anchor is a **spherical point constraint**, not a hinge. If the tip is below the COM, the rod is an inverted pendulum and axis-tilt objectives fight gravity. Hang the tip **above** the COM (or add an axial hinge / orientation stabilizer).

## F2 — Stiff solref + high kp causes free-joint blow-ups
`solref≈0.004` with `kp≈40` produced DOF-12 NaNs. Use milder contact/equality solref (~0.008), lower kp, free-joint damping, and terminate unstable episodes.

## F3 — Stage 0 can exceed 180° with hanging tip + stabilizer
Open-loop and trained PPO both achieved large +axial rotation with tip error ≪2 cm. Best recorded Stage 0 eval: success_rate 0.95, mean rotation ~528°.

## F4 — Abrupt removal of Stage 0 assists breaks transfer
Resuming into Stage 1 (no connect, no stabilizer) yielded negative mean rotation and high drops despite Stage 0 mastery. Assists should be faded or replaced with an intermediate curriculum stage.

## F5 — Mass randomization must restore baselines
Stage 2 `body_mass *= U` without restore drifts unboundedly. Always restore baseline mass/inertia (and friction) before applying a fresh scale.

## FIND-20260723-006: Independent checkpoint selection is required during assist fade
- Confidence: high
- Supporting runs: `20260723-0130-stage1-stab020-lowlr-seed0`
- Applies to: resumed PPO assist-fade training
- Does not apply to: unevaluated algorithms

### Finding
The final checkpoint can be worse than a periodic checkpoint even with low KL and zero clip fraction.

### Evidence
At stabilizer 0.20, step 301480 passed (199.58°, 16.04 mm, drop 0.15); final failed (164.66°, 18.14 mm, drop 0.30).

### Implication
Evaluate and retain periodic checkpoints; do not automatically promote `final_model.zip`.

### Caveats
Observed on one training seed.

## FIND-20260723-007: Stabilizer removal has a sharp performance cliff
- Confidence: medium
- Supporting runs: `20260723-0130-stage1-stab020-lowlr-seed0`, `20260723-0200-stage1-stab018-lowlr-seed0`
- Related debug issues: DBG-20260723-003
- Applies to: current reward, observation and 9-action debug hand

### Finding
Scale 0.20 passes, while 0.18 fails all aggregate gates and 0.10–0.00 produces severe drop/negative rotation.

### Evidence
0.20: 199.58°, 16.04 mm, drop 0.15. 0.18 best: 177.18°, 20.06 mm, drop 0.30. Scale 0: -5.49°, drop 0.95.

### Implication
Further training should change measurement/reward of stabilizer dependence rather than repeat the same 25k fine-tune.

### Caveats
Multi-training-seed validation is not yet available.

## FIND-20260723-008: Model capacity helps but does not replace reward design
- Confidence: medium
- Supporting runs: `20260723-0610-capacity512-stab018-seed0`, `20260723-0611-capacity256-resetopt-stab018-seed0`, `20260723-0820-capacity512-stab018-tiltw010-rot160-seed0`
- Applies to: current 48-observation, 9-action PPO task

### Finding
A function-preserving 2x512 expansion adapted substantially better than a matched 2x256 reset-optimizer control, but capacity alone did not pass the task. The larger model passed only after reward balance was corrected.

### Evidence
Matched capacity result: 126.39° versus 37.79°, both at drop 0.05. The 512/rot160 candidate later passed stabilizer 0.18 at 229.57° and stabilizer 0.15 at 200.00°.

### Implication
Retain 2x512 for Stage 2, but diagnose reward components before increasing capacity again.

### Caveats
One training seed.

## FIND-20260723-009: Correct Stage 2 keeps the endpoint joint
- Confidence: high
- Supporting runs: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Applies to: intended Stage 2 mechanics

### Finding
Stage 2 keeps the point connect active as a ball/universal joint and removes only the external axis stabilizer.

### Evidence
With connect active, mean tip error was 4.03 mm while all failures were axis-tilt terminations. The earlier connect-off visual run showed 12 cm endpoint error and is retained only as a misconfigured control.

### Implication
All Stage 2 train/eval/video commands must pass `--tip-connect --axis-stabilizer-scale 0`.

## FIND-20260723-010: Increasing absolute tilt weight does not teach Stage 2 recovery
- Confidence: high
- Supporting runs: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`, `20260723-1230-stage2-tipjoint-tiltw025-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current Stage 2 PPO transfer from the stabilizer-0.15 parent

### Finding
Increasing axis-tilt penalty weight from 0.10 to 0.25 amplified the intended reward component but did not reduce axis-tilt terminations.

### Evidence
The weighted component moved from -2.03 to -5.01 per step. Both conditions ended 20/20 episodes on axis tilt, with mean final tilt near 41°.

### Implication
Do not keep increasing this absolute penalty. Test a recovery/progress-shaped signal or curriculum that supplies a local corrective learning signal.

### Caveats
One training seed and one parent checkpoint.

## FIND-20260723-011: One-step tilt-recovery shaping does not fix direct assist removal
- Confidence: medium
- Supporting runs: `20260723-1500-stage2-tilt-recovery40-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current Stage 2 PPO transfer with tip joint on, stabilizer zero, and recovery scale 40

### Finding
A clipped reward for decreasing tilt increased mean axial rotation slightly but did not improve episode survival after direct stabilizer removal.

### Evidence
Mean rotation increased from 1.13° to 4.22°, while all six checkpoints still ended 20/20 evaluation episodes on axis tilt. Final tilt stayed near 41.4° and drop stayed 1.0.

### Implication
Do not extend or merely rescale this recovery term. Test the curriculum transition and Stage 2 randomization separately.

### Caveats
One training seed and one recovery scale; this does not rule out richer observations or longer-horizon recovery objectives.

## FIND-20260723-012: Stage 2 failure is an assist cliff, not randomization transfer
- Confidence: high
- Supporting runs: `20260723-1600-stage2-rand-adapt-stab012-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current 2x512 checkpoint and Stage 2 mass/friction distribution

### Finding
The stabilizer-0.12 policy transfers to Stage 2 randomization almost unchanged, while performance collapses nonlinearly below stabilizer 0.10.

### Evidence
The fixed parent achieved 178.51° and drop 0.05 at scale 0.12, and 176.55° with drop 0 at 0.10. Scale 0.08 fell to 132.42° with drop 0.40; scale 0.02 and 0 had 20/20 tilt terminations.

### Implication
Treat 0.10→0.08 as the next curriculum transition. Do not attribute the Stage 2 zero-stabilizer failure primarily to mass/friction randomization.

### Caveats
One checkpoint and fixed evaluation seeds; the exact cliff may move with a different trained policy.
## FIND-20260724-013: Hard 20-step contact gate does not bootstrap three-contact behavior
- Confidence: medium
- Supporting runs: `20260724-2200-stage0-contact30-seed0`, `20260724-2230-stage0-contact-gate-seed0`
- Related debug issues: DBG-20260724-002
- Applies to: fresh Stage 0 PPO from the current reset distribution
- Does not apply to: contact-friendly resets or curricula with a grace period

### Finding
Requiring a +5 rolling contact-reward sum within the first 20 steps does not bootstrap simultaneous three-finger contact.

### Evidence
All five gated checkpoints had 0% three-contact and finger2 occupancy over fixed seeds 0–19. The best checkpoint had 17/20 contact-support terminations and 0% success.

### Implication
Do not tune this threshold further from the same reset distribution. Change initialization or curriculum exposure first.

### Caveats
One training seed and a 25k-step budget. The gate may be useful after the policy already has a three-contact behavior.

## FIND-20260823-014: Flip the Allegro hand at its subtree root
- Confidence: medium
- Supporting runs: `reversed_world_x_180_20260823-161731`
- Applies to: Allegro index/middle/thumb preview geometry around the current vertical rod
- Does not apply to: policy compatibility or training-reset force safety

### Finding
A 180° world-X flip about the rod-center pivot can be represented as one transform on the `palm` root; no child joint frame needs or should receive an independent rotation.

### Evidence
Both bottom-revolute and bottom-point-connect preview models compiled with unchanged palm-relative transforms and retained 3/3 geometric and force contacts after 0.2 s.

### Implication
If the reversed orientation is accepted, implement it as an explicit root-transform option and preserve the current MJCF as the default baseline.

### Caveats
This is deterministic preview evidence only. A0 lost index contact after 0.2 s at both the 10 mm-clearance placement and the later 30 mm-clearance placement with the thumb root moved 10 mm inward; C3 retained 3/3 in both. No policy was evaluated, so reset dynamics and policy transfer remain unvalidated.

## FIND-20260823-015: Palm-root pose files preserve Allegro kinematics
- Confidence: high
- Supporting runs: `20260823-hand-pose-interface-validation`
- Applies to: current Allegro revolute and point-connect task models
- Does not apply to: arbitrary MJCFs without a world-parented `palm` body

### Finding
An absolute model-frame transform applied only to the world-parented `palm` body moves the full hand rigidly while preserving all palm-relative body transforms, joint positions, and joint axes.

### Evidence
Deterministic comparisons across both physics variants found zero changed non-palm local transforms and zero changed joint positions/axes. Both retained 12-D actions and 48-D observations, and a configured reset returned finite observations.

### Implication
Future candidate hand placements should use the versioned hand-pose configuration instead of editing each finger subtree or replacing the validated MJCF defaults.

### Caveats
Preserved kinematics do not imply preserved contact quality, reset stability, or policy performance. Each newly saved pose still requires a reset/contact audit before training.

## FIND-20260823-016: Palm-pose validity does not imply grasp-reset validity
- Confidence: high
- Supporting runs: `20260823-1730-two-phase-force-pose-smoke-seed0`
- Related debug issues: `DBG-20260823-003`
- Applies to: rigid Allegro palm-root pose files combined with fixed reset/grasp joint vectors
- Does not apply to: poses with separately validated pose-specific joint resets

### Finding
A valid palm-root transform can preserve all kinematics and model interfaces while
moving a previously valid joint-space grasp outside the three-tip contact basin.

### Evidence
The newly saved, schema-valid pose produced 0.00 three-tip occupancy and 0 N index
normal-force p95 on both fixed and unseen revolute seed sets. All episodes ended
on the contact-support gate, despite finite 48-D observations and loadable PPO
checkpoints.

### Implication
Every new palm pose requires a joint-reset/contact-force audit in both physics
modes and both mass endpoints before long training. Do not infer task progress
from unwrapped rod displacement when support-gated rotation reward is zero.

### Caveats
The corrective revolute joint vector is now recorded in
`configs/hand_grasps/my_grasp_revolute_shared.json`; it is not robust for
tip-connect `s=400`.

## FIND-20260823-017: Pose-specific reset support does not ensure policy support
- Confidence: high
- Supporting runs: `20260823-1740-my-grasp-shared-reset-audit-seed0`, `20260823-1820-two-phase-grasp-recovery-cont-seed0`, `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Related debug issues: `DBG-20260823-003`, `DBG-20260823-004`
- Applies to: saved-pose revolute `s=400` PPO with the recovered companion grasp
- Does not apply to: alternative optimizers, rewards, or success thresholds

### Finding
A reset that retains 3/3 contacts under zero action across fixed noise seeds can
still produce a policy that violates support after learning useful rotation.

### Evidence
The reset passed 10/10 `s=400` seeds for 100 steps. The 200k policy rotated
200.56°/192.06° on fixed/unseen sets but violated support in 20%/50% of episodes.
At 300k, violation rose to 100%/90% while total force p95 rose from about
164–169 N to 231–234 N.

### Implication
Keep reset robustness and policy robustness as separate gates. Select checkpoints
using termination and force metrics in addition to total rotation, and do not
accept the reset's static force as the rotating-policy force reference.

### Caveats
One training seed and one PPO lineage. The evidence does not distinguish reward
misspecification from optimizer instability.

## FIND-20260823-018: Ungated rotation credit exposes unsupported rotation
- Confidence: medium
- Supporting runs: `20260823-1830-two-phase-grasp-recovery-cont2-seed0`, `20260823-1858-rotation-credit-ablation-A-s400-seed0`, `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`
- Related debug issues: `DBG-20260823-004`
- Applies to: current saved-pose revolute `s=400` PPO lineage
- Does not apply to: policies trained with a stronger continuous contact-loss objective

### Finding
Zeroing rotation reward below three contacts suppresses learned transient angle,
but removing that rule does not create sustained supported manipulation. If hard
support termination is also removed, the policy learns a high-angle,
low-three-contact behavior.

### Evidence
Against a matched continuation from the same checkpoint, A raises fixed/unseen
angle from 183.18°/188.97° to 294.01°/277.39° but retains 20/20 support
terminations. B reaches 374.04°/379.20° while three-contact occupancy falls to
0.288/0.335. All sustained-omega success rates are zero.

### Implication
Do not treat higher unwrapped angle or return from ungated credit as task
improvement. Retain the original acceptance gate and pair rotation credit with an
explicit continuous support objective before considering curriculum progression.

### Caveats
One training seed. B changes two factors and is diagnostic rather than a clean
single-factor causal estimate.

## FIND-20260823-019: Contact scaling alone does not coordinate finger gait
- Confidence: medium
- Supporting runs: `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`, `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Related debug issues: `DBG-20260823-004`
- Applies to: current revolute `s=400` policy lineage with ungated rotation credit and no support termination
- Does not apply to: explicit gait-phase observations or leave-return objectives

### Finding
Reducing the full contact reward ladder to 10% increases finger leave/recontact
and net angle, but worsens support and does not create sustained axial rotation.

### Evidence
C records mean leave-return events on all three fingers and increases fixed/unseen
angle to 533.97°/546.98°. Relative to B, three-contact occupancy falls from
0.288/0.335 to 0.179/0.262, >=2-contact fraction falls to 0.604/0.578, and
zero-contact gaps reach 1.76/1.68 s. Sustained-omega success remains zero.

### Implication
Do not continue tuning contact reward scale alone. Continuous gait likely needs
phase information or an objective for timely recontact.

### Caveats
One scale and one training seed. Large angle is not ballistic-free success.

## FIND-20260823-020: Friction-to-force response changes under policy transfer
- Confidence: medium
- Supporting runs: `20260823-1855-net-angle-C-force-curriculum-seed0`
- Related debug issues: `DBG-20260823-005`
- Applies to: C-policy revolute mass transfer with per-friction PPO retraining
- Does not apply to: fixed open-loop actions or untrained contact statics

### Finding
The physical `mu proportional to mass` rule is a useful initial estimate but is
not a reliable closed-loop force-calibration direction after the policy retrains.

### Evidence
The physical estimates match the 98.058 N reference at `s=200` (98.208 N) and
`s=100` (96.651 N). At `s=50`, lowering `mu` from 0.5 to 0.396 raises force from
77.670 to 81.298 N. At `s=25`, however, lowering `mu` from 0.25 through 0.168,
0.101, and 0.10 lowers force from 65.903 to 53.386 N despite task success 1.0.
A separately predeclared extension gives 54.669 N at 0.05 and 49.427 N at 0.025.
The sequence from 0.10 downward is non-monotone and remains far below target;
>=2-contact occupancy drops to 0.157 at scale 0.025.

### Implication
Use physical scaling only to seed calibration. Before repeated adaptation,
identify the local measured response under a fixed parent and predeclare how
non-monotone or policy-dependent responses will be handled.
Do not lower friction further for this `s=25` lineage: friction-only reduction is
insufficient under the tested transfer protocol.

### Caveats
One PPO seed. This establishes failure of the declared decreasing-friction path
through scale 0.025, not a universal claim about every possible policy or force
controller.

## FIND-20260823-021: Learned-policy force cannot gate required force
- Confidence: high
- Supporting runs: `20260823-1855-net-angle-C-force-curriculum-seed0`, `20260823-2015-proportional-physics-C-seed0`
- Related debug issues: `DBG-20260823-005`

### Finding
Conditioned normal force from an adapting policy is not an estimator of minimum
required pressing force. The historical conditioning retained only about
10–18% of steps and changed trajectory/contact geometry across stages.

### Evidence
Removing this invalid gate while scaling the full friction vector and explicit
rod passive dynamics allowed every revolute stage from `s=400` through `s=1` to
pass fixed and unseen net-angle success at 1.0.

### Implication
Keep policy contact force as a diagnostic. Use a fixed controlled preload/torque
protocol for required-force questions.

### Caveats
The controlled sweep did not demonstrate exact invariance: its strict tracking
criterion found no passing preload through multiplier 3 at `s=400` or `s=25`,
while `s=1` first passed at multiplier 1.5 (9.31 N median). Contact compliance,
hand dynamics, and friction-cone coupling remain mass-dependent limitations.

## FIND-20260823-022: Revolute net-angle transfer reaches s1 but support collapses
- Confidence: high
- Supporting runs: `20260823-2015-proportional-physics-C-seed0`
- Related debug issues: `DBG-20260823-006`

### Finding
The declared net-angle gate accepts all corrected revolute stages, but this does
not imply supported finger gaiting. Fixed >=2-contact fraction falls from 0.775
at `s=400` to 0.0146 at `s=1` while net angle rises from 544 to 11,385 degrees.

### Implication
The result permits the requested phase transition under the declared gate, but
contact quality must remain visible and the behavior should not be described as
supported manipulation.

### Caveats
Revolute mounting prevents physical rod drop. The same lineage fails immediately
after transfer to tip-connect, where lateral tilt and physical drop are possible.
