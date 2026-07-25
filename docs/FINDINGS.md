# Findings

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
