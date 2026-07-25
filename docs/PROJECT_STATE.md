# Project State

## Current Objective
Correct finger 2's model geometry so its motion plane can intersect the rod, then revalidate three-fingertip reachability before training.

## Current Best Result
Stage 1 with solref 0.10 / stabilizer 0.15 passed using the 2x512 policy: mean rotation 200.00°, tip error 12.15 mm, success 0.50, drop 0.10.

## Best Checkpoint
- Stage 1: `runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip`
- Stage 2 baseline videos: `runs/20260723-1200-stage2-tip-joint-no-axis-stabilizer/videos/`

## Active Configuration
- Stage 0: tip connect `solref=0.008` + axis stabilizer 1.0
- Stage 1 best: tip connect on, `solref=0.10`, stabilizer 0.15, tilt weight 0.10, rotation scale 160, 2x512
- Stage 2: tip connect on (`solref=0.10`, ball/universal joint), stabilizer 0, mass/friction randomization on

## What Is Working
Stage 0; Stage 1 assist fade through stabilizer 0.15; function-preserving 2x256→2x512 expansion; explicit tip-joint/stabilizer/reward CLI parameters; dense checkpoint selection; representative videos.

## Known Problems
- Corrected Stage 2 baseline has 20/20 axis-tilt terminations, mean rotation 1.13°, mean tilt 41.32°, while endpoint error remains only 4.03 mm.
- Under Stage 2 randomization, stabilizer 0.10 retains 176.55°/5.57 mm/drop 0, but 0.08 falls to 132.42°/13.31 mm/drop 0.40.
- Continued training can regress; final checkpoint is not reliably best.

## Current Hypotheses
The dominant failure is a nonlinear curriculum cliff between stabilizer 0.10 and 0.08, not Stage 2 randomization. Stronger absolute tilt punishment and clipped local recovery shaping do not bridge it.

New ideas to test (recorded 2026-07-24, NOT yet verified):
- H-A (single-finger cause): The policy may rotate the rod using effectively one finger, which pushes the rod off-axis and induces the Stage 2 axis tilt. Proposed fix: a reward requiring all three fingertips to hold positive contact simultaneously so the fingers work together. See EXP-20260724-001. Verify the single-finger premise on existing checkpoints (per-fingertip contact logging) before training.
- H-B (stabilizer dependence / transfer value): Stage 1 uses the axis stabilizer but Stage 2 does not, so Stage 1 pretraining might teach reliance on external orientation torque and could be neutral or harmful for stabilizer-free Stage 2. Open question whether Stage 0/1 pretraining actually benefits Stage 2. Assumption only; deferred to later verification. See EXP-20260724-002.
- H-C resolved: Three-finger contact is impossible in the current geometry. Finger 2 moves at fixed world Y=+0.04 m while the rod lies near Y=-0.05 m, leaving a ≥66.66 mm surface gap. Contact detection works for fingers 0 and 1. See DBG-20260724-001 and EXP-20260724-004.

## Most Recent Experiment
EXP-20260724-004 `20260724-1730-contact-reachability`: 60,000 configurations found 4,166 two-contact states, zero three-contact states, and a best finger-2 gap of 66.66–68.27 mm. Geometry root cause confirmed.

## Next Recommended Experiment
Run EXP-20260724-005: change only finger-2 base placement/orientation so its motion plane intersects the rod, then repeat the identical reachability and dynamic-settling verification. Do not resume reward training until all three contacts are reproducible.

Queued ideas (unverified, recorded 2026-07-24):
1. Diagnose per-fingertip contact on existing checkpoints to confirm/refute the single-finger premise, then add a three-finger simultaneous-contact reward (EXP-20260724-001).
2. Compare stabilizer-pretrained vs less-/non-stabilizer Stage 2 to test whether Stage 0/1 pretraining helps Stage 2 (EXP-20260724-002).

## Blocked Items
None.

## Important Commands
```bash
source .venv/bin/activate
python scripts/eval_policy.py checkpoints/stage0/final_model.zip --stage 0
python scripts/eval_policy.py runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip --stage 2 --tip-connect --tip-connect-solref 0.10 --axis-stabilizer-scale 0 --axis-tilt-penalty-weight 0.10 --rotation-reward-scale 160
```
