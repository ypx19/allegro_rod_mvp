# Corrected-Geometry Discrete-Contact Retraining

## Question
Does making finger 2 spatially reachable allow the steep discrete contact reward to produce three-finger coordination and reduce Stage 2 tilt failure?

## Change from Failed Baseline
Only the adopted `f2_j0` axis differs from EXP-20260724-003. Parent checkpoint, reward mapping, Stage 2 dynamics, randomization, network, optimizer, 25k budget, and evaluation seeds are identical.

## Zero-Shot Baseline
- Rotation: 1.26°
- Tip error: 4.55 mm
- Drop rate: 1.00
- Axis-tilt terminations: 20/20
- Contact occupancy: 80.18% zero, 19.82% one, 0% two/three

## Final Result
- Rotation: 1.44°
- Tip error: 3.98 mm
- Drop rate: 1.00
- Axis-tilt terminations: 20/20
- Contact occupancy: 80.55% zero, 19.45% one, 0% two/three
- Mean contact reward: -8.201 per step
- Mean force penalty: 0

## Best Checkpoint
`checkpoints/ppo_rod_60600_steps.zip` by mean rotation:
- Rotation: 1.53°
- Tip error: 4.11 mm
- Drop rate: 1.00
- Axis-tilt terminations: 20/20
- Two-/three-contact occupancy: 0%

## Interpretation
The geometry correction is necessary but not sufficient. The policy starts from the legacy two-contact initialization with spatial finger 2 open and never explores the verified three-contact region. The steep reward remains effectively sparse and averages about -8 per step.

## Decision
Reject all retrained checkpoints. Retain geometry commit `080e367`; do not revert the corrected finger.

## Artifacts
- `metrics.csv`
- `plots/contact_occupancy.png`
- `videos/stage2_best_00_seed0_rot10deg.mp4`
- `checkpoints/*_eval.json`
- `logs/train.log`

## Recommended Next Step
EXP-20260724-007: change only the reset grasp to the verified moderate-force three-contact seed-0 configuration, confirm three-contact reset across randomized rod phases, and run a short smoke test before any formal retraining. This isolates initialization/exploration from reward design.
