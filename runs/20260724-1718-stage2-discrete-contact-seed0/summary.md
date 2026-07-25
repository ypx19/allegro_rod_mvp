# Experiment Summary

## Question
Can a steep simultaneous-contact reward (`0→-10, 1→-1, 2→0.1, 3→10`) make the Stage 2 policy coordinate all three fingertips and avoid axis tilt?

## Change from Baseline
Only the contact reward mapping changed from the legacy linear mapping. The parent checkpoint, Stage 2 dynamics, tip joint, stabilizer 0, all other reward terms, optimizer, and evaluation seeds remained fixed.

## Final Metrics
- Mean axis rotation: 0.92° (baseline 1.13°)
- Mean tip error: 4.85 mm (baseline 4.03 mm)
- Drop rate: 1.00 (baseline 1.00)
- Axis-tilt terminations: 20/20 (baseline 20/20)
- Contact occupancy: 79.73% zero, 20.27% one, 0% two, 0% three
- Mean discrete contact reward: -8.147 per step

## Best Checkpoint
`checkpoints/ppo_rod_55600_steps.zip` by mean rotation:
- Rotation: 1.53°
- Tip error: 4.62 mm
- Drop rate: 1.00
- Axis-tilt terminations: 20/20
- Contact occupancy: 81.86% zero, 18.14% one, 0% two, 0% three

## Artifacts
- Machine-readable comparison: `metrics.csv`
- Contact comparison plot: `plots/contact_reward_evaluation.png`
- Representative failure: `videos/stage2_best_00_seed0_rot8deg.mp4`
- Fixed-seed checkpoint evaluations: `checkpoints/*_eval.json`
- Training log: `logs/train.log`

## Important Observations
The contact reward dominated at roughly -8 per step but never produced a two- or three-finger contact. Training success fell to zero and explained variance remained negative. A separate stabilizer-0.10 control also had 0% two-/three-finger contact despite 176.55° rotation and zero drops.

## Failure Mode
Reward specification failure / exploration failure, with a possible contact-measurement or mechanical-reachability issue.

## Conclusion
Reject this reward mapping. The evidence contradicts the hypothesis that a steep terminal contact ladder alone can elicit coordinated contact from the current policy and geometry.

## Recommended Next Step
Before more training, verify that each fingertip and all three simultaneously can register rod contact under the current model. Use controlled joint configurations/action search, record per-finger force and geometry distance, and visually verify the best configurations. If three contacts are unreachable or the detector is wrong, fix that correctness issue first; otherwise design a gradual per-finger contact-acquisition curriculum.
