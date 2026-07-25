# Stage 0 Three-Contact Reward +30

## Change
Fresh Stage 0 PPO training with spatial finger 2 and the discrete contact mapping `0→-10, 1→-1, 2→0.1, 3→30`. No rolling contact gate was active.

## Result
Increasing the simultaneous-contact reward did not produce a three-contact evaluation step.

- Best task-success checkpoint: `ppo_rod_5000_steps.zip`
- Best success rate: 0.15
- Rotation: 107.9°
- Drop rate: 0.85
- Three-contact occupancy: 0%
- Two-contact occupancy: 16.3%

The 15,000-step checkpoint reached 226.0° mean rotation but dropped in 20/20 episodes. All checkpoints had 0% three-contact occupancy.

## Interpretation
The larger terminal reward is insufficient by itself. Training return increased while task success remained poor, and axis-tilt termination dominated.

## Artifacts
- `metrics.csv`
- `videos/stage0_success_00_seed2_rot222deg.mp4`
- `checkpoints/*_eval.json`
- `logs/train.log`

## Decision
Reject the run as a task solution. Use it as the ungated baseline for the requested rolling contact-support termination experiment.
