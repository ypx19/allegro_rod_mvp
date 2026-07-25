# Stage 1 Spatial-Finger Stabilizer-1.0 Training

## Requested Configuration
- Curriculum stage: 1
- Axis stabilizer scale: 1.0
- Tip connection: enabled, solref 0.10
- Corrected spatial finger 2
- Discrete contact reward: `0→-10, 1→-1, 2→0.1, 3→10`
- PPO adaptation: 25,000 steps, learning rate 1e-5, seed 0

## Pre-training Baseline
- Rotation: 232.59°
- Tip error: 13.85 mm
- Success rate: 0.95
- Drop rate: 0.05
- Passed gate: yes

## Best Trained Checkpoint
`checkpoints/ppo_rod_65600_steps.zip`

- Rotation: 269.30°
- Tip error: 18.86 mm
- Success rate: 0.90
- Drop rate: 0.05
- Two-contact step fraction: 0.91%
- Three-contact step fraction: 0%
- Passed gate: yes

## Checkpoint Selection
Later checkpoints reached 295–318° rotation but exceeded the 20 mm tip-error gate. The 65,600-step checkpoint is selected because it has the highest success rate among passing post-training checkpoints.

## Demo
- `videos/stage1_success_00_seed0_rot239deg.mp4`
- Demo seed: 0
- Demo rotation: 239°
- Successful episode: yes

## Interpretation
Training with stabilizer 1.0 preserves long episodes and improves mean rotation relative to the pre-training policy. It does not teach three-finger contact: no evaluation step has all three fingertips contacting. The stabilizer is doing substantial task-stabilization work, so this is an assisted Stage 1 result and not evidence of stabilizer-free Stage 2 success.

## Decision
Adopt `ppo_rod_65600_steps.zip` as the best assisted Stage 1 checkpoint for this geometry/reward combination. Retain the three-contact initialization experiment as the next contact-coordination test.
