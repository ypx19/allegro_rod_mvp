# Recovered-reset Phase R smoke

## Question
Does the pose-specific companion grasp eliminate the unsupported-reset failure
and expose a nonzero supported-rotation reward?

## Change from Baseline
Only the Allegro reset/grasp companion differs from the original saved-pose smoke.

## Result
Checkpoint plumbing passed and rotation reward became nonzero, but the 2,048-step
policy rotated in the wrong direction and ended on support loss. This is a smoke,
not an accepted task result.

## Key Settings
- num_envs: 2
- device: cuda:0
- net_arch: [512, 256, 128]
- vec_normalize: True
- steps: 2048
- seed: 0

## Artifacts
- Config: `config.yaml`
- Metadata: `metadata.json`
- Metrics: `metrics.csv`
- Checkpoints: `checkpoints/`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- TensorBoard: `tb/`

## Conclusion
The reset fix is ready for a meaningful performance budget; this policy is rejected.

## Recommended Next Step
Run 100k at the same `s=400`, friction, pose, grasp, and reward.
