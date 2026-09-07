# Phase R s400 initial performance run

## Question
Does 100k PPO steps from the recovered saved-pose reset pass the strict Phase R gate?

## Change from Baseline
The pose-specific companion grasp replaces the incompatible default reset. Mass
400, friction scale 4.0, and the reduced +0.3 contact bonus are fixed.

## Result
Training plumbing passed; the task gate failed. Fixed/unseen rotation was
179.12°/180.49° and every episode terminated on contact support.

## Key Settings
- num_envs: 32
- device: cuda:0
- net_arch: [512, 256, 128]
- vec_normalize: True
- steps: 100000
- seed: 0

## Artifacts
- Config: `config.yaml`
- Metadata: `metadata.json`
- Metrics: `metrics.csv`
- Checkpoints: `checkpoints/`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- TensorBoard: `tb/`

## Conclusion
Reject as a task checkpoint. Rotation reward is now nonzero, confirming supported
rotation signal, but no force reference is accepted.

## Recommended Next Step
Warm-start one controlled 100k extension because rotation reached the threshold
neighborhood but support remained unresolved.
