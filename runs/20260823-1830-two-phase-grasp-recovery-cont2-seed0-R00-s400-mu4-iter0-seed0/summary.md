# Phase R s400 final adaptive extension

## Question
Does a final 100k continuation from the best 200k policy resolve the remaining
support and sustained-omega gate failures?

## Change from Baseline
Only cumulative PPO exposure changed from 200k to 300k.

## Result
Training plumbing passed; behavior regressed. Fixed/unseen rotation was
183.18°/188.97°, violation rate 1.0/0.9, success 0/0, and total force p95
231.39/233.83 N.

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
Reject. Keep the 200k parent as the best checkpoint, but it also does not pass.

## Recommended Next Step
Stop identical training. Preserve per-step omega/contact traces for a discriminating
evaluation/reward ablation before another policy run.
