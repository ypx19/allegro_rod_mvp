# Ablation A: ungated rotation credit

## Question
Does crediting genuine rotation on fewer-than-three-contact steps improve the
strict `s=400` task gate while retaining the original support termination?

## Change from Baseline
Relative to the matched 300k control, only
`rotation_requires_three_contacts=false`. Both use the same 200k warm start and
100k continuation budget.

## Result
Training passed; the task gate failed. Fixed/unseen rotation increased to
294.01°/277.39°, but all 20 episodes terminated on contact support and success
remained zero.

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
- Fixed/unseen evaluations: `eval_fixed.json`, `eval_unseen.json`
- Video: `videos/revolute_best_00_seed10002_rot359deg_tilt0deg_steps76_contact_support.mp4`

## Conclusion
Reject for task progression. The higher angle is unsupported transient rotation.

## Recommended Next Step
Run separately identified B only as a diagnostic because A remains entirely
masked by support termination.
