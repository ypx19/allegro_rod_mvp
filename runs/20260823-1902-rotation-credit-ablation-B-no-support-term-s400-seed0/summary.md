# Ablation B: rotation credit without support termination

## Question
Does disabling support termination reveal sustained rotation that ablation A's
hard gate masks?

## Change from Baseline
This intentionally changes two factors from control: rotation credit is ungated
and support-gate termination is disabled. Contact occupancy, rolling gate status,
and discrete contact reward remain measured.

## Result
Training passed; original task quality failed. Fixed/unseen rotation reached
374.04°/379.20°, but three-contact occupancy collapsed to 0.288/0.335 and
sustained-omega success remained zero.

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
- Video: `videos/revolute_best_00_seed10003_rot478deg_tilt0deg_steps500_none.mp4`

## Conclusion
Reject. Full-horizon completion is caused by disabling termination and is not task success.

## Recommended Next Step
Keep the original acceptance gate. Design a continuous contact-loss penalty
ablation separately if further training is authorized.
