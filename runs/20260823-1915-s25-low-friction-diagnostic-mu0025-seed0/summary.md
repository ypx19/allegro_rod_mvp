# EXP-infra: Parallel MuJoCo + CUDA PPO

## Question
Does SubprocVecEnv + CUDA + net_arch [512,256,128] + VecNormalize train Stage 0 without NaNs and produce a loadable checkpoint?

## Change from Baseline
Only the training stack changed relative to `scripts/train.py` (DummyVecEnv, CPU, [256,256]). Stage 0 reward and physics are unchanged.

## Result
Training and load smoke passed. Fixed/unseen net-angle success is 1.0/1.0, but
the fixed conditioned force median is 49.427 N, below the 78.447 N gate.

## Key Settings
- num_envs: 32
- device: cuda:4
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
Reject for curriculum progression. Scale 0.025 further reduces conditioned force
and support; friction-only adaptation is insufficient in the declared extension.

## Evaluation
- Fixed/unseen angle: 3073.89°/3191.96°.
- Conditioned force median: 49.427/47.272 N.
- Conditioned force p95: 95.461/80.244 N.
- Eligible fraction: 0.100/0.105.
- Contact distribution fixed: `[0.2710,0.5724,0.1432,0.0134]`.
- Contact distribution unseen: `[0.2820,0.5660,0.1404,0.0116]`.
- Axial-slip proxy p95: `1.925e-15/1.928e-15 m/s`.
- Tip error max below `5.1e-17 m`; numerical instability: 0/0.

## Recommended Next Step
Stop at `s=25`; do not run lower mass or tip-connect.
