# EXP-infra: Parallel MuJoCo + CUDA PPO

## Question
Does SubprocVecEnv + CUDA + net_arch [512,256,128] + VecNormalize train Stage 0 without NaNs and produce a loadable checkpoint?

## Change from Baseline
Only the training stack changed relative to `scripts/train.py` (DummyVecEnv, CPU, [256,256]). Stage 0 reward and physics are unchanged.

## Result
Training and load smoke passed. Fixed/unseen net-angle success is 1.0/1.0, but
the fixed conditioned force median is 54.669 N, below the 78.447 N gate.

## Key Settings
- num_envs: 32
- device: cuda:1
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
Reject for curriculum progression. Scale 0.05 does not restore the conditioned
pressing-force target.

## Evaluation
- Fixed/unseen angle: 3026.01°/2802.79°.
- Conditioned force median: 54.669/54.972 N.
- Conditioned force p95: 100.573/105.380 N.
- Eligible fraction: 0.176/0.170.
- Contact distribution fixed: `[0.1786,0.5614,0.2368,0.0232]`.
- Contact distribution unseen: `[0.1790,0.5552,0.2390,0.0268]`.
- Axial-slip proxy p95: `1.886e-15/1.850e-15 m/s`.
- Tip error max below `6.6e-17 m`; numerical instability: 0/0.

## Recommended Next Step
Evaluate only the separately predeclared scale 0.025; do not add scales.
