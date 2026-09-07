# EXP-infra: Parallel MuJoCo + CUDA PPO

## Question
Does SubprocVecEnv + CUDA + net_arch [512,256,128] + VecNormalize train Stage 0 without NaNs and produce a loadable checkpoint?

## Change from Baseline
Only the training stack changed relative to `scripts/train.py` (DummyVecEnv, CPU, [256,256]). Stage 0 reward and physics are unchanged.

## Result
passed. Wrote 12 metrics rows; final checkpoint at /data/ypx/allegro_rod_mvp/runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/checkpoints/final_model.zip Load smoke passed.

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
Adopt parallel training plumbing for subsequent DexScrew-style EXPs.

## Recommended Next Step
EXP-A0: revolute MJCF + DexScrew-style ω reward core, n_envs=8, 2e5 smoke.
