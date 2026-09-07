# EXP-infra: Parallel MuJoCo + CUDA PPO

## Question
Can a 10x lower PPO learning rate recover the heavy tip-connect `s=400`
transition after the original transfer showed excessive KL and zero success?

## Change from Baseline
Only the training stack changed relative to `scripts/train.py` (DummyVecEnv, CPU, [256,256]). Stage 0 reward and physics are unchanged.

## Result
Training and checkpoint loading are finite, but the task gate fails. Fixed/unseen
net-angle success is 0.0/0.0; rotation is 177.02°/175.45°; every episode
terminates on `axis_tilt`; drop rate is 1.0/1.0. Tip error remains below
4.04 mm and numerical-instability episodes are zero.

## Key Settings
- num_envs: 32
- device: cuda:1
- net_arch: [512, 256, 128]
- vec_normalize: True
- steps: 100000
- seed: 0
- learning_rate: 3e-5
- full contact friction vector: [7.2, 0.2, 0.004]
- effective pair vector: [7.2, 7.2, 0.2, 0.004, 0.004]

## Artifacts
- Config: `config.yaml`
- Metadata: `metadata.json`
- Metrics: `metrics.csv`
- Checkpoints: `checkpoints/`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- TensorBoard: `tb/`
- Fixed evaluation: `eval_fixed.json`
- Unseen evaluation: `eval_unseen.json`
- Failure video: `videos/tip_connect_best_00_seed10000_rot179deg_tilt86deg_steps34_axis_tilt.mp4`

## Conclusion
Reject this checkpoint. Lower learning rate fixes PPO KL (about 0.026–0.029)
but not lateral tilt. Phase T stops at `s=400`.

## Recommended Next Step
Run no lower mass. Isolate tip-connect tilt with a controlled grasp or tilt
curriculum experiment.
