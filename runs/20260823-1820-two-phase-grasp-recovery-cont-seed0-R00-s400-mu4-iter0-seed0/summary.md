# Phase R s400 adaptive continuation

## Question
Does another 100k steps from the near-threshold 100k policy pass the strict
saved-pose revolute `s=400` gate?

## Change from Baseline
Only cumulative training exposure changed. The fixed palm pose, companion grasp,
mass 400, friction scale 4.0, +0.3 contact reward, and evaluation seeds remained fixed.

## Result
Training and checkpoint reload passed, but the task gate failed. Fixed/unseen
rotation reached 200.56°/192.06°; violations were 0.20/0.50 and sustained-omega
success was 0/0. This is the best checkpoint in the lineage, not an accepted stage.

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
- Video: `videos/revolute_best_00_seed10000_rot211deg_tilt0deg_steps500_none.mp4`
- Fixed evaluation:
  `../20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0-cal-fixed/`
- Unseen evaluation:
  `../20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0-cal-unseen/`

## Conclusion
Reject as a Phase R stage. Do not record its 163.98 N fixed total-force p95 as an
accepted pressing-force reference.

## Recommended Next Step
Compare per-step omega/contact traces against the 100k checkpoint. The subsequent
300k checkpoint regressed, so do not continue identical training.
