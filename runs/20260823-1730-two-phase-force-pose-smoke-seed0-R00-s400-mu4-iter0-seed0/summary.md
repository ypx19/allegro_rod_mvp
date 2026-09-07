# Strict Phase R `s=400` Saved-Pose Smoke

## Question
Does the newly saved palm pose provide a supported rotating state from which the
strict revolute mass/friction curriculum can begin?

## Change from Baseline
- Required pose `configs/hand_poses/my_grasp.json` (SHA-256 `2d8ac7f...e7c9f`).
- Revolute mass/inertia scale 400; rod and all pad sliding-friction scale 4.
- Three-contact bonus +0.3 with the unchanged binary support/rotation gates.

## Result
Training plumbing and checkpoint reload passed, but the task gate failed on both
fixed and unseen seeds. Success was 0, three-tip occupancy 0, and violation rate
1.0. No force reference was accepted.

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
- Fixed calibration: `../20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0-cal-fixed/`
- Unseen calibration: `../20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0-cal-unseen/`
- Representative failure: `videos/revolute_best_00_seed10000_rot222deg_tilt0deg_steps25_contact_support.mp4`
- Endpoint gate plot: `plots/smoke_gate_metrics.png`

## Evaluation Metrics
- fixed: 224.09 deg displacement, 0 rotation reward, 0.00 support occupancy,
  1.00 violation rate, total force p95 49.78 N, index p95 0 N;
- unseen: 228.66 deg displacement, 0 rotation reward, 0.00 support occupancy,
  1.00 violation rate, total force p95 48.75 N, index p95 0 N.

## Conclusion
Reject this pose/reset combination for long training. Unsupported hinge motion is
not task success, and the force statistic cannot become a curriculum reference.

## Recommended Next Step
Find a pose-specific grasp joint reset with stable 3/3 contact in both physics
modes and both mass endpoints, then repeat only this first strict stage.
