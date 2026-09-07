# Ablation C: reduced-contact-pressure finger gait

## Question
Does reducing the complete contact reward to 10% enable coordinated,
leave-and-return finger gaiting at revolute `s=400`?

## Change from Baseline
Relative to B, only `contact_reward_scale` changes from 1.0 to 0.10. The scaled
0/1/2/3 ladder is `[-1,-0.1,+0.01,+0.03]`.

## Result
Training and loading passed; gait/task criteria failed. Fixed/unseen angle reached
533.97°/546.98° with leave-return events on every finger, but >=2-contact
fraction was only 0.604/0.578 and zero-contact gaps reached 1.76/1.68 s.

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
- Fixed/unseen metrics: `eval_fixed.json`, `eval_unseen.json`
- Enhanced comparisons: `comparisons/gait_comparison.json`
- Video: `videos/revolute_best_00_seed10009_rot770deg_tilt0deg_steps500_none.mp4`

## Conclusion
Reject. Large angle includes long under-supported phases and sustained-omega
success remains zero.

## Later Net-Angle Reevaluation
The user subsequently replaced the gait task gate with explicit positive
net-angle success. Under `success_mode=net_angle`, C passes 10/10 fixed and 10/10
unseen episodes: net angle >=pi, tilt <0.25 rad, tip error <0.02 m, finite/stable,
and no physical drop. Legacy omega-hold success remains 0/20 and the poor contact
metrics above remain unchanged diagnostics.

The accepted Phase-R force reference is 98.058 N: median total contact-frame
normal force conditioned on `axial_omega >0.5 rad/s AND contact_count >=2`.
Only 18.16% of fixed steps are eligible; the excluded 81.84% remains reported.
This later criterion permitted C to seed the mass curriculum without erasing the
historical rejection under the original support/omega criteria.

## Recommended Next Step
See `EXP-20260823-015`: progression reached accepted `s=50` and then blocked at
`s=25` conditioned-force calibration.
