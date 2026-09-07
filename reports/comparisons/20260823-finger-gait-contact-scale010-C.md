# Run Comparison
## Compared Runs
- Strict best: `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- A, ungated rotation credit: `20260823-1858-rotation-credit-ablation-A-s400-seed0`
- B, no support termination: `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`
- C, scaled contact reward: `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`

All trainable ablations use the same 200k parent, 100k budget, seed, pose, grasp,
revolute `s=400`, friction scale 4.0, network, optimizer, and normalization state.

## Experimental Difference
C differs from B only by `contact_reward_scale: 1.0 -> 0.10`.
The raw discrete ladder `[-10,-1,+0.1,+0.3]` becomes
`[-1,-0.1,+0.01,+0.03]` for contact counts 0/1/2/3.

## Evaluation Protocol
Ten deterministic fixed episodes (10000–10009) and ten unseen episodes
(20000–20009), 20 seconds each. Gait metrics include net and cumulative absolute
rotation, completed cycles, per-tip leave-return events and longest contact loss,
fractions with at least one/two contacts, longest zero-contact interval, rolling
window occupancy, omega hold, force, constraint error, and numerical instability.

## Metric Comparison
Fixed / unseen:
- Strict: rotation 200.56°/192.06°, >=2-contact fraction 1.000/1.000,
  3-contact occupancy 0.975/0.960, success 0/0.
- A: 294.01°/277.39°, >=2 fraction 0.980/0.987, occupancy 0.878/0.871,
  all episodes terminate on support.
- B: 374.04°/379.20°, >=2 fraction 0.912/0.894, occupancy 0.288/0.335,
  cumulative absolute rotation 921.70°/878.76°.
- C: 533.97°/546.98°, >=2 fraction 0.604/0.578, occupancy 0.179/0.262,
  cumulative absolute rotation 1040.33°/982.70°.

C per-tip mean leave-return events:
- Fixed: `[8.3, 4.1, 2.8]`
- Unseen: `[7.9, 5.7, 2.9]`

C complete zero-contact maximum is 1.76/1.68 s; mean completed net cycles is
1.1/1.1, with only one episode per set reaching two cycles. Maximum mean omega
hold is 1.20/1.19 s and success remains zero. Force p95 is 158.40/167.61 N.
Maximum tip/constraint error remains below `5e-17 m`; numerical instability is zero.

Machine-readable metrics:
`runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/comparisons/gait_comparison.json`

## Training Curves
`reports/comparisons/20260823-finger-gait-contact-scale010-C.png`

## Representative Videos
`runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/videos/revolute_best_00_seed10009_rot770deg_tilt0deg_steps500_none.mp4`

The 501-frame H.264 video decodes through its final frame at 640x480, 25 FPS.

## Success Cases
C demonstrates repeated leave-and-return events on all three fingers and more
than one mean net cycle without numerical or constraint failure.

## Failure Cases
C fails the predeclared gait criteria: >=2-contact fraction is below 0.70,
zero-contact intervals exceed 0.5 s, and three-contact occupancy is worse than B.
It also fails the original ten-second omega criterion.

## Statistical Caveats
One training seed and one predeclared reward scale. Full-horizon completion is not
task success because support termination is intentionally disabled.

## Conclusion
Reducing the contact ladder enables more finger motion and larger net angle, but
does not produce coordinated continuous gaiting. The policy spends too much time
with one or zero contacts and does not sustain axial speed.

## Recommendation
Stop scale tuning. The next `s=400`-only experiment should add an explicit
leave-and-return phase signal or observation, rather than further weakening
contact pressure. Keep mass and tip-connect progression paused.

## Later Success-Definition Update
After this report, the user explicitly replaced the gait task gate with positive
net-angle success: net angle >=pi, tilt <0.25 rad, tip error <0.02 m, stable and
no physical drop. Under that definition C passes 1.0/1.0 fixed/unseen while
legacy omega success remains 0.0/0.0. Contact diagnostics above are preserved but
are no longer gates. See
`reports/comparisons/20260823-net-angle-C-force-curriculum.md`.
