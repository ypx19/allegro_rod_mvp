# Run Comparison
## Compared Runs
- Starting best checkpoint (200k): `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- Matched control (+100k, credit gated): `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Ablation A (+100k, credit ungated): `20260823-1858-rotation-credit-ablation-A-s400-seed0`
- Ablation B (+100k, credit and support termination ungated):
  `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`

All three continuation runs warm-start from the same 200k model and
VecNormalize state. They use identical seed, budget, pose, grasp, mass, friction,
reward weights, optimizer, network, and evaluation sets.

## Experimental Difference
- Control: `rotation_requires_three_contacts=true`,
  `contact_support_termination_enabled=true`.
- A: only `rotation_requires_three_contacts=false`.
- B: A plus `contact_support_termination_enabled=false`. The 25-step/18-hit
  gate is still measured, and the discrete contact reward remains active.

## Evaluation Protocol
Ten fixed episodes beginning at seed 10000 and ten unseen episodes beginning at
seed 20000; deterministic policy; 20 seconds maximum per episode.

## Metric Comparison
Fixed / unseen:
- Starting 200k: rotation 200.56°/192.06°, violations 0.20/0.50,
  success 0/0.
- Matched control: rotation 183.18°/188.97°, 3-contact occupancy
  0.975/0.975, violations 1.0/0.9, force p95 231.39/233.83 N.
- A: rotation 294.01°/277.39°, occupancy 0.878/0.871, violations 1.0/1.0,
  force p95 184.22/191.48 N.
- B: rotation 374.04°/379.20°, occupancy 0.288/0.335, original-gate quality
  failure on every seed, force p95 230.82/217.18 N.
- All runs: sustained-omega success 0. Maximum mean hold remained below 1.02 s.

Machine-readable metrics:
`reports/comparisons/20260823-rotation-credit-ablation-metrics.json`

## Training Curves
`reports/comparisons/20260823-rotation-credit-ablation.png`

## Representative Videos
- A support failure:
  `runs/20260823-1858-rotation-credit-ablation-A-s400-seed0/videos/revolute_best_00_seed10002_rot359deg_tilt0deg_steps76_contact_support.mp4`
- B unsupported full episode:
  `runs/20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0/videos/revolute_best_00_seed10003_rot478deg_tilt0deg_steps500_none.mp4`

Both H.264 files were decoded through their final frame at 640x480 and 25 FPS.

## Success Cases
None under the original task-quality gate.

## Failure Cases
A receives more rotation credit and accumulates more angle but still loses the
rolling support gate in 20/20 episodes. B completes the time horizon only because
that termination is disabled; its 3-contact occupancy collapses below 0.34 and it
still never sustains target omega.

## Statistical Caveats
One training seed. A isolates one factor against the matched control. B is a
two-factor diagnostic and must not be interpreted as the same ablation.

## Conclusion
The reward-credit rule suppresses measurable rotation learning, but removing it
alone does not solve contact retention or sustained rotation. Removing hard
termination reveals a strongly unsupported high-angle behavior rather than task
success.

## Recommendation
Keep the original gate for task acceptance. The next smallest experiment is a
reward-component ablation that penalizes contact loss continuously while keeping
rotation credit ungated and hard termination enabled; predeclare its scale before
training and compare from the same 200k parent.
