# Run Comparison
## Compared Runs
- Prior default-pose plumbing baseline: `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Saved-pose strict smoke: `20260823-1730-two-phase-force-pose-smoke-seed0`

## Experimental Difference
The candidate explicitly uses `configs/hand_poses/my_grasp.json`, starts at
revolute `s=400`, reduces the three-contact bonus from +3.0 to +0.3, and uses an
explicit rod-plus-fingertip sliding-friction multiplier of 4.0. This is not a
single-factor performance comparison; it is a correctness-gate comparison.

## Evaluation Protocol
The baseline's selected-stage metrics used five deterministic episodes under its
stage protocol. The candidate uses fixed seeds 10000–10001 and unseen seeds
20000–20001 under the newly predeclared strict gate. Therefore absolute reward
and rotation values are not treated as causal comparisons.

## Metric Comparison
- Baseline A0 three-contact occupancy: 0.984; candidate fixed/unseen: 0.000/0.000.
- Baseline A0 drop rate: 0.60; candidate fixed/unseen: 1.00/1.00.
- Candidate sustained-omega success: 0.00/0.00.
- Candidate index normal-force p95: 0.00/0.00 N.
- Candidate total normal-force p95: 49.78/48.75 N.
- Candidate unwrapped displacement: 224.09/228.66 degrees, but support-gated
  rotation reward is exactly 0.0/step and every episode ends at 25 steps.

## Training Curves
The 2,048-step candidate smoke has episode length fixed at 25 and mean return
approximately -202 to -212. This is a plumbing budget, not a learning curve.

## Representative Videos
- Candidate contact-support failure:
  `runs/20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0/videos/revolute_best_00_seed10000_rot222deg_tilt0deg_steps25_contact_support.mp4`
  (26 decoded frames, 640x480, 25 FPS).

## Success Cases
None in the candidate fixed or unseen evaluations.

## Failure Cases
All four candidate evaluation episodes lose required support. The index pad has
zero measured normal force throughout. Bottom tip-connect endpoint smokes also
have zero three-contact occupancy; the heavy endpoint terminates on rod height.

## Statistical Caveats
Only two episodes per seed set were used because this was a smoke gate. The
failure is nevertheless deterministic across both sets and agrees with direct
endpoint checks. No claim about trainability after a corrected reset is made.

## Conclusion
The candidate is invalid as a long-training initial condition. Its large hinge
displacement is unsupported transient motion, not task success. The revolute
`s=1` gate did not pass, so Phase T was correctly not started.

## Recommendation
Keep the saved palm pose unchanged and search only for a compatible reset/grasp
joint vector. Require stable three-tip contact at both mass endpoints and in both
physics modes before repeating the strict `s=400` revolute smoke.
