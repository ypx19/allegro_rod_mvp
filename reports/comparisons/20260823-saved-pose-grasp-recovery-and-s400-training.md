# Run Comparison
## Compared Runs
- Reset failure: `20260823-1730-two-phase-force-pose-smoke-seed0`
- Reset audit: `20260823-1740-my-grasp-shared-reset-audit-seed0`
- 100k: `20260823-1810-two-phase-grasp-recovery-seed0`
- 200k: `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- 300k: `20260823-1830-two-phase-grasp-recovery-cont2-seed0`

## Experimental Difference
The palm pose, `s=400`, explicit friction scale 4.0, reward, and evaluation
protocol remain fixed. The reset audit changes only Allegro joint reset/preload.
The policy comparison then changes only cumulative PPO exposure.

## Evaluation Protocol
Reset: seeds 0–9, 100 zero-action steps. Policy: deterministic fixed seeds
10000–10009 and unseen seeds 20000–20009, 20-second episodes.

## Metric Comparison
- Old reset: 0 three-tip occupancy and 100% support termination.
- Recovered revolute reset: 10/10 passes at both `s=400` and `s=1`.
- 100k policy: fixed/unseen rotation 179.12°/180.49°; violations 1.0/1.0.
- 200k policy: 200.56°/192.06°; violations 0.20/0.50.
- 300k policy: 183.18°/188.97°; violations 1.0/0.9.
- Sustained-omega success remains 0 at every training budget.

## Training Curves
`reports/comparisons/20260823-grasp-recovery-s400-training.png`

## Representative Videos
`runs/20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0/videos/revolute_best_00_seed10000_rot211deg_tilt0deg_steps500_none.mp4`

## Success Cases
The representative 200k episode completes 500 policy steps without early
termination and rotates 211.2°. It does not satisfy the 10-second omega hold.

## Failure Cases
At 200k, 2/10 fixed and 5/10 unseen episodes terminate early. At 300k this
regresses to 10/10 and 9/10, respectively.

## Statistical Caveats
Only one training seed was used. Reset robustness uses ten perturbation seeds;
policy conclusions need multiple training seeds after the failure mechanism is
better isolated.

## Conclusion
The companion grasp resolves the revolute reset bug, but identical PPO
continuation does not pass the first task gate. The 200k checkpoint is the best
of this lineage; 300k is a negative result.

## Recommendation
Do not anneal mass or start tip-connect. Preserve per-step omega/contact traces
from 100k and 200k and use them for the smallest reward-versus-evaluation
discrimination before another training run.
