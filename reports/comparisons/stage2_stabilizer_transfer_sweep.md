# Stage 2 Stabilizer Transfer Sweep

## Compared Configuration
One fixed 2x512 checkpoint (`ppo_rod_84200_steps.zip`) was evaluated under Stage 2 mass/friction randomization while only the external axis-stabilizer scale changed.

## Evaluation Protocol
Twenty deterministic episodes, seeds 0–19, active tip ball/universal joint at solref 0.10, tilt weight 0.10, recovery scale 0, and rotation scale 160.

## Metric Comparison
| Stabilizer | Rotation | Tip error | Final tilt | Drop | Axis-tilt terminations |
|---:|---:|---:|---:|---:|---:|
| 0.12 | 178.51° | 8.25 mm | 12.07° | 0.05 | 1/20 |
| 0.10 | 176.55° | 5.57 mm | 10.11° | 0.00 | 0/20 |
| 0.08 | 132.42° | 13.31 mm | 25.84° | 0.40 | 8/20 |
| 0.06 | 80.01° | 15.85 mm | 37.88° | 0.85 | 17/20 |
| 0.04 | 54.57° | 13.32 mm | 39.91° | 0.90 | 18/20 |
| 0.02 | 24.90° | 8.70 mm | 41.48° | 1.00 | 20/20 |
| 0.00 | 4.97° | 3.33 mm | 41.15° | 1.00 | 20/20 |

## Conclusion
Stage 2 randomization is not the main failure: the same checkpoint remains near the gate at stabilizer 0.12 and 0.10. The dominant nonlinear cliff begins between 0.10 and 0.08 and becomes universal axis-tilt failure by 0.02.

## Recommendation
Use 0.10 as the next curriculum stage, select checkpoints densely, and avoid direct 0.12→0 removal.
