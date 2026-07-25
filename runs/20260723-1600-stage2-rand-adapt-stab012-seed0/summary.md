# Stage 2 Randomization Adaptation at Stabilizer 0.12

## Question
Can the near-gate stabilizer-0.12 policy adapt to Stage 2 mass/friction randomization when assist strength is held fixed?

## Change from Baseline
Only Stage 2 randomization was introduced. The tip joint, stabilizer 0.12, reward settings, 2x512 network, learning rate, and fixed evaluation protocol were retained.

## Result
The untrained parent transferred almost unchanged: 178.51° rotation, 8.25 mm tip error, drop 0.05. Therefore Stage 2 randomization is not the dominant transfer failure.

The 15k adaptation run was harmful. The first 3k checkpoint fell to 173.79°, and the final checkpoint fell to 66.70°. Drop stayed at or below 0.05, indicating conservative loss of rotation rather than axis instability.

## Stabilizer Sweep
Using the unmodified parent under Stage 2 randomization:

| Stabilizer | Rotation | Tip error | Drop | Tilt terminations |
|---:|---:|---:|---:|---:|
| 0.12 | 178.51° | 8.25 mm | 0.05 | 1/20 |
| 0.10 | 176.55° | 5.57 mm | 0.00 | 0/20 |
| 0.08 | 132.42° | 13.31 mm | 0.40 | 8/20 |
| 0.06 | 80.01° | 15.85 mm | 0.85 | 17/20 |
| 0.04 | 54.57° | 13.32 mm | 0.90 | 18/20 |
| 0.02 | 24.90° | 8.70 mm | 1.00 | 20/20 |
| 0.00 | 4.97° | 3.33 mm | 1.00 | 20/20 |

## Visual Evidence
- Best periodic success: `videos/best_periodic/stage2_success_00_seed1_rot195deg.mp4`
- Final conservative failure: `videos/final/stage2_best_00_seed7_rot97deg.mp4`
- Sweep plot: `plots/stabilizer_transfer_sweep.png`

## Decision
Reject continued adaptation at 0.12 and retain the original step-84200 parent. The performance cliff is localized between stabilizer 0.10 and 0.08.

## Next Step
Run a bounded 5k adaptation at stabilizer 0.10 with 1k checkpoint selection, starting from the unmodified step-84200 parent.
