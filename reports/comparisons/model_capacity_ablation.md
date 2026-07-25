# Model Capacity Ablation

## Question
Is the 2x256 model too small for stabilizer fade?

## Controlled Setup
The 2x256 source policy was embedded into 2x512 with zero-initialized added units. Initial deterministic actions matched within 1.79e-7. A 2x256 reset-optimizer control matched exactly. Both trained for 25k steps at stabilizer 0.18.

## Results
| Model | Parameters | Rotation | Tip error | Drop | Gate |
|---|---:|---:|---:|---:|:---:|
| 2x256 reset control | 159,251 | 37.79° | 8.22 mm | 5% | fail |
| 2x512 | 580,627 | 126.39° | 8.22 mm | 5% | fail |
| 2x256, rot128 | 159,251 | 171.84° | 20.37 mm | 30% | fail |
| 2x512, rot128 | 580,627 | 163.43° | 8.84 mm | 5% | fail |
| 2x512, rot160, stab .18 | 580,627 | 229.57° | 14.08 mm | 10% | pass |
| 2x512, rot160, stab .15 | 580,627 | 200.00° | 12.15 mm | 10% | pass |

## Conclusion
Capacity was a real secondary limitation: the larger model learned substantially more rotation in the matched reset-optimizer comparison and was much more stable under rot128. It was not sufficient alone; reward balance also had to change.

## Caveats
All training used seed 0. Multi-seed capacity validation remains required.
