# Run Summary

- Question: Can fine-tuning reduce the stabilizer from 0.20 to 0.18?
- Change: only stabilizer scale; tip solref remained 0.10.
- Result: failed. Best periodic checkpoint reached 177.18°, 20.06 mm tip error and 30% drop.
- Best checkpoint within this failed run: `checkpoints/ppo_rod_326480_steps.zip`.
- Conclusion: reject. The evidence suggests a sharp assist-removal cliff.
- Next: revise the reward/curriculum rather than add steps under the same setup.
