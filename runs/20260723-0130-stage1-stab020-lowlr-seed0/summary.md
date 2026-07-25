# Run Summary

- Question: Can fine-tuning reduce the stabilizer from 0.25 to 0.20?
- Change: only stabilizer scale; tip solref remained 0.10.
- Result: periodic checkpoint passed; final checkpoint regressed.
- Best checkpoint: `checkpoints/ppo_rod_301480_steps.zip`.
- Best metrics: mean rotation 199.58°, mean tip error 16.04 mm, success rate 0.40, drop rate 0.15.
- Final metrics: 164.66°, 18.14 mm, success 0.35, drop 0.30 (rejected).
- Videos: `videos/stage1_success_00_seed0_rot180deg.mp4`, `videos/stage1_success_01_seed1_rot237deg.mp4`.
- Conclusion: adopt the periodic checkpoint only; checkpoint selection must use independent evaluation.
- Next: test stabilizer 0.18.
