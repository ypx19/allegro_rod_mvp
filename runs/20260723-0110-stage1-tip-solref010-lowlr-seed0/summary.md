# Run Summary

- Question: Can low-LR fine-tuning preserve rotation while weakening tip solref from 0.05 to 0.10?
- Change: only tip solref; stabilizer remained 0.25.
- Result: passed. Final mean rotation 223.96°, mean tip error 11.89 mm, success rate 0.55, drop rate 0.05.
- Best checkpoint: `checkpoints/final_model.zip`.
- Baseline: 211.37°, 15.36 mm, success 0.45, drop 0.05.
- Conclusion: adopt; correct low-LR resume improved rotation and endpoint control.
- Next: reduce stabilizer from 0.25 to 0.20.
