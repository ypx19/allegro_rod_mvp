# Project State

## Current Objective
Bottom-tip mass curriculum @s=400: C0 revolute → C1–C4 hard tip-connect → **C5 free tip + tip-error reward**. Smoke stops after C5.

## Progress snapshot (2026-08-08)
| Step | Status | Notes |
|---|---|---|
| Top-hang s=400 soft-tilt C4 | **pass (VN)** | success=1.0; preferred two-finger gait |
| VecNormalize in curriculum eval | fixed | prior gates were false negatives |
| `tip_anchor=bottom` | done | relocates site/equality/hinge at runtime |
| C5 free tip + tip reward | done | `--no-tip-connect`, tip_penalty_scale=8, tip_sigma=0.015 |
| Bottom-tip s=400 smoke C0→C5 | **running** | see `runs/curricula/` |

## Active Configuration
- Tip at **bottom** (inverted pendulum); μ_cap=4; tilt_term=1.2; start **s=400**
- C5: hard equality off; tip-error reward carries fixed-tip skill
- Smoke: `--stop-after-c5` (no mass anneal until C5 OK)

## Known Problems
Historical DBG-002: bottom tip collapses as inverted pendulum at light mass — heavy s + soft tilt kill is the hypothesized fix.

## Next Recommended Experiment
If bottom C5 passes smoke, full budget + anneal free tip toward s=1.
