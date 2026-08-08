# Project State

## Current Objective
Bottom-tip s=400 smoke with **mandatory three-finger contact** (dense discrete bonus + hard 1s window + no ω credit without 3 contacts).

## Progress snapshot (2026-08-08)
| Step | Status | Notes |
|---|---|---|
| Bottom revolute reach fix | done | rod spans z∈[-0.07,+0.07]; all 3 fingers can touch |
| C1 without 3-touch gate | **fail** | ep_len~34, tilt_frac=1 |
| DexScrew contact bonus bug | fixed | bonus was computed but never added |
| 3-touch required + hard gate | **running** | `20260808-0224-...-3touch-smoke` |

## Active Configuration (bottom tip)
- `contact_reward_mode=discrete`, `three_contact_reward=3.0`
- window 25 / thresh 18 (≥72% 3-contact or `contact_support` kill)
- `three_contact_required`: strip rotation reward if contacts < 3
- C0→C1 hard tip → C5 free tip; smoke stops after C5

## Next Recommended Experiment
Watch whether C0 learns sustained 3-contact before tip transfer.
