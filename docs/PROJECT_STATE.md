# Project State

## Current Objective
Mass–friction curriculum (s=400→1 auto). Hypothesis: s=40 still too light vs finger force; start heavier so tip-connect episodes survive long enough to learn tilt rejection.

## Progress snapshot (2026-08-08)
| Step | Status | Notes |
|---|---|---|
| Env `--rod-mass-scale` + μ×s | done | then capped via `--rod-friction-cap` (default 4) |
| Tip solref ∝ 1/√s | done | stiffer tip when heavy |
| Curriculum driver | done | `scripts/run_mass_friction_curriculum.py` |
| C0/C1 s=40 soft tilt 1.2 | **fail gate** | ep_len~20–90 then collapse; videos all `axis_tilt` ~70° |
| C0/C1 s=400 soft tilt 1.2 smoke | **running** | `20260808-0124-...` |

## Qualitative preference (video inspect)
Mass-curriculum tip gaits (esp. soft-tilt **C4** `20260808-0115-...-C4-retry3`) look **closer to the desired behavior** than prior Arm A→B / B2 transfers: **two fingers cooperating** on the rod, not a one-finger thrash. Prefer this curriculum path even though s=40 did not pass gates — next lever is **heavier start (s=400)** to keep that gait surviving longer.

## Best Checkpoint
- Qualitative ref videos: `runs/20260808-0115-massfric-s40-smoke-softTilt12-C4-retry3-tip-s40-subproc8/videos/final/`
- Prior C0 s=40 (ladder obsolete): `runs/20260808-0052-massfric-s40-smoke-C0-revolute-s40-subproc8/checkpoints/final_model.zip`
- Track: `runs/curricula/<id>/CURRICULUM_PROGRESS.md`

## Active Configuration
- C0/C1 start **s=400**; anneal `s/=√2`; μ = min(s, 4)×baseline; tip solref /= √s
- Tip curriculum default `--tilt-terminate-rad 1.2` (vs old 0.7)
- DexScrew ω-hold; no adaptive reward-mass in this series

## Known Problems
s=40 tip-connect still dies on hard tilt kill under transferred revolute policy; videos show brief spin then tip-over — but gait quality is encouraging.

## Next Recommended Experiment
Smoke C0→C1 at s=400; if ep_len ≫ 40 and tilt_frac drops, run full auto ladder toward s=1.
