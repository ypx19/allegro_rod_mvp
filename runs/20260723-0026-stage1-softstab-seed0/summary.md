# Summary: 20260723-0026-stage1-softstab-seed0
## Status
rejected / interrupted

## Result
Stage1 with axis_stabilizer_scale=0.5 and tip connect OFF was numerically unstable.
- ep_len_mean collapsed to ~5
- ep_rew_mean ~-14 (immediate terminations)
- Frequent MuJoCo NaN at DOF 12
- Did not complete a meaningful 250k policy improvement under the gate metrics

## Decision
Reject this assist design. Next: soft tip-connect fade (EXP-003).
