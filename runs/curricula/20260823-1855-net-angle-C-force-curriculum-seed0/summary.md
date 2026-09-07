# Net-angle C force-matched curriculum

## Result
Revolute task transfer passed through `s=50` and stopped at `s=25` because the
conditioned pressing-force gate could not be met within the declared calibration
budget and safe friction lower bound. No lower mass or tip-connect stage ran.

## Parent
`runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/checkpoints/final_model.zip`

## Gate
- Per-set success rate >=0.5 under `success_mode=net_angle`.
- Per episode: positive net angle >=pi, tilt <0.25 rad, tip error <0.02 m,
  finite/stable, no physical drop.
- Fixed conditioned force median within 20% of 98.058 N.
- Force condition: `axial_omega >0.5 rad/s AND contact_count >=2`.
- Contact occupancy is diagnostic only.

## Accepted Stages
- `s=400, mu=4`: 98.058 N, fixed/unseen success 1.0/1.0.
- `s=200, mu=2`: 98.208 N, success 1.0/1.0.
- `s=100, mu=1`: 96.651 N, success 1.0/1.0.
- `s=50, mu=0.396040`: 81.298 N, success 1.0/1.0.

Best accepted checkpoint:
`runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/checkpoints/final_model.zip`

## Blocking Stage
At `s=25`, `mu=0.25/0.168020/0.100933/0.10` produced fixed conditioned medians
65.903/58.906/54.900/53.386 N, all below the 78.447 N lower limit. Task success
remained 1.0/1.0, so force calibration alone blocked progression.

## Artifacts
- Full machine-readable state: `state.json`
- Comparison:
  `reports/comparisons/20260823-net-angle-C-force-curriculum.md`
- Force plot:
  `reports/comparisons/20260823-net-angle-C-force-curriculum.png`
- Accepted `s=50` video:
  `runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/videos/net_angle-iter1/revolute_success_00_seed10000_rot2881deg_tilt0deg_steps500_none.mp4`

The video was decoded as 501 frames, 640x480, 25 FPS, 20.04 seconds.
