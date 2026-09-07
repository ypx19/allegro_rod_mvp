# Run Comparison
## Compared Runs
- C `s=400`: `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Curriculum: `20260823-1855-net-angle-C-force-curriculum-seed0`

## Experimental Difference
Only revolute rod mass and explicit rod/fingertip sliding-friction scale change.
The policy is warm-started stage by stage with C's reward, contact permissions,
pose, grasp, and net-angle success definition unchanged.

## Evaluation Protocol
Ten deterministic fixed episodes (10000–10009) and ten unseen episodes
(20000–20009), 20 seconds each. Task acceptance requires per-set net-angle
success >=0.5. Force acceptance uses fixed-set median total contact-frame normal
force on `axial_omega >0.5 rad/s AND contact_count >=2` steps, within 20% of the
98.058 N C reference (`78.447–117.670 N`). Excluded timestep fractions remain
reported.

## Stage Results
- `s=400, mu=4`: success 1.0/1.0; angle 533.97°/546.98°; force 98.058 N;
  eligible fraction 0.182.
- `s=200, mu=2`: success 1.0/1.0; angle 1144.59°/1169.50°; force 98.208 N;
  eligible fraction 0.159.
- `s=100, mu=1`: success 1.0/1.0; angle 2044.64°/2196.78°; force 96.651 N;
  eligible fraction 0.118.
- `s=50, mu=0.5`: task passed; force 77.670 N, just below the band.
- `s=50, mu=0.396040`: success 1.0/1.0; angle 2828.65°/2775.10°; force
  81.298 N; eligible fraction 0.077. Accepted.
- `s=25, mu=0.25/0.168020/0.100933/0.10`: task success 1.0/1.0 throughout,
  but force 65.903/58.906/54.900/53.386 N. All rejected.

## Contact Diagnostics
Three-contact occupancy declines from 0.179/0.262 at `s=400` to 0.027/0.030 at
accepted `s=50`, and 0.020/0.020 at the final `s=25` trial. This does not enter
the authorized net-angle gate, but it shows increasingly sparse three-tip support.
No accepted or rejected stage had physical drops, numerical instability, or
meaningful revolute tip error.

## Representative Video
Force plot:
`reports/comparisons/20260823-net-angle-C-force-curriculum.png`

`runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/videos/net_angle-iter1/revolute_success_00_seed10000_rot2881deg_tilt0deg_steps500_none.mp4`

The video decodes as 501 frames at 640x480, 25 FPS, 20.04 seconds.

## Statistical Caveats
One training seed. Each friction trial retrains from the same accepted parent, so
the measured force response combines environment friction and policy adaptation.
The `s=25` trials do not prove global impossibility over all safe friction values.

## Conclusion
Net-angle task transfer succeeds through every attempted mass, but force matching
is accepted only through `s=50`. The declared calibration saturates at the safe
lower friction bound at `s=25`.

## Recommendation
Block `s=12.5` and below and all tip-connect work. A future continuation should
predeclare a local friction-response identification experiment at `s=25` rather
than assume the physical inverse-force response remains valid after retraining.
