# Corrected proportional-physics curriculum

## Methodology correction

Historical EXP-015/016 force values are preserved as actual learned-policy
contact forces. They used only about 10–18% conditioned steps and are not
minimum required pressing force, so they are no longer a stage gate.

For fixed geometry and trajectory:

`tau_required = I*alpha + b*omega + tau_external`

and friction capacity is approximated by:

`tau_friction <= N*(mu_slide*r + mu_torsion)`.

If `I`, `b`, every other axial torque, `mu_slide`, and `mu_torsion` scale by
`s/400`, the common factor cancels and analytical `N_min` is constant. This is
an approximation because the MuJoCo hand/contact system includes compliant
contacts, elliptic friction coupling, fixed hand inertia/actuators, and
distributed moving contacts.

## MuJoCo friction audit

Rod and fingertip geoms all have priority 0. MuJoCo combines equal-priority
friction inputs by element-wise maximum. Its contact vector expands the three
geom inputs as `[slide, slide, torsion, rolling, rolling]`. Therefore both the
rod and all three pads are scaled in `full_vector` mode.

| s | scale | per-geom `[slide,torsion,rolling]` | effective pair five-vector |
|---:|---:|---|---|
| 400 | 4 | `[7.2,0.2,0.004]` | `[7.2,7.2,0.2,0.004,0.004]` |
| 200 | 2 | `[3.6,0.1,0.002]` | `[3.6,3.6,0.1,0.002,0.002]` |
| 100 | 1 | `[1.8,0.05,0.001]` | `[1.8,1.8,0.05,0.001,0.001]` |
| 50 | 0.5 | `[0.9,0.025,0.0005]` | `[0.9,0.9,0.025,0.0005,0.0005]` |
| 25 | 0.25 | `[0.45,0.0125,0.00025]` | `[0.45,0.45,0.0125,0.00025,0.00025]` |
| 12.5 | 0.125 | `[0.225,0.00625,0.000125]` | `[0.225,0.225,0.00625,0.000125,0.000125]` |
| 6.25 | 0.0625 | `[0.1125,0.003125,0.0000625]` | `[0.1125,0.1125,0.003125,0.0000625,0.0000625]` |
| 3.125 | 0.03125 | `[0.05625,0.0015625,0.00003125]` | `[0.05625,0.05625,0.0015625,0.00003125,0.00003125]` |
| 1.5625 | 0.015625 | `[0.028125,0.00078125,0.000015625]` | `[0.028125,0.028125,0.00078125,0.000015625,0.000015625]` |
| 1 | 0.01 | `[0.018,0.0005,0.00001]` | `[0.018,0.018,0.0005,0.00001,0.00001]` |

## Axial torque audit

Revolute hinge damping/armature at the `s=400` reference are 0.02 and 0.001;
frictionloss is zero. Corrected mode scales all three by `s/400`. The axis
stabilizer is zero and no policy-rollout external torque is applied.

For point-connect, the free-joint rotational damping/armature reference values
are 0.05 and 0.002 per DOF and are also proportional. A contact-disabled
1 rad/s axial test gives equality-constraint generalized axial torque
`3.28e-7`, `-1.65e-8`, and `-1.97e-11 N m` at `s=400,25,1`, respectively:
point-connect adds no meaningful axial resistance. This test correctly maps
free-joint generalized velocity to the rod's world axial velocity.

## Controlled preload validation

Attempt A used `tau_400=5 N m`; every declared preload lost support immediately,
so it provides only a failed upper-demand validation. Attempt B used the
separately predeclared `tau_400=0.5 N m`. The analytical estimate is
`N_min=1.838 N` at all three masses.

| s | first passing multiplier | measured median N | conclusion |
|---:|---:|---:|---|
| 400 | none through 3.0 | 24.144 N at 3.0 | lower bound only; strict tracking failed |
| 25 | none through 3.0 | 19.561 N at 3.0 | lower bound only; max omega remained 0.439 rad/s |
| 1 | 1.5 | 9.314 N | passed; 0.139 deg max error, 0.0352 rad/s max omega |

The controlled simulator does not exhibit exact invariance. Fixed contact
compliance, hand-side dynamics/actuators, contact distribution, and friction
ellipse coupling are the remaining approximation limits. These measurements do
not gate policy stages.

## Revolute results

Every stage passes fixed and unseen net-angle success at 1.0/1.0 with no
numerical instability.

| s | fixed/unseen rotation (deg) | fixed/unseen >=2-contact fraction | checkpoint |
|---:|---:|---:|---|
| 400 | 544.43 / 616.82 | 0.7746 / 0.6774 | C parent |
| 200 | 924.92 / 1098.19 | 0.5306 / 0.5110 | `R01-s200-mu2` |
| 100 | 2088.03 / 1676.95 | 0.3672 / 0.3884 | `R02-s100-mu1` |
| 50 | 4168.63 / 4134.15 | 0.1496 / 0.1586 | `R03-s50-mu0.5` |
| 25 | 5437.15 / 5151.57 | 0.0918 / 0.0912 | `R04-s25-mu0.25` |
| 12.5 | 7917.74 / 7937.44 | 0.0424 / 0.0404 | `R05-s12.5-mu0.125` |
| 6.25 | 9645.73 / 9766.73 | 0.0344 / 0.0354 | `R06-s6.25-mu0.0625` |
| 3.125 | 10255.29 / 10280.77 | 0.0278 / 0.0276 | `R07-s3.125-mu0.03125` |
| 1.5625 | 11101.50 / 11122.11 | 0.0248 / 0.0238 | `R08-s1.5625-mu0.015625` |
| 1 | 11384.68 / 11337.89 | 0.0146 / 0.0180 | `R09-s1-mu0.01` |

The net-angle gate permits Phase T, but support collapse is severe and remains
reported rather than hidden.

## Tip-connect result and blocker

Heavy tip-connect training was attempted, extended once, and retried once with
only learning rate reduced from `3e-4` to `3e-5`. The final retry has fixed and
unseen success 0.0/0.0, rotation 177.02/175.45 degrees, drop 1.0/1.0, tip error
below 4.04 mm, no numerical instability, and `axis_tilt` termination in all 20
episodes. Lower tip-connect masses were not started.

Exact blocker: uncontrolled lateral tilt after the revolute-to-point-connect
dynamics transition. It is not a learned-policy force mismatch and not
mass-independent axial resistance from the point constraint.

## Artifacts

- Curriculum state:
  `runs/curricula/20260823-2015-proportional-physics-C-seed0/state.json`
- Plot: `reports/comparisons/20260823-proportional-physics-C-curriculum.png`
- Controlled validation A:
  `runs/20260823-2010-proportional-required-force-validation/results.json`
- Controlled validation B:
  `runs/20260823-2012-proportional-required-force-validation-B/results.json`
- Accepted revolute video:
  `runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/videos/revolute_success_00_seed10000_rot11271deg_tilt0deg_steps500_none.mp4`
- Tip failure video:
  `runs/20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0/videos/tip_connect_best_00_seed10000_rot179deg_tilt86deg_steps34_axis_tilt.mp4`

Both videos were decoded from first through last frame with `imageio-ffmpeg`.
