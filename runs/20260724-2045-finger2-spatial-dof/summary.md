# Spatial Finger-2 Geometry Validation

## Change
Changed only `f2_j0` from local Z rotation to local X rotation. The first axis is now orthogonal to the two distal Z flexion axes, giving the fingertip a spatial three-DoF chain.

## Controlled Baseline
Finger base pose, distal axes, link lengths, joint limits, actuators, solver, rod, environment, observations, and rewards were unchanged.

## Reachability Results

| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Finger 2 minimum distance before | +66.66 mm | +68.27 mm | +67.92 mm |
| Finger 2 minimum distance after | -15.20 mm | -14.96 mm | -15.21 mm |
| Geometric three-contact samples | 28 | 33 | 26 |
| Best settled contact count | 3 | 3 | 3 |
| Settled finger forces | 33.5/37.2/9.5 N | 43.6/38.6/7.3 N | 240.1/228.1/8.4 N |

## Intermediate Validation Results Preserved
The initial replay chose the deepest simultaneous overlap and retained at most two contacts. A refined shallow-candidate replay retained at most one. These were validation-selection failures, not geometry failures. Their complete outputs are preserved as:

- `preliminary_deep_candidate_*`
- `refined_single_candidate_*`

The final method dynamically replayed every geometric three-contact candidate and retained the strongest settled candidate.

## Interpretation
The axis change passes geometric and dynamic reachability on all three fixed seeds. Seed 2's candidate has excessive force and must not be used as initialization. Seed 0 provides the preferred moderate-force verified grasp.

## Visual Evidence
- `images/spatial_finger2_three_contact.png`

## Decision
Adopt the spatial first joint. Proceed to a separate retraining experiment, while monitoring force penalties and using a new run ID.

## Next Step
EXP-20260724-006: retrain Stage 2 with the corrected geometry and discrete contact reward, preserving all previous configuration choices and evaluating contact occupancy, force, rotation, tilt terminations, and reward components at every checkpoint.
