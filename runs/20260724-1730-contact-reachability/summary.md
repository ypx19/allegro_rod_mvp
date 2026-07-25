# Contact Reachability Validation

## Question
Can each fingertip—and all three simultaneously—contact the rod under the current model geometry and joint limits?

## Method
No learning or reward change. For each of seeds 0, 1, and 2:

- reset the Stage 2 environment with tip connect 0.10 and stabilizer 0.10;
- hold the rod pose fixed;
- uniformly sample 20,000 configurations across all declared joint limits;
- compute exact signed MuJoCo capsule-to-capsule distances;
- replay the closest simultaneous candidate for 100 dynamics steps;
- retain the normal reset and closest finger-2 pose as visual evidence.

## Results

| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Finger 0 minimum distance | -23.99 mm | -23.99 mm | -23.99 mm |
| Finger 1 minimum distance | -24.00 mm | -24.00 mm | -24.00 mm |
| Finger 2 minimum distance | +66.66 mm | +68.27 mm | +67.92 mm |
| Three-contact samples | 0/20,000 | 0/20,000 | 0/20,000 |
| Two-contact samples | 1,437 | 1,383 | 1,346 |

At the normal reset, fingers 0 and 1 both register contact:

- Seed 0: 32.73 N and 32.52 N
- Seed 1: 47.89 N and 47.14 N
- Seed 2: 53.89 N and 52.71 N

Finger 2 registers 0 N for all three resets.

## Root Cause
Finger 2 is mounted at world `y=+0.04 m` and rotated so its entire planar chain moves in XZ while retaining that Y coordinate. The rod axis is near `y=-0.05 m`. The 90 mm plane separation exceeds the combined fingertip/rod collision radii of 24 mm, leaving a theoretical surface gap of approximately 66 mm. Joint motion cannot close this gap.

The contact detector is not the cause: it correctly reports simultaneous strong contact for fingers 0 and 1.

## Decision
The existing three-contact reward target is mechanically impossible. Do not run more three-contact reward training against this model.

## Artifacts
- `reachability.json`: per-seed configurations, exact distances, forces, and histograms
- `metrics.csv`: compact machine-readable results
- `images/contact_reachability_comparison.png`: normal two-contact reset versus closest finger-2 pose
- `search.log`: complete search output

## Recommended Next Step
Run EXP-20260724-005 as a geometry-only correction. Move or reorient finger 2 so its motion plane intersects the rod, then repeat this exact reachability test and require reproducible three-finger force contact before any policy training.
