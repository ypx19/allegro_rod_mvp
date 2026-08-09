# Plan: Finger2 configuration space for bottom-tip rod rotation

- Status: planned
- Branch: `plan/finger2-cs-bottom-tip`
- Date: 2026-08-08
- Scope: kinematic CS + short dynamic contact probes; **bottom-tip only** (no top-hang comparison)
- Out of scope for this plan: PPO training, reward changes, multi-seed RL

## Motivation

Prior work showed finger2 can **touch** the rod after the spatial-joint fix
(`20260724-2045-finger2-spatial-dof`). Bottom-tip curricula currently **stand**
(~20 s) but only achieve ~50–56° spin before the ω-hold gate fails.

Open question: under `tip_anchor=bottom`, is finger2’s reachable contact set
suitable for producing **usable axial torque and a sustained spin gait**, or is
low spin partly a kinematic/workspace limit?

## Goal

Decide whether finger2’s configuration space is suitable for rotating the
bottom-tip rod (with fingers 0/1 as context), using numerical and visual
evidence—not intuition from videos alone.

## Suitability criteria (pre-registered)

Treat CS as suitable only if **all** hold:

1. **Contact reach:** finger2 fingertip can achieve signed distance ≤ 0 on the
   rod surface over a nontrivial fraction of rod azimuth (not one lucky pose).
2. **Torque geometry:** at those contacts, fingertip normals have a meaningful
   component that can generate torque about the rod’s long axis (not only
   radial squeeze / axial push that fights the tip support).
3. **Path connectivity:** there exist joint-space paths that keep contact (or
   re-acquire it) while rod phase advances by at least ~90–180° without
   requiring fingertips to teleport.
4. **Dynamic settle:** some of those poses stabilize under short PD holding +
   small open-loop twist without immediate tilt collapse.

Reject suitability if (1) fails, or (1) holds but (2)/(3) fail systematically.

## Method

Reuse patterns from `scripts/check_contact_reachability.py`, retargeted to
bottom tip and rotation metrics. Do **not** train PPO in this investigation.

### Phase 0 — Fix question and logging

- Register pass/fail criteria above in the run `summary.md`.
- Create a unique run id, e.g. `YYYYMMDD-HHMM-finger2-cs-bottom-static`.
- Keep reward, PPO, network, and mass curriculum unchanged.

### Phase 1 — Static joint-space map (FK workspace)

Controlled factors:

- current `models/three_finger_rod.xml` geometry;
- `tip_anchor=bottom`;
- rod pose fixed at nominal reset (tip equality on);
- fingers 0/1 held at reset grasp or coarsely gridded;
- **only finger2** densely sampled in its 3 joint ranges (~10k–50k samples for
  smoke; scale up if inconclusive).

Per-sample outputs:

- tip2 world position / signed distance to rod capsule;
- contact flag + contact normal;
- moment arm / projected torque sign about rod axis (cw vs ccw);
- whether contact is mid-barrel vs tip/endcap.

Artifacts:

- tip workspace cloud colored by contact vs no-contact;
- histogram of torque-about-axis capability;
- fraction of azimuth bins with at least one positive-torque contact;
- CSV + `config.yaml` + `metadata.json` + `summary.md`.

**Hypothesis A:** finger2 can touch but only in a narrow azimuth band or with
mostly non-spin normals → CS unsuitable for continuous rotation.

**Early stop:** if finger2 min distance ≫ 0 or τ_axis histogram is unimodal
near 0, conclude Phase 1 “no” and skip Phase 2–3 expansion.

### Phase 2 — Contact manifold vs rod phase

- Freeze fingers 0/1 in good 2-contact grasps (reset / prior three-contact
  candidates).
- Sweep rod free-joint axial phase in N bins (e.g. 36 × 10°).
- For each phase, sample/optimize finger2 joints for shallow contact + max
  |τ_axis|.

Metrics:

- % of phases with finger2 contact;
- % with τ_axis above a small threshold;
- largest contiguous phase interval with contact (“spin continuity”);
- gap size = phases needing finger2 lift/relocate.

**Hypothesis B:** contacts exist but phase coverage has large holes → gait
needs regrasps finger2 cannot execute fast enough under bottom-tip balance;
explains stand-but-slow-spin.

### Phase 3 — Short dynamic probes (no learning)

From best static candidates:

1. Hold PD targets at contact pose; simulate T seconds; log tip error, tilt,
   forces, ω.
2. Open-loop finger2 joint trajectories that sweep through the high-|τ| path
   while 0/1 hold.
3. Optional: tiny scripted “push then slide” cycles for one half-turn.

Success criteria:

- mean |tilt| stays below curriculum soft threshold for ≥2–5 s;
- cumulative unwrapped axis rotation ≥ 90° (stretch: 180°) with finger2
  force > 0 for ≥30% of steps;
- no NaN / tip blow-up.

**Hypothesis C:** static geometry looks fine but dynamics tip under any useful
tangential force → problem is bottom-tip balance / contact compliance, not CS
emptiness.

### Phase 4 — Decision tree

| Outcome | Interpretation | Next step |
|---|---|---|
| Contact rare / τ_axis≈0 | CS unsuitable for bottom-tip spin | Geometry change (base pose / joint axes / ranges), then re-run Phase 1 only |
| Contact + τ ok, phase holes large | Reachable but not continuous | Scripted regrasp curriculum or add DOF before RL |
| Static+paths ok, dynamic collapses | CS ok; balance/reward issue | Prefer reward/gate/soft-tilt experiments |
| Dynamic ≥90° scripted spin | CS adequate | RL exploration/init is the bottleneck; seed resets near Phase-2 manifold |

## What stays fixed

Reward mapping, PPO hyperparameters, network architecture, mass curriculum,
and default success metrics remain unchanged until this probe decides.

## Evidence requirements

Per `AGENTS.md` run hygiene:

- `runs/<run_id>/{config.yaml,metadata.json,metrics.csv,summary.md}`
- plots: workspace, phase coverage, τ vs phase
- ≥1 video per dynamic probe class
- entries in `docs/EXPERIMENT_LOG.md` and, if durable, `docs/FINDINGS.md`
- update `docs/PROJECT_STATE.md` after results

## What this plan does not resolve

Whether a *learned* policy will find the manifold (exploration). It only
answers whether the manifold exists and can produce rotation under
open-loop/dynamic hold.

## Smallest first slice

Implement Phase 1 as a bottom-tip variant of `check_contact_reachability.py`
with torque-about-axis scoring, then gate Phase 2/3 on those numbers.

## Related prior artifacts

- `FIND-20260724-001` / `EXP-20260724-005`: finger2 was geometrically excluded, then spatially corrected
- `FIND-20260724-002`: reachability alone did not produce multi-contact RL behavior
- `scripts/check_contact_reachability.py`: existing geometric search harness
- Bottom-tip C4 partial: stands ~20 s, ~50–56° spin (`docs/media/partial-bottom-tip-c4-stand.mp4`)
