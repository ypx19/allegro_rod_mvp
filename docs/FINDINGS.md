# Findings

## FIND-20260918-013: Tip penalty ×5 does not clear free-tip 0.5 cm gate
- Confidence: high (900k one-factor from EXP-009)
- Supporting runs: `20260918-1829-s1-freetip-tipscale1-900k-seed0`
- Related: FIND-20260918-011/012
- Applies to: free tip + tip-stop 0.5 cm palm_down
- Does not apply to: tip-connect

### Finding
Raising palm-down tip penalty scale from 0.2 to **1.0** for 900k does not
reduce tip_error_mean below ~6 mm or eliminate tip_error deaths. Fixed length
rises modestly (239→298); unseen flat. Free tip still unsolved.

### Evidence
EXP-010 vs EXP-009 eval; demos `docs/media/s1-freetip-tipscale1-900k-*.mp4`.

### Implication
Do not keep stacking tip weight alone; try soft tip or a different tip
objective.

### Caveats
Single seed; KL/clip high late in run.

## FIND-20260918-012: +600k on tip-stop 0.5 cm free tip yields only modest length gains
- Confidence: high (continue from EXP-008; fixed/unseen)
- Supporting runs: `20260918-1820-s1-freetip-tipstop005-cont600k-seed0`
- Related: FIND-20260918-011
- Applies to: free tip + tip-stop 0.5 cm from EXP-005 stack
- Does not apply to: tip-connect

### Finding
An additional 600k steps raises eval length from ~195 to ~240 and keeps
tip_error as the sole death mode, but tip_error_mean stays ~6 mm and success
remains 0. Pure longer training under this gate is slow / insufficient for
20 s free-tip holds.

### Evidence
EXP-009 vs EXP-008 eval tables; demos `docs/media/s1-freetip-tipstop005-cont600k-*.mp4`.

### Implication
Prefer a new free-tip factor over another budget-only continue.

### Caveats
Single seed; cumulative ~900k from EXP-005 free-tip path.

## FIND-20260918-011: Tip-stop 0.5 cm flips free-tip deaths from tilt to tip_error
- Confidence: high (one-factor vs EXP-007; fixed/unseen)
- Supporting runs: `20260918-1812-s1-freetip-tipstop005-from-exp005-seed0`
- Related debug issues: FIND-20260918-010
- Applies to: free tip from EXP-005 palm-down at s=1
- Does not apply to: tip-connect (tip error ≈0)

### Finding
Tightening tip-error termination from 12 cm to **0.5 cm** on free tip from
EXP-005 shifts nearly all deaths to `tip_error`, cuts tip error ~10×, cuts
final tilt from ~24° to ~3°, and roughly doubles eval length (84→195). Free
tip still fails success/drop gates within ~8 s.

### Evidence
EXP-008 eval vs EXP-007; demos `docs/media/s1-freetip-tipstop005-*.mp4`.

### Implication
Use 0.5 cm tip stop as the free-tip working gate. Next levers: tip reward
strength, soft tip, or longer budget — not loosening the stop back to 12 cm.

### Caveats
Single seed; 300k; success still 0.

## FIND-20260918-010: Free tip from EXP-005 needs more than disabling equality
- Confidence: high (zero-shot + 300k one-factor fine-tune)
- Supporting runs: `20260918-1753-s1-freetip-from-exp005-seed0`;
  zero-shot `20260918-1753-freetip-from-exp005-zeroshot`
- Related debug issues: tip-holding curriculum / soft-tip history
- Applies to: EXP-005 palm-down PPO our_hand at s=1
- Does not apply to: soft-tip solref fade or stronger tip-error shaping (untested)

### Finding
With user-accepted EXP-005 as Phase-T, turning tip equality off and relying
on existing palm-down tip-error reward + `tip_error>0.12` stop fails:
zero-shot length 147 / drop 1.0; 300k fine-tune length **84/87**, success 0,
drop 1.0. Deaths are mostly `axis_tilt`, not tip_error. Do not claim free-tip
phase passed.

### Evidence
EXP-007 eval tables; demos `docs/media/s1-freetip-from-exp005-*.mp4`.

### Implication
Next free-tip attempt must introduce a *new* controlled factor (soft tip,
stronger tip penalty, assist, etc.). Keep EXP-005 tip-connect as best live
checkpoint.

### Caveats
Single seed; 300k may be short for free tip; tip stop at 12 cm is loose vs
success tip <2 cm.

## FIND-20260918-009: Palm-down tip solref 0.004 raises spin but hurts tilt survival on our_hand
- Confidence: high (one-factor 300k vs EXP-005; fixed/unseen eval)
- Supporting runs: `20260918-1559-s1-palm-down-ppo-solref004-seed0` vs
  `20260918-1528-s1-palm-down-ppo-tilt035-seed0`
- Related debug issues: historical solref softening notes; DBG-20260918-002
- Applies to: our Allegro tip-connect + palm-down PPO/pose at s=1
- Does not apply to: the bundled palm_down XML/policy (other geoms/actuation)

### Finding
Copying the palm-down bundle's runtime tip-connect solref **0.004** onto the
EXP-005 our_hand recipe (was 0.008) increases online return/length and eval
rotation (238°/229° vs 207°/181°) and unseen net-angle success (0.70 vs 0.50),
but **breaks 20 s survival**: length 454/434, drop 0.2/0.3, final tilt ~7–9°,
`axis_tilt` deaths, `passed=false`. Soft tip equality **0.008** remains the
better default for this stack.

### Evidence
EXP-006 eval_fixed/unseen vs EXP-005 tables; demos
`docs/media/s1-palm-down-ppo-solref004-seed6.mp4` (success) and
`...-seed10000.mp4` (axis_tilt @ 223). Curves:
`runs/20260918-1559-.../plots/train_curves_vs_005.png`.

### Implication
Do not assume literal hyperparameter copy from the palm-down bundle is
optimal on our MJCF. Prefer EXP-005 solref 0.008 when continuing mass-up.
Online return alone would have falsely suggested adopting 0.004.

### Caveats
Single seed. Bundle still uses 0.004 successfully — residual differences
(XML, solimp, contacts) are not isolated here.

## FIND-20260918-008: Tip-only contact sensing still allows proximal wedges; PD vs contact is jamming, not a logic clash
- Confidence: high (obs/actuator code + contact audit; conceptual, not a new train)
- Supporting runs: `20260918-1528-s1-palm-down-ppo-tilt035-seed0`; FIND-20260918-007
- Related debug issues: `DBG-20260918-001`, `DBG-20260918-002`
- Applies to: tip-connect Allegro policies with tip-only `_touch` / `contact_count`
  and position actuators (`kp`/`kv`, force-limited)
- Does not apply to: claims that the policy “sees” proximal contact; torque-control
  hardware without an equivalent PD+contact solver story

### Finding
Two linked considerations for future work:

1. **Sensing gap.** The policy observation exposes only tip↔rod forces
   (`_touch`), plus proprioception (`hand_q`/`hand_v`) and rod state
   (tilt, linvel, ω, tip error). Proximal/medial/distal contacts are **not**
   in the obs or in `contact_count`. Multi-link wedges (FIND-20260918-007)
   are therefore learned **indirectly**: jammed joint angles/velocities,
   tip-load sharing, and rewarded rod dynamics — not from an explicit
   proximal-contact bit.

2. **Actuator vs contact force.** Actuators are position servos
   (`kp=20`, `kv=1`, `forcerange≈±3`), not open-loop torques. MuJoCo solves
   PD joint forces and contact constraints in one step. Commanding more
   flexion while a proximal link is already against the rod raises contact
   normal force toward a jammed equilibrium (high Fn, possibly saturated
   actuators). That is expected quasi-static jamming, not a software
   double-application bug — but it does explain large audited normals
   (tens–100+ N) and is a risk for sim instability and hardware transfer.

### Evidence
`env.py` `_touch` / `_frame_obs` (tips only); XML `<position kp="20" … forcerange="-3 3"/>`;
contact audit proximal Fmean 67.8 N (EXP-005 seed 6). Labeled anatomy:
`docs/media/finger-link-labels.png`.

### Implication
Future directions worth treating as first-class options (not yet chosen):
- all-link contact diagnostics / optional privileged or student contact features
  beyond tips;
- penalties or gates on non-tip normal force / actuator saturation;
- hardware transfer assumptions: tip tactile alone may not match the sim grasp;
- mass-up / solref changes may interact with jamming force magnitude more than
  with tip `contact_count`.

### Caveats
No experiment yet that adds proximal sensors or force penalties. Do not change
`contact_count` semantics silently (many gates depend on tip-only).

## FIND-20260918-007: Both success gaits are multi-link wedges, not fingertip-only
- Confidence: high (full contact enumeration; EXP-005 + bundle; 2 seeds each)
- Supporting runs: `20260918-1528-s1-palm-down-ppo-tilt035-seed0`; bundle `palm_down_screwdriver`
- Related debug issues: `DBG-20260918-001`
- Applies to: s=1 tip-connect screwdriver policies (our transfer and the bundle)
- Does not apply to: s=400 T00; claims based on `contact_count` occupancy

### Finding
Enumerating every MuJoCo rod contact (normal force ≥ 0.05 N) shows both the
EXP-005 success rollout and the palm-down bundle rollout load the rod with
**non-fingertip finger links on ~100% of steps**. The **proximal** phalanx is
a dominant contributor. In EXP-005 seed 6 the proximal link averages 67.8 N
(max 163.5 N), exceeding the tip average of 46.5 N; medial and distal also
contribute. The bundle is the same pattern but less proximal-dominant
(tip 53 N > proximal 28 N). So the working strategy is a multi-link wedge of
the rod against the proximal/palm region, with the tip as one of several
supports — not a clean fingertip gait.

### Evidence
`contact_audit.json` (EXP-005): seed 6 proximal 100%/Fmean 67.8/Fmax 163.5,
tip 100%/46.5, medial 45%, distal 77%, finger_base 29%; non-tip 100% of steps.
`contact_audit_palm_down.json`: seed 7000 tip 100%/53.1, proximal 88.8%/27.7,
distal 78.6%, medial 23.8%; non-tip 99.8% of steps.

### Implication
`contact_count` / ≥2-tip occupancy (FIND-20260918-001, EXP-005 tables)
describe **tips only** and understate the grasp. Do not call these
"2-contact fingertip gaits." When judging contact strategy, enumerate all
rod-hand geoms. The transfer relies on the proximal link even more than the
bundle, which may matter for real-hardware transfer and for higher-mass runs.

### Caveats
Two seeds each. Force is instantaneous normal component. Not yet folded into
`eval_policy.py`; see DBG-20260918-001.

## FIND-20260918-006: Palm-down PPO is decisive for s=1 tracker tip-connect on our_hand
- Confidence: high (300k; 20/20 full-horizon eval; vs EXP-004 one-factor PPO)
- Supporting runs: `20260918-1528-s1-palm-down-ppo-tilt035-seed0`; baseline EXP-004
- Related debug issues: `DBG-20260823-006` (s=400 still open)
- Applies to: palm-down translation + default qpos, tracker reward, 0.35 kill, s=1
- Does not apply to: T00 s=400, free-tip, claiming three-contact gait

### Finding
Replacing T00 PPO (`[512,256,128]`, LR 3e-5, ent 0, log_std 0) with the
palm-down PPO recipe (`[256,256]`, LR 3e-4, ent 0.005, log_std −1) at
otherwise identical s=1 / tracker / 0.35 kill yields **20/20 full 20 s**
episodes, drop 0, final tilt ~2°, tip <1 mm, and net-angle success
0.80/0.50. EXP-004 under T00 PPO died by ~50–60 steps. The working contact
pattern is a 1–2 finger gait.

### Evidence
Eval fixed/unseen passed=true; length 500/500; terminations none×20.
Online length 394, return +366, success_rate 0.42. Demos:
`docs/media/s1-palm-down-ppo-tilt035-seed6.mp4` (success, 210°),
`docs/media/s1-palm-down-ppo-tilt035-seed10000.mp4` (20 s hold, ~180°,
net_angle miss). Curves: `plots/train_curves_vs_004.png`.

### Implication
At s=1 the palm-down transfer is now reproducible on our stack without the
bundled XML. Prefer this PPO when continuing mass-up or solref ablations.
T00 s=400 remains a separate problem; do not assume this PPO alone fixes it.

### Caveats
Changed four PPO knobs together (as the bundle recipe). KL/clip_fraction
are high late in training. Unseen success 0.5 is mostly near-π misses.
Single seed. solref still 0.008 (bundle 0.004). No obs[36] rewrite.

## FIND-20260918-005: s=1 improves tracker+0.35 survival but does not reproduce palm-down success under T00 PPO
- Confidence: high (300k train; 20 eval episodes; curves vs EXP-003)
- Supporting runs: `20260918-1515-s1-palm-down-tracker-tilt035-seed0`; parent EXP-003 `20260918-1417-...`
- Related debug issues: `DBG-20260823-006`
- Applies to: palm-down translation + default qpos, tracker reward, 0.35 kill, T00 PPO, solref 0.008
- Does not apply to: palm-down bundle PPO/solref/obs rewrite; free-tip

### Finding
Lowering mass from 400 to 1 with an otherwise fixed EXP-003 stack and a
300k budget raises online length to 77 and return to +57 (EV 0.91), and
raises eval length/rotation/≥2-contact occupancy versus s=400. It does
**not** yield eval success, 20 s holds, or zero tilt deaths. Failures split
between `axis_tilt` and a new `unstable` mode. s=1 is helpful and required
context for further transfer, but not sufficient alone under T00 PPO.

### Evidence
Eval fixed/unseen: success 0, length 58.1/46.3, rot 138.8°/155.3°,
terms axis_tilt 6+7 and unstable 4+3, recontact 0, ≥2-contact 0.84/0.93.
Curves: `plots/train_curves_vs_003.png`. Demo seed 6: 105 steps then
axis_tilt (`docs/media/s1-palm-down-tracker-tilt035-seed10000.mp4` is the
unstable 48-step seed).

### Implication
Next one-factor changes should stay at s=1 and close remaining gaps vs the
verified bundle (solref 0.004 or palm-down PPO hyperparams), not return to
s=400 pose swaps.

### Caveats
Steps budget also rose 100k→300k (intentional match to palm-down). At the
106k mark online length was already 42 > EXP-003's 28, so mass—not only
extra steps—contributes. Single seed.

## FIND-20260918-004: dθ/dt vs cvel ω diverge mainly after support loss, not during 2-contact spin
- Confidence: high (EXP-003 seeds 6 and 10000; CSV + fresh rollout agree)
- Supporting runs: `20260918-1417-t00-palm-down-tracker-tilt035-seed0`
- Related debug issues: `DBG-20260823-006`
- Applies to: tip-connect free rod at s=400 under palm-down tracker; EXP-003 obs[36]
- Does not apply to: claiming this is why palm-down s=1 succeeds

### Finding
On EXP-003 episodes, reward ω (`dθ/dt`) and body-twist `_axial_omega`
(`cvel` projection) agree while `n_contact >= 2` (|gap| mean ≈ 0.06 rad/s,
max ≈ 0.28). They diverge after support collapse (`n <= 1`): seed 6 max
|gap| = 2.80 rad/s; seed 10000 max |gap| = 0.41. `obs[36]×10` equals
`cvel` to ~1e-7, confirming EXP-003's policy sees body-twist while the
tracker reward uses `dθ/dt`. Mean spin rates are similar (~1.17 rad/s,
near the tracker peak). The mismatch is a **collapse-phase** effect, not a
steady-gait disagreement.

### Evidence
`runs/20260918-1417-t00-palm-down-tracker-tilt035-seed0/plots/omega_dtheta_vs_cvel_seed{6,10000}.png`
and `.../omega_dtheta_vs_cvel_summary.json`. Seed 6: corr 0.72 overall,
|gap|mean 0.065 (n≥2) vs 0.627 (n≤1). Seed 10000: corr 0.98,
|gap|mean 0.063 (n≥2) vs 0.180 (n≤1).

### Implication
Do not treat the obs[36] rewrite as the primary fail-vs-succeed explanation
for EXP-003 vs the palm-down bundle. During the productive 2-contact phase
the two ω's nearly match; the big gap appears only after 2→1 contact loss,
alongside the ω spike and tilt rise. Mass s=400 remains the higher-leverage
untested split. Fixing obs[36] to `dθ/dt` is still good hygiene for a
palm-down-faithful recipe, but it is unlikely to rescue T00 alone.

### Caveats
Only two collapse episodes (25 steps each). No s=1 comparison yet.

## FIND-20260918-003: Palm-down tracker + 0.35 rad kill does not stabilize T00 at s=400
- Confidence: high (100k train; 20 eval episodes; length curve vs EXP-002)
- Supporting runs: `20260918-1417-t00-palm-down-tracker-tilt035-seed0`; parent `20260918-1343-t00-palm-down-translation-seed0`
- Related debug issues: `DBG-20260823-006`
- Applies to: T00 s=400, palm-down XML translation + default qpos, native `reward_style=palm_down`, tilt kill 0.35
- Does not apply to: palm-down bundle at s=1, other PPO sizes, solref 0.004

### Finding
Copying the palm-down **training objective** (tracker reward peaked at 1 rad/s
plus a 0.35 rad tilt kill) onto the T00 pose/grasp at **s=400** still yields
20/20 `axis_tilt`. Episodes get **shorter** (eval 29 vs EXP-002's 36; online
48→28) while Monitor return rises on a different scale. Final tilt sits at
the new gate (~24–25°) instead of ~80°. Rotation collapses to ~60° because
the policy is killed before the DexScrew spin-then-fall can accumulate angle.

### Evidence
Eval `eval_fixed.json` / `eval_unseen.json`: success 0, drop 1.0,
`termination_reasons.axis_tilt=10` each, length 28.8/28.9, recontact 0.
Online `metrics.csv`: return −256→−4, length 47.7→28.4, EV 0.30.
Videos: 25-step tilt-kills,
`docs/media/t00-palm-down-tracker-tilt035-seed10000.mp4`.
Curves: `runs/20260918-1417-t00-palm-down-tracker-tilt035-seed0/plots/train_curves_vs_002.png`.

### Implication
The palm-down screwdriver result is not explained by reward+tilt-kill alone
at T00 mass. The remaining high-leverage split vs the verified bundle is
**s=1** (and secondarily PPO size / solref 0.004 / 300k). Do not treat a
rising tracker return as stability.

### Caveats
Reward and kill changed together. Observation was not rewritten at index 36.
Single seed. 100k vs bundle 300k.

## FIND-20260918-002: Palm-down translation is not a T00 fix; grasp qpos is pose-specific
- Confidence: high (reset 10 seeds × 2 grasps × 2 poses; T00 100k eval 20/20)
- Supporting runs: `20260918-1343-t00-palm-down-translation-seed0`; baseline A `20260914-1638-t00-ablation-A-hist1-rewbase-seed0`
- Related debug issues: `DBG-20260823-006`
- Applies to: T00 s=400 DexScrew, `my_grasp` vs palm-down XML translation, heavy vs default Allegro qpos
- Does not apply to: palm-down s=1 tracker-reward policy, free-tip, real robot

### Finding
The 52 mm palm translation from the palm-down XML does **not** make T00 succeed.
A 100k from-scratch T00 run at that pose still dies 20/20 on `axis_tilt` (length
36, ~210–217°, recontact 0), worse than from-scratch A on `my_grasp` (length 54,
305°, recontact 0.17).

The T00 heavy grasp and the palm-down translation are incompatible: heavy qpos
at the new pose is 10/10 **zero** tip-rod contacts; env-default qpos at
`my_grasp` is 10/10 zero contacts and drops. Default qpos at the palm-down
translation is 10/10 **three-contact** under zero actions. Each palm location
needs its own joints.

### Evidence
Reset smoke `runs/20260918-1343-t00-palm-down-translation-seed0/reset_smoke.json`.
Eval `eval_fixed.json` / `eval_unseen.json`: `termination_reasons.axis_tilt=10`
each; `support_collapse.recontact_success_rate=0`.

### Implication
Do not treat palm-down success as “move `my_grasp` 52 mm.” Pose and grasp are
a pair. Remaining T00 gaps vs palm-down are mass, reward, solref, PPO, and
horizon actually lived.

### Caveats
This run changed translation **and** qpos versus A (heavy qpos cannot be held
fixed). Single seed. 100k only.

## FIND-20260918-001: Palm-down tip-connect screwdriving is stable at mass-scale 1 with a 2-contact gait
- Confidence: high (bundled 100×20s + 30×60s bounded; independent 8×20s + 5×60s new seeds; alignment tests)
- Supporting runs: `20260918-1225-palm-down-screwdriver-verify`; bundle `palm_down_screwdriver/`
- Related debug issues: `DBG-20260823-006` (not resolved; different mass/pose/reward)
- Applies to: palm-down XML, bottom `mjEQ_CONNECT`, `s=1`, stabilizer 0, bundled 300k PPO, 20 s original / 60 s with obs[37] clipped to 3 turns
- Does not apply to: Phase T `s=400`, free-tip (no equality), `my_grasp` 52 mm palm offset, real robot, DexScrew +3 contact bonus policies

### Finding
A CPU PPO trained from scratch on palm-down tip-connect at mass-scale 1 produces continuous axial rotation with max tilt <3.1° and max tip error <1.0 mm. The working contact pattern is **2-finger**, not 3-finger: about 73% of steps have 2 tip contacts, 26% have 1, 1–4% have 3, and **zero** steps have 0. Original 20 s episodes pass 100/100 bundled and 8/8 independent. Raw 60 s fails a speed-continuity gate because cumulative-turn observation 37 leaves the 20 s training range; clipping it to 3 turns restores 10/10, 20/20, and 5/5 independent 60 s success without changing physics.

### Evidence
Independent orig 8×20s: success 1.0, mean 3.284 turns, worst tilt 2.985°, worst tip 0.919 mm. Independent bounded 5×60s: success 1.0, mean 10.358 turns, worst tilt 2.967°, worst tip 0.927 mm. Independent orig 2×60s: duration 60 s, terminated false, tilt <3°, tip <1 mm, `positive_fraction` 0.929. Videos: `docs/media/palm-down-screwdriver-20s-seed7000.mp4`, `docs/media/palm-down-screwdriver-60s-seed4000.mp4`. Alignment tests confirm palm −Z, `mjEQ_CONNECT`, free joint, zero stabilizer.

### Implication
“Stable tip-connect screwdriving” is achieved in this geometry/reward/mass setting. Phase T remains blocked at `s=400`. The T00 hypothesis that ≥2-contact recovery is physically infeasible is **not a universal statement**: at `s=1` palm-down, a 2-contact gait is the successful strategy. Support-termination and three-contact rotation gates would hide this gait. Do not mix this protocol with `eval_policy.py` success.

### Caveats
Tip **position** is constrained; only orientation is policy-stabilized. 60 s unbounded needs the obs clip. Palm translation differs from `my_grasp` by 52 mm. Single training seed. No domain randomization.

## FIND-20260914-004: 300k A/B vs C/D share the seed-6 2→1→0→axis_tilt sequence
- Confidence: medium (one deterministic seed; matches the 1748 video returns)
- Supporting runs: `20260914-1805-t00-ablation-demos-300k`, videos `20260914-1748-t00-ablation-videos-300k`, ckpts `20260914-1712-t00-ablation-{A,B,C,D}-continue300k-seed0` step 306432
- Related debug issues: `DBG-20260823-006`, FIND-20260914-001, FIND-20260914-003
- Applies to: T00 300k continue policies, seed 6, tilt_terminate=1.2
- Does not apply to: claiming task success, other seeds, λ retunes, T01

### Finding
On the same seed-6 collapse demo used for the transfer timeline, the four
300k continue policies all follow **2-contact hold → 1-contact support loss →
n=0 → axis_tilt**. None recontact. Support-aware C/D lose 2+ contact a few
steps later (50 / 53 vs A 49 / B 43) and then gate rotation / apply wobble,
but that does not change the narrative or prevent tilt-kill. D’s post-loss
window is the shortest (9 steps).

### Evidence
Re-rolled traces (not video-filename parsing):

| Cond | 2+→≤1 | n=0 | Kill | Loss→kill | Recontact |
|---|---:|---:|---:|---:|---|
| A | 49 | 57 | 65 | 16 | no |
| B | 43 | 54 | 61 | 18 | no |
| C | 50 | 60 | 65 | 15 | no |
| D | 53 | 57 | 62 | 9 | no |

Returns match the 1748 export (A +278.7 / B +258.7 / C +213.7 / D +204.2).
Page: `docs/pages/t00-ablation-demos-300k/`.

### Implication
Do not treat C/D as a collapse-timeline fix. Do not retune λ. Next work stays
on physical recoverability of ≥2-contact support.

### Caveats
Single seed 6. One checkpoint (306432). 3-contact exists only at reset.


## FIND-20260914-002: 4-frame history already lengthens T00 episodes at 100k
- Confidence: medium (online length only; low on collapse/eval)
- Supporting runs: `20260914-1638-t00-ablation-A-hist1-rewbase-seed0`, `...-B-hist4-rewbase-seed0`, `...-C-hist1-rewsup-seed0`, `...-D-hist4-rewsup-seed0`
- Related debug issues: `DBG-20260823-006`; revises FIND-20260914-001 for the 100k snapshot
- Applies to: T00 bottom tip-connect s=400, from-scratch PPO `[512,256,128]`, online Monitor at ~100k, seed 0
- Does not apply to: eval success, re-contact, GRU/LSTM, other seeds, claiming the task is solved

### Finding
At the 100k cutoff, 4-frame observation history already produces longer online
episodes than hist1. B (192-D hist4, baseline reward) and D (192-D hist4,
support-aware) last 53.7 / 54.6 steps versus A / C at 40.0 / 46.8. History
stacking is effective as an early online-length signal before full convergence.

This revises FIND-20260914-001’s “H1 not supported at history_len=4” for the
100k *training* snapshot only. H1 is still unproven on collapse and eval metrics.

### Evidence
Last `metrics.csv` row (step 106,496), Monitor episode length: A 40.0, B 53.7,
C 46.8, D 54.6. Length panel of
`reports/comparisons/20260914-t00-ablation-train-curves.png`. Standalone page:
`docs/pages/t00-ablation-history.html`. Seed-6 collapse demo (inherit later
with this page): `runs/20260914-t00-seed6-tilt-collapse/index.html` and
`docs/pages/t00-seed6-tilt-collapse/index.html`. Integration note:
`docs/pages/README.md`.

A has the highest train return (+22.1) but the shortest length. C/D returns are
lower (−47.5 / −17.9) under the wobble penalty, not as a failed-train signal.

### Implication
Do not discard frame stacking solely from the 100k eval (80/80 `axis_tilt`).
Keep history as an observability factor when reading later 300k curves. Do not
treat longer episodes as contact recovery.

### Caveats
100k is not converged (returns/lengths still rising). Eval success is 0.
Single seed 0. Episode length ≠ re-contact. Reward hacking of rotation before
tilt-death remains plausible for A’s high return.
The 300k continuation (FIND-20260914-003) shows this early length gap
**narrowed**: at 303k, A 62.4 / B 64.4 / C 68.6 / D 62.4.


## FIND-20260914-003: Hist4 length lead at 100k does not persist at 300k
- Confidence: medium (online Monitor only; single seed)
- Supporting runs: `20260914-1712-t00-ablation-{A,B,C,D}-continue300k-seed0` plus parents `20260914-1638-…`
- Related debug issues: FIND-20260914-002, FIND-20260914-001, `DBG-20260823-006`
- Applies to: T00 from-scratch PPO, seed 0, cumulative ~300k snapshot
- Does not apply to: eval success, re-contact, GRU/LSTM, other seeds

### Finding
The 100k online-length advantage of 4-frame history (B/D > A/C) **narrowed
and largely disappeared** by 300k. C (hist1 + support-aware) is longest
(68.6). B is only +2 vs A (64.4 vs 62.4). D equals A (62.4). Treat the 100k
hist4 length signal as early / transient, not as a lasting T00 fix.

### Evidence
Step 303,104 Monitor: return A +221.8, B +197.9, C +169.6, D +164.1;
length A 62.4, B 64.4, C 68.6, D 62.4; KL ~0.01; EV 0.975–0.992.
Figure: `reports/comparisons/20260914-t00-ablation-train-curves-300k.png`.
Page: `docs/pages/t00-ablation-history.html`.

### Implication
Do not adopt hist4 as a T00 solution on the 100k length snapshot. Do not
retune λ. Next work stays on physical recoverability, not another PPO trick.

### Caveats
Single seed. Eval not re-run at 300k. Returns and lengths are still rising
(not a plateau). Jobs actually reached ~410k because SB3 adds the current
timestep counter when `reset_num_timesteps=False`; the claim uses the 303k
slice. Episode length ≠ contact recovery.


## FIND-20260914-001: Short history and support-gated rotation do not stop T00 support collapse
- Confidence: medium
- Supporting runs: `20260914-1638-t00-ablation-A-hist1-rewbase-seed0`, `...-B-hist4-rewbase-seed0`, `...-C-hist1-rewsup-seed0`, `...-D-hist4-rewsup-seed0`; published T00 smoke `20260914-t00-ablation-eval-smoke`
- Related debug issues: `DBG-20260823-006`
- Applies to: T00 bottom tip-connect s=400, from-scratch PPO `[512,256,128]`, 100k *eval*, `history_len=4`, rotation contact scales `{0,0.1,1.0}`, `λ_wobble=0.5`
- Does not apply to: GRU/LSTM, longer history, λ sweeps, revolute-transfer fine-tunes, other masses; the 100k *online-length* snapshot (see FIND-20260914-002)

### Finding
Neither 160 ms frame stacking nor contact-gated rotation plus a low-support
wobble penalty teaches T00 re-contact. All 80 eval episodes still die on
`axis_tilt`. On **eval** collapse metrics, history-only (B) is ~A. Support-aware
reward alone (C) shortens episodes and increases post-loss `‖ω_perp‖` (~8 vs
~2 rad/s). Both together (D) does not recover A. Do not treat this as a
coefficient-tuning problem next.

Revision (100k online Monitor, FIND-20260914-002): H1 has an early
episode-length signal (B/D > A/C). That does not overturn the eval/collapse
rejection.

### Evidence
Fixed/unseen recontact: A 0.17/0.09, B 0.09/0.09, C 0.00/0.00, D 0.00/0.00.
Eval episode length: A 54/54, C 37/37. Every split is 10/10 `axis_tilt`.
Comparison: `reports/comparisons/20260914-t00-ablation-A-B-C-D.md`.

### Implication
The failure is not isolated by cheap temporal observability or this particular
credit-assignment patch. Next experiments should test physical recoverability
of ≥2-contact support (grasp geometry, contact mechanics, force-signal latency).

### Caveats
Single training seed. From-scratch A is not the published revolute-transfer T00
(177°, length ~35). A longer budget or GRU could still matter; this 2×2 does not
test those. Online length at 100k (FIND-20260914-002) is a separate, weaker
signal than eval re-contact.


## FIND-20260824-001: DexScrew tilt-growth penalty fires during monotonic collapse but does not restore the 0.25 rad gate
- Confidence: medium
- Supporting runs: `20260824-0000-dexscrew-tilt-growth-s100-tip-seed0`; recovery sibling `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`; zero-shot parent `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Related debug issues: `DBG-20260823-006`, FIND-20260823-002, FIND-20260723-011
- Applies to: DexScrew `reward_style` on bottom tip-connect after revolute transfer, growth `g=50`, deadzone 0.05 rad, gate 0.25 rad, clip 0.05 rad, 65,536-step fine-tune
- Does not apply to: larger growth scales; two-sided Δtilt; tilt assists; longer budgets; other masses

### Finding
A one-sided tilt-growth penalty
`r_growth = -50 * clip(curr−prev, 0, 0.05) * w(curr)` with
`w=clip((θ−0.05)/0.20, 0, 1)` occupies ~3–4% of Σ|reward| during monotonic
tip-connect collapse, versus ~0.2–0.4% for recovery-only shaping. Final/max
tilt still saturates near 80°, and net-angle success stays 0. After the
fine-tune, rotation share rises (38–42%) while mean growth remains ~−0.5
versus rotation +5.2 to +6.5.

### Evidence
EXP-20260824-001: 65,536-step fine-tune from accepted revolute `s=100`.
Fixed/unseen success 0.0/0.0. Rotation 280.12°/264.34° versus zero-shot
182.31°/169.09° and recovery 190.95°/223.07°. Decode-checked failures reach
399° rotation with 85° final tilt.

### Implication
Do not further increase this growth scale as the next one-factor test, and do
not start other tip-connect masses. Growth solved the “term never fires”
problem left by recovery, but not the lateral-tilt failure.

### Caveats
One training seed and a short fine-tune. This does not rule out a tilt assist,
grasp change, or a much stronger state penalty.

## FIND-20260823-002: DexScrew one-sided tilt recovery does not fire during monotonic tip-connect collapse
- Confidence: medium
- Supporting runs: `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`; zero-shot parent `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0`
- Related debug issues: `DBG-20260823-006`, FIND-20260723-011
- Applies to: DexScrew `reward_style` on bottom tip-connect after revolute transfer, recovery `scale=50`, deadzone/clip 0.05 rad
- Does not apply to: two-sided stage-path recovery; longer budgets; other masses
  (tilt-growth at `g=50` was tested in FIND-20260824-001 and also missed the gate)

### Finding
A one-sided back-to-balance term that pays only when tilt decreases has ~0.2–0.4%
reward mass when every evaluated episode’s max tilt equals its final tilt. The
policy still terminates on ~80° lateral tilt; net-angle success stays 0.

### Evidence
EXP-20260823-018: 65,536-step fine-tune from accepted revolute `s=100`. Fixed/unseen
success 0.0/0.0 versus zero-shot 0.0/0.0. Mean recovery +0.032/+0.056 per step
versus tilt penalty −3.83/−3.34. Decode-checked failures reach 333° rotation with
81° final tilt.

### Implication
Do not further increase this recovery scale as the next one-factor test. Prefer a
signal that is nonzero while tilt is growing, or change grasp/assist dynamics.

### Caveats
One training seed and a short fine-tune. This does not rule out two-sided
recovery or a tilt curriculum.

## FIND-20260823-001: Allegro-aligned wrap reset removes the sparse three-contact initialization failure
- Confidence: high
- Supporting runs: `20260823-geometry-reachability-allegro-tip-bottom-v2`, `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Related debug issues: `DBG-20260724-002`, `DBG-20260823-001`
- Applies to: 12-DoF Allegro index/middle/thumb primitive model with ramped bottom-tip reset
- Does not apply to: the legacy 9-DoF surrogate or arbitrary rod/palm placements

### Finding
Using official Allegro joint frames with an optimized shallow-to-preload reset makes simultaneous three-fingertip support reproducible instead of exploration-sparse.

### Evidence
Both revolute and point-connect reset audits produced three contacts on 50/50 fixed seeds. The final curriculum smoke maintained three-contact occupancy of 1.0 at nominal mass and zero stabilizer torque.

### Implication
Future training should not increase contact reward to discover the third finger. Contact initialization is now solved; optimize rotation reward balance and curriculum duration instead.

### Caveats
The middle primitive pad radius is 2 mm larger than the Menagerie collision primitive so the adjacent distal box does not occlude pad contact. Revolute reset has occasional high transient index force and still needs visual artifact review.

## FIND-20260802-002: Unconstrained long PPO blows up `std` after Stage 0 success peak
- Confidence: high
- Supporting runs: `20260802-0220-exp-infra-subproc64-1e9-seed0` (EXP-20260802-002)
- Related debug issues: `DBG-20260802-001`
- Applies to: long SB3 PPO runs with default `ent_coef=0.01` and unclipped `log_std` on this Stage 0 stack
- Does not apply to: claiming Stage 0 cannot reach ~0.9 success; Arm A reward changes

### Finding
Scaling to 1e9 steps **did** produce high online Stage 0 success (~0.88 at 1e6, ~0.93 at 5e6), then policy `std` exploded (→1e18) and training crashed with NaN action means at ~3.17e7 steps. Pure “train longer” without entropy/`log_std` control is unsafe.

### Evidence
Console milestones in EXP-20260802-002; crash traceback in `logs/console.log`; best ckpt `ppo_rod_5000000_steps.zip`.

### Implication
Use dense checkpoints, early-stop/select peak policies, set `ent_coef→0` and/or clip `log_std`, and save VecNormalize every checkpoint before multi-day jobs. Then resume Arm A.

### Caveats
Single seed; online success not yet confirmed with `eval_policy.py`. Peak may shift with different hyperparams.

## FIND-20260802-001: Parallel CUDA stack works; 2e5 Stage 0 steps insufficient for success under new net
- Confidence: medium (updated by FIND-20260802-002)
- Supporting runs: `20260802-0216-exp-infra-plumbing-check-seed0`, `20260802-0217-exp-infra-subproc8-cuda-seed0` (EXP-20260802-001); follow-up `20260802-0220-exp-infra-subproc64-1e9-seed0` (EXP-20260802-002, failed after peak)
- Related debug issues: `DBG-20260802-001` (post-peak divergence)
- Applies to: DexScrew-style MuJoCo+SB3 parallel training on Stage 0 with `[512,256,128]` + VecNormalize
- Does not apply to: claiming Stage 0 is unlearnable; Arm A revolute/ω reward; legacy CPU `train.py` Stage 0 quality

### Finding
`scripts/train_parallel.py` (SubprocVecEnv + CUDA + 3-layer MLP + VecNormalize) trains and reloads. Under unchanged Stage 0 reward, **2e5** total env-steps left **`success_rate=0`**, but longer training later reached ~0.9 online success by ~5e6 (then diverged — FIND-20260802-002).

### Evidence
EXP-20260802-001 metrics/TB/load smoke; EXP-20260802-002 console milestones.

### Implication
Adopt the parallel stack. Treat short smokes as infra gates only. Prefer ~5e6–1e7 budgets with std control over unconstrained 1e9.

### Caveats
Single seed; formal 20-episode eval of the 5e6 ckpt still pending.

## FIND-20260730-001: DexScrew “good rotation” is revolute-constrained sim + real BC, not free-object RL
- Confidence: high
- Supporting runs: literature / code review of `references/dexscrew` (arXiv 2512.02011)
- Related debug issues: Stage 2 stabilizer cliff (PROJECT_STATE)
- Applies to: curriculum design for tip-connect / axis assists; interpreting external screwdriving RL claims
- Does not apply to: claiming our Stage 1 degrees equal their real progress ratios

### Finding
DexScrew trains sim rotation on a **fixed base + revolute joint** (threads omitted), distills a proprio student, then finishes screwdriving with **real tactile + history behavior cloning** (~95% progress). Direct sim2real of their rotation policy is only ~42% progress and never completes. Their published success does not imply free-object Stage 2 PPO should work.

### Evidence
Paper §§III–IV and configs in `configs/task/XHandHoraScrewDriver.yaml`; reward on nut DOF ω in `dexscrew/tasks/xhand_hora.py`. Full write-up: `reports/comparisons/dexscrew_vs_allegro_rod_mvp.md`.

### Implication
Treat tip-connect / hinge-style constraints as Stage-A skill learning (closer to DexScrew), not a temporary crutch to delete before gaits are solid. Optional ablations: true revolute rod; pose-diff/work penalties.

### Caveats
Different hand, simulator, and task; comparison is methodological, not a matched eval.

## FIND-20260724-002: Geometry reachability alone does not overcome sparse contact exploration
- Confidence: medium
- Supporting runs: `20260724-2045-finger2-spatial-dof`, `20260724-2100-spatial-finger2-retrain-seed0`
- Related debug issues: `DBG-20260724-002`
- Applies to: corrected spatial finger geometry with legacy reset and discrete -10/-1/0.1/10 contact reward
- Does not apply to: future three-contact initialization or dense approach shaping

### Finding
After finger 2 became geometrically reachable, a matched 25k retraining run still never produced a multi-contact evaluation step. Reachability was necessary but insufficient because the legacy reset and policy did not enter the three-contact basin.

### Evidence
EXP-005 dynamically verified three-contact grasps. EXP-006 evaluated zero two-/three-contact occupancy at all checkpoints, with 20/20 tilt terminations and best rotation 1.53°.

### Implication
Test reset/exploration coverage before further reward scaling. The next discriminating factor is three-contact initialization, not a larger contact bonus.

### Caveats
One training seed and a 25k adaptation budget. This does not prove that longer or from-scratch training can never discover contact.

## FIND-20260724-001: Finger 2 is kinematically excluded from rod contact
- Confidence: high
- Supporting runs: `20260724-1730-contact-reachability`
- Related debug issues: `DBG-20260724-001`
- Applies to: `models/three_finger_rod.xml` at commit `46fa9b8`
- Does not apply to: future geometry after EXP-20260724-005

### Finding
Finger 2 cannot contact the rod under the current model at any joint configuration because its planar chain remains at world Y=+0.04 m while the rod lies near Y=-0.05 m. The minimum observed surface gap is 66.66 mm, consistent with the 90 mm plane separation minus 24 mm combined collision radii.

### Evidence
A 60,000-configuration, three-seed bounded search found 4,166 two-contact configurations and no three-contact configuration. Fingers 0 and 1 each achieved approximately -24 mm signed distance and registered simultaneous 32–54 N reset forces; finger 2 remained at least 66.66–68.27 mm away.

### Implication
Do not use a three-contact reward or interpret its failure as an RL exploration problem until finger-2 geometry is corrected and reachability is reverified.

### Caveats
The conclusion is specific to the current simplified planar hand model and rod placement. Geometry changes require rerunning the reachability protocol.

## F1 — Stage 0 needs a stable tip constraint under gravity
A MuJoCo `<connect>` tip anchor is a **spherical point constraint**, not a hinge. If the tip is below the COM, the rod is an inverted pendulum and axis-tilt objectives fight gravity. Hang the tip **above** the COM (or add an axial hinge / orientation stabilizer).

## F2 — Stiff solref + high kp causes free-joint blow-ups
`solref≈0.004` with `kp≈40` produced DOF-12 NaNs. Use milder contact/equality solref (~0.008), lower kp, free-joint damping, and terminate unstable episodes.

## F3 — Stage 0 can exceed 180° with hanging tip + stabilizer
Open-loop and trained PPO both achieved large +axial rotation with tip error ≪2 cm. Best recorded Stage 0 eval: success_rate 0.95, mean rotation ~528°.

## F4 — Abrupt removal of Stage 0 assists breaks transfer
Resuming into Stage 1 (no connect, no stabilizer) yielded negative mean rotation and high drops despite Stage 0 mastery. Assists should be faded or replaced with an intermediate curriculum stage.

## F5 — Mass randomization must restore baselines
Stage 2 `body_mass *= U` without restore drifts unboundedly. Always restore baseline mass/inertia (and friction) before applying a fresh scale.

## FIND-20260723-006: Independent checkpoint selection is required during assist fade
- Confidence: high
- Supporting runs: `20260723-0130-stage1-stab020-lowlr-seed0`
- Applies to: resumed PPO assist-fade training
- Does not apply to: unevaluated algorithms

### Finding
The final checkpoint can be worse than a periodic checkpoint even with low KL and zero clip fraction.

### Evidence
At stabilizer 0.20, step 301480 passed (199.58°, 16.04 mm, drop 0.15); final failed (164.66°, 18.14 mm, drop 0.30).

### Implication
Evaluate and retain periodic checkpoints; do not automatically promote `final_model.zip`.

### Caveats
Observed on one training seed.

## FIND-20260723-007: Stabilizer removal has a sharp performance cliff
- Confidence: medium
- Supporting runs: `20260723-0130-stage1-stab020-lowlr-seed0`, `20260723-0200-stage1-stab018-lowlr-seed0`
- Related debug issues: DBG-20260723-003
- Applies to: current reward, observation and 9-action debug hand

### Finding
Scale 0.20 passes, while 0.18 fails all aggregate gates and 0.10–0.00 produces severe drop/negative rotation.

### Evidence
0.20: 199.58°, 16.04 mm, drop 0.15. 0.18 best: 177.18°, 20.06 mm, drop 0.30. Scale 0: -5.49°, drop 0.95.

### Implication
Further training should change measurement/reward of stabilizer dependence rather than repeat the same 25k fine-tune.

### Caveats
Multi-training-seed validation is not yet available.

## FIND-20260723-008: Model capacity helps but does not replace reward design
- Confidence: medium
- Supporting runs: `20260723-0610-capacity512-stab018-seed0`, `20260723-0611-capacity256-resetopt-stab018-seed0`, `20260723-0820-capacity512-stab018-tiltw010-rot160-seed0`
- Applies to: current 48-observation, 9-action PPO task

### Finding
A function-preserving 2x512 expansion adapted substantially better than a matched 2x256 reset-optimizer control, but capacity alone did not pass the task. The larger model passed only after reward balance was corrected.

### Evidence
Matched capacity result: 126.39° versus 37.79°, both at drop 0.05. The 512/rot160 candidate later passed stabilizer 0.18 at 229.57° and stabilizer 0.15 at 200.00°.

### Implication
Retain 2x512 for Stage 2, but diagnose reward components before increasing capacity again.

### Caveats
One training seed.

## FIND-20260723-009: Correct Stage 2 keeps the endpoint joint
- Confidence: high
- Supporting runs: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Applies to: intended Stage 2 mechanics

### Finding
Stage 2 keeps the point connect active as a ball/universal joint and removes only the external axis stabilizer.

### Evidence
With connect active, mean tip error was 4.03 mm while all failures were axis-tilt terminations. The earlier connect-off visual run showed 12 cm endpoint error and is retained only as a misconfigured control.

### Implication
All Stage 2 train/eval/video commands must pass `--tip-connect --axis-stabilizer-scale 0`.

## FIND-20260723-010: Increasing absolute tilt weight does not teach Stage 2 recovery
- Confidence: high
- Supporting runs: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`, `20260723-1230-stage2-tipjoint-tiltw025-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current Stage 2 PPO transfer from the stabilizer-0.15 parent

### Finding
Increasing axis-tilt penalty weight from 0.10 to 0.25 amplified the intended reward component but did not reduce axis-tilt terminations.

### Evidence
The weighted component moved from -2.03 to -5.01 per step. Both conditions ended 20/20 episodes on axis tilt, with mean final tilt near 41°.

### Implication
Do not keep increasing this absolute penalty. Test a recovery/progress-shaped signal or curriculum that supplies a local corrective learning signal.

### Caveats
One training seed and one parent checkpoint.

## FIND-20260723-011: One-step tilt-recovery shaping does not fix direct assist removal
- Confidence: medium
- Supporting runs: `20260723-1500-stage2-tilt-recovery40-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current Stage 2 PPO transfer with tip joint on, stabilizer zero, and recovery scale 40

### Finding
A clipped reward for decreasing tilt increased mean axial rotation slightly but did not improve episode survival after direct stabilizer removal.

### Evidence
Mean rotation increased from 1.13° to 4.22°, while all six checkpoints still ended 20/20 evaluation episodes on axis tilt. Final tilt stayed near 41.4° and drop stayed 1.0.

### Implication
Do not extend or merely rescale this recovery term. Test the curriculum transition and Stage 2 randomization separately.

### Caveats
One training seed and one recovery scale; this does not rule out richer observations or longer-horizon recovery objectives.

## FIND-20260723-012: Stage 2 failure is an assist cliff, not randomization transfer
- Confidence: high
- Supporting runs: `20260723-1600-stage2-rand-adapt-stab012-seed0`
- Related debug issues: DBG-20260723-005
- Applies to: current 2x512 checkpoint and Stage 2 mass/friction distribution

### Finding
The stabilizer-0.12 policy transfers to Stage 2 randomization almost unchanged, while performance collapses nonlinearly below stabilizer 0.10.

### Evidence
The fixed parent achieved 178.51° and drop 0.05 at scale 0.12, and 176.55° with drop 0 at 0.10. Scale 0.08 fell to 132.42° with drop 0.40; scale 0.02 and 0 had 20/20 tilt terminations.

### Implication
Treat 0.10→0.08 as the next curriculum transition. Do not attribute the Stage 2 zero-stabilizer failure primarily to mass/friction randomization.

### Caveats
One checkpoint and fixed evaluation seeds; the exact cliff may move with a different trained policy.
## FIND-20260724-013: Hard 20-step contact gate does not bootstrap three-contact behavior
- Confidence: medium
- Supporting runs: `20260724-2200-stage0-contact30-seed0`, `20260724-2230-stage0-contact-gate-seed0`
- Related debug issues: DBG-20260724-002
- Applies to: fresh Stage 0 PPO from the current reset distribution
- Does not apply to: contact-friendly resets or curricula with a grace period

### Finding
Requiring a +5 rolling contact-reward sum within the first 20 steps does not bootstrap simultaneous three-finger contact.

### Evidence
All five gated checkpoints had 0% three-contact and finger2 occupancy over fixed seeds 0–19. The best checkpoint had 17/20 contact-support terminations and 0% success.

### Implication
Do not tune this threshold further from the same reset distribution. Change initialization or curriculum exposure first.

### Caveats
One training seed and a 25k-step budget. The gate may be useful after the policy already has a three-contact behavior.

## FIND-20260823-014: Flip the Allegro hand at its subtree root
- Confidence: medium
- Supporting runs: `reversed_world_x_180_20260823-161731`
- Applies to: Allegro index/middle/thumb preview geometry around the current vertical rod
- Does not apply to: policy compatibility or training-reset force safety

### Finding
A 180° world-X flip about the rod-center pivot can be represented as one transform on the `palm` root; no child joint frame needs or should receive an independent rotation.

### Evidence
Both bottom-revolute and bottom-point-connect preview models compiled with unchanged palm-relative transforms and retained 3/3 geometric and force contacts after 0.2 s.

### Implication
If the reversed orientation is accepted, implement it as an explicit root-transform option and preserve the current MJCF as the default baseline.

### Caveats
This is deterministic preview evidence only. A0 lost index contact after 0.2 s at both the 10 mm-clearance placement and the later 30 mm-clearance placement with the thumb root moved 10 mm inward; C3 retained 3/3 in both. No policy was evaluated, so reset dynamics and policy transfer remain unvalidated.

## FIND-20260823-015: Palm-root pose files preserve Allegro kinematics
- Confidence: high
- Supporting runs: `20260823-hand-pose-interface-validation`
- Applies to: current Allegro revolute and point-connect task models
- Does not apply to: arbitrary MJCFs without a world-parented `palm` body

### Finding
An absolute model-frame transform applied only to the world-parented `palm` body moves the full hand rigidly while preserving all palm-relative body transforms, joint positions, and joint axes.

### Evidence
Deterministic comparisons across both physics variants found zero changed non-palm local transforms and zero changed joint positions/axes. Both retained 12-D actions and 48-D observations, and a configured reset returned finite observations.

### Implication
Future candidate hand placements should use the versioned hand-pose configuration instead of editing each finger subtree or replacing the validated MJCF defaults.

### Caveats
Preserved kinematics do not imply preserved contact quality, reset stability, or policy performance. Each newly saved pose still requires a reset/contact audit before training.

## FIND-20260823-016: Palm-pose validity does not imply grasp-reset validity
- Confidence: high
- Supporting runs: `20260823-1730-two-phase-force-pose-smoke-seed0`
- Related debug issues: `DBG-20260823-003`
- Applies to: rigid Allegro palm-root pose files combined with fixed reset/grasp joint vectors
- Does not apply to: poses with separately validated pose-specific joint resets

### Finding
A valid palm-root transform can preserve all kinematics and model interfaces while
moving a previously valid joint-space grasp outside the three-tip contact basin.

### Evidence
The newly saved, schema-valid pose produced 0.00 three-tip occupancy and 0 N index
normal-force p95 on both fixed and unseen revolute seed sets. All episodes ended
on the contact-support gate, despite finite 48-D observations and loadable PPO
checkpoints.

### Implication
Every new palm pose requires a joint-reset/contact-force audit in both physics
modes and both mass endpoints before long training. Do not infer task progress
from unwrapped rod displacement when support-gated rotation reward is zero.

### Caveats
The corrective revolute joint vector is now recorded in
`configs/hand_grasps/my_grasp_revolute_shared.json`; it is not robust for
tip-connect `s=400`.

## FIND-20260823-017: Pose-specific reset support does not ensure policy support
- Confidence: high
- Supporting runs: `20260823-1740-my-grasp-shared-reset-audit-seed0`, `20260823-1820-two-phase-grasp-recovery-cont-seed0`, `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Related debug issues: `DBG-20260823-003`, `DBG-20260823-004`
- Applies to: saved-pose revolute `s=400` PPO with the recovered companion grasp
- Does not apply to: alternative optimizers, rewards, or success thresholds

### Finding
A reset that retains 3/3 contacts under zero action across fixed noise seeds can
still produce a policy that violates support after learning useful rotation.

### Evidence
The reset passed 10/10 `s=400` seeds for 100 steps. The 200k policy rotated
200.56°/192.06° on fixed/unseen sets but violated support in 20%/50% of episodes.
At 300k, violation rose to 100%/90% while total force p95 rose from about
164–169 N to 231–234 N.

### Implication
Keep reset robustness and policy robustness as separate gates. Select checkpoints
using termination and force metrics in addition to total rotation, and do not
accept the reset's static force as the rotating-policy force reference.

### Caveats
One training seed and one PPO lineage. The evidence does not distinguish reward
misspecification from optimizer instability.

## FIND-20260823-018: Ungated rotation credit exposes unsupported rotation
- Confidence: medium
- Supporting runs: `20260823-1830-two-phase-grasp-recovery-cont2-seed0`, `20260823-1858-rotation-credit-ablation-A-s400-seed0`, `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`
- Related debug issues: `DBG-20260823-004`
- Applies to: current saved-pose revolute `s=400` PPO lineage
- Does not apply to: policies trained with a stronger continuous contact-loss objective

### Finding
Zeroing rotation reward below three contacts suppresses learned transient angle,
but removing that rule does not create sustained supported manipulation. If hard
support termination is also removed, the policy learns a high-angle,
low-three-contact behavior.

### Evidence
Against a matched continuation from the same checkpoint, A raises fixed/unseen
angle from 183.18°/188.97° to 294.01°/277.39° but retains 20/20 support
terminations. B reaches 374.04°/379.20° while three-contact occupancy falls to
0.288/0.335. All sustained-omega success rates are zero.

### Implication
Do not treat higher unwrapped angle or return from ungated credit as task
improvement. Retain the original acceptance gate and pair rotation credit with an
explicit continuous support objective before considering curriculum progression.

### Caveats
One training seed. B changes two factors and is diagnostic rather than a clean
single-factor causal estimate.

## FIND-20260823-019: Contact scaling alone does not coordinate finger gait
- Confidence: medium
- Supporting runs: `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`, `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Related debug issues: `DBG-20260823-004`
- Applies to: current revolute `s=400` policy lineage with ungated rotation credit and no support termination
- Does not apply to: explicit gait-phase observations or leave-return objectives

### Finding
Reducing the full contact reward ladder to 10% increases finger leave/recontact
and net angle, but worsens support and does not create sustained axial rotation.

### Evidence
C records mean leave-return events on all three fingers and increases fixed/unseen
angle to 533.97°/546.98°. Relative to B, three-contact occupancy falls from
0.288/0.335 to 0.179/0.262, >=2-contact fraction falls to 0.604/0.578, and
zero-contact gaps reach 1.76/1.68 s. Sustained-omega success remains zero.

### Implication
Do not continue tuning contact reward scale alone. Continuous gait likely needs
phase information or an objective for timely recontact.

### Caveats
One scale and one training seed. Large angle is not ballistic-free success.

## FIND-20260823-020: Friction-to-force response changes under policy transfer
- Confidence: medium
- Supporting runs: `20260823-1855-net-angle-C-force-curriculum-seed0`
- Related debug issues: `DBG-20260823-005`
- Applies to: C-policy revolute mass transfer with per-friction PPO retraining
- Does not apply to: fixed open-loop actions or untrained contact statics

### Finding
The physical `mu proportional to mass` rule is a useful initial estimate but is
not a reliable closed-loop force-calibration direction after the policy retrains.

### Evidence
The physical estimates match the 98.058 N reference at `s=200` (98.208 N) and
`s=100` (96.651 N). At `s=50`, lowering `mu` from 0.5 to 0.396 raises force from
77.670 to 81.298 N. At `s=25`, however, lowering `mu` from 0.25 through 0.168,
0.101, and 0.10 lowers force from 65.903 to 53.386 N despite task success 1.0.
A separately predeclared extension gives 54.669 N at 0.05 and 49.427 N at 0.025.
The sequence from 0.10 downward is non-monotone and remains far below target;
>=2-contact occupancy drops to 0.157 at scale 0.025.

### Implication
Use physical scaling only to seed calibration. Before repeated adaptation,
identify the local measured response under a fixed parent and predeclare how
non-monotone or policy-dependent responses will be handled.
Do not lower friction further for this `s=25` lineage: friction-only reduction is
insufficient under the tested transfer protocol.

### Caveats
One PPO seed. This establishes failure of the declared decreasing-friction path
through scale 0.025, not a universal claim about every possible policy or force
controller.

## FIND-20260823-021: Learned-policy force cannot gate required force
- Confidence: high
- Supporting runs: `20260823-1855-net-angle-C-force-curriculum-seed0`, `20260823-2015-proportional-physics-C-seed0`
- Related debug issues: `DBG-20260823-005`

### Finding
Conditioned normal force from an adapting policy is not an estimator of minimum
required pressing force. The historical conditioning retained only about
10–18% of steps and changed trajectory/contact geometry across stages.

### Evidence
Removing this invalid gate while scaling the full friction vector and explicit
rod passive dynamics allowed every revolute stage from `s=400` through `s=1` to
pass fixed and unseen net-angle success at 1.0.

### Implication
Keep policy contact force as a diagnostic. Use a fixed controlled preload/torque
protocol for required-force questions.

### Caveats
The controlled sweep did not demonstrate exact invariance: its strict tracking
criterion found no passing preload through multiplier 3 at `s=400` or `s=25`,
while `s=1` first passed at multiplier 1.5 (9.31 N median). Contact compliance,
hand dynamics, and friction-cone coupling remain mass-dependent limitations.

## FIND-20260823-022: Revolute net-angle transfer reaches s1 but support collapses
- Confidence: high
- Supporting runs: `20260823-2015-proportional-physics-C-seed0`
- Related debug issues: `DBG-20260823-006`

### Finding
The declared net-angle gate accepts all corrected revolute stages, but this does
not imply supported finger gaiting. Fixed >=2-contact fraction falls from 0.775
at `s=400` to 0.0146 at `s=1` while net angle rises from 544 to 11,385 degrees.

### Implication
The result permits the requested phase transition under the declared gate, but
contact quality must remain visible and the behavior should not be described as
supported manipulation.

### Caveats
Revolute mounting prevents physical rod drop. The same lineage fails immediately
after transfer to tip-connect, where lateral tilt and physical drop are possible.
