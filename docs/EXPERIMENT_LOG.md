# Experiment Log

## EXP-20260914-002: T00 2×2 continuation from ~100k to cumulative 300k
- Run ID: `20260914-1712-t00-ablation-{A,B,C,D}-continue300k-seed0`
- Date: 2026-09-14
- Status: completed
- Parent or baseline run: `20260914-1638-t00-ablation-{A,B,C,D}-hist{1|4}-rew{base|sup}-seed0` (EXP-20260914-001, ~106k)
- Git commit: dirty (ablation + history page)
- Git branch: `PhaseT_tilt_solving`
- Random seed: 0
- Device: cuda:4 (`CUDA_VISIBLE_DEVICES=4`, `--device cuda:0`)
- Duration: ~24 min wall (A–D sequential; A already finished ~410k because SB3 adds the current counter)
- Checkpoint: `runs/20260914-1712-t00-ablation-{A,B,C,D}-continue300k-seed0/checkpoints/final_model.zip`

### Question
Does the 100k online episode-length advantage of 4-frame history (B/D > A/C)
persist, narrow, or reverse by a cumulative 300k steps? Do the curves look
closer to a plateau?

### Hypothesis
If history stacking is a real early observability effect rather than a
transient, B and D should remain longer than A and C at 300k. Support-aware
C/D should still show lower return from the wobble penalty. Eval success is
not expected to leave 0.

### Change from Baseline
No hyperparameter or physics change. Resume each 100k `final_model.zip` plus
VecNormalize with `--continue-timesteps` and `total_timesteps=300000`.
New run directories so 100k artifacts are not overwritten. SB3 2.9
`reset_num_timesteps=False` **adds** `num_timesteps`, so jobs continued to
~410k; the reported snapshot is the 303,104 row.

### Configuration
- Same as EXP-20260914-001 except resume to a 300k snapshot
- Resume-in-new-dir, not from scratch
- Sequential A→B→C→D on GPU 4

### Success Criteria
Record last-step return / length / KL / EV for A–D. State whether the B/D
length advantage persisted. Do **not** treat longer episodes as task success.
Eval is not required for this continuation unless a condition plateaus with
clear recovery.

### Result
Completed. The 100k B/D length lead **narrowed**. At 303k, C is longest
(68.6); B is only +2 vs A; D equals A. Returns still rising. EV ~0.98.
No eval re-run. No NaN. Task still unsolved.

### Key Metrics
| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Last-step return (100k) | +22.1 | −8.9 | −47.5 | −17.9 |
| Last-step length (100k) | 40.0 | 53.7 | 46.8 | 54.6 |
| Return (303k) | +221.8 | +197.9 | +169.6 | +164.1 |
| Length (303k) | 62.4 | 64.4 | 68.6 | 62.4 |
| KL (303k) | 0.0104 | 0.0105 | 0.0094 | 0.0108 |
| EV (303k) | 0.987 | 0.975 | 0.992 | 0.984 |

### Visual Evidence
- 100k curves: `reports/comparisons/20260914-t00-ablation-train-curves.png`
- 300k curves: `reports/comparisons/20260914-t00-ablation-train-curves-300k.png`
- Page: `docs/pages/t00-ablation-history.html`
- Seed-6 demo: `runs/20260914-t00-seed6-tilt-collapse/index.html`
- 300k A/B/C/D videos (ckpt 306432, seeds 6 and 10000; all `axis_tilt`):
  `runs/20260914-1748-t00-ablation-videos-300k/`
  (grid: `ABCD_grid_seed6.mp4`; browser copy: `docs/pages/t00-ablation-videos-300k/`)
- Seed-6 A–D contact/tilt **frame sliders** (re-rolled traces, ckpt 306432):
  `runs/20260914-1805-t00-ablation-demos-300k/index.html`
  (inheritable: `docs/pages/t00-ablation-demos-300k/`; live
  http://127.0.0.1:8767/pages/t00-ablation-demos-300k/?v=scrub20260915)

### Interpretation
H1’s early online-length signal does not persist as a hist4 advantage at
300k. C/D still have lower return (wobble penalty), not failed training.
EV/KL look stable; return and length have not plateaued. This is not a
solved-task claim.

### Decision
reject hist4 as a lasting T00 fix on length. Keep FIND-20260914-002 as an
early-only snapshot. Do not retune λ. Do not start T01.

### Next Step
Probe whether ≥2-contact recovery is physically feasible under the current
tip-connect grasp. Inherit the ablation page and the seed-6 timeline into
the public project page when that site is updated.


## EXP-20260914-001: T00 2×2 ablation (observation history × support-aware reward)
- Run ID: `20260914-1638-t00-ablation-{A,B,C,D}-hist{1|4}-rew{base|sup}-seed0`
- Date: 2026-09-14
- Status: completed
- Parent or baseline run: `20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0` (published transfer T00; A–D trained from scratch)
- Git commit: `c948861` (dirty: ablation implementation)
- Git branch: `PhaseT_tilt_solving`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009; traces 6 and 10000
- Device: cuda:4 (`CUDA_VISIBLE_DEVICES=4`)
- Duration: ~9.1 min wall (A–D sequential, ~800–2000 fps)
- Checkpoint: `runs/20260914-1638-t00-ablation-{A,B,C,D}-hist{1|4}-rew{base|sup}-seed0/checkpoints/final_model.zip`

### Question
Does short observation history (H1) and/or a support-aware rotation/wobble
objective (H2) reduce T00 support collapse, even if net rotation temporarily
falls?

### Hypothesis
H1: the 48-D single-frame MLP cannot distinguish unloading vs recovery, so
4-frame stacks (160 ms) should raise re-contact rate.
H2: rotation reward remains large while support is released, so contact-gated
rotation (×1 / ×0.1 / ×0) plus `-0.5 ||ω_perp||^2` on `n_contact<2` should
make recovery preferable to spinning until `axis_tilt`.
If B≈A and C≈A but D≫A, both factors are required together.

### Change from Baseline
Only `obs_history_len` ∈ {1,4} and `support_aware_reward_enabled` ∈ {false,true}.
T00 physics, grasp, s=400, μ=4, tilt kill 1.2 rad, PPO `[512,256,128]`,
32 envs, n_steps 256, batch 256, LR `3e-5`, ent_coef 0, 100k steps, seed 0,
and eval seeds are shared. No GRU/LSTM, no joint torque, no T01, no new
contact termination.

### Configuration
- Algorithm: SB3 PPO from scratch on T00 (not a new curriculum stage)
- Environment: bottom tip-connect, `s=400`, μ=4, solref 0.008, stabilizer 0
- Reward terms: existing DexScrew + discrete contact ×0.10; C/D add support gating and wobble
- Observation space: A/C 48-D; B/D 192-D stacked
- Action space: 12-D
- Network: [512, 256, 128] (input dim only changes)
- Learning rate: 3e-5
- Number of environments: 32
- Training steps: 100,000
- Curriculum stage: T00 only
- Evaluation protocol: net-angle, 10 fixed + 10 unseen; traces seeds 6 and 10000

### Success Criteria
Directional success does **not** require net-angle ≥ 0.50. Treat as positive if
re-contact rate, 1-contact duration, post-loss ω_perp, axis_tilt-after-loss
fraction, and episode length improve even if rotation drops (e.g. 177°→150°
but length 34→150). If A≈B≈C≈D, do not retune λ; inspect grasp/mechanics next.

### Result
Completed. No condition resolved support collapse. All 80 eval episodes terminate
on `axis_tilt`. Versus from-scratch A (the matched control):
- B (history): similar or slightly worse collapse metrics; recontact 0.09 vs A's 0.17/0.09.
- C (support-aware reward): **worse** — episode length 37 vs 54, recontact 0,
  post-loss Δω_perp ~8 vs ~2, 100% of axis_tilt deaths preceded by support loss.
- D (both): no complementary win; recontact 0, length ~50, rotation highest (327°).

From-scratch A itself rotates more and lives longer than the published *transfer*
T00 (305° / 54 steps vs 177° / 35), but that is a different initialization, not
evidence that history or the new reward helped.

### Key Metrics
| Metric | T00 transfer | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Success rate | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Episode length (fixed) | ~35 | 54.0 | 50.0 | 36.5 | 49.7 |
| Rotation progress (fixed) | 177.0° | 304.9° | 297.0° | 275.2° | 326.5° |
| Recontact success rate (fixed) | 0.00 | 0.17 | 0.09 | 0.00 | 0.00 |
| Fraction axis_tilt preceded by support loss (fixed) | ~1 (seed 6) | 0.40 | 0.80 | 1.00 | 0.50 |
| Δω_perp after support loss (fixed) | — | 2.18 | 2.87 | 7.87 | 2.45 |

### Visual Evidence
- Diagnostic traces: `runs/20260914-1638-t00-ablation-{A,B,C,D}-*/traces/{fixed,unseen}/`
- Comparison: `reports/comparisons/20260914-t00-ablation-A-B-C-D.md`
- Known failure replay: `runs/20260914-t00-seed6-tilt-collapse/index.html`

### Interpretation
H1 is not supported at `history_len=4`. H2 as implemented (rotation gate +
λ=0.5 wobble) does not induce re-contact and makes C's unsupported interval more
violent. D is not a joint rescue. This matches the predeclared "none of A/B/C/D
improves collapse → do not retune λ" branch.

### Decision
reject B, C, and D as T00 fixes. Keep A only as the from-scratch control. Do not
start T01 and do not sweep `low_support_wobble_scale`.

### Next Step
Inspect whether ≥2-contact recovery is physically feasible with the current
tip-connect grasp, including load redistribution in the force observations and
actuator delay. A T-Balance / grasp-redesign experiment is the next smallest
discriminating step.


## EXP-20260824-001: DexScrew tilt-growth penalty at tip-connect s=100
- Run ID: `20260824-0000-dexscrew-tilt-growth-s100-tip-seed0`
- Date: 2026-08-24
- Status: completed
- Parent or baseline run: `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0` (zero-shot tip-connect s=100)
- Git commit: `b0b76d4` (dirty: DexScrew growth implementation; recovery left at default 0)
- Git branch: `cursor/teleop-hand-keyboard`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Device: cuda:4
- Duration: ~46 s training (65,536 steps, ~1758 fps)
- Checkpoint: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/checkpoints/final_model.zip`

### Question
Does a default-preserving DexScrew tilt-growth penalty, scaled so a 0.05 rad
increase near the 0.25 rad success gate competes with rotation (`2.5 * ω` at
`ω=1`), raise net-angle success at bottom tip-connect `s=100` above 0.50?

### Hypothesis
If collapse is monotonic, a one-sided recovery term never fires. A growth
penalty that is nonzero while tilt is increasing, especially around/above
0.25 rad, should provide on-policy samples and reduce terminal tilt versus
zero-shot and versus recovery-only shaping.

### Change from Baseline
Only DexScrew tilt-growth: `--axis-tilt-growth-scale 50` (default 0). Recovery
stays 0. Formula:
`r_growth = -50 * clip(curr − prev, 0, 0.05) * clip((curr − 0.05) / 0.20, 0, 1)`.
Existing tilt-state penalty, 48-D observations, LR `3e-4`, C gait flags, saved
pose, recovered grasp, and full-vector proportional friction at `s=100` are
fixed. Parent VecNormalize is reused.

### Configuration
- Algorithm: SB3 PPO resume
- Environment: bottom tip-connect, `s=100`, `mu=1.0`, solref 0.008
- Reward terms: DexScrew + discrete contact ×0.10 + tilt penalty 1.0 + growth 50 + recovery 0
- Observation space: 48-D
- Action space: 12-D
- Network: [512, 256, 128]
- Optimizer: PPO default Adam
- Learning rate: 3e-4
- Batch size: 256
- Horizon: n_steps 256
- Number of environments: 32
- Training steps: 65,536
- Domain randomization: none
- Curriculum stage: tip-connect s=100 only
- Evaluation protocol: net-angle, 10 fixed + 10 unseen episodes

### Success Criteria
Fixed and unseen net-angle success each ≥ 0.50. Otherwise reject and do not
start other masses.

### Result
Failed the gate. Trained success 0.0/0.0 versus zero-shot 0.0/0.0 and recovery
0.0/0.0. Rotation rose from 182.31/169.09° to 280.12/264.34° (recovery was
190.95/223.07°). Final/max tilt stayed ~80–83°. All 20 trained episodes
terminate on `axis_tilt`. Growth share of Σ|r| is 0.033/0.035 versus recovery
share 0.002/0.004 on the sibling run. Mean growth −0.452/−0.547 versus tilt
penalty −2.93/−3.17.

### Key Metrics
| Metric | Zero-shot fixed | Current fixed | Change |
|---|---:|---:|---:|
| Success rate | 0.00 | 0.00 | 0.00 |
| Mean return (online end) | — | −0.74 | — |
| Position error | 0.00067 m | 0.00077 m | +0.00010 m |
| Rotation progress | 182.31° | 280.12° | +97.81° |
| Episode length (online) | 22.7 smoke | 40.3 | +17.6 |
| Constraint violation rate | 1.00 | 1.00 | 0.00 |

Unseen rotation 169.09° → 264.34°; unseen success remained 0.

### Visual Evidence
- Training curve: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/plots/train_return_length_kl.png`
- Evaluation video: none succeeded
- Failure-case video: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/videos/tip_connect_best_00_seed10005_rot399deg_tilt85deg_steps65_axis_tilt.mp4`
- Zero-shot failure: `runs/20260824-0000-dexscrew-tilt-growth-s100-tip-seed0/videos/zeroshot/tip_connect_best_00_seed10000_rot355deg_tilt74deg_steps63_axis_tilt.mp4`
- Comparison: `reports/comparisons/20260824-dexscrew-tilt-growth-s100.md`

### Interpretation
Measured: tilt collapse remains monotonic (max tilt = final tilt). Growth does
fire (~3–4% of |reward|), so the recovery-run diagnosis was correct that a
signal while increasing is on-policy here. That was not sufficient to hold
tilt below 0.25 rad. Rotation share rose to 38–42% while growth stayed ~−0.5
per step versus rotation +5.2 to +6.5. This rejects growth-at-g=50 as a Phase T
gate solution. PPO KL ~0.18 and clip fraction ~0.64 remain high, similar to
recovery.

### Decision
reject

### Next Step
Do not start other masses. Do not retune this growth scale as the next
one-factor test. Prefer a tilt assist, grasp change, or other
dynamics/initialization intervention.

## EXP-20260823-018: DexScrew back-to-balance tilt recovery at tip-connect s=100
- Run ID: `20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: `20260823-2015-proportional-physics-C-seed0-R02-s100-mu1-seed0` (zero-shot tip-connect s=100)
- Git commit: `b0b76d4` (dirty: DexScrew recovery implementation)
- Git branch: `cursor/teleop-hand-keyboard`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Device: cuda:4
- Duration: ~45 s training (65,536 steps, ~1800 fps)
- Checkpoint: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/checkpoints/final_model.zip`

### Question
Does a default-preserving DexScrew one-sided tilt-recovery term, scaled to
compete with rotation (`2.5 * ω` at `ω=1`) near the 0.25 rad success gate,
raise net-angle success at bottom tip-connect `s=100` above the 0.50 gate?

### Hypothesis
If the dominant failure is tilt that grows through the 0.25 rad gate and then
saturates the quadratic penalty above 0.75 rad, rewarding `Δtilt < 0` while
`tilt > 0.05` will produce recoverable episodes without needing a larger
absolute penalty.

### Change from Baseline
Only DexScrew tilt recovery: `--axis-tilt-recovery-scale 50` (default 0).
Formula: `r_recovery = 50 * clip(prev − current, 0, 0.05) * 1[current > 0.05]`.
Existing tilt penalty, 48-D observations, LR `3e-4`, C gait flags, saved pose,
recovered grasp, and full-vector proportional friction at `s=100` are fixed.
Parent VecNormalize is reused.

### Configuration
- Algorithm: SB3 PPO resume
- Environment: bottom tip-connect, `s=100`, `mu=1.0`, solref 0.008
- Reward terms: DexScrew + discrete contact ×0.10 + tilt penalty 1.0 + recovery 50
- Observation space: 48-D
- Action space: 12-D
- Network: [512, 256, 128]
- Optimizer: PPO default Adam
- Learning rate: 3e-4
- Batch size: 256
- Horizon: n_steps 256
- Number of environments: 32
- Training steps: 65,536
- Domain randomization: none
- Curriculum stage: tip-connect s=100 only
- Evaluation protocol: net-angle, 10 fixed + 10 unseen episodes

### Success Criteria
Fixed and unseen net-angle success each ≥ 0.50. Otherwise reject and do not
start other masses.

### Result
Failed the gate. Trained success 0.0/0.0 versus zero-shot 0.0/0.0. Rotation
rose from 182.31/169.09° to 190.95/223.07°. Final/max tilt stayed ~80–83°.
All 20 trained episodes terminate on `axis_tilt`. Recovery share of Σ|r| is
0.002/0.004. Mean recovery +0.032/+0.056 versus tilt penalty −3.83/−3.34.

### Key Metrics
| Metric | Zero-shot fixed | Current fixed | Change |
|---|---:|---:|---:|
| Success rate | 0.00 | 0.00 | 0.00 |
| Mean return (online end) | — | −27.3 | — |
| Position error | 0.00067 m | 0.00073 m | +0.00006 m |
| Rotation progress | 182.31° | 190.95° | +8.64° |
| Episode length (online) | 22.7 smoke | 41.0 | +18.3 |
| Constraint violation rate | 1.00 | 1.00 | 0.00 |

Unseen rotation 169.09° → 223.07°; unseen success remained 0.

### Visual Evidence
- Training curve: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/plots/train_return_length_kl.png`
- Evaluation video: none succeeded
- Failure-case video: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/tip_connect_best_00_seed10006_rot333deg_tilt81deg_steps46_axis_tilt.mp4`
- Zero-shot failure: `runs/20260823-2350-dexscrew-tilt-recovery-s100-tip-seed0/videos/zeroshot/tip_connect_best_00_seed10000_rot355deg_tilt74deg_steps63_axis_tilt.mp4`
- Comparison: `reports/comparisons/20260823-dexscrew-tilt-recovery-s100.md`

### Interpretation
Measured: tilt collapse remains monotonic (max tilt = final tilt), so the
one-sided recovery term has almost no on-policy samples. Rotation can still
exceed 180° while the 0.25 rad tilt gate fails. This supports the hypothesis
that a scale-only tilt-penalty increase was the wrong next knob, but it
rejects the hypothesis that adding recovery-only shaping at scale 50 is
sufficient at this budget. PPO KL ~0.20 and clip fraction ~0.65 remain high.

### Decision
reject

### Next Step
Do not start other masses. If continuing reward work, add a signal that is
nonzero while tilt is increasing near 0.25 rad, or change grasp/assist
dynamics rather than paying only for already-decreasing tilt.

## EXP-20260808-003 / Bottom-tip C0–C5 (hard tip → free tip reward)
- Progress: `runs/curricula/*/CURRICULUM_PROGRESS.md`
- Date: 2026-08-08
- Status: **running**
- Change: `--tip-anchor bottom` for C0–C4 hard tip; **C5** disables tip equality and raises DexScrew tip-error penalty (`scale=8`, `sigma=0.015`)
- Smoke: s=400, soft tilt 1.2, stop after C5
- Motivation: user wants support-tip skill then soft fixed tip without hard constraint

## EXP-20260808-002 / Mass–friction curriculum start s=400
- Progress: `runs/curricula/*/CURRICULUM_PROGRESS.md`
- Date: 2026-08-08
- Status: **smoke tip PASS with VN** (driver aborted on pre-fix eval)
- Change: C0/C1 `--start-scale 400` (was 40); keep μ_cap=4, tilt_term=1.2, tip solref/√s
- Motivation: videos at s=40 still look under-damped / force-dominated; heavier rod to buy episode length
- Result: top-hang C4 tip@400 **success=1.0** with VecNormalize; preferred two-finger cooperating gait

## EXP-20260808-001 / Mass–friction curriculum (s=40 → auto)
- Progress: `runs/curricula/*/CURRICULUM_PROGRESS.md`
- Date: 2026-08-08
- Status: **aborted at s=40**
- Script: `scripts/run_mass_friction_curriculum.py`

### Findings so far
- **C0 revolute @s=40:** works (ep_len=500). Ckpt: `.../20260808-0052-...-C0-.../final_model.zip`
- **C1 tip @s=40, μ×40:** worse than B1 (ep_len~10, all tilt kills)
- **C1 tip @s=40, μ_cap=4 + tip solref/√s:** still ep_len~7–12 under policy
- **Random probe:** s=1 and s=40 both ~40-step mean under random actions — mass alone ≠ free episode length
- **C1 + tilt_terminate=1.2:** early online ep_len ~50–90 then collapsed; gate failed (tilt_frac=1); videos ~22–34 steps, ~110° rot then axis_tilt ~70°
- **Qualitative (video):** C4 gait is **more like the target** than prior B1/B2 — **two fingers cooperating**. Keep mass curriculum as preferred path despite failed numeric gates; raise start mass (EXP-002).

### Design updates mid-flight
μ capped (`--rod-friction-cap 4`); tip solref /= √s; curriculum tip kill default **1.2 rad**.

### Result
s=40 insufficient for tip-connect *survival*, but gait style is preferred over previous transfer curriculums → try s=400 (EXP-002).

## EXP-20260802-007 / EXP-A1→B1: revolute then tip-connect+tilt transfer
- A1: `20260802-1439-exp-a1-revolute-sharedObs-omegaHold10-subproc64-1e6-seed0` (**passed** online; videos at `videos/ckpt_800k/`)
- B1: `20260802-1439-exp-b1-from-a1-tipconnect-tilt-subproc64-1e6-seed0` (**failed** task gate)
- First A1 stamp `20260802-1323-…` aborted at ~33k; relaunched via `setsid`.
- Date: 2026-08-02
- Status: completed
- Device: CUDA GPU5, 64 envs × 1e6 each, `ent_coef=0`
- Script: `scripts/run_revolute_then_tilt_curriculum.sh`

### Change from prior Arm A/B
1. Shared obs layout (dim **42**) — hand q/v + contacts + tip/ω/axis/tilt/linvel so PPO weights transfer.
2. Success = sustain `ω > 0.5` for **10 s** (episode 20 s); angle metric only.
3. Tip-connect tilt punishment `dexscrew_tilt_scale=1.0`; B1 resumes A1 with **fresh VecNormalize**.

### Question
Does revolute gait transfer to tip-connect+tilt better than training tip-connect from scratch (EXP-B0)?

### Result
**A1 yes / B1 no.** B1 online end: `ep_len≈18.8`, success≈0. Eval@final (20 eps): success=0, drop=1.0, all `axis_tilt` terminations, tilt≈43.9°, ω-hold=0. Component probe @600k: |rot|≈67%, |tilt|≈29% of Σ|terms|. Transfer alone does not fix tip-connect tilt collapse.

### Decision
Keep A1; next EXP-B2 adds online EMA mass balancing targeting 45% rot / 45% tilt (still hard tilt term).

## EXP-20260802-008 / EXP-B2: adaptive 45/45 rot–tilt mass
- Smoke: `20260802-1542-exp-b2-smoke-adaptiveMass-subproc8-2e5-seed0` — **infra pass**; mass/rot 0.81→0.45, mass/tilt 0.18→0.45.
- Full: `20260802-1555-exp-b2-from-a1-adaptiveMass45-subproc64-1e6-seed0`
- Change: [`allegro_rod_mvp/adaptive_mass.py`](../allegro_rod_mvp/adaptive_mass.py) EMA balancer; `--adaptive-reward-mass`; TB `mass/*`.
- Date: 2026-08-02
- Status: completed (mass targets hit; **task gate failed**)

### Question
Does online 45/45 rot–tilt mass balancing fix tip-connect tilt collapse vs B1?

### Result
**Mass yes, task no.** Training TB: mass/rot 0.80→0.45, mass/tilt 0.18→0.48; tilt_scale floored at 0.5. Online `ep_len` worsened (~18→12). Eval@final+VecNorm (20 eps): success=0, drop=1.0, all `axis_tilt`, tilt≈44.7°, rot≈74°. Videos: `.../videos/final/`. Vs B1: same failure mode; mass rebalancing alone insufficient under hard tilt termination.

### Decision
Next one-factor: soften/remove hard `axis_tilt > 0.7` termination (keep tilt reward); keep adaptive mass.

## EXP-20260802-005 / EXP-A0: Arm A revolute + DexScrew ω reward
- Run ID: `20260802-1250-exp-a0-revolute-dexscrew-subproc8-seed0`
- Date: 2026-08-02
- Status: completed (strong)
- Device: CUDA GPU5, 8 envs, 2e5 steps, ent_coef=0

### Question
Does revolute hinge + DexScrew ω/prox/pose/energy reward learn axial gait quickly?

### Result
**Yes.** Online success≈0.99; deterministic eval 20 seeds: rot≈366°, success≈0.85. Policy std stayed ~0.7.

### Artifacts
- Ckpt: `runs/20260802-1250-exp-a0-revolute-dexscrew-subproc8-seed0/checkpoints/final_model.zip`
- Videos: `.../videos/revolute_ep*_rot*.mp4`

### Decision
Adopt Arm A recipe as DexScrew-track baseline.

## EXP-20260802-006 / EXP-B0: Arm B tip-connect + DexScrew + tilt (ω sign fix)
- Run ID: `20260802-1305-exp-b0-tipconnect-dexscrew-tilt-omegaSignFix-subproc64-1e6-seed0`
- Date: 2026-08-02
- Status: completed (failed task gate; stress-test result)
- Parent: first B0 attempt `20260802-1255-...` had inverted ω sign (negative rotation); fixed then rerun.

### Question
Does the shared DexScrew reward + tilt penalty transfer to tip-connect with stabilizer 0?

### Result
**Not at this budget/config.** Final eval: rot≈-1.3°, success=0.00, tilt≈36.2°, terms={'none': 14, 'axis_tilt': 6}. Episodes stay short (tilt terminations dominate). Arm A succeeds; Arm B remains the free-orientation stress test (matches plan claim bar).

### Decision
Keep Arm A; revise Arm B (tilt scale / tip solref / mild stab) or accept as negative result vs revolute.

### Next Step
Compare report; optional B0 retune tilt_scale or tip stiffness one-factor.

## EXP-20260802-003: Stage0→1 transfer from 5e6 parallel ckpt (1e6 steps)
- Run ID: `20260802-1225-stage1-from-s0-5e6-subproc64-1e6-seed0`
- Date: 2026-08-02
- Status: completed (passed eval gate)
- Parent or baseline run: `20260802-0220-exp-infra-subproc64-1e9-seed0` / `ppo_rod_5000000_steps.zip`
- Git commit: dirty tree (`scripts/train_parallel.py` + docs)
- Git branch: `main`
- Random seed: 0
- Device: CUDA GPU5, 64 envs
- Duration: ~310 s (~3253 fps)
- Checkpoint: `runs/20260802-1225-stage1-from-s0-5e6-subproc64-1e6-seed0/checkpoints/final_model.zip` (+ `vecnormalize.pkl`)

### Question
Does the strong Stage 0 parallel policy (~0.9 online SR at 5e6) transfer into Stage 1 (softer tip spring, stab 0.15, tilt w=0.10, rot scale 160) after 1e6 additional parallel env-steps?

### Hypothesis
With `ent_coef=0` (DBG-20260802-001 mitigation) and Stage 1 assists, the policy retains rotation and meets the eval gate (rot>180°, tip<0.02 m, drop≤0.15).

### Change from Baseline
Resume `ppo_rod_5000000_steps.zip` into Stage 1 config; 1e6 steps; 64 envs; `ent_coef=0`; fresh VecNormalize (parent stats missing); VecNormalize saved every checkpoint.

### Configuration
- Algorithm: SB3 PPO resume
- Stage: 1; tip-connect on; solref=0.10; stabilizer=0.15; tilt_w=0.10; rot_scale=160
- Network: [512,256,128]; n_envs=64; n_steps=128; batch=512; steps=1e6; ent_coef=0
- Evaluation: 20-seed `eval_policy.py` after training

### Success Criteria
Eval gate pass on Stage 1; finite losses; no std explosion; VecNormalize artifacts saved.

### Result
**Passed** Stage 1 eval gate (20 seeds). rot_mean=1019.3°, tip=0.0183 m, success=0.55, drop=0.05, terminations={'none': 19, 'axis_tilt': 1}. Online end success≈0.37, std≈32.5 (no NaN). Fresh VecNormalize + ent_coef=0.

### Key Metrics
| Metric | Value |
|---|---:|
| Success rate | 0.55 |
| Rotation deg mean | 1019.3 |
| Tip error m | 0.0183 |
| Drop rate | 0.05 |
| Passed gate | True |

### Visual Evidence
- Videos: `runs/20260802-1225-stage1-from-s0-5e6-subproc64-1e6-seed0/videos/`
- Eval: `eval_final.json`

### Decision
Adopt; proceed to Stage 2.

### Next Step
EXP-20260802-004.

## EXP-20260802-004: Stage1→2 transfer (stab 0, 1e6 parallel steps)
- Run ID: `20260802-1231-stage2-from-s1-subproc64-1e6-seed0`
- Date: 2026-08-02
- Status: completed (failed eval gate)
- Parent or baseline run: `20260802-1225-stage1-from-s0-5e6-subproc64-1e6-seed0`
- Random seed: 0
- Device: CUDA GPU5, 64 envs
- Duration: ~256 s (~3926 fps)
- Checkpoint: final + mid `ppo_rod_200000_steps.zip`

### Question
Does the Stage 1 parallel policy survive Stage 2 (tip-connect, stab 0) after 1e6 steps?

### Hypothesis
ent_coef=0 + Stage1 VecNormalize keeps rot>180° and drop≤0.15 with fewer than 20/20 tilt terminations.

### Change from Baseline
Stage 1→2, stabilizer 0; resume Stage1 final+vecnormalize; 1e6 steps; ent_coef=0.

### Result
**Failed gate.** Final: rot=155.1°, tip=0.0136, success=0.05, drop=0.65, terms={'axis_tilt': 13, 'none': 7}, mean_tilt=32.9°.
200k: rot=590.7°, tip=0.0216, drop=0.70, terms={'axis_tilt': 14, 'none': 6}.

### Key Metrics
| Ckpt | rot° | tip m | success | drop | tilt/20 | passed |
|---|---:|---:|---:|---:|---:|:---:|
| 200k | 590.7 | 0.0216 | 0.05 | 0.70 | 14 | no |
| final | 155.1 | 0.0136 | 0.05 | 0.65 | 13 | no |

### Visual Evidence
- `runs/20260802-1231-stage2-from-s1-subproc64-1e6-seed0/videos/final/`
- `runs/20260802-1231-stage2-from-s1-subproc64-1e6-seed0/videos/ckpt_200k/`

### Interpretation
Stage0→1 works. Stage2 stab=0 still tilt/drop limited; more steps from 200k→1e6 hurt rotation.

### Decision
Reject Stage2 final. Keep Stage1 + Stage2-200k diagnostics. Prefer Arm A / tilt-aware next.

### Next Step
EXP-A0 revolute+ω, or Stage2 tilt single-factor ablation.

## EXP-20260802-002: Scale Stage 0 parallel budget to 1e9 env-steps
- Run ID: `20260802-0220-exp-infra-subproc64-1e9-seed0`
- Date: 2026-08-02
- Status: failed (NaN / std explosion at ~3.17e7 steps; did not reach 1e9)
- Parent or baseline run: `20260802-0217-exp-infra-subproc8-cuda-seed0`
- Git commit: `057f5e3` (dirty: uncommitted `scripts/train_parallel.py` + docs)
- Git branch: `main`
- Random seed: 0
- Device: CUDA (`CUDA_VISIBLE_DEVICES=5`, RTX 3090), host `batiquitos.ucsd.edu`
- Duration: ~2.8 h (~3362 fps); wall ~10140 s until crash
- Checkpoint: best pre-collapse `checkpoints/ppo_rod_5000000_steps.zip` (also 10M–30M post-collapse)

### Question
Does a much larger interaction budget (**1e9** total env-steps across parallel workers) produce Stage 0 success under the EXP-infra stack when 2e5 steps only moved return from ~-200 to ~-5 with `success_rate=0`?

### Hypothesis
The 2e5 smoke was too short for the 3-layer CUDA policy + VecNormalize to reach the existing Stage 0 success gate; scaling total steps to 1e9 will either yield sustained rotation success or show a clear plateau that justifies moving to Arm A (ω/revolute) instead of more Stage 0 wall-clock.

### Change from Baseline
Relative to `20260802-0217-exp-infra-subproc8-cuda-seed0` only:
1. `total_timesteps`: 2e5 → **1e9** (SB3 sum across envs).
2. Throughput: `num_envs` 8→64, `n_steps` 256→128, `batch_size` 256→512, `checkpoint_freq` 5e6.
Unchanged: Stage 0 reward/physics, `net_arch [512,256,128]`, VecNormalize, CUDA PPO, `scripts/train_parallel.py`.

### Configuration
- Algorithm: SB3 PPO (`MlpPolicy`)
- Environment: `RodRotationEnv` Stage 0 (tip connect default, axis stabilizer default)
- Reward terms: unchanged Stage 0 (rotation, tip, tilt, contact, proximity, force, action-rate)
- Observation space: unchanged (48-D)
- Action space: 9-D joint position targets in [-1, 1]
- Network: `pi`/`vf` `[512, 256, 128]`
- Optimizer: Adam (SB3 default)
- Learning rate: 3e-4
- Batch size: 512
- Horizon: `n_steps=128` per env (rollout = 64×128 = 8192)
- Number of environments: 64 (`SubprocVecEnv`)
- Training steps: 1_000_000_000 (actual ~31_719_424 before crash)
- Domain randomization: off (Stage 0)
- Curriculum stage: 0
- Evaluation protocol: online `ep_rew_mean` / `success_rate`; formal eval pending on 5e6 ckpt
- Code entrypoint: `scripts/train_parallel.py`

### Success Criteria
- Rising `ep_rew_mean`, nonzero success during training, finite losses.
- At ≥1e7 / 5e7 / 1e8 checkpoints: rotation/tip/drop competitive with legacy Stage 0 when evaluated, or a documented plateau with success≈0.
- Infra remains stable (no NaNs; checkpoints + VecNormalize load).

### Result
**Rejected as a pure long-budget strategy.** Training briefly succeeded then diverged:
- ~1e6 steps: `ep_rew_mean≈105`, `success_rate≈0.88`, `std≈1.4`
- ~5e6 steps: `ep_rew_mean≈277`, `success_rate≈0.93`, `std≈32`
- ~1e7+: success→0, return negative, `std` → 1e4…1e18
- Crash: `ValueError` NaN in Gaussian action `loc` during `PPO.train()` (~3.17e7 steps)
No `final_model` / `vecnormalize.pkl` saved (exception path). Last checkpoint on disk: `ppo_rod_30000000_steps.zip` (post-collapse).

### Key Metrics
| Metric | Baseline (0217 @2e5) | Best (~5e6) | At crash (~3.17e7) |
|---|---:|---:|---:|
| Success rate | 0 | **0.93** | 0 |
| Mean return | ~-5.4 | **~277** | ~-172 |
| Policy std | ~1.2 | ~32 | ~1e18 |
| Episode length | ~268 | (longer while succeeding) | ~47 |

### Visual Evidence
- Training curve: `runs/20260802-0220-exp-infra-subproc64-1e9-seed0/tb/`
- Metrics CSV: `runs/20260802-0220-exp-infra-subproc64-1e9-seed0/metrics.csv`
- Console log: `runs/20260802-0220-exp-infra-subproc64-1e9-seed0/logs/console.log`
- Summary: `runs/20260802-0220-exp-infra-subproc64-1e9-seed0/summary.md`
- Evaluation video: pending (recommend eval of 5e6 ckpt)

### Interpretation
**Fact:** longer training *did* reach high online Stage 0 success by ~5e6 steps — the 2e5 smoke was too short. **Fact:** without entropy/`log_std` control, continued training destroyed the policy via std explosion → NaNs. **Hypothesis:** budget alone is insufficient; need stability knobs (e.g. `ent_coef=0`, std clip) or stop at best checkpoint.

### Decision
Reject unconstrained 1e9 continuation. Revise: stabilize PPO std or early-stop/select mid checkpoints; then Arm A.

### Next Step
1. `eval_policy.py` on `ppo_rod_5000000_steps.zip` (and note missing VecNormalize at crash — may need matching norm stats from TB/run or re-eval carefully).
2. Optional short rerun with `ent_coef=0` / capped `log_std`.
3. Or proceed to EXP-A0 with lessons from DBG-20260802-001.

### Code Modification (this track)
New file `scripts/train_parallel.py` (not yet committed): SubprocVecEnv factory, CUDA device, configurable `net_arch`, VecNormalize save/load, run artifacts (`config.yaml`, `metadata.json`, `metrics.csv`, `summary.md`), stub `--reward-style` / `--physics`. Left `scripts/train.py` unchanged for legacy CPU recipes.

## EXP-20260802-001: SubprocVecEnv + CUDA + 3-layer MLP + VecNormalize (Stage 0 stack only)
- Run ID: `20260802-0217-exp-infra-subproc8-cuda-seed0`
- Date: 2026-08-02
- Status: completed (infra passed; task learning weak at 2e5)
- Parent or baseline run: `scripts/train.py` DummyVecEnv + CPU + `net_arch [256,256]`
- Git commit: `057f5e3` (dirty: uncommitted `scripts/train_parallel.py` + docs)
- Git branch: `main`
- Random seed: 0
- Device: CUDA (RTX 3090), 8 parallel envs
- Duration: ~214 s (~939 fps)
- Checkpoint: `runs/20260802-0217-exp-infra-subproc8-cuda-seed0/checkpoints/final_model.zip` (+ `vecnormalize.pkl`)
- Plumbing check: `20260802-0216-exp-infra-plumbing-check-seed0` (4096 steps)

### Question
Does `SubprocVecEnv` + CUDA + `net_arch [512,256,128]` + `VecNormalize` train Stage 0 without NaNs and produce a loadable checkpoint + normalization stats?

### Hypothesis
Only the training stack changes; Stage 0 reward/physics stay fixed. Parallel MuJoCo workers plus a CUDA MLP should complete a 2e5-step smoke with finite losses and reloadable artifacts.

### Change from Baseline
New trainer `scripts/train_parallel.py`: 8× `SubprocVecEnv`, `device=cuda`, `net_arch [512,256,128]`, VecNormalize (obs+reward, clip_obs=10), `n_steps=256`, `batch_size=256`. Stage 0 reward/physics unchanged. `scripts/train.py` untouched.

### Configuration
- Algorithm: SB3 PPO (`MlpPolicy`)
- Environment: `RodRotationEnv` Stage 0
- Reward terms: unchanged Stage 0 defaults (`rotation_reward_scale=16`, tilt weight 1.0, linear contact)
- Observation space: 48-D
- Action space: 9-D
- Network: `pi`/`vf` `[512, 256, 128]`
- Optimizer: Adam
- Learning rate: 3e-4
- Batch size: 256
- Horizon: `n_steps=256` (rollout = 8×256 = 2048)
- Number of environments: 8
- Training steps: 200_000 (actual 200_704)
- Domain randomization: off
- Curriculum stage: 0
- Evaluation protocol: in-trainer load smoke (5 steps); no 20-seed eval (infra gate only)

### Success Criteria
No NaNs; `n_envs≥8` trains; checkpoint and `vecnormalize.pkl` load and take steps.

### Result
Infra criteria **passed**. 200704 env steps; 97 finite `metrics.csv` rows; `train_value_loss` ~1e-4 at end; load smoke OK. Task signal weak: `ep_rew_mean` improved ~-204→~-5.4 but `success_rate` stayed 0 (informational; not the infra gate).

### Key Metrics
| Metric | Baseline (random / early) | Current (@2e5) | Change |
|---|---:|---:|---:|
| Success rate | 0 | 0 | none |
| Mean return | ~-204 | ~-5.4 | improved, still negative |
| Position error | n/a | n/a | no formal eval |
| Rotation progress | n/a | n/a | no formal eval |
| Episode length | ~56 | ~268 | longer episodes |
| Constraint violation rate | n/a | n/a | — |

### Visual Evidence
- Training curve: `runs/20260802-0217-exp-infra-subproc8-cuda-seed0/tb/`
- Metrics CSV: `runs/20260802-0217-exp-infra-subproc8-cuda-seed0/metrics.csv`
- Console/log: `runs/20260802-0217-exp-infra-subproc8-cuda-seed0/logs/`
- Summary: `runs/20260802-0217-exp-infra-subproc8-cuda-seed0/summary.md`
- Evaluation video: not generated (infra-only)

### Interpretation
**Fact:** parallel CUDA plumbing works and is reproducible via saved VecNormalize. **Fact:** at 2e5 steps this config did not reach Stage 0 success. **Hypothesis (tested in EXP-20260802-002):** budget, not plumbing, is the bottleneck for Stage 0 under the new net.

### Decision
Adopt infra stack. Revise next step: scale budget (EXP-20260802-002) before Arm A.

### Next Step
EXP-20260802-002 (1e9 total steps, 64 envs). Arm A remains queued after scale evidence.

## EXP-20260724-002: Does Stage 0/1 pretraining help or hurt Stage 2? (idea, unverified)
- Run ID: TBD
- Date: 2026-07-24
- Status: idea / planned
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0` (pretrained) vs a from-scratch Stage 2 control
- Random seed: 0 (multi-seed later)
- Device: CPU

### Question
Is a policy pretrained under the Stage 1 axis stabilizer actually beneficial for Stage 2 (which has no stabilizer), or does stabilizer-era pretraining induce a dependence that harms unassisted Stage 2?

### Hypothesis (falsifiable, not yet tested)
The Stage 1 stabilizer may teach the policy to rely on external orientation torque, so transferring that policy into stabilizer-free Stage 2 could be worse than, or no better than, training Stage 2 more directly. This is only an assumption; it has NOT been verified.

### Change from Baseline
Compare, under identical Stage 2 config (tip joint on, stabilizer 0, same rewards/network/eval):
1. resume from the stabilizer-faded Stage 1 parent (current approach); vs
2. a control that reaches Stage 2 with less/zero stabilizer exposure (e.g. shorter stabilizer schedule, or from-scratch Stage 2 with tip joint).

### Success Criteria
Define before interpreting: a meaningful difference in Stage 2 axis-tilt termination rate and mean rotation between pretrained vs control on seeds 0–19. If pretrained ≈ control, stabilizer dependence is not the main issue.

### Note
Deferred. Record now so the assumption is not silently assumed true. Revisit after EXP-20260724-001.

## EXP-20260724-001: Three-finger simultaneous-contact reward (idea, planned)
- Run ID: TBD
- Date: 2026-07-24
- Status: idea / planned
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0` (near-gate 2x512 parent)
- Random seed: 0 (multi-seed later)
- Device: CPU

### Question
Does the policy currently rotate the rod using effectively one finger, and does that single-finger contact push the rod off-axis (induce tilt), causing the Stage 2 axis-tilt terminations?

### Hypothesis (falsifiable, not yet tested)
Requiring all three fingertips to maintain positive contact with the rod will force coordinated multi-finger manipulation, reduce the lateral push that tilts the rod, and lower axis-tilt terminations while preserving positive axial rotation.

### Change from Baseline
Add a reward term that rewards simultaneous positive contact signal on all three fingertips (or penalizes fewer than 3 in contact). Keep tip joint, stabilizer schedule, rotation/tilt weights, network, and optimizer otherwise fixed. Change only this one term.

### Pre-experiment diagnostic (do first, cheap)
Before training, confirm the premise on existing checkpoints: log per-fingertip contact state during Stage 1/Stage 2 eval and measure how often only one finger is in contact, and whether single-finger contact correlates with tilt onset. If the premise is false, revise the idea before spending training.

### Success Criteria
Define before interpreting. Diagnostic: fewer than 20/20 axis-tilt terminations under Stage 2 with stabilizer 0. Gate: rotation >180°, tip error <0.02 m, drop ≤0.15 on seeds 0–19. Watch for reward hacking (fingers just touching without producing rotation).

### Risks / watch-outs
- Contact bonus could dominate and cause the policy to grip without rotating (log raw + weighted component and rotation together).
- Contact signal definition and threshold must be documented in METRICS.md.
- Must be tested as a single-factor change per AGENTS.md §2.2.

## EXP-20260723-016: Stage 2 short adaptation at stabilizer 0.10
- Run ID: `20260723-1700-stage2-stab010-shortadapt-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0`
- Random seed: 0
- Device: CPU

### Question
Can a bounded 5k adaptation cross the rotation gate at the last stable side of the measured assist cliff?

### Hypothesis
Reducing stabilizer only from 0.12 to 0.10 and selecting every 1k steps can recover the 3.45° needed to pass without inducing drop.

### Change from Baseline
Only external stabilizer scale changes 0.12→0.10 under Stage 2. Tip joint, rewards, network, optimizer settings, and evaluation remain fixed.

### Success Criteria
At least one checkpoint passes rotation >180°, tip error <0.02 m, and drop ≤0.15 over seeds 0–19.

## EXP-20260723-015: Stage 2 randomization adaptation at stabilizer 0.12
- Run ID: `20260723-1600-stage2-rand-adapt-stab012-seed0`
- Date: 2026-07-23
- Status: completed (failed training; diagnostic control supported)
- Parent or baseline run: `20260723-1045-capacity512-stab012-denseckpt-seed0`
- Random seed: 0
- Device: CPU

### Question
Can the near-gate stabilizer-0.12 policy adapt to Stage 2 mass/friction randomization when assist strength is held fixed?

### Hypothesis
If direct stabilizer removal is the primary curriculum cliff, the policy should retain useful rotation and survival when only Stage 2 randomization is introduced.

### Change from Baseline
Only curriculum stage/randomization changes from Stage 1 to Stage 2. Tip joint, stabilizer 0.12, absolute tilt weight 0.10, recovery scale 0, rotation scale 160, network, and optimizer settings remain fixed.

### Success Criteria
Standard fixed-seed Stage 2 gate: rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
Before training, the parent transferred at 178.51°, 8.25 mm, and drop 0.05. The first 3k checkpoint fell to 173.79° and the final checkpoint to 66.70°, while drop stayed ≤0.05.

### Interpretation
Stage 2 randomization itself is not the dominant failure. Continued training at fixed stabilizer 0.12 caused conservative loss of rotation.

### Additional Diagnostic
A fixed-policy stabilizer sweep localized the cliff: scale 0.10 retained 176.55° with drop 0, while 0.08 fell to 132.42° with drop 0.40. Scale 0.02 and 0 produced 20/20 tilt terminations.

### Decision
reject the adapted checkpoints; retain the unmodified parent

### Next Step
Short, densely checkpointed Stage 2 adaptation at stabilizer 0.10.

## EXP-20260723-014: Stage 2 local tilt-recovery shaping
- Run ID: `20260723-1500-stage2-tilt-recovery40-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Random seed: 0
- Device: CPU
- Duration: ~15 seconds training
- Checkpoint: `runs/20260723-1500-stage2-tilt-recovery40-seed0/checkpoints/final_model.zip`

### Question
Does a local reward for reducing tilt teach active recovery where a larger absolute penalty failed?

### Hypothesis
Adding clipped `40 * (previous_tilt - current_tilt)` with range [-2, 2] will reduce axis-tilt terminations while preserving the existing absolute tilt weight and rotation incentive.

### Change from Baseline
Only the tilt-recovery shaping term is added. Absolute tilt weight remains 0.10.

### Success Criteria
Standard gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support: fewer than 20/20 axis-tilt terminations.

### Result
All six checkpoints had 20/20 axis-tilt terminations. The final checkpoint was best by mean rotation: 4.22°, 4.73 mm tip error, 41.40° final tilt, success 0, and drop 1.0. The recovery component was active at -0.283 per step.

### Key Metrics
| Metric | Baseline | Recovery 40 | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 4.22° | +3.09° |
| Tip error | 4.03 mm | 4.73 mm | +0.70 mm |
| Final tilt | 41.32° | 41.40° | +0.08° |
| Drop rate | 1.00 | 1.00 | 0 |
| Axis-tilt terminations | 20/20 | 20/20 | 0 |

### Visual Evidence
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_00_seed0_rot14deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_01_seed6_rot11deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/videos/stage2_best_02_seed13_rot10deg.mp4`
- `runs/20260723-1500-stage2-tilt-recovery40-seed0/plots/stage2_reward_strategy_comparison.png`

### Interpretation
The small rotation increase is measurable, but the experiment fails its diagnostic criterion because survival did not change. Direct removal of the stabilizer remains the dominant failure.

### Decision
reject

### Next Step
Hold stabilizer at 0.12 while enabling Stage 2 randomization to isolate curriculum mismatch from randomization adaptation.

## EXP-20260723-013: Stage 2 stronger axis-deviation penalty
- Run ID: `20260723-1230-stage2-tipjoint-tiltw025-seed0`
- Date: 2026-07-23
- Status: completed (failed)
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Random seed: 0
- Device: CPU

### Question
Is Stage 2 failing because axis deviation is under-penalized?

### Hypothesis
Increasing axis-tilt penalty weight from 0.10 to 0.25 will reduce axis-tilt terminations and raise survival without eliminating positive axial rotation.

### Change from Baseline
Only `axis_tilt_penalty_weight`: 0.10→0.25.

### Success Criteria
On 20 deterministic episodes, rotation >180°, tip error <0.02 m, and drop ≤0.15. A diagnostic improvement requires fewer than 20 axis-tilt terminations without numerical instability.

### Result
All six evaluated checkpoints had 20/20 axis-tilt terminations. Best rotation was 1.89°; final was 1.76°, 4.61 mm tip error, 41.62° tilt, success 0, drop 1.0. No NaN/Inf was observed.

### Key Metrics
| Metric | Baseline | Weight 0.25 | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 1.76° | +0.63° |
| Tip error | 4.03 mm | 4.61 mm | +0.58 mm |
| Final tilt | 41.32° | 41.62° | +0.30° |
| Drop rate | 1.00 | 1.00 | 0 |
| Weighted tilt term | -2.03 | -5.01 | -2.98 |

### Interpretation
Measured fact: the increased penalty was active. Measured fact: it did not reduce tilt terminations. The evidence contradicts the hypothesis that insufficient absolute tilt weight is the primary cause.

### Decision
reject

### Next Step
Use a recovery-shaped tilt signal or curriculum rather than increasing the same absolute penalty again.

## EXP-20260723-012: Corrected Stage 2 baseline
- Run ID: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Status: completed (failed)

### Result
With tip joint active and stabilizer off: rotation 1.13°, tip error 4.03 mm, final tilt 41.32°, success 0, drop 1.0; 20/20 ended on axis tilt.

### Decision
Increase the axis-deviation penalty in a controlled experiment.

## EXP-20260723-011: 2x512 assist fade
- Supporting runs: `20260723-0820-capacity512-stab018-tiltw010-rot160-seed0`, `20260723-0900-capacity512-stab015-tiltw010-rot160-seed0`, `20260723-1000-capacity512-stab012-tiltw010-rot160-seed0`
- Status: completed

### Result
Stabilizer 0.18 passed at 229.57°/14.08 mm/drop 0.10. Stabilizer 0.15 passed at 200.00°/12.15 mm/drop 0.10. Stabilizer 0.12 narrowly failed; dense selection peaked at 179.19°/8.13 mm/drop 0.05.

### Decision
Adopt the stabilizer-0.15 checkpoint as the Stage 2 parent.

## EXP-20260723-010: Model-capacity ablation
- Candidate: `20260723-0610-capacity512-stab018-seed0`
- Matched control: `20260723-0611-capacity256-resetopt-stab018-seed0`
- Status: completed

### Result
Function-preserving expansion increased parameters 159,251→580,627 with maximum initial action error 1.79e-7. After matched 25k training, 512 reached 126.39° versus 37.79° for 256, both with drop 0.05.

### Interpretation
Capacity is a secondary limitation, not a sufficient fix. Reward balance was also required.

## EXP-20260723-009: Reward-balance bracket
- Supporting runs: `20260723-0300-stage1-stab018-tiltw010-seed0`, `20260723-0340-stage1-stab018-tiltw010-rot64-seed0`, `20260723-0410-stage1-stab018-tiltw010-rot256-seed0`, `20260723-0450-stage1-stab018-tiltw010-rot128-seed0`
- Status: completed

### Result
Tilt weight 0.10 with rotation 16 collapsed to inactivity; rotation 64 remained conservative; rotation 256 was unstable; rotation 128 approached the gate. The 512 model with rotation 160 later passed.

### Decision
Use tilt weight 0.10 / rotation scale 160 for assist fade; revisit tilt weight specifically for Stage 2.

## EXP-20260723-007: Stabilizer fade 0.20 to 0.18
- Run ID: `20260723-0200-stage1-stab018-lowlr-seed0`
- Date: 2026-07-23
- Status: completed (failed gate)
- Parent or baseline run: EXP-20260723-006
- Random seed: 0
- Device: CPU
- Duration: ~20 seconds training
- Checkpoint: `runs/20260723-0200-stage1-stab018-lowlr-seed0/checkpoints/ppo_rod_326480_steps.zip`

### Question
Can low-LR fine-tuning preserve the gate when stabilizer scale changes 0.20→0.18?

### Hypothesis
The small reduction can be adapted within 25k steps.

### Success Criteria
Rotation >180°, tip error <0.02 m, drop ≤0.15 over seeds 0–19.

### Result
Rejected. Best checkpoint: 177.18°, 20.06 mm, success 0.25, drop 0.30.

### Decision
investigate a new reward/curriculum limitation.

## EXP-20260723-006: Stabilizer fade 0.25 to 0.20
- Run ID: `20260723-0130-stage1-stab020-lowlr-seed0`
- Date: 2026-07-23
- Status: completed
- Parent or baseline run: EXP-20260723-005R
- Random seed: 0
- Device: CPU
- Checkpoint: `runs/20260723-0130-stage1-stab020-lowlr-seed0/checkpoints/ppo_rod_301480_steps.zip`

### Question
Can low-LR fine-tuning reduce the stabilizer to 0.20?

### Success Criteria
Rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
Periodic checkpoint passed: 199.58°, 16.04 mm, success 0.40, drop 0.15. Final checkpoint failed and was rejected.

### Decision
adopt periodic checkpoint.

## EXP-20260723-005R: Tip solref fade 0.05 to 0.10 with verified low LR
- Run ID: `20260723-0110-stage1-tip-solref010-lowlr-seed0`
- Date: 2026-07-23
- Status: completed
- Parent or baseline run: `20260723-0040-stage1-soft-tip-stab025-seed0`
- Random seed: 0
- Device: CPU
- Checkpoint: `runs/20260723-0110-stage1-tip-solref010-lowlr-seed0/checkpoints/final_model.zip`

### Result
Passed: rotation 223.96°, tip error 11.89 mm, success 0.55, drop 0.05.

### Decision
adopt.

## EXP-20260723-005: Stage 1 tip solref 0.10 low-LR fine-tune (invalid)
- Run ID: `20260723-0100-stage1-tip-solref010-seed0`
- Date: 2026-07-23
- Status: failed
- Parent or baseline run: `20260723-0040-stage1-soft-tip-stab025-seed0`

### Question
Can a low-learning-rate fine-tune preserve the Stage 0 policy while weakening the tip constraint?

### Change from Baseline
Planned: tip solref `0.05→0.10`, LR `3e-4→1e-5`, entropy coefficient `0.01→0`.

### Result
Invalid experiment. The log showed LR remained `3e-4`; rollout success fell from 0.97 to 0.41.

### Interpretation
This run does not test the stated hypothesis because the LR override was not applied (DBG-20260723-004).

### Decision
reject

### Next Step
Verify the LR fix with a 1024-step smoke test, then rerun under a new run ID.

## EXP-20260723-003: Stage 1 soft tip-connect fade
- Run ID: `20260723-0034-stage1-soft-tip-seed0`
- Date: 2026-07-23
- Status: running
- Parent or baseline run: EXP-20260723-001 Stage 1 (connect off)
- Checkpoint: `runs/20260723-0034-stage1-soft-tip-seed0/checkpoints/`
### Question
Does keeping tip `<connect>` active but softer (`solref` 0.05) enable Stage 1 transfer from Stage 0?
### Hypothesis
A softer tip spring preserves tip locality while requiring more finger work than Stage 0; eval gate beats connect-off baseline (success 0, rot -38.6°).
### Change from Baseline
Only Stage1 equality: remains **active** with `solref[0]=0.05` (was disabled). Axis stabilizer stays 0.
### Success Criteria
`eval_policy` Stage 1 gate.
### Result
Pending.
### Decision
Pending.

## EXP-20260723-002: Stage 1 with 50% axis stabilizer
- Run ID: `20260723-0026-stage1-softstab-seed0`
- Date: 2026-07-23
- Status: failed / rejected
- Parent or baseline run: EXP-20260723-001 Stage 1 (stabilizer 0)
### Question
Does half-strength vertical axis stabilizer (tip connect off) restore Stage 1 transfer?
### Hypothesis
scale=0.5 fixes abrupt assist removal.
### Result
Rejected. `ep_len_mean≈5`, `ep_rew_mean≈-14`, frequent DOF-12 NaNs. See `runs/20260723-0026-stage1-softstab-seed0/summary.md`.
### Interpretation
xfrc orientation spring without firm tip constraint is unstable in this scene.
### Decision
reject
### Next Step
EXP-20260723-003 soft tip-connect fade.

## EXP-20260723-001: Hanging tip + Stage0 axis stabilizer curriculum
- Run ID: `20260723-0010-hanging-tip-stabilizer-curriculum`
- Date: 2026-07-23
- Status: completed (Stage0 pass; Stage1/2 failed within budget)
- Checkpoint: `checkpoints/stage0/final_model.zip` (best)
### Question
Can Stage 0 succeed with a stable tip anchor and axis objective, and does that transfer to Stage 1/2?
### Hypothesis
Hanging tip + Stage-0 stabilizer enables Stage 0; abrupt Stage 1 removal still transfers via tip/tilt rewards.
### Result
Stage 0 **pass** (success 0.95, rot ~528°). Stage 1/2 **fail** (negative mean rotation). Artifacts under `runs/20260723-0010-hanging-tip-stabilizer-curriculum/` and `videos/stage0_hanging/`.
### Decision
revise Stage 1 assist fade (see EXP-002/003).

## EXP-20260722-001: Bottom-tip MVP curriculum (pre-hanging)
- Run ID: `20260722-bottom-tip-mvp-attempts`
- Status: failed / superseded
### Result
Bottom tip + axis objective → inverted pendulum collapse (DBG-002). Superseded by hanging tip.
### Decision
reject
## EXP-20260724-003: Stage 2 steep discrete contact reward
- Run ID: `20260724-1718-stage2-discrete-contact-seed0`
- Date: 2026-07-24
- Status: completed; rejected
- Parent or baseline run: `20260723-1200-stage2-tip-joint-no-axis-stabilizer`
- Git commit: `8e262a006c7a427034cdcc3a5715321d4400e326` (preserved baseline)
- Git branch: `main`
- Random seed: 0
- Device: CPU

### Question
Can a steep simultaneous-contact reward make the policy coordinate all three fingertips and prevent the Stage 2 axis-tilt collapse?

### Hypothesis
Replacing only the linear contact bonus with the discrete ladder `0→-10, 1→-1, 2→0.1, 3→10` will produce nonzero three-contact occupancy and fewer than 20/20 axis-tilt terminations.

### Change from Baseline
Only contact reward mapping changes. Tip joint, stabilizer 0, Stage 2 randomization, other rewards, parent checkpoint, network, optimizer, and evaluation seeds remain fixed.

### Configuration
- Algorithm: PPO
- Environment: Stage 2, tip connect solref 0.10, stabilizer 0
- Reward terms: rotation 160, tilt penalty weight 0.10, discrete contact reward
- Observation space: 48-dimensional baseline observation
- Action space: 9 normalized joint commands
- Network: policy/value 2x512
- Optimizer: Adam, resumed optimizer
- Learning rate: 1e-5
- Batch size: 128
- Horizon: 1024
- Number of environments: 1
- Training steps: 25,000
- Domain randomization: Stage 2 mass/friction
- Curriculum stage: 2
- Evaluation protocol: deterministic seeds 0–19 every 5k

### Success Criteria
Gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support: nonzero three-contact step fraction and fewer than 20/20 axis-tilt terminations. Reject reward hacking if contact improves while rotation collapses.

### Pre-experiment Diagnostic
The baseline has 0 contacts on 80.47% of steps, 1 contact on 19.53%, and never 2 or 3. Only fingertip 2 contacts (19.53% of steps); all 20 episodes end on axis tilt. This supports the single-finger/lost-contact premise.

### Result
All five periodic checkpoints and the final checkpoint had 20/20 axis-tilt terminations, zero two-contact steps, and zero three-contact steps. Final rotation was 0.92°, tip error 4.85 mm, drop 1.0, with the contact component averaging -8.147 per step. The best checkpoint by rotation reached only 1.53°.

### Key Metrics
| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| Rotation | 1.13° | 0.92° | -0.21° |
| Tip error | 4.03 mm | 4.85 mm | +0.82 mm |
| Drop rate | 1.00 | 1.00 | 0 |
| Three-contact step fraction | 0 | 0 | 0 |
| One-contact step fraction | 0.1953 | 0.2027 | +0.0074 |
| Mean contact reward | +0.050 | -8.147 | -8.197 |

### Visual Evidence
- `runs/20260724-1718-stage2-discrete-contact-seed0/plots/contact_reward_evaluation.png`
- `runs/20260724-1718-stage2-discrete-contact-seed0/videos/stage2_best_00_seed0_rot8deg.mp4`

### Interpretation
Measured fact: the steep reward was active and dominated the reward scale. Measured fact: it did not create any multi-finger contact or reduce tilt terminations. A stabilizer-0.10 control also produced no multi-finger contact despite 176.55° rotation and zero drops, so three-contact reachability/detection is not yet validated.

### Decision
reject

### Next Step
Run EXP-20260724-004 to validate per-finger and simultaneous-contact mechanical reachability before attempting another contact-reward training run.

## EXP-20260724-004: Three-fingertip contact reachability validation
- Run ID: `20260724-1730-contact-reachability`
- Date: 2026-07-24
- Status: completed (failed reachability; root cause confirmed)
- Parent or baseline run: `20260724-1718-stage2-discrete-contact-seed0`
- Git commit: `46fa9b8` at experiment start
- Git branch: `main`
- Random seed: deterministic sweep plus fixed random-search seeds
- Device: CPU

### Question
Can the current hand geometry, joint limits, and contact detector produce valid contact for each fingertip and all three simultaneously?

### Hypothesis
At least one collision-free configuration within actuator limits should register all three fingertips above 0.05 N while keeping tip error below 0.02 m and axis tilt below 0.25 rad.

### Change from Baseline
No learning and no reward change. Search controlled joint configurations/actions around the initial grasp and record per-finger force, fingertip-to-rod distance, pose, and rendered evidence.

### Success Criteria
Find and reproduce at least one configuration for each individual fingertip and at least one simultaneous three-contact configuration across three resets. If none is found after a documented bounded search, inspect collision groups/contact-force measurement and revise the grasp or target from three contacts to a mechanically supported requirement.

### Planned Evidence
- Machine-readable search results with qpos, forces, distances, and pose errors
- Images/video of best one-, two-, and three-contact candidates
- Regression test for per-finger contact detection if a detector issue is found
- Explicit go/no-go decision for another contact-reward training experiment

### Result
Across 60,000 configurations over seeds 0–2, fingers 0 and 1 each reached approximately -24 mm signed distance, but finger 2 remained 66.66–68.27 mm from the rod surface. The search found 4,166 two-contact states and zero three-contact states. Normal resets registered simultaneous 32–54 N force on fingers 0 and 1, confirming the detector works.

### Key Metrics
| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Finger 0 min distance | -23.99 mm | -23.99 mm | -23.99 mm |
| Finger 1 min distance | -24.00 mm | -24.00 mm | -24.00 mm |
| Finger 2 min distance | +66.66 mm | +68.27 mm | +67.92 mm |
| Two-contact samples | 1,437 | 1,383 | 1,346 |
| Three-contact samples | 0 | 0 | 0 |

### Visual Evidence
- `runs/20260724-1730-contact-reachability/images/contact_reachability_comparison.png`

### Interpretation
The hypothesis is rejected: three-finger contact is mechanically impossible in the current model. Finger 2 moves in an XZ plane fixed at world Y=+0.04 m while the rod lies near Y=-0.05 m. The resulting 90 mm plane separation is larger than the combined 24 mm collision radii.

### Decision
Investigate and correct the model geometry. Do not train another three-contact reward until the same test produces reproducible three-contact force.

### Next Step
EXP-20260724-005: change only finger-2 placement/orientation so its motion plane intersects the rod, then rerun the identical no-learning reachability test.

## EXP-20260724-005: Correct finger-2 contact geometry
- Run ID: `20260724-2045-finger2-spatial-dof`
- Date: 2026-07-24
- Status: completed (adopt)
- Parent or baseline run: `20260724-1730-contact-reachability`
- Device: CPU

### Question
Does aligning finger 2's motion plane with the rod make three-fingertip contact reproducibly reachable without introducing instability?

### Hypothesis
Changing only finger 2's base placement/orientation will reduce its minimum signed distance from at least +66 mm to ≤0 and produce a dynamically settled three-contact configuration on all three fixed reset seeds.

### Change from Baseline
Geometry only: adjust finger 2's base Y coordinate and/or orientation. Do not change rewards, observations, actuators, solver settings, rod geometry, or other fingers.

### Success Criteria
1. Finger 2 individual minimum signed distance ≤0 on seeds 0–2.
2. At least one three-contact configuration settles above 0.05 N on every fingertip across three resets.
3. Environment checks remain finite and stable.
4. Rendered evidence confirms genuine fingertip-rod contact rather than interpenetration or another collision artifact.

### Next Step if Successful
Establish a corrected-geometry baseline before deciding whether gradual per-finger contact shaping is still needed.

### Change Implemented
Changed only `f2_j0` axis from local Z (`0 0 1`) to local X (`1 0 0`). The two distal axes remain local Z, creating one nonparallel abduction/adduction axis followed by two flexion axes.

### Result
Finger 2 minimum signed distance improved from +66.66–68.27 mm to -14.96–15.21 mm. The 60,000-sample search found 87 geometric three-contact configurations. Exhaustive dynamic replay of those candidates found a settled three-contact grasp for every seed.

### Key Metrics
| Metric | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Three-contact samples | 28 | 33 | 26 |
| Settled contact count | 3 | 3 | 3 |
| Finger 0 force | 33.52 N | 43.58 N | 240.06 N |
| Finger 1 force | 37.17 N | 38.62 N | 228.06 N |
| Finger 2 force | 9.50 N | 7.35 N | 8.39 N |

### Visual Evidence
- `runs/20260724-2045-finger2-spatial-dof/images/spatial_finger2_three_contact.png`

### Interpretation
The geometry correction succeeds. Seed 2 has excessive force, so the result is reachability evidence rather than an acceptable initialization. The moderate-force seed-0 candidate is suitable as a reference.

### Decision
adopt

### Next Step
Run EXP-20260724-006 as a separate corrected-geometry training experiment, tracking contact occupancy, per-finger forces, excessive-force penalty, rotation, and tilt terminations.

## EXP-20260724-006: Corrected-geometry discrete-contact retraining
- Run ID: `20260724-2100-spatial-finger2-retrain-seed0`
- Date: 2026-07-24
- Status: completed (failed)
- Parent checkpoint: `runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip`
- Geometry baseline: `20260724-2045-finger2-spatial-dof`
- Random seed: 0
- Device: CPU

### Question
With finger 2 now spatially capable of contact, can the discrete contact reward produce sustained three-finger contact and reduce Stage 2 axis-tilt terminations?

### Hypothesis
The corrected geometry will yield nonzero three-contact occupancy during training/evaluation, unlike EXP-20260724-003, and reduce axis-tilt terminations below 20/20 while retaining positive rotation.

### Change from EXP-20260724-003
Only the adopted finger-2 first-joint axis differs. Reward mapping, parent checkpoint, Stage 2 environment, tip joint, stabilizer 0, randomization, network, optimizer, learning rate, checkpoint frequency, and fixed evaluation seeds remain unchanged.

### Success Criteria
Primary gate: rotation >180°, tip error <0.02 m, drop ≤0.15. Diagnostic support requires nonzero three-contact step fraction and fewer than 20/20 axis-tilt terminations. Reject force exploitation if excessive-force penalty or fingertip forces dominate while rotation does not improve.

### Result
Every periodic checkpoint and the final checkpoint retained 0% two-/three-contact occupancy and 20/20 axis-tilt terminations. Final rotation was 1.44°, and the best checkpoint reached 1.53°. The contact reward averaged approximately -8.1 to -8.3 per step; force penalty remained zero because the policy never reached additional contacts.

### Key Metrics
| Metric | Zero-shot | Best | Final |
|---|---:|---:|---:|
| Rotation | 1.26° | 1.53° | 1.44° |
| Tip error | 4.55 mm | 4.11 mm | 3.98 mm |
| Drop rate | 1.00 | 1.00 | 1.00 |
| Three-contact fraction | 0 | 0 | 0 |
| Axis-tilt terminations | 20/20 | 20/20 | 20/20 |

### Visual Evidence
- `runs/20260724-2100-spatial-finger2-retrain-seed0/plots/contact_occupancy.png`
- `runs/20260724-2100-spatial-finger2-retrain-seed0/videos/stage2_best_00_seed0_rot10deg.mp4`

### Interpretation
Measured fact: finger 2 is reachable after EXP-005. Measured fact: PPO never visits any multi-contact state from the legacy reset distribution during fixed-seed evaluation. The evidence suggests an initialization/exploration failure rather than remaining geometric impossibility.

### Decision
reject checkpoints; retain corrected geometry

### Next Step
EXP-20260724-007: initialize from the verified moderate-force three-contact grasp as the only change, validate reset robustness first, then run a bounded training control if the smoke gate passes.

## EXP-20260724-007: Three-contact reset initialization
- Run ID: `planned-20260724-three-contact-reset`
- Date: 2026-07-24
- Status: planned
- Parent: `20260724-2100-spatial-finger2-retrain-seed0`
- Device: CPU

### Question
Is the corrected policy failing because the reset distribution starts outside the reachable three-contact basin?

### Hypothesis
Replacing only `_GRASP_QPOS` with the verified moderate-force seed-0 three-contact configuration will produce three-contact resets across randomized rod phases and make the discrete reward observable early enough for PPO to preserve contact.

### Change from Baseline
Reset joint configuration only. Keep corrected geometry, reward, observation, action space, Stage 2 dynamics, parent checkpoint, optimizer, and evaluation unchanged.

### Pre-training Gate
Across seeds 0–19 after settling: finite dynamics, all three forces >0.05 N in at least 90% of resets, median force below 50 N per finger, tip error <0.02 m, and no immediate tilt termination.

### Training Success Criteria
At least one checkpoint must show nonzero three-contact evaluation occupancy and fewer than 20/20 tilt terminations. Full task gate remains rotation >180°, tip error <0.02 m, drop ≤0.15.
## EXP-20260724-008: Stage 1 training with stabilizer 1.0
- Run ID: `20260724-2130-spatial-stage1-stabilizer1-train-seed0`
- Date: 2026-07-24
- Status: completed
- Parent checkpoint: `runs/20260723-0900-capacity512-stab015-tiltw010-rot160-seed0/checkpoints/final_model.zip`
- Geometry commit: `080e367`
- Random seed: 0
- Device: CPU

### Question
Can Stage 1 PPO training with stabilizer 1.0 preserve rotation and endpoint stability while using the spatial finger and discrete contact reward?

### Hypothesis
Strong axis assistance will prevent tilt collapse and allow a post-training checkpoint to pass rotation >180°, tip error <0.02 m, and drop ≤0.15.

### Change from Parent
Stage 1 stabilizer is 1.0, corrected spatial finger 2 is active, and the discrete contact reward is used. Tip solref 0.10, network, optimizer, and fixed evaluation seeds remain controlled.

### Success Criteria
At least one post-training checkpoint passes rotation >180°, tip error <0.02 m, and drop ≤0.15 on deterministic seeds 0–19.

### Result
Three of six post-training checkpoints passed. The selected 65,600-step checkpoint reached 269.30° rotation, 18.86 mm tip error, 0.90 success rate, and 0.05 drop. Later checkpoints rotated farther but failed the endpoint-error gate.

### Key Metrics
| Checkpoint | Rotation | Tip error | Success | Drop | Passed |
|---|---:|---:|---:|---:|---|
| Pre-training | 232.59° | 13.85 mm | 0.95 | 0.05 | yes |
| 55,600 | 217.90° | 14.8 mm | 0.80 | 0.05 | yes |
| 60,600 | 245.30° | 17.3 mm | 0.75 | 0.05 | yes |
| 65,600 | 269.30° | 18.86 mm | 0.90 | 0.05 | yes |
| Final | 317.90° | 22.6 mm | 0.65 | 0.05 | no |

### Visual Evidence
- `runs/20260724-2130-spatial-stage1-stabilizer1-train-seed0/videos/stage1_success_00_seed0_rot239deg.mp4`

### Interpretation
The stabilizer prevents tilt collapse and supports strong assisted rotation. Training trades endpoint accuracy for rotation after 65,600 steps. No checkpoint produces three-contact occupancy, so the contact-coordination objective remains unmet.

### Decision
Adopt the 65,600-step checkpoint for assisted Stage 1 demonstrations; reject later checkpoints.

### Next Step
Use the verified three-contact reset initialization if the next objective remains coordinated contact. Treat this checkpoint as assisted and do not compare it directly with stabilizer-free Stage 2.
## EXP-20260724-009: Stage 0 simultaneous-contact reward +30
- Run ID: `20260724-2200-stage0-contact30-seed0`
- Date: 2026-07-24
- Status: completed (failed)
- Parent: none; fresh PPO
- Random seed: 0
- Device: CPU

### Question
Does increasing only the three-contact reward from +10 to +30 cause a fresh Stage 0 policy to discover simultaneous three-finger contact?

### Success Criteria
Nonzero three-contact occupancy and the standard task gate: rotation >180°, tip error <0.02 m, drop ≤0.15.

### Result
No evaluated checkpoint produced a three-contact step. The best success checkpoint reached 107.9° rotation, success 0.15, and drop 0.85. The 15k checkpoint reached 226.0° but dropped in all episodes.

### Decision
reject; activate the predefined 20-step rolling contact gate in EXP-20260724-010

### Next Step
Terminate when the rolling 20-step accumulated contact reward is below +5, and require the same gate for success.

## EXP-20260724-010: Stage 0 rolling simultaneous-contact gate
- Run ID: `20260724-2230-stage0-contact-gate-seed0`
- Date: 2026-07-24
- Status: completed (failed)
- Parent or baseline run: `20260724-2200-stage0-contact30-seed0`
- Git commit: `bab18a7` at launch
- Git branch: `main`
- Random seed: 0
- Device: CPU
- Duration: 25,000 environment steps
- Checkpoint: `runs/20260724-2230-stage0-contact-gate-seed0/checkpoints/ppo_rod_20000_steps.zip` (best rotation)

### Question
Does terminating when a rolling 20-step contact reward is below +5 cause fresh Stage 0 PPO to discover and maintain simultaneous three-finger contact?

### Hypothesis
Because at least one three-contact step is mathematically required to pass each full window, the policy will learn three-finger support rather than remain in a one-/two-finger solution.

### Change from Baseline
Only the rolling 20-step/+5 contact gate was added to termination and success. Geometry, Stage 0 stabilizer, reward table, PPO settings, seed, and training budget remained fixed.

### Configuration
- Algorithm: PPO
- Environment: Stage 0, spatial finger2
- Reward terms: contact `0:-10, 1:-1, 2:+0.1, 3:+30`; rotation scale 160; tilt weight 0.10
- Network: 2x256 policy and value networks
- Learning rate: 0.0003
- Training steps: 25,000
- Curriculum stage: 0
- Evaluation protocol: deterministic, fixed seeds 0–19, every 5k checkpoint

### Success Criteria
Nonzero three-contact occupancy, fewer contact-support terminations over training, rotation >180°, tip error <0.02 m, and drop ≤0.15.

### Result
No checkpoint produced a single three-contact evaluation step or a successful episode. Finger2 contact occupancy remained 0%. The best checkpoint at 20k reached 149.93° mean rotation and 0.63 mm mean tip error, but 17/20 episodes terminated on the contact gate.

### Key Metrics
| Metric | Baseline best-success checkpoint | Current 20k | Change |
|---|---:|---:|---:|
| Success rate | 0.15 | 0.00 | -0.15 |
| Mean rotation | 107.9° | 149.93° | +42.03° |
| Tip error | not recorded here | 0.63 mm | — |
| Three-contact occupancy | 0.000 | 0.000 | 0 |
| Contact-support terminations | n/a | 17/20 | new |
| Drop/termination rate | 0.85 | 1.00 | +0.15 |

### Visual Evidence
- Plot: `runs/20260724-2230-stage0-contact-gate-seed0/plots/contact_gate_evaluation.png`
- Representative failure: `runs/20260724-2230-stage0-contact-gate-seed0/videos/stage0_best_00_seed17_rot167deg.mp4`

### Interpretation
Measured fact: the gate is active and rejects unsupported windows. Measured fact: finger2 never contacts during fixed-seed evaluation. The evidence supports an exploration-failure/curriculum-mismatch diagnosis: the hard gate shortens episodes without creating a path to three-contact behavior.

### Decision
revise

### Next Step
Use the verified settled three-contact reset distribution and add an initial gate grace period, while holding the reward table and gate threshold fixed.

## EXP-20260823-001: Allegro three-fingertip reset reachability
- Run ID: `20260823-geometry-reachability-allegro-tip-bottom-v2`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: `20260724-2045-finger2-spatial-dof`
- Git branch: `cursor/teleop-hand-keyboard`
- Random seed: 0–2
- Device: CPU
- Checkpoint: none

### Question
Can an Allegro V3-derived index/middle/thumb model begin the bottom-tip task in a dynamically settled three-fingertip wrap?

### Hypothesis
Exact Allegro joint frames plus a bounded ramp from a shallow IK solution to a preload target will produce all three fingertip contacts without excessive median force.

### Change from Baseline
- Replaced the planar 9-DoF surrogate with a 12-DoF index/middle/thumb model.
- Preserved official Allegro joint axes, ranges, link offsets, and thumb opposition frame.
- Used primitive collision geometry, a 2 mm middle-pad allowance, and a ramped reset.

### Success Criteria
- Three contacts on at least 90% of fixed-seed resets.
- Median force below 50 N per fingertip.
- Finite dynamics in both revolute and point-connect variants.

### Result
Passed. A 50-seed reset audit produced three contacts on 50/50 resets in each physics variant. The recorded three-seed point-connect audit had settled forces of 12.91/18.40/6.20 N, 13.48/20.16/5.25 N, and 11.26/19.50/4.74 N.

### Key Metrics
- Point-connect three-contact reset rate: 1.00 (50/50).
- Revolute three-contact reset rate: 1.00 (50/50 after seed-phase settling).
- Maximum point-connect force in the 50-seed audit: 29.55 N.
- Observation/action shape: 48 / 12 in both physics variants.

### Visual Evidence
No rendered media was generated in this validation pass.

### Interpretation
The previous exploration failure caused by an unreachable/non-contacting third finger is removed for the new model. Random 12-DoF search remains sparse, but the optimized reset is reproducible.

### Decision
adopt

### Next Step
Run a short revolute PPO smoke, then execute every retained-connect curriculum transition.

## EXP-20260823-002: Allegro retained-connect curriculum smoke
- Run ID: `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Date: 2026-08-23
- Status: completed (plumbing passed; task failed)
- Parent or baseline run: `20260808-0224-...-3touch-smoke` documentation
- Git branch: `cursor/teleop-hand-keyboard`
- Random seed: 0
- Device: NVIDIA RTX 3090
- Duration: approximately 22 minutes
- Checkpoint: `runs/20260823-0405-allegro-tip-bottom-smoke-seed0-C3-mass-1-s1-retry0/checkpoints/final_model.zip`

### Question
Can the new 12-DoF policy and VecNormalize state transfer from revolute through progressively less-assisted bottom point-connect stages to nominal mass?

### Hypothesis
The verified wrap reset will keep contact observable at every stage and prevent the immediate C1/C5 collapse seen with the surrogate.

### Change from Baseline
- New Allegro-like hand and 12-action/48-observation policy trained from scratch.
- Final point-connect is retained; no free-tip C5.
- Eight stages: A0, B0–B3, then mass 4→2→1.
- Each smoke stage used 20k steps and selected among 10k/20k/final checkpoints.

### Success Criteria
Plumbing: every stage trains, saves/loads, transfers matching VecNormalize state, and evaluates finitely. Task: final success ≥0.5, tip error <2 cm, drop ≤0.15, three-contact occupancy ≥0.72, and zero stabilizer torque.

### Result
Plumbing passed all eight stages. The final task gate failed because sustained-ω success was 0. The final policy survived every 20 s episode with all three contacts and low tip error, but rotated only 40.11°.

### Key Metrics
| Metric | A0 selected | Final selected |
|---|---:|---:|
| Success rate | 0.00 | 0.00 |
| Mean rotation | 0.01° | 40.11° |
| Tip error | 0.00 mm | 1.40 mm |
| Three-contact occupancy | 0.984 | 1.000 |
| Drop rate | 0.60 | 0.00 |
| Stabilizer torque max mean | 0.000 | 0.000 |
| Mean rotation reward / step | 0.002 | 0.085 |
| Mean contact reward / step | 2.950 | 3.000 |

### Visual Evidence
- Eight-stage selected-checkpoint video index: `runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/INDEX.md`
- Machine-readable video/config/metric manifest: `runs/curricula/20260823-0405-allegro-tip-bottom-smoke-seed0/stage_videos_20260823-130800/metadata.json`
- Deterministic seed 0 was retained for seven stages. C1 uses fixed seed 7 because seed 0 terminated at 2.84 s while seed 7 provided a representative 4.92 s rollout. B0 seed 0 ends at 2.76 s, but seeds 1–9 fail similarly, so the short episode is representative.
- Every MP4 includes a readable overlay with stage/checkpoint, seed, rotation, tip error, contact count, tilt, success, and termination. All eight files fully decoded at 640×480, 25 FPS.

### Interpretation
Measured fact: the model solves the prior contact-initialization and immediate-collapse failures. Measured fact: contact reward dominates rotation reward by about 35:1 at the final checkpoint. The evidence suggests a reward-specification failure: holding still with three contacts is a high-return local optimum.

### Decision
revise

### Next Step
Change only the three-contact bonus from +3.0 to +0.3 while keeping the 72% hard support window and no-rotation-credit gate. Train A0 long enough to evaluate angular-speed learning before another full curriculum.

## EXP-20260823-003: Reversed Allegro world-X geometry preview
- Run ID: `reversed_world_x_180_20260823-161731`
- Date: 2026-08-23
- Status: completed (preview; awaiting orientation confirmation)
- Parent or baseline run: validated Allegro MJCF geometry from `20260823-geometry-reachability-allegro-tip-bottom-v2`
- Random seed: none
- Device: CPU / MuJoCo EGL renderer
- Duration: 5 s per video
- Checkpoint: none; no policy loaded

### Question
Does a rigid 180° world-X transform of the complete palm subtree produce a clear downward-facing orientation preview while keeping the existing grasp meaningfully positioned around the vertical rod?

### Hypothesis
Rotating only the palm root about the rod-center pivot `(0, -0.05, 0)` will preserve every Allegro-relative joint frame and map the validated wrap to the opposite side of the rod without losing all three fingertip contacts.

### Change from Baseline
- Pre-multiplied the palm root by quaternion `(0, 1, 0, 0)` about world pivot `(0, -0.05, 0)`.
- Changed no palm-relative child body or joint transform.
- Generated separate preview MJCF files; validated defaults remain unchanged.
- Configured explicit bottom anchors and visual-only world-axis/anchor markers.

### Success Criteria
- Both preview MJCFs compile and render.
- Videos decode at 640×480 and 25 FPS.
- Distinct bottom-anchor marker remains visible throughout each camera orbit.
- The settled preview retains 3/3 fingertip-to-rod geometric and force contacts.

### Result
All preview gates passed. Both 125-frame MP4s decoded at 640×480 and 25 FPS. The yellow revolute marker and magenta point-connect marker were detected in every frame. Both scenes retained 3/3 contacts after 0.2 s.

### Key Metrics
- A0 settled signed distances: `[-0.509, -0.216, -0.462]` mm; forces `[113.44, 31.89, 52.97]` N.
- C3 settled signed distances: `[-0.572, -0.945, -0.820]` mm; forces `[6.38, 26.06, 17.72]` N.
- A0 marker pixels/frame: minimum 10, maximum 184.
- C3 marker pixels/frame: minimum 5, maximum 240.

### Visual Evidence
- Index: `runs/previews/reversed_world_x_180_20260823-161731/INDEX.md`
- Metadata: `runs/previews/reversed_world_x_180_20260823-161731/metadata.json`
- A0 video: `runs/previews/reversed_world_x_180_20260823-161731/reversed_A0_bottom_revolute.mp4`
- C3 video: `runs/previews/reversed_world_x_180_20260823-161731/reversed_C3_bottom_point_connect.mp4`

### Interpretation
The rigid root transform preserves initial grasp contact under both constraints. This is only geometry/constraint evidence; it does not establish policy validity or task success. The high A0 preview force also means the transformed pose should not be promoted to a training reset without a separate reset audit.

### Decision
Await visual orientation confirmation; do not replace the validated default.

### Next Step
If the orientation is accepted, expose this transform as an explicit training configuration and run a multi-seed reset/contact-force audit before any policy training.

## EXP-20260823-004: Reversed hand with 10 mm palm clearance
- Run ID: `reversed_world_x_180_clearance10mm_20260823-162310`
- Date: 2026-08-23
- Status: completed (preview; A0 contact gate failed)
- Parent or baseline run: `reversed_world_x_180_20260823-161731`
- Random seed: none
- Device: CPU / MuJoCo EGL renderer
- Duration: 5 s per video
- Checkpoint: none; no policy loaded

### Question
What contact state remains when the already-flipped hand is translated rigidly upward until the palm collision bottom is 10 mm above the fixed rod top?

### Hypothesis
A world-Z-only translation of the full palm subtree will achieve exact clearance without modifying Allegro-relative frames, though fingertip contact may change because the finite rod and constraints remain fixed.

### Change from Baseline
- Added only `+0.032386 m` world-Z translation to the flipped palm root.
- Kept the 180° world-X orientation, rod, bottom anchors, joint values, and every child transform fixed.
- Targeted palm-bottom Z `0.080 m` above rod-top Z `0.070 m`.

### Success Criteria
- Measured initial clearance equals 10 mm.
- Both models compile and both 125-frame videos fully decode at 640×480, 25 FPS.
- Constraint markers remain visible.
- Contact is measured after 0.2 s without joint retuning.

### Result
Placement and media checks passed. C3 retained 3/3 geometric and force contacts. A0 retained only 2/3: index-to-rod distance became +15.63 mm and index normal force became 0 N.

### Key Metrics
- Applied ΔZ: `+0.032386 m`.
- Initial palm bottom / rod top / clearance: `0.080 / 0.070 / 0.010 m`.
- A0 after 0.2 s: distances `[15.634, -0.024, -0.456]` mm; forces `[0.00, 15.60, 51.36]` N; 2/3 contacts.
- C3 after 0.2 s: distances `[-0.702, -1.049, -1.457]` mm; forces `[11.42, 29.64, 39.95]` N; 3/3 contacts.

### Visual Evidence
- Index: `runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/INDEX.md`
- Metadata: `runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/metadata.json`
- A0: `runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_A0_bottom_revolute.mp4`
- C3: `runs/previews/reversed_world_x_180_clearance10mm_20260823-162310/reversed_C3_bottom_point_connect.mp4`

### Interpretation
The requested clearance is geometrically exact, but it is not contact-neutral under A0 dynamics. This preview does not validate a shared training reset or policy transfer.

### Decision
Preserve as a visual option pending orientation feedback; do not promote to the validated default.

### Next Step
If this placement is accepted visually, decide explicitly whether A0 must preserve 3/3 contact before designing a separate controlled reset adjustment.

## EXP-20260823-005: Configurable Allegro palm-root pose interface
- Run ID: `20260823-hand-pose-interface-validation`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: current validated Allegro geometry
- Random seed: 3 for reset smoke
- Device: CPU / MuJoCo EGL
- Duration: <1 second focused tests; ~2.4 seconds environment smoke
- Checkpoint: none

### Question
Can one versioned pose file move the complete Allegro hand identically in both physics variants without altering its internal kinematics or policy signature?

### Hypothesis
Applying an absolute model-frame transform only to the world-parented `palm` body will preserve every palm-relative body/joint frame and keep the 12-action/48-observation interface unchanged.

### Change from Baseline
- Added an opt-in hand-pose JSON loader and interactive editor.
- Added no default pose and changed no MJCF geometry or reset joint values.

### Success Criteria
- Identical configured palm transform in revolute and point-connect models.
- All non-palm local body transforms and all joint positions/axes unchanged.
- Malformed JSON, non-unit quaternion, and incompatible model variants rejected.
- Reset observation remains finite with shape 48; action shape remains 12.

### Result
All criteria passed in five dedicated pose tests, including overwrite protection. The combined pose/contact regression suite passed 9/9 tests, `py_compile` passed for every changed Python file, and `scripts/check_env.py` passed. The headless editor check exited immediately with actionable display instructions.

### Key Metrics
- Dedicated pose tests: 5/5 passed.
- Combined pose/contact tests: 9/9 passed.
- Action/observation dimensions: 12 / 48 for both variants.
- Changed non-palm body transforms: 0.
- Changed joint positions/axes: 0.

### Visual Evidence
No live viewer artifact was produced because this validation session had no graphical display. The editor adds green/cyan rod endpoint markers and physics-specific yellow/magenta anchor markers when run with a display.

### Interpretation
The code-level and model-level evidence supports the hypothesis. Live GUI key handling still requires a display-backed manual check before relying on the editor ergonomics.

### Decision
adopt as an opt-in configuration interface; do not promote any edited pose to the training default.

### Next Step
Open the editor on a graphical session, save a candidate pose to a new path, then run the existing multi-seed reset/contact-force audit before training with it.

## EXP-20260823-005: Thirty-millimeter clearance with thumb root inward
- Run ID: `reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817`
- Date: 2026-08-23
- Status: completed (preview; A0 contact gate failed)
- Parent or baseline run: `reversed_world_x_180_clearance10mm_20260823-162310`
- Random seed: none
- Device: CPU / MuJoCo EGL renderer
- Duration: 5 s per video
- Checkpoint: none; no policy loaded

### Question
What geometry and contact state result from raising the current flipped hand another 20 mm and moving its thumb-root body exactly 10 mm closer to the fixed rod axis?

### Hypothesis
A measured radial XY translation plus a rigid +20 mm Z increment will achieve both placement targets without changing any Allegro-relative frame, but A0 may continue to lose index contact.

### Change from Baseline
- Relative Z translation: `+0.020000 m`; total post-flip Z translation: `+0.052386 m`.
- MuJoCo reference positions: thumb-root XY `[-0.064705, -0.058490]` m; rod-axis XY `[0, -0.05]` m.
- Applied XY translation: `[+0.009915014, +0.001300958]` m.
- Kept rod, anchors, orientation, joint values, and child transforms fixed.

### Success Criteria
- Palm-bottom clearance is 30 mm above the rod top.
- Thumb-root radial distance decreases by exactly 10 mm.
- Both models compile and both 125-frame videos fully decode at 640×480, 25 FPS.
- Contact and forces are measured after 0.2 s without retuning.

### Result
Both placement targets and media checks passed. Thumb-root distance decreased from 65.259613 to 55.259613 mm. C3 retained 3/3 contacts; A0 retained 2/3 and again lost index contact.

### Key Metrics
- Palm bottom / rod top / clearance: `0.100 / 0.070 / 0.030 m`.
- A0 after 0.2 s: distances `[8.203, -0.232, -0.658]` mm; forces `[0.00, 34.58, 132.93]` N; 2/3 contacts.
- C3 after 0.2 s: distances `[-0.633, -1.116, -0.927]` mm; forces `[8.47, 31.36, 24.11]` N; 3/3 contacts.

### Visual Evidence
- Index: `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/INDEX.md`
- Metadata: `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/metadata.json`
- A0: `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_A0_bottom_revolute.mp4`
- C3: `runs/previews/reversed_world_x_180_clearance30mm_thumb10mmcloser_20260823-162817/reversed_C3_bottom_point_connect.mp4`

### Interpretation
The measured rigid placement achieves the requested geometry but does not restore A0 three-contact support. C3 contact preservation does not establish policy or reset validity.

### Decision
Preserve as a reversible preview; do not promote to the validated default.

### Next Step
Await visual feedback before any explicit, separately documented joint or reset adjustment.

## EXP-20260823-006: Headless Web hand-pose editor validation
- Run ID: `20260823-hand-pose-web-validation`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: `20260823-hand-pose-interface-validation`
- Random seed: 0 for deterministic scene reset
- Device: CPU / MuJoCo EGL
- Duration: <2 seconds focused tests; live API smoke <1 second
- Checkpoint: none

### Question
Can the validated palm-root pose interface be used through a practical local Web UI on a headless server without changing the saved schema or hand kinematics?

### Hypothesis
A standard-library HTTP server, plain browser client, and MuJoCo EGL renderer can expose responsive pose/camera controls while continuing to apply and serialize only the `palm` root through `allegro_rod_mvp.hand_pose`.

### Change from Baseline
- Added `scripts/edit_hand_pose_web.py` and deterministic backend/API tests.
- Replaced the keyboard viewer with the Web UI as the primary documented interface.
- Kept the existing JSON schema, environment integration, model defaults, and keyboard script unchanged.

### Success Criteria
- Initial state, pose update/reset, save/load/overwrite protection, malformed requests, and confined paths pass deterministic tests.
- Revolute and bottom point-connect scenes both produce non-empty decodable PNG renders under EGL without a display.
- A live HTTP smoke returns state, accepts a pose update, and returns a 640×480 PNG.

### Result
All four focused tests passed. `py_compile` passed. A live server bound to `127.0.0.1`, used EGL with no display, accepted an API pose update, and returned a decodable 46,099-byte 640×480 PNG.

### Key Metrics
- Focused tests: 4/4 passed.
- Physics variants rendered: 2/2.
- Live render: 640×480 PNG, 46,099 bytes.
- Changed child/joint frames: 0 by construction; the existing five pose-plumbing regression tests remain the kinematic check.

### Visual Evidence
- Live MuJoCo image was decoded during the API smoke.
- No browser-layout screenshot was produced because no browser automation namespace or Chromium executable was available in this session.

### Interpretation
The Web transport and headless rendering path are validated independently of training behavior. The saved file is still produced by `make_hand_pose`/`write_hand_pose`, so it remains compatible with the environment and artifact hashing.

### Decision
adopt as the primary documented pose-editing interface; retain the keyboard viewer as an optional display-backed fallback.

### Next Step
Use the Web UI to save a new candidate pose, then run the existing multi-seed reset/contact-force audit before opting that pose into training.

## EXP-20260823-007: Saved-pose two-phase force-curriculum smoke
- Run ID: `20260823-1730-two-phase-force-pose-smoke-seed0`
- Date: 2026-08-23
- Status: failed at Phase R `s=400`; Phase T not started
- Parent or baseline run: `20260823-0405-allegro-tip-bottom-smoke-seed0`
- Git commit: `33d484e` (dirty working tree preserved)
- Git branch: `main`
- Random seed: training 0; fixed evaluation 10000–10001; unseen 20000–20001
- Device: RTX 3090 GPU 1 for strict smoke; CPU for endpoint PPO smokes
- Duration: strict smoke 19.2 s; endpoint PPO smokes approximately 87 s wall-clock in parallel
- Checkpoint: `runs/20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0/checkpoints/final_model.zip`

### Question
Does the newly saved palm pose support the strict revolute `s=400 -> 1` phase,
with measured normal-force-preserving friction control, well enough to unlock an
independent bottom tip-connect `s=400 -> 1` phase?

### Hypothesis
The saved pose will retain three fingertips during initial revolute rotation, and
explicitly scaling rod/pad sliding friction will permit force calibration as mass
is annealed.

### Change from Baseline
- Used `configs/hand_poses/my_grasp.json` explicitly (SHA-256 `2d8ac7f...e7c9f`).
- Reduced the continuing three-contact reward from +3.0 to +0.3.
- Added explicit rod-plus-three-pad sliding-friction control and contact normal-force metrics.
- Declared a ten-stage schedule: `400, 200, 100, 50, 25, 12.5, 6.25, 3.125, 1.5625, 1`.
- Kept tip equality solref, damping, torsional/rolling friction, and actuator settings fixed.

### Configuration
- Algorithm: PPO
- Environment: 12-DoF Allegro, bottom revolute for Phase R
- Reward terms: DexScrew rotation/proximity/pose/energy plus +0.3 three-contact bonus
- Observation space: 48-D
- Action space: 12-D
- Network: `[512, 256, 128]`
- Optimizer: Adam
- Learning rate: 3e-4
- Batch size: 256
- Horizon: 256
- Number of environments: 2 (smoke)
- Training steps: 2,048 at attempted strict stage; 512 per endpoint plumbing smoke
- Domain randomization: off while explicit mass/friction ladder owns physics
- Curriculum stage: Phase R stage 0, `s=400`, friction scale 4.0
- Evaluation protocol: deterministic fixed and unseen seed sets, 2 episodes each (smoke only)

### Success Criteria
Predeclared in `docs/METRICS.md`: success >=0.5, mean unwrapped rotation >180
degrees, endpoint error <2 cm for tip-connect, three-tip occupancy >=0.72,
violation rate <=0.15, positive rotation reward not dominated by contact reward,
and total normal-force p95 within +/-20% of each phase's accepted `s=400`
reference at lower mass stages.

### Result
Rejected at the first revolute stage. The saved pose never produced three-tip
support in evaluation; every episode terminated at the 25-step support window.
No force reference was accepted, no lower mass was attempted, and Phase T was not started.

### Key Metrics
- Fixed: success 0.00, rotation 224.09 degrees, three-tip occupancy 0.00,
  drop/violation 1.00, index p95 0 N, total force p95 49.78 N.
- Unseen: success 0.00, rotation 228.66 degrees, three-tip occupancy 0.00,
  drop/violation 1.00, index p95 0 N, total force p95 48.75 N.
- Rotation reward: 0.0/step because unsupported motion receives no credit.
- Contact reward: -3.48/-3.51 per step on fixed/unseen sets.

### Visual Evidence
- Representative failure:
  `runs/20260823-1730-two-phase-force-pose-smoke-seed0-R00-s400-mu4-iter0-seed0/videos/revolute_best_00_seed10000_rot222deg_tilt0deg_steps25_contact_support.mp4`
- Video integrity: 26 frames, 640x480, 25 FPS, all frames decoded.
- Curriculum state:
  `runs/curricula/20260823-1730-two-phase-force-pose-smoke-seed0/state.json`

### Interpretation
The apparent >180-degree displacement is unsupported transient hinge motion, not
task success. The evidence rejects the contact-basin hypothesis and prevents a
meaningful pressing-force reference from being defined. Longer PPO would train
almost exclusively on 25-step contact-support failures.

### Decision
investigate a new bug (`DBG-20260823-003`); do not launch long training.

### Next Step
Run a pose-specific kinematic/reset search that changes only the initial Allegro
joint vector (not the saved palm pose), requiring 3/3 contacts and finite stable
dynamics in revolute and bottom tip-connect at `s=400` and `s=1`. Then repeat the
same strict Phase R `s=400` smoke.

## EXP-20260823-008: Saved-pose companion grasp recovery
- Run ID: `20260823-1740-my-grasp-shared-reset-audit-seed0`
- Date: 2026-08-23
- Status: completed with one rejected condition
- Parent or baseline run: `20260823-1730-two-phase-force-pose-smoke-seed0`
- Random seed: 0–9
- Device: CPU
- Duration: 4.7 s final audit; bounded search candidates preserved in session evidence
- Checkpoint: none

### Question
Can one 12-DoF reset/grasp vector support the fixed saved palm pose at both mass
endpoints and in both physics modes?

### Hypothesis
A shallow, joint-limit-safe three-pad preload will provide stable support without
moving the palm root.

### Change from Baseline
- Changed only reset/grasp joint targets and reset ramp.
- Preserved `my_grasp.json` byte-for-byte.
- Used a -1 mm signed tip/rod distance target, `grasp_ramp_steps=1`,
  `grasp_hold_steps=100`, and reset noise 0.0075 rad.

### Success Criteria
Ten fixed-noise seeds per mode/mass, 100 steps each, 3/3 occupancy 1.0, finite
observations, no support termination, no non-tip rod collision, and bounded
constraint error.

### Result
The shared vector passes both revolute masses and tip-connect `s=1`, but fails
heavy tip-connect. It is therefore recorded as a revolute companion, not a
universal preset.

### Key Metrics
- Revolute `s=400`: 10/10; median forces `[3.276, 3.851, 1.187]` N.
- Revolute `s=1`: 10/10; median forces `[2.238, 3.471, 1.238]` N.
- Tip-connect `s=400`: 0/10; minimum occupancy 0.765; median forces
  `[0.382, 1.023, 1.089]` N.
- Tip-connect `s=1`: 10/10; median forces `[0.711, 1.352, 1.139]` N.
- Maximum non-tip rod contacts: 0 in all conditions.

### Visual Evidence
- Metrics: `runs/20260823-1740-my-grasp-shared-reset-audit-seed0/metrics.csv`
- Summary: `runs/20260823-1740-my-grasp-shared-reset-audit-seed0/summary.json`

### Interpretation
One shared vector is insufficient under the declared robustness criterion because
the heavy equality-constrained rod intermittently loses the index pad. Separate
tip-connect preload work is justified only after Phase R passes.

### Decision
Adopt for revolute; reject as universal.

### Next Step
Train and gate revolute `s=400` using the companion config.

## EXP-20260823-009: Recovered-reset revolute s400 adaptive training
- Run IDs: `20260823-1800-two-phase-grasp-recovery-smoke-seed0`, `20260823-1810-two-phase-grasp-recovery-seed0`, `20260823-1820-two-phase-grasp-recovery-cont-seed0`, `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Date: 2026-08-23
- Status: failed at Phase R `s=400`; Phase T not started
- Parent or baseline run: EXP-20260823-008
- Random seed: training 0; fixed evaluation 10000–10009; unseen 20000–20009
- Device: RTX 3090 GPU 1
- Duration: 18.5 s smoke; approximately 69, 79, and 79 s per 100k increment
- Best checkpoint: `runs/20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0/checkpoints/final_model.zip`

### Question
Does the corrected reset enable an accepted supported-rotation policy at the
first revolute mass stage?

### Hypothesis
Once reset support is restored and the contact bonus is reduced to +0.3, PPO
will learn positive supported rotation without sacrificing the rolling contact gate.

### Change from Baseline
Only training exposure was increased: 2k, 100k, 200k, then 300k cumulative steps.
Mass `s=400`, friction scale 4.0, pose, grasp, reward, and evaluation seeds stayed fixed.

### Success Criteria
The predeclared two-phase gate in `METRICS.md`.

### Result
The 200k checkpoint crossed the total-rotation threshold but still failed support
and sustained-speed gates. A final equal-budget extension regressed violations
and increased normal force, so training stopped.

### Key Metrics
- 2k fixed/unseen: rotation -28.26°/-37.00°, violations 1.0/1.0.
- 100k: 179.12°/180.49°, violations 1.0/1.0.
- 200k: 200.56°/192.06°, violations 0.20/0.50, total force p95
  163.98/168.72 N, success 0/0.
- 300k: 183.18°/188.97°, violations 1.0/0.9, total force p95
  231.39/233.83 N, success 0/0.

### Visual Evidence
- Comparison plot: `reports/comparisons/20260823-grasp-recovery-s400-training.png`
- Representative 200k rollout:
  `runs/20260823-1820-two-phase-grasp-recovery-cont-seed0-R00-s400-mu4-iter0-seed0/videos/revolute_best_00_seed10000_rot211deg_tilt0deg_steps500_none.mp4`
- Video integrity: 501 frames, H.264, 640x480, 25 FPS, 20.04 s; first and last frames decoded.

### Interpretation
Reset geometry is no longer the dominant failure. The current optimizer/reward
lineage improves angle but does not produce sustained speed and becomes less
stable with further training.

### Decision
Investigate `DBG-20260823-004`; do not accept a pressing-force reference, anneal
mass, or start Phase T.

### Next Step
Preserve per-step omega traces for 100k and 200k checkpoints and run a
checkpoint/evaluation ablation before changing reward or success definitions.

## EXP-20260823-010: Rotation-credit and support-termination ablations
- Run IDs: `20260823-1858-rotation-credit-ablation-A-s400-seed0`, `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`
- Date: 2026-08-23
- Status: completed; both rejected by original task-quality gate
- Parent or baseline run: `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- Matched control: `20260823-1830-two-phase-grasp-recovery-cont2-seed0`
- Random seed: training 0; fixed evaluation 10000–10009; unseen 20000–20009
- Device: RTX 3090 GPU 1
- Duration: approximately 61–62 s training per ablation
- Checkpoints: each run's `checkpoints/final_model.zip`

### Question
Does the rule that zeroes rotation reward below three contacts prevent the policy
from learning useful rotation, and does the hard support termination mask that effect?

### Hypothesis
A will increase genuine rotation credit without changing contact termination. If
all A episodes still terminate on support, B will reveal whether longer ungated
trajectories develop sustained supported rotation.

### Change from Baseline
- Control, A, and B all warm-start from the same 200k model and VecNormalize state
  and train for 100k with identical hyperparameters.
- A changes only `rotation_requires_three_contacts: true -> false`.
- B retains A and additionally changes
  `contact_support_termination_enabled: true -> false`. It is explicitly a
  two-factor diagnostic.

### Configuration
- Algorithm/network/optimizer: PPO, `[512,256,128]`, Adam, learning rate 3e-4
- Environment: revolute, bottom anchor, `s=400`, friction scale 4.0
- Reward: DexScrew plus discrete +0.3 three-contact bonus
- Contact gate: 25 steps / 18 three-contact hits, still computed in all runs
- Evaluation: deterministic 10 fixed and 10 unseen episodes, 20 s each

### Success Criteria
Original two-phase gate: success >=0.5, rotation >180°, three-contact occupancy
>=0.72, violations <=0.15, positive dominant rotation reward, and 10-second
sustained omega. Disabling termination does not waive the occupancy criterion.

### Result
A increased angle and rotation reward but terminated all episodes on support. B
completed all horizons only because termination was disabled; occupancy collapsed
and sustained-omega success stayed zero. Neither permits curriculum progression.

### Key Metrics
- Matched control fixed/unseen: rotation 183.18°/188.97°, occupancy 0.975/0.975,
  violations 1.0/0.9, success 0/0.
- A: rotation 294.01°/277.39°, occupancy 0.878/0.871, support terminations
  10/10 and 10/10, force p95 184.22/191.48 N, success 0/0.
- B: rotation 374.04°/379.20°, occupancy 0.288/0.335, original-gate failure
  10/10 and 10/10, force p95 230.82/217.18 N, success 0/0.
- Maximum mean omega hold: control 0.948 s, A 1.012 s, B 0.924 s.

### Visual Evidence
- Comparison: `reports/comparisons/20260823-rotation-credit-ablation.md`
- Plot: `reports/comparisons/20260823-rotation-credit-ablation.png`
- A video: `runs/20260823-1858-rotation-credit-ablation-A-s400-seed0/videos/revolute_best_00_seed10002_rot359deg_tilt0deg_steps76_contact_support.mp4`
- B video: `runs/20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0/videos/revolute_best_00_seed10003_rot478deg_tilt0deg_steps500_none.mp4`
- Both videos decoded through their final H.264 frame at 640x480, 25 FPS.

### Interpretation
The credit rule suppresses transient rotation learning, but it is not the sole
cause of task failure. Without hard termination, PPO exploits under-supported
rotation instead of acquiring sustained three-tip manipulation.

### Decision
Reject A and B for progression. Do not define a force reference, reduce mass, or
start tip-connect.

### Next Step
The smallest discriminating follow-up is a separately predeclared continuous
contact-loss penalty ablation with A's ungated rotation credit and the original
hard termination restored.

## EXP-20260823-011: Continuous missing-contact penalty
- Run ID: none allocated
- Date: 2026-08-23
- Status: interrupted before training by user redirect
- Parent or baseline run: `20260823-1820-two-phase-grasp-recovery-cont-seed0`

### Question
Would a penalty proportional to missing fingertip contacts retain A's rotation
while reducing support termination?

### Result
Interrupted before any run directory, checkpoint, or training process was
created. The unexecuted penalty implementation was removed rather than silently
included in the redirected experiment.

### Decision
Do not run; preserve this negative/redirected planning result.

## EXP-20260823-012: Predeclared s400 finger-gait contact-scale ablation C
- Run ID: `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- Matched diagnostic baseline: `20260823-1902-rotation-credit-ablation-B-no-support-term-s400-seed0`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Device: RTX 3090 GPU 1

### Question
Does substantially reducing contact reward pressure allow repeatable
leave-and-return finger gaits without collapsing into unsupported ballistic spin?

### Hypothesis
Scaling the complete discrete contact component by 0.10 will make transient
one/two-contact phases inexpensive while retaining a -1 penalty for zero contact.

### Change from Baseline
Relative to B, only `contact_reward_scale` changes from 1.0 to 0.10.
Rotation credit remains ungated and support termination remains disabled.

### Configuration
The scaled contact ladder for counts 0/1/2/3 is exactly
`[-1.0, -0.1, +0.01, +0.03]`; raw ladder remains
`[-10.0, -1.0, +0.1, +0.3]`.

### Scale Rationale
In A fixed evaluation, rotation reward averaged +4.069/step and contact reward
+0.222/step; contact counts 0/1/2/3 occupied 0.0043/0.0158/0.1022/0.8777.
Scale 0.10 preserves a meaningful complete-loss penalty (-1) but makes one- and
two-contact gait phases at least an order of magnitude smaller than typical
rotation credit. Exactly one scale is predeclared.

### Success Criteria
This diagnostic is accepted as gait evidence only if fixed and unseen sets have:
finite 20-second rollouts; no numerical instability; mean net rotation >360°;
at least one leave-return event per fingertip on average; >=2-contact fraction
>=0.70; complete unsupported longest duration <=0.5 s; and improved three-contact
occupancy over B while preserving repeated rotation cycles. Original curriculum
success additionally still requires the ten-second omega hold and original task
quality; no lower mass or tip-connect progression is authorized regardless.

### Result
C increases net rotation and produces leave-return events on every finger, but
reduces support below both the predeclared gait criterion and B. Sustained omega
success remains zero. No second contact scale was tried.

### Key Metrics
- Fixed/unseen net rotation: 533.97°/546.98°.
- Cumulative absolute rotation: 1040.33°/982.70°.
- Mean completed net cycles: 1.1/1.1; episodes with >=2 cycles: 1/10 and 1/10.
- Mean per-tip leave-return events: `[8.3,4.1,2.8]` /
  `[7.9,5.7,2.9]`.
- >=1-contact fraction: 0.946/0.929; >=2-contact fraction: 0.604/0.578.
- Three-contact occupancy: 0.179/0.262 versus B 0.288/0.335.
- Longest zero-contact interval: 1.76/1.68 s.
- Omega hold mean maximum: 1.20/1.19 s; success 0/0.
- Force p95: 158.40/167.61 N; numerical instability 0/0.

### Visual Evidence
- Report: `reports/comparisons/20260823-finger-gait-contact-scale010-C.md`
- Plot: `reports/comparisons/20260823-finger-gait-contact-scale010-C.png`
- Video: `runs/20260823-1812-finger-gait-contact-scale010-C-s400-seed0/videos/revolute_best_00_seed10009_rot770deg_tilt0deg_steps500_none.mp4`
- Video integrity: 501 H.264 frames, 640x480, 25 FPS, 20.04 s; first and last frames decoded.

### Interpretation
The lower contact ladder encourages disengagement and recontact, but not a stable
continuous gait. Large angle includes long one/zero-contact intervals and cannot
be called supported manipulation or ballistic-free success.

### Decision
Reject. Stop at this one scale. Keep all lower mass and tip-connect stages paused.

### Next Step
At `s=400` only, expose or reward an explicit leave-then-return gait phase rather
than further reducing contact reward pressure.

## EXP-20260823-013: Predeclared two-support finger-gait lattice D
- Run ID: `20260823-1840-finger-gait-two-support-D-s400-seed0`
- Date: 2026-08-23
- Status: completed; not selected after C passed the revised gate
- Parent or baseline run: `20260823-1820-two-phase-grasp-recovery-cont-seed0`
- Comparisons: strict, A, B, and C
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Device: RTX 3090 GPU 1

### Question
Can a contact lattice that treats two- and three-contact support equally permit
one-finger gaiting while discouraging one/zero-contact spinning?

### Hypothesis
The exact 0/1/2/3 lattice `[-2.0,-0.5,0.0,0.0]` removes pressure against
single-finger release while making one- and zero-support states costly.

### Change from Baseline
Relative to B, only `contact_reward_mode` changes from `discrete` to
`gait_two_support`. Reward scale is 1.0. Rotation credit and support termination
remain disabled. The model and VecNormalize state warm-start from the same 200k
parent as A/B/C, not from C.

### Lattice Rationale
C's fixed distribution is 0/1/2/3 =
0.0536/0.3422/0.4250/0.1792 and rotation reward averages +0.861/step. Under the
new lattice, its measured-state expected contact contribution would be
`-2*0.0536 - 0.5*0.3422 = -0.2783/step`, about 32% of rotation reward: strong
enough to distinguish one from two contacts without dominating genuine rotation.
Exactly one lattice is predeclared.

### Success Criteria
On both fixed and unseen sets: >=2-contact fraction >=0.80; longest <2-contact
interval <=1.0 s; longest zero-contact interval <=0.5 s; mean >=1 leave-return
event for each finger; net rotation >360°; at least 3/10 episodes complete >=2
net positive cycles; mean maximum omega hold >=1.5 s; finite rollouts, zero
numerical instability, and tip/constraint error <0.02 m. Forces are reported.
High angle with long zero/one-contact periods is rejected. Lower mass and
tip-connect remain paused regardless of outcome.

### Result
D completed before the success-definition redirect. Fixed/unseen angle was
479.40°/537.43° and >=2-contact fraction improved to 0.717/0.732, but it missed
the predeclared 0.80 support threshold and longest below-two-contact intervals
reached 9.00/9.84 s. It was preserved and not substituted for C.

### Decision
Reject under D's declared gait-quality criteria. The later curriculum uses C
because C passed the newly authorized net-angle task gate.

## EXP-20260823-014: C net-angle reevaluation and force reference
- Run ID: `20260823-1847-C-net-angle-reeval-fixed-s400-seed10000`
- Companion runs: unseen `20260823-1847-C-net-angle-reeval-unseen-s400-seed20000`;
  conditioned-force iter1 runs at `20260823-1851-C-conditioned-force-reference-*`
- Date: 2026-08-23
- Status: completed
- Parent or baseline run: `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Random seed: fixed 10000–10009; unseen 20000–20009

### Question
Does C pass the newly declared per-episode positive net-angle criterion on both
held-out sets, and what is its supported-positive-rotation pressing-force
reference?

### Success Criteria
Per episode: net unwrapped angle >=pi, tilt <0.25 rad, tip error <0.02 m,
finite/stable, and no physical drop. Fixed and unseen success rates must each be
>=0.5. Contact quality is diagnostic.

### Result
C passes 10/10 fixed and 10/10 unseen episodes. Mean net angle is
533.97°/546.98°, tip error is below `5e-17 m`, physical drop and numerical
instability are zero. Legacy omega-hold success remains 0/20.

The predeclared pressing-force condition is `axial_omega > 0.5 rad/s AND
contact_count >= 2`. The fixed-set conditioned total median is 98.058 N and is
the Phase-R reference; 908/5000 steps (18.16%) are eligible and 81.84% are
explicitly excluded. Unseen median is 95.296 N over 16.90% eligible steps.

### Decision
Adopt C as the `s=400` parent and permit the monotone revolute curriculum.

## EXP-20260823-015: Net-angle C force-matched revolute curriculum
- Run ID: `20260823-1855-net-angle-C-force-curriculum-seed0`
- Date: 2026-08-23
- Status: failed at force calibration; task policy remained successful
- Parent or baseline run: `20260823-1812-finger-gait-contact-scale010-C-s400-seed0`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Device: RTX 3090 GPU 1
- Duration: 700.7 s
- Best accepted checkpoint:
  `runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/checkpoints/final_model.zip`

### Question
Can C transfer through `400,200,100,50,25,...,1` while retaining fixed+unseen
net-angle success and the `s=400` conditioned force within +/-20%?

### Change from Baseline
Only rod mass and explicitly recorded rod/pad sliding-friction scale change by
stage. C's reward, permissive contact settings, palm pose, grasp, optimizer,
normalization transfer, success mode, seeds, and 100k-step stage budget remain
fixed.

### Success Criteria
Both fixed and unseen net-angle success rates >=0.5. The fixed conditioned total
normal-force median must lie in `[78.447,117.670] N`. Contact occupancy remains
diagnostic.

### Result
- `s=400, mu=4.0`: accepted parent, success 1.0/1.0, force 98.058 N.
- `s=200, mu=2.0`: accepted, success 1.0/1.0, force 98.208 N.
- `s=100, mu=1.0`: accepted, success 1.0/1.0, force 96.651 N.
- `s=50, mu=0.5`: task passed but force 77.670 N failed narrowly.
- `s=50, mu=0.396040`: accepted, success 1.0/1.0, force 81.298 N.
- `s=25, mu=0.25/0.168020/0.100933/0.10`: task success remained 1.0/1.0,
  but fixed conditioned medians were 65.903/58.906/54.900/53.386 N.

At the final `s=25` trial, fixed/unseen angle was 3126.81°/3047.00°, eligible
force fractions were 15.70%/14.48%, tip error was negligible, and drop rate was
zero. Thus task success did not mask the force failure.

### Visual Evidence
- Accepted `s=50` video:
  `runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/videos/net_angle-iter1/revolute_success_00_seed10000_rot2881deg_tilt0deg_steps500_none.mp4`
- Integrity: 501 decoded frames, 640x480, 25 FPS, 20.04 s.
- State: `runs/curricula/20260823-1855-net-angle-C-force-curriculum-seed0/state.json`

### Interpretation
The physical `mu(s)` initialization works through `s=100`; one bounded correction
recovers `s=50`. At `s=25`, reducing friction toward the safe lower bound lowers,
rather than restores, the learned policy's conditioned force. The declared
calibration budget cannot establish equivalence. This does not prove that every
safe friction value is impossible.

### Decision
Stop at `s=25`. Do not run `s=12.5` or below and do not enter tip-connect.

## EXP-20260823-016: Predeclared below-bound s25 friction diagnostic
- Run IDs:
  - `20260823-1915-s25-low-friction-diagnostic-mu005-seed0`
  - `20260823-1915-s25-low-friction-diagnostic-mu0025-seed0`
- Date: 2026-08-23
- Status: completed; both scales rejected
- Parent checkpoint:
  `runs/20260823-1855-net-angle-C-force-curriculum-seed0-R03-s50-mu0.39604-iter1-seed0/checkpoints/final_model.zip`
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009
- Training budget: exactly 100000 steps per scale, independently from the same
  accepted `s=50` parent and VecNormalize state

### Question
Was the original friction-scale lower bound 0.10 conservative, and can friction
alone restore the `s=400` conditioned pressing-force target at `s=25`?

### Predeclared Scales and Vectors
Exactly two additional scales will be tested; no post-result additions:
- scale 0.05: rod and each pad `[0.09, 0.05, 0.001]`;
- scale 0.025: rod and each pad `[0.045, 0.05, 0.001]`.

For reference, scale 0.10 is `[0.18, 0.05, 0.001]`. Only sliding friction is
scaled; torsional 0.05 and rolling 0.001 remain fixed. All coefficients are
strictly positive, finite, directly accepted by MuJoCo, and remain within a
physically plausible low-friction diagnostic range. This explicitly extends the
diagnostic lower bound to 0.025 without changing or reinterpreting
`EXP-20260823-015`.

### Fixed Controls
Revolute `s=25`; saved palm pose and grasp; C reward and permissive contact
settings; net-angle success; network, optimizer, normalization, 32 environments,
and 100k budget. No force reward or task-reward change.

### Success Criteria
The first scale in declared order (0.05, then 0.025) is accepted only if:
- fixed and unseen net-angle success rates are each >=0.50;
- fixed conditioned force median lies in `[78.447,117.670] N`;
- rollouts are finite with no numerical instability.

Report conditioned-force eligibility, tip error, 0/1/2/3 contact distribution,
>=1 and >=2 contact fractions, axial-slip proxy, all-step/conditioned force p95,
and whether force increases monotonically as friction falls from 0.10 to 0.05 to
0.025. If neither passes, stop at `s=25`; lower masses and tip-connect remain
blocked.

### Result
Both policies pass fixed+unseen net-angle success at 1.0/1.0 with zero numerical
instability and negligible revolute tip error, but both miss the force band.

- Scale 0.05: fixed/unseen conditioned medians 54.669/54.972 N, eligibility
  0.176/0.170, conditioned p95 100.573/105.380 N, all-step p95
  74.628/78.996 N, and >=2-contact fraction 0.260/0.266.
- Scale 0.025: medians 49.427/47.272 N, eligibility 0.100/0.105, conditioned
  p95 95.461/80.244 N, all-step p95 62.193/61.157 N, and >=2-contact fraction
  0.157/0.152.

Fixed force is 53.386 N at the prior scale 0.10, 54.669 N at 0.05, and
49.427 N at 0.025. It is not monotone in the physically expected increasing
direction as friction falls. The axial-slip proxy p95 remains around
`1.85e-15–1.93e-15 m/s`; it is a kinematic proxy and does not override the
measured contact/support degradation.

### Visual Evidence
No stage was accepted, so no acceptance video was generated. Full metrics and
logs are in both run directories. Comparison:
`reports/comparisons/20260823-s25-low-friction-bound-extension.md`.
Plot: `reports/comparisons/20260823-s25-low-friction-bound-extension.png`.
Machine-readable comparison:
`reports/comparisons/20260823-s25-low-friction-bound-extension.json`.

### Decision
Reject both scales. The prior block remains: friction-only adaptation is
insufficient at `s=25` under this transfer protocol. Do not run lower mass or
tip-connect.

## EXP-20260823-017: Corrected proportional-physics curriculum
- Curriculum ID: `20260823-2015-proportional-physics-C-seed0`
- Controlled validation: `20260823-2010-proportional-required-force-validation`
- Date: 2026-08-23
- Status: completed Phase R; failed Phase T at s400
- Parent: C `s=400` checkpoint
- Random seed: training 0; fixed 10000–10009; unseen 20000–20009

### Question
Does a full friction vector and every explicit rod axial passive term scaled
proportionally from the `s=400` reference permit net-angle policy transfer
without using learned-policy contact force as a gate?

### Physics Declaration
- Mass/inertia: existing `body_mass/body_inertia * s`.
- Friction multiplier: exactly `4*s/400` at
  `400,200,100,50,25,12.5,6.25,3.125,1.5625,1`.
- Scale all three per-geom friction inputs on rod and all pads.
- Scale rod DOF damping, armature, and frictionloss by `s/400` relative to their
  `s=400` model values.
- Axis stabilizer is exactly zero.
- No other applied torque is used during policy rollouts.
- Point-connect is not entered unless revolute reaches `s=1`; its axial
  constraint torque must be audited before Phase T.

### Controlled Validation Declaration
At `s=400,25,1`, apply axial disturbance `tau=5*s/400 N m` while tracking the
stationary angular trajectory for two seconds. Sweep exactly preload multipliers
`[0,0.25,0.5,0.75,1,1.5,2,3]`. Pass thresholds: angle error <5 degrees, axial
speed <0.2 rad/s, >=2 contacts for >=90% of steps, finite values. Compare the
first passing median total normal force against
`N_min ~= tau/(mu_slide*0.01 + mu_torsion)`.

Validation attempt A is preserved under
`20260823-2010-proportional-required-force-validation`. Its 5 N m reference
disturbance immediately ejects the contacts at every preload and is outside the
validated hand/contact holding regime, so it cannot estimate a minimum. Before
policy curriculum interpretation, validation B is separately predeclared at a
0.5 N m `s=400` reference (10% of A), with the identical mass scaling, preload
sweep, seeds, duration, and thresholds. No further torque values will be added.

### Stage Gate
Only net-angle success: fixed and unseen success rates each >=0.50, with
per-episode positive net angle >=pi, tilt <0.25 rad, tip error <0.02 m,
finite/stable, and no physical drop. Policy contact force and gait metrics are
diagnostic only.

### Tip-Connect Transfer Contingency
The first `s=400` tip-connect transfer and one same-hyperparameter extension
both produce zero training success with episode length near 35 and PPO KL
roughly 0.6–0.8. This is an optimizer incompatibility at the physics transition,
not evidence about lower mass. One controlled retry is predeclared from the
original T00 checkpoint with only learning rate reduced from `3e-4` to `3e-5`.
If fixed or unseen net-angle success remains below 0.50, Phase T stops at
`s=400`; no other hyperparameter or lower mass will be tried.

### Result
Phase R accepted all ten stages. Fixed/unseen success is 1.0/1.0 at every mass,
with zero numerical instability and zero revolute drop. Mean fixed/unseen net
angle progresses from 544.43/616.82 degrees at `s=400` to
11,384.68/11,337.89 degrees at `s=1`. Contact quality degrades sharply:
fixed/unseen >=2-contact fraction is 0.0146/0.0180 at `s=1`.

The controlled preload validation's analytical estimate is constant:
18.382 N for attempt A and 1.838 N for attempt B. Attempt A ejects contacts at
all declared preloads. In B, no preload through multiplier 3 meets the strict
tracking criterion at `s=400` or `s=25`; measured force at the upper bound is
24.144 and 19.561 N respectively. `s=1` first passes at multiplier 1.5 with
9.314 N median force. This disproves exact simulator-level invariance under the
current compliant hand/contact system, while preserving the proportional
analytical derivation.

The point-connect axial audit finds effectively zero equality-constraint axial
resistance at 1 rad/s with contacts disabled. Phase T nevertheless fails at
`s=400`. The predeclared low-LR retry records fixed/unseen success 0.0/0.0,
rotation 177.02/175.45 degrees, drop 1.0/1.0, and `axis_tilt` on all 20 episodes.

### Decision
Adopt the corrected measurement methodology and the revolute `s=1` checkpoint.
Stop Phase T at `s=400`; do not start lower tip-connect masses. The exact blocker
is lateral tilt after the revolute-to-point-connect dynamics transition, not
policy-force mismatch or point-connect axial resistance.

### Artifacts
- State: `runs/curricula/20260823-2015-proportional-physics-C-seed0/state.json`
- Controlled sweeps: `runs/20260823-2010-proportional-required-force-validation/`
  and `runs/20260823-2012-proportional-required-force-validation-B/`
- Revolute video:
  `runs/20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0/videos/revolute_success_00_seed10000_rot11271deg_tilt0deg_steps500_none.mp4`
- Tip failure video:
  `runs/20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0/videos/tip_connect_best_00_seed10000_rot179deg_tilt86deg_steps34_axis_tilt.mp4`
- Comparison: `reports/comparisons/20260823-proportional-physics-C-curriculum.md`
