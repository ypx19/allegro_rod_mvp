# Run Comparison
## Compared Runs
- **DexScrew** ([x-robotics-lab/dexscrew](https://github.com/x-robotics-lab/dexscrew), arXiv [2512.02011](https://arxiv.org/abs/2512.02011)) — reference clone at `references/dexscrew/` (gitignored).
- **allegro_rod_mvp** — this repository (MuJoCo + SB3 PPO rod-axis rotation curriculum).

## Experimental Difference
Not an apples-to-apples metric comparison. DexScrew is a full sim→teleop→real BC pipeline on an XHand; allegro_rod_mvp is a Mac-local MuJoCo smoke curriculum on a 3-finger surrogate aiming at learnable free-rod axial rotation.

## Evaluation Protocol
| | DexScrew | allegro_rod_mvp |
|---|---|---|
| Sim metric | Episode reward / length; nut DOF angular velocity | Unwrapped axis rotation (°), tip error (m), drop rate, tilt |
| Real metric | Progress ratio (% of fastening rotations), completion time | N/A (sim only so far) |
| Claimed “good” result | Real BC + tactile + history: **95% ± 13%** screwdriving progress | Assisted Stage 1: **~269°**, tip **18.9 mm**, success **0.90** |

## Metric Comparison
Direct numerical comparison of “rotation quality” is misleading: DexScrew rewards **revolute-joint nut/handle angular velocity**; we reward **free-body unwrapped axial Δθ** under tip-connect / stabilizer assists.

## Training Curves
- DexScrew: Isaac Gym, **8192** envs, oracle PPO **1.5×10⁹** env steps, then ProprioAdapt distillation.
- allegro_rod_mvp: SB3 PPO, **1** env, CPU, stage budgets ~150k–400k steps.

## Representative Videos
- DexScrew: project page / `assets/DexScrew.gif`; deploy under `xhand-deploy/`.
- Ours: `runs/20260723-1200-stage2-tip-joint-no-axis-stabilizer/videos/`, Stage 1 spatial-finger checkpoints under `runs/20260724-2130-…`.

## Success Cases
- DexScrew sim: learns finger gaits under **fixed base + revolute joint** (threads omitted by design).
- DexScrew real: BC with tactile + history completes screwdriving reliably (~95% progress).
- Ours: Stage 0/1 with tip connect + axis stabilizer achieves large positive rotation and low drop.

## Failure Cases
- DexScrew: paper states sim policies **cannot complete** real fastening; direct sim2real screwdriving **41.6%** progress, **never finishes**.
- Ours: Stage 2 stabilizer fade cliff (0.10→0.08→0.0) → axis-tilt terminations; hard three-contact gate fails from current reset.

## Statistical Caveats
- Different hands (XHand 12-DoF vs 9-DoF surrogate), simulators, object models, and success definitions.
- DexScrew’s published headline numbers are **real-world BC**, not pure RL sim success.
- Our best numbers are **assisted** Stage 1, not unconstrained free-object Stage 2.

## Conclusion
DexScrew’s “good rotation policy” is not evidence that free-object PPO works for screwdriving. It is evidence that:

1. A **hard 1-DoF revolute constraint** + dense ω reward is enough to induce finger gaits in sim.
2. That gait is a **motion prior** for teleop, not a finished task policy.
3. Reliable screwdriving needs **real tactile + temporal history BC**.

Our Stage 0 tip-connect (+ optional stabilizer) is the closest analogue to their revolute abstraction. Our dominant failure — collapsing when lateral orientation assists are removed — is consistent with their claim that sim alone misses the dynamics needed for unconstrained contact-rich fastening.

## Recommendation
1. Treat tip-connect / revolute-style constraints as the **intended Stage-A rotation skill**, not a temporary crutch to delete before gaits are solid.
2. Consider DexScrew-style **pose-diff / work / torque** penalties and **proximity-to-handle** shaping alongside rotation.
3. Prefer **privileged teacher → proprio student** if/when scaling past single-env MuJoCo.
4. Do not expect Stage 2 free-object RL to match DexScrew’s real-world claims; those claims rest on BC + tactile after skill-assisted teleop.
5. Optional follow-up: port a “revolute handle” MuJoCo variant (fixed axis, reward on joint ω) as a controlled ablation against free-rod Stage 0.

---

# Detailed Method Comparison

## 1. Task and physics abstraction

| Dimension | DexScrew | allegro_rod_mvp |
|---|---|---|
| Goal | Finger gait that spins nut/handle about fastening axis; real fastening via BC | Positive unwrapped rod spin about local +x while tip stays near a world target |
| Object model | **Revolute joint** between fixed base and nut/handle primitives (no threads) | Capsule rod on **free joint**; optional tip `<connect>` equality |
| Axis assist | Kinematic (revolute enforces axis) | Soft **axis stabilizer** `xfrc` (faded by stage) |
| Tip / axial constraint | Implicit in revolute + fixed base | Tip connect (ball); Stage 2 practice keeps connect, fades stabilizer |
| Hand | Real XHand, 12 actions, 5 fingertips | Mesh-free 3-finger surrogate, 9 actions |
| Simulator | Isaac Gym, 200 Hz physics, 20 Hz control | MuJoCo, dt 0.002, ~25 Hz policy |

**Paper quote (paraphrased):** simplified revolute models induce rotational behavior; they deliberately defer thread/tactile physics to real data.

## 2. Learning pipeline

```text
DexScrew:
  Oracle PPO (priv + point cloud)
    → ProprioAdapt student (proprio history → latent)
      → JIT deploy as rotation skill
        → Skill-assisted teleop (+ downward / wrist) + tactile
          → Behavior cloning (tactile + history)

allegro_rod_mvp:
  Stage 0 PPO (tip connect + strong stabilizer)
    → Stage 1 fade stabilizer / tighten tip σ
      → Stage 2 tip connect + DR, stabilizer → 0  [current cliff]
```

DexScrew README lists four steps; the published “good” screwdriving result is step 4 (BC), not step 1 alone. Direct sim2real of the rotation student is only ~42% progress and never finishes.

## 3. Observations and actions

| | DexScrew oracle | DexScrew student | allegro_rod_mvp |
|---|---|---|---|
| Proprio | Joint pos + targets, history | Same + 30-step proprio hist for adapt | qpos/qvel (31) |
| Object | Privileged pose/vel, mass, friction, COM, scale, nut DOF, fingertips | Predicted latent only | Tip error, axial ω, unwrapped θ, sin/cos, stage id |
| Contact | Optional binary tactile / GPU contact in priv | Real tactile only at BC stage | 3 normal forces + local contact xy |
| Geometry | 100-pt object point cloud | Distilled away | None |
| Action | 12-D, PD torque targets, scale 0.05 | Same | 9-D Δq, scale 0.25 |

## 4. Reward design

**DexScrew** (`compute_hand_reward` / config):
- `rotate_reward`: clip(nut DOF ω, −4…4) × **2.5**
- `rotate_penalty`: excess ω above curriculum threshold × **−0.3**
- `proximity_reward`: thumb+index near nut × **2.0**
- `pose_diff_penalty`: stay near init grasp × **−0.1** (thumb masked)
- `torque_penalty` × **−3.0**, `work_penalty` × **−0.01**
- `pc_z_dist_penalty`: point-cloud height span × **−1.0**

**allegro_rod_mvp**:
- Axial Δθ (clipped) × rotation scale (**16** default, **160** in best Stage 1)
- Tip MSE, axis-tilt MSE, lateral ω, contact count / discrete contact, proximity (f0/f1), force, action rate
- Terminal −15 on drop/unstable

Shared idea: reward **positive spin** + keep fingers near the object. DexScrew leans on **energy/work** and **grasp anchoring**; we lean on **tip error / tilt** because the object can still tip under a soft connect.

## 5. Domain randomization and reset

DexScrew DR (aggressive): mass, COM, friction **0.5–8**, discrete scales **0.85–1.25**, PD gains, obs/action noise, ±5° object tilt, random forces.

Ours Stage 2: friction **U(0.8,1.5)**, mass scale **U(0.85,1.15)**; Stage 0/1 grasp noise ±0.05 and random axial phase.

DexScrew reset: pre-defined inclined grasp, terminate on finger–nut distance, nut stagnation (60-step history), loss of nut contact, or screw joint near upper limit (~100 turns).

Ours: settle with stabilizer; terminate on tip error, height, axis tilt >0.7 rad, contact gate, unstable.

## 6. Scale and algorithm

| | DexScrew | allegro_rod_mvp |
|---|---|---|
| Algo | Custom GPU PPO + ProprioAdapt (Hora/PenSpin lineage) | SB3 PPO |
| Envs | 8192 | 1 |
| Net | MLP [512,256,128] + priv MLP + point MLP | [256,256] → expanded [512,512] |
| LR | 5e-3 (adaptive KL) | 3e-4 (finetune 1e-5) |
| Entropy | 0 | 0.01 → 0 |
| Episode | 800 control steps (~40 s) | 300 steps (~12 s) |

## 7. Implications for this project

### What DexScrew supports
- **Hard kinematic simplification is intentional and published**, not a bug. Their revolute joint is stronger than our tip connect: it removes lateral DOFs entirely.
- A rotation policy that “looks good” in sim can still be **incomplete for the real task**; they say so explicitly.
- **Privileged info** materially helps oracle training (their Fig. 6 ablation).
- Real reliability needed **tactile + history**, not more sim fidelity alone.

### What it does *not* support
- Claiming that free-rod Stage 2 PPO should work if we just tune rewards harder.
- Claiming their sim rotation success rate matches our Stage 1 degrees metric.

### Smallest useful follow-ups (ordered)
1. **Revolute-rod ablation** in MuJoCo: fix tip and axis with a hinge; reward hinge ω; compare gait emergence vs Stage 0 tip-connect.
2. Add **pose-diff / work** penalties while keeping current Stage 0 fixed otherwise (one-factor experiment).
3. Log per-finger contact on best Stage 1 ckpt (existing H-A) before adding three-contact rewards.
4. Defer free-object Stage 2 until a revolute/tip-constrained gait is stable under DR without the soft stabilizer.

## Sources
- Local clone: `references/dexscrew/`
- Paper HTML: https://arxiv.org/html/2512.02011v1
- Task/reward: `references/dexscrew/dexscrew/tasks/xhand_hora.py`
- Configs: `references/dexscrew/configs/task/XHandHoraScrewDriver.yaml`, `configs/train/XHandHoraScrewDriver.yaml`
- Our state: `docs/PROJECT_STATE.md`, `allegro_rod_mvp/env.py`
