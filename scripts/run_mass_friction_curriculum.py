#!/usr/bin/env python3
"""Auto-gated mass–friction curriculum with optional bottom tip + free-tip C5.

C0 revolute @s0 → C1 hard tip-connect @s0 → retries → (pass) → C5 free tip
(no equality; tip-error reward) → optional anneal s→1 under free tip.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

PROGRESS_NAME = "CURRICULUM_PROGRESS.md"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def write_progress(curr_dir: Path, text: str) -> None:
    path = curr_dir / PROGRESS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    print(text, flush=True)


def run_train(
    *,
    run_id: str,
    physics: str,
    rod_mass_scale: float,
    steps: int,
    num_envs: int,
    resume: str | None,
    device: str,
    seed: int,
    checkpoint_freq: int,
    notes: str,
    friction_cap: float = 4.0,
    tilt_terminate_rad: float = 0.7,
    tip_anchor: str = "bottom",
    tip_hard_constraint: bool = True,
    tip_penalty_scale: float = 0.5,
    tip_sigma: float = 0.025,
    contact_reward_mode: str = "linear",
    three_contact_reward: float = 10.0,
    contact_window_steps: int = 0,
    contact_window_threshold: float = 0.0,
    three_contact_required: bool = False,
) -> Path:
    cmd = [
        str(PY),
        str(ROOT / "scripts" / "train_parallel.py"),
        "--run-id",
        run_id,
        "--physics",
        physics,
        "--reward-style",
        "dexscrew",
        "--axis-stabilizer-scale",
        "0",
        "--rod-mass-scale",
        str(rod_mass_scale),
        "--rod-friction-cap",
        str(friction_cap),
        "--tilt-terminate-rad",
        str(tilt_terminate_rad),
        "--tip-anchor",
        tip_anchor,
        "--dexscrew-tip-penalty-scale",
        str(tip_penalty_scale),
        "--dexscrew-tip-sigma",
        str(tip_sigma),
        "--contact-reward-mode",
        contact_reward_mode,
        "--three-contact-reward",
        str(three_contact_reward),
        "--contact-window-steps",
        str(contact_window_steps),
        "--contact-window-threshold",
        str(contact_window_threshold),
        "--omega-success-threshold",
        "0.5",
        "--omega-success-hold-seconds",
        "10",
        "--num-envs",
        str(num_envs),
        "--n-steps",
        "256",
        "--batch-size",
        "256",
        "--steps",
        str(steps),
        "--ent-coef",
        "0.0",
        "--checkpoint-freq",
        str(checkpoint_freq),
        "--device",
        device,
        "--seed",
        str(seed),
        "--notes",
        notes,
    ]
    if three_contact_required:
        cmd += ["--three-contact-required"]
    if physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    elif tip_hard_constraint:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref",
            "0.008",
            "--dexscrew-tilt-scale",
            "1.0",
        ]
    else:
        # C5+: free tip, soft tip-error reward keeps tip near reset target.
        cmd += [
            "--no-tip-connect",
            "--dexscrew-tilt-scale",
            "1.0",
        ]
    if resume:
        cmd += ["--resume", resume]

    print("RUN:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    run_dir = ROOT / "runs" / run_id
    ckpt = run_dir / "checkpoints" / "final_model.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing final ckpt: {ckpt}")
    return run_dir


def online_stats(run_dir: Path, last_n: int = 8) -> dict:
    """Read recent metrics.csv rows for ep_len / success proxies."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return {"ep_len_mean": 0.0, "n_rows": 0}
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {"ep_len_mean": 0.0, "n_rows": 0}
    recent = rows[-last_n:]
    lens = [float(r["episode_length"]) for r in recent if r.get("episode_length")]
    rews = [float(r["episode_return"]) for r in recent if r.get("episode_return")]
    stds = []
    for r in recent:
        for k, v in r.items():
            if k and "std" in k and v not in (None, ""):
                try:
                    stds.append(float(v))
                except ValueError:
                    pass
    return {
        "ep_len_mean": float(sum(lens) / len(lens)) if lens else 0.0,
        "ep_rew_mean": float(sum(rews) / len(rews)) if rews else 0.0,
        "train_std_mean": float(sum(stds) / len(stds)) if stds else float("nan"),
        "n_rows": len(rows),
    }


def eval_gate(
    *,
    model: Path,
    physics: str,
    rod_mass_scale: float,
    episodes: int,
    seed: int,
    out: Path,
    friction_cap: float = 4.0,
    tilt_terminate_rad: float = 1.2,
    tip_anchor: str = "bottom",
    tip_hard_constraint: bool = True,
    tip_penalty_scale: float = 0.5,
    tip_sigma: float = 0.025,
    contact_reward_mode: str = "linear",
    three_contact_reward: float = 10.0,
    contact_window_steps: int = 0,
    contact_window_threshold: float = 0.0,
    three_contact_required: bool = False,
) -> dict:
    vn = model.parent / "vecnormalize.pkl"
    cmd = [
        str(PY),
        str(ROOT / "scripts" / "eval_policy.py"),
        str(model),
        "--physics",
        physics,
        "--reward-style",
        "dexscrew",
        "--axis-stabilizer-scale",
        "0",
        "--rod-mass-scale",
        str(rod_mass_scale),
        "--rod-friction-cap",
        str(friction_cap),
        "--tilt-terminate-rad",
        str(tilt_terminate_rad),
        "--tip-anchor",
        tip_anchor,
        "--dexscrew-tip-penalty-scale",
        str(tip_penalty_scale),
        "--dexscrew-tip-sigma",
        str(tip_sigma),
        "--contact-reward-mode",
        contact_reward_mode,
        "--three-contact-reward",
        str(three_contact_reward),
        "--contact-window-steps",
        str(contact_window_steps),
        "--contact-window-threshold",
        str(contact_window_threshold),
        "--omega-success-threshold",
        "0.5",
        "--omega-success-hold-seconds",
        "10",
        "--episode-seconds",
        "20",
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    if three_contact_required:
        cmd += ["--three-contact-required"]
    if vn.exists():
        cmd += ["--vecnormalize", str(vn)]
    if physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    elif tip_hard_constraint:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref",
            "0.008",
            "--dexscrew-tilt-scale",
            "1.0",
        ]
    else:
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "1.0"]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return json.loads(out.read_text())


def tip_tilt_fraction(metrics: dict) -> float:
    terms = metrics.get("termination_reasons") or {}
    total = sum(int(v) for v in terms.values()) or 1
    return float(terms.get("axis_tilt", 0)) / float(total)


def tip_error_fail_fraction(metrics: dict) -> float:
    terms = metrics.get("termination_reasons") or {}
    total = sum(int(v) for v in terms.values()) or 1
    return float(terms.get("tip_error", 0)) / float(total)


def gate_pass(online: dict, metrics: dict, *, min_ep_len: float, free_tip: bool = False) -> tuple[bool, str]:
    reasons = []
    ok = True
    ep_len = float(online.get("ep_len_mean", 0.0))
    if ep_len < min_ep_len:
        ok = False
        reasons.append(f"ep_len {ep_len:.1f} < {min_ep_len}")
    tilt_frac = tip_tilt_fraction(metrics)
    if metrics.get("physics_mode") == "tip_connect" or "axis_tilt" in (
        metrics.get("termination_reasons") or {}
    ):
        if tilt_frac > 0.40:
            ok = False
            reasons.append(f"tilt_term_frac {tilt_frac:.2f} > 0.40")
    if free_tip:
        tip_fail = tip_error_fail_fraction(metrics)
        tip_err = float(metrics.get("tip_error_m_mean", 1.0))
        if tip_fail > 0.40:
            ok = False
            reasons.append(f"tip_error_term_frac {tip_fail:.2f} > 0.40")
        if tip_err > 0.03:
            ok = False
            reasons.append(f"tip_error_m_mean {tip_err:.4f} > 0.03")
    sr = float(metrics.get("success_rate", 0.0))
    hold = float(metrics.get("omega_hold_seconds_max_mean", 0.0))
    if sr < 0.20 and hold < 2.0:
        ok = False
        reasons.append(f"success {sr:.2f} < 0.20 and hold_max_mean {hold:.2f} < 2.0")
    std = float(online.get("train_std_mean", float("nan")))
    if math.isfinite(std) and std >= 5.0:
        ok = False
        reasons.append(f"train_std {std:.2f} >= 5")
    return ok, ("; ".join(reasons) if reasons else "all gates passed")


def next_scale(s: float, gamma: float) -> float:
    return max(1.0, s / gamma)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mass–friction auto curriculum")
    parser.add_argument("--curriculum-id", type=str, default=None)
    parser.add_argument("--start-scale", type=float, default=400.0)
    parser.add_argument("--friction-cap", type=float, default=4.0)
    parser.add_argument(
        "--tilt-terminate-rad",
        type=float,
        default=1.2,
        help="Tip-connect hard tilt kill threshold during curriculum (default 1.2; revolute ignores).",
    )
    parser.add_argument(
        "--tip-anchor",
        choices=["top", "bottom"],
        default="bottom",
        help="Tip / hinge location (default bottom for this ladder).",
    )
    parser.add_argument("--c5-tip-penalty-scale", type=float, default=8.0)
    parser.add_argument("--c5-tip-sigma", type=float, default=0.015)
    parser.add_argument("--c5-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--stop-after-c5",
        action="store_true",
        help="Stop after free-tip C5 passes (skip mass anneal).",
    )
    parser.add_argument("--gamma", type=float, default=math.sqrt(2.0))
    parser.add_argument("--c0-steps", type=int, default=1_000_000)
    parser.add_argument("--c1-steps", type=int, default=1_000_000)
    parser.add_argument("--chunk-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--min-ep-len", type=float, default=150.0)
    parser.add_argument("--skip-c0", action="store_true", help="Start from existing C0 ckpt")
    parser.add_argument("--c0-ckpt", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Short budgets for plumbing")
    args = parser.parse_args()

    if args.smoke:
        args.c0_steps = min(args.c0_steps, 200_000)
        args.c1_steps = min(args.c1_steps, 200_000)
        args.c5_steps = min(args.c5_steps, 200_000)
        args.chunk_steps = min(args.chunk_steps, 100_000)
        args.num_envs = min(args.num_envs, 8)
        args.min_ep_len = min(args.min_ep_len, 40.0)
        args.eval_episodes = min(args.eval_episodes, 10)
        args.stop_after_c5 = True

    cid = args.curriculum_id or f"{now_stamp()}-massfric-s{args.start_scale:g}-{args.tip_anchor}"
    curr_dir = ROOT / "runs" / "curricula" / cid
    curr_dir.mkdir(parents=True, exist_ok=True)
    state_path = curr_dir / "state.json"
    write_progress(
        curr_dir,
        f"# Curriculum {cid}\n"
        f"- start_scale={args.start_scale} tip_anchor={args.tip_anchor} "
        f"friction_cap={args.friction_cap} tilt_term={args.tilt_terminate_rad} "
        f"gamma={args.gamma:.4f} smoke={args.smoke} stop_after_c5={args.stop_after_c5}\n"
        f"- C0 revolute → C1 hard tip@s → (retries) → C5 free tip + tip-error reward"
        f"{'' if args.stop_after_c5 else ' → anneal free tip to s=1'}\n"
        f"- Started {datetime.now().isoformat(timespec='seconds')}",
    )

    s0 = float(args.start_scale)
    # Bottom tip is an inverted pendulum: enforce 3-finger contact (dense + hard window).
    if args.tip_anchor == "bottom":
        contact_kwargs = dict(
            contact_reward_mode="discrete",
            three_contact_reward=3.0,
            contact_window_steps=25,  # 1 s @ 25 Hz
            contact_window_threshold=18.0,  # ≥72% of window must be 3-contact
            three_contact_required=True,
        )
    else:
        contact_kwargs = dict(
            contact_reward_mode="linear",
            three_contact_reward=10.0,
            contact_window_steps=0,
            contact_window_threshold=0.0,
            three_contact_required=False,
        )
    write_progress(
        curr_dir,
        f"- contact: mode={contact_kwargs['contact_reward_mode']} "
        f"3touch_rew={contact_kwargs['three_contact_reward']} "
        f"window={contact_kwargs['contact_window_steps']}/"
        f"{contact_kwargs['contact_window_threshold']} "
        f"three_contact_required={contact_kwargs['three_contact_required']}",
    )
    tip_kwargs = dict(
        friction_cap=args.friction_cap,
        tilt_terminate_rad=args.tilt_terminate_rad,
        tip_anchor=args.tip_anchor,
        **contact_kwargs,
    )

    # ---- C0: heavy revolute ----
    if args.skip_c0 and args.c0_ckpt:
        c0_ckpt = Path(args.c0_ckpt)
        c0_dir = c0_ckpt.parent.parent
        write_progress(curr_dir, f"## C0 skipped\nUsing `{c0_ckpt}`")
    else:
        c0_id = f"{cid}-C0-revolute-s{s0:g}-subproc{args.num_envs}"
        write_progress(curr_dir, f"## C0 revolute s={s0:g} tip={args.tip_anchor}\nTraining `{c0_id}` …")
        c0_dir = run_train(
            run_id=c0_id,
            physics="revolute",
            rod_mass_scale=s0,
            steps=args.c0_steps,
            num_envs=args.num_envs,
            resume=None,
            device=args.device,
            seed=args.seed,
            checkpoint_freq=max(50_000, args.c0_steps // 5),
            notes=f"C0 heavy revolute s={s0} tip={args.tip_anchor} curriculum {cid}",
            tip_hard_constraint=False,
            **tip_kwargs,
        )
        c0_ckpt = c0_dir / "checkpoints" / "final_model.zip"
        c0_eval = eval_gate(
            model=c0_ckpt,
            physics="revolute",
            rod_mass_scale=s0,
            episodes=args.eval_episodes,
            seed=args.seed,
            out=curr_dir / "C0_eval.json",
            tip_hard_constraint=False,
            **tip_kwargs,
        )
        write_progress(
            curr_dir,
            f"C0 done. online={online_stats(c0_dir)} eval_success={c0_eval.get('success_rate')} "
            f"rot={c0_eval.get('axis_rotation_deg_mean')}",
        )

    # ---- C1: hard tip-connect at same heavy s ----
    c1_id = f"{cid}-C1-tipHard-s{s0:g}-subproc{args.num_envs}"
    write_progress(curr_dir, f"## C1 tip-connect (hard) s={s0:g}\nResume C0 → `{c1_id}` …")
    c1_dir = run_train(
        run_id=c1_id,
        physics="tip_connect",
        rod_mass_scale=s0,
        steps=args.c1_steps,
        num_envs=args.num_envs,
        resume=str(c0_ckpt),
        device=args.device,
        seed=args.seed,
        checkpoint_freq=max(50_000, args.c1_steps // 5),
        notes=f"C1 hard tip s={s0} tip={args.tip_anchor} from C0; curriculum {cid}",
        tip_hard_constraint=True,
        tip_penalty_scale=0.5,
        **tip_kwargs,
    )
    prev_ckpt = c1_dir / "checkpoints" / "final_model.zip"
    s = s0
    retries = 0
    stage_idx = 1
    free_tip = False
    tip_penalty = 0.5
    tip_sigma = 0.025
    c5_done = False

    state = {
        "curriculum_id": cid,
        "scale": s,
        "last_ckpt": str(prev_ckpt),
        "stage": "C1_done",
        "tip_anchor": args.tip_anchor,
        "free_tip": False,
    }
    state_path.write_text(json.dumps(state, indent=2))

    # ---- Gate / retry / C5 / anneal ----
    while True:
        online = online_stats(Path(prev_ckpt).parent.parent)
        metrics = eval_gate(
            model=prev_ckpt,
            physics="tip_connect",
            rod_mass_scale=s,
            episodes=args.eval_episodes,
            seed=args.seed + stage_idx,
            out=curr_dir / f"eval_s{s:g}_{'free' if free_tip else 'hard'}_r{retries}.json",
            tip_hard_constraint=not free_tip,
            tip_penalty_scale=tip_penalty,
            tip_sigma=tip_sigma,
            **tip_kwargs,
        )
        passed, detail = gate_pass(
            online, metrics, min_ep_len=args.min_ep_len, free_tip=free_tip
        )
        tilt_frac = tip_tilt_fraction(metrics)
        write_progress(
            curr_dir,
            f"### Gate at s={s:g} free_tip={free_tip} (retries={retries})\n"
            f"- online={online}\n"
            f"- success={metrics.get('success_rate')} hold_max={metrics.get('omega_hold_seconds_max_mean')} "
            f"tilt_frac={tilt_frac:.2f} tip_err={metrics.get('tip_error_m_mean')} "
            f"drop={metrics.get('drop_rate')}\n"
            f"- result: {'PASS' if passed else 'FAIL'} — {detail}",
        )

        if not passed:
            retries += 1
            if retries > args.max_retries:
                write_progress(
                    curr_dir,
                    f"## ABORT at s={s:g} free_tip={free_tip}\n"
                    f"Exceeded max_retries={args.max_retries}. last_ckpt=`{prev_ckpt}`",
                )
                state["aborted"] = True
                state_path.write_text(json.dumps(state, indent=2))
                return 2
            stage_idx += 1
            tag = "freeTip" if free_tip else "tipHard"
            run_id = f"{cid}-C{stage_idx}-retry{retries}-{tag}-s{s:g}-subproc{args.num_envs}"
            write_progress(curr_dir, f"Retry {retries} → `{run_id}`")
            run_dir = run_train(
                run_id=run_id,
                physics="tip_connect",
                rod_mass_scale=s,
                steps=args.chunk_steps,
                num_envs=args.num_envs,
                resume=str(prev_ckpt),
                device=args.device,
                seed=args.seed,
                checkpoint_freq=max(50_000, args.chunk_steps // 2),
                notes=f"Retry {'free' if free_tip else 'hard'} tip s={s} curriculum {cid}",
                tip_hard_constraint=not free_tip,
                tip_penalty_scale=tip_penalty,
                tip_sigma=tip_sigma,
                **tip_kwargs,
            )
            prev_ckpt = run_dir / "checkpoints" / "final_model.zip"
            state = {
                "curriculum_id": cid,
                "scale": s,
                "last_ckpt": str(prev_ckpt),
                "stage": run_id,
                "retries": retries,
                "free_tip": free_tip,
            }
            state_path.write_text(json.dumps(state, indent=2))
            continue

        # Passed gate.
        retries = 0
        if not free_tip and not c5_done:
            # ---- C5: drop hard tip, strengthen tip-error reward ----
            stage_idx += 1
            c5_id = f"{cid}-C5-freeTip-s{s:g}-subproc{args.num_envs}"
            write_progress(
                curr_dir,
                f"## C5 free tip (no equality) s={s:g}\n"
                f"tip_penalty_scale={args.c5_tip_penalty_scale} tip_sigma={args.c5_tip_sigma}\n"
                f"Resume hard-tip → `{c5_id}` …",
            )
            free_tip = True
            tip_penalty = float(args.c5_tip_penalty_scale)
            tip_sigma = float(args.c5_tip_sigma)
            c5_dir = run_train(
                run_id=c5_id,
                physics="tip_connect",
                rod_mass_scale=s,
                steps=args.c5_steps,
                num_envs=args.num_envs,
                resume=str(prev_ckpt),
                device=args.device,
                seed=args.seed,
                checkpoint_freq=max(50_000, args.c5_steps // 5),
                notes=f"C5 free tip + tip-error reward s={s} curriculum {cid}",
                tip_hard_constraint=False,
                tip_penalty_scale=tip_penalty,
                tip_sigma=tip_sigma,
                **tip_kwargs,
            )
            prev_ckpt = c5_dir / "checkpoints" / "final_model.zip"
            c5_done = True
            state = {
                "curriculum_id": cid,
                "scale": s,
                "last_ckpt": str(prev_ckpt),
                "stage": c5_id,
                "free_tip": True,
            }
            state_path.write_text(json.dumps(state, indent=2))
            continue  # gate C5 next iteration

        if free_tip and args.stop_after_c5:
            write_progress(
                curr_dir,
                f"## DONE after C5 (stop_after_c5)\n- ckpt=`{prev_ckpt}`\n"
                f"- eval={json.dumps({k: metrics.get(k) for k in ['success_rate','drop_rate','axis_rotation_deg_mean','tip_error_m_mean','omega_hold_satisfied_rate','termination_reasons','passed']})}",
            )
            state = {
                "curriculum_id": cid,
                "scale": s,
                "last_ckpt": str(prev_ckpt),
                "stage": "complete_c5",
                "free_tip": True,
                "final_eval": metrics,
            }
            state_path.write_text(json.dumps(state, indent=2))
            return 0

        if s <= 1.0 + 1e-9:
            write_progress(
                curr_dir,
                f"## DONE s=1.0 free_tip={free_tip}\n- ckpt=`{prev_ckpt}`\n"
                f"- eval={json.dumps({k: metrics.get(k) for k in ['success_rate','drop_rate','axis_rotation_deg_mean','tip_error_m_mean','omega_hold_satisfied_rate','termination_reasons','passed']})}",
            )
            state = {
                "curriculum_id": cid,
                "scale": 1.0,
                "last_ckpt": str(prev_ckpt),
                "stage": "complete",
                "free_tip": free_tip,
                "final_eval": metrics,
            }
            state_path.write_text(json.dumps(state, indent=2))
            return 0

        # Anneal mass under free tip (after C5).
        s_next = next_scale(s, args.gamma)
        if s_next >= s - 1e-9 and s > 1.0:
            s_next = 1.0
        write_progress(curr_dir, f"Advance s {s:g} → {s_next:g} (free_tip={free_tip})")
        s = s_next
        stage_idx += 1
        run_id = f"{cid}-C{stage_idx}-freeTip-s{s:g}-subproc{args.num_envs}"
        run_dir = run_train(
            run_id=run_id,
            physics="tip_connect",
            rod_mass_scale=s,
            steps=args.chunk_steps,
            num_envs=args.num_envs,
            resume=str(prev_ckpt),
            device=args.device,
            seed=args.seed,
            checkpoint_freq=max(50_000, args.chunk_steps // 2),
            notes=f"Annealed free tip s={s} curriculum {cid}",
            tip_hard_constraint=False,
            tip_penalty_scale=tip_penalty,
            tip_sigma=tip_sigma,
            **tip_kwargs,
        )
        prev_ckpt = run_dir / "checkpoints" / "final_model.zip"
        state = {
            "curriculum_id": cid,
            "scale": s,
            "last_ckpt": str(prev_ckpt),
            "stage": run_id,
            "free_tip": True,
        }
        state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
