#!/usr/bin/env python3
"""Auto-gated mass–friction curriculum: heavy revolute → heavy tip-connect → anneal s→1.

C0/C1 start at s=400 (mass, inertia, μ coupled). Tip-connect stages resume prior ckpt
with fresh VecNormalize. Scale only drops when online+eval gates pass.
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
    if physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref",
            "0.008",
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
    out = {
        "ep_len_mean": float(sum(lens) / len(lens)) if lens else 0.0,
        "ep_rew_mean": float(sum(rews) / len(rews)) if rews else 0.0,
        "train_std_mean": float(sum(stds) / len(stds)) if stds else float("nan"),
        "n_rows": len(rows),
    }
    return out


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
    if vn.exists():
        cmd += ["--vecnormalize", str(vn)]
    if physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref",
            "0.008",
            "--dexscrew-tilt-scale",
            "1.0",
        ]
    # eval returns 1 on fail; we still want the JSON
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return json.loads(out.read_text())


def tip_tilt_fraction(metrics: dict) -> float:
    terms = metrics.get("termination_reasons") or {}
    total = sum(int(v) for v in terms.values()) or 1
    return float(terms.get("axis_tilt", 0)) / float(total)


def gate_pass(online: dict, metrics: dict, *, min_ep_len: float) -> tuple[bool, str]:
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
        args.chunk_steps = min(args.chunk_steps, 100_000)
        args.num_envs = min(args.num_envs, 8)
        args.min_ep_len = min(args.min_ep_len, 40.0)
        args.eval_episodes = min(args.eval_episodes, 10)

    cid = args.curriculum_id or f"{now_stamp()}-massfric-s{args.start_scale:g}"
    curr_dir = ROOT / "runs" / "curricula" / cid
    curr_dir.mkdir(parents=True, exist_ok=True)
    state_path = curr_dir / "state.json"
    write_progress(
        curr_dir,
        f"# Curriculum {cid}\n"
        f"- start_scale={args.start_scale} friction_cap={args.friction_cap} "
        f"tilt_term={args.tilt_terminate_rad} gamma={args.gamma:.4f} smoke={args.smoke}\n"
        f"- C0 revolute → C1 tip@s → auto anneal to s=1 (μ capped; tip solref ∝ 1/√s)\n"
        f"- Started {datetime.now().isoformat(timespec='seconds')}",
    )

    s0 = float(args.start_scale)
    # ---- C0: heavy revolute ----
    if args.skip_c0 and args.c0_ckpt:
        c0_ckpt = Path(args.c0_ckpt)
        c0_dir = c0_ckpt.parent.parent
        write_progress(curr_dir, f"## C0 skipped\nUsing `{c0_ckpt}`")
    else:
        c0_id = f"{cid}-C0-revolute-s{s0:g}-subproc{args.num_envs}"
        write_progress(curr_dir, f"## C0 revolute s={s0:g}\nTraining `{c0_id}` …")
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
            notes=f"C0 heavy revolute s={s0} friction_cap={args.friction_cap} curriculum {cid}",
            friction_cap=args.friction_cap,
            tilt_terminate_rad=args.tilt_terminate_rad,
        )
        c0_ckpt = c0_dir / "checkpoints" / "final_model.zip"
        c0_eval = eval_gate(
            model=c0_ckpt,
            physics="revolute",
            rod_mass_scale=s0,
            episodes=args.eval_episodes,
            seed=args.seed,
            out=curr_dir / "C0_eval.json",
        )
        write_progress(
            curr_dir,
            f"C0 done. online={online_stats(c0_dir)} eval_success={c0_eval.get('success_rate')} "
            f"rot={c0_eval.get('axis_rotation_deg_mean')}",
        )

    # ---- C1: tip-connect at same heavy s ----
    c1_id = f"{cid}-C1-tip-s{s0:g}-subproc{args.num_envs}"
    write_progress(curr_dir, f"## C1 tip-connect s={s0:g}\nResume C0 → `{c1_id}` …")
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
        notes=f"C1 tip-connect s={s0} friction_cap={args.friction_cap} from C0; curriculum {cid}",
        friction_cap=args.friction_cap,
        tilt_terminate_rad=args.tilt_terminate_rad,
    )
    prev_ckpt = c1_dir / "checkpoints" / "final_model.zip"
    s = s0
    retries = 0
    stage_idx = 1

    state = {
        "curriculum_id": cid,
        "scale": s,
        "last_ckpt": str(prev_ckpt),
        "stage": "C1_done",
    }
    state_path.write_text(json.dumps(state, indent=2))

    # ---- Auto anneal ----
    while s > 1.0 + 1e-9:
        online = online_stats(Path(prev_ckpt).parent.parent)
        metrics = eval_gate(
            model=prev_ckpt,
            physics="tip_connect",
            rod_mass_scale=s,
            episodes=args.eval_episodes,
            seed=args.seed + stage_idx,
            out=curr_dir / f"eval_s{s:g}_r{retries}.json",
        )
        passed, detail = gate_pass(online, metrics, min_ep_len=args.min_ep_len)
        # For tip-connect also use eval drop/tilt as tilt fraction proxy if online len weak
        tilt_frac = tip_tilt_fraction(metrics)
        write_progress(
            curr_dir,
            f"### Gate at s={s:g} (retries={retries})\n"
            f"- online={online}\n"
            f"- success={metrics.get('success_rate')} hold_max={metrics.get('omega_hold_seconds_max_mean')} "
            f"tilt_frac={tilt_frac:.2f} drop={metrics.get('drop_rate')}\n"
            f"- result: {'PASS' if passed else 'FAIL'} — {detail}",
        )
        if passed:
            s_next = next_scale(s, args.gamma)
            if s_next >= s - 1e-9 and s > 1.0:
                s_next = 1.0
            write_progress(curr_dir, f"Advance s {s:g} → {s_next:g}")
            s = s_next
            retries = 0
            stage_idx += 1
            run_id = f"{cid}-C{stage_idx}-tip-s{s:g}-subproc{args.num_envs}"
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
                notes=f"Annealed tip-connect s={s} curriculum {cid}",
                friction_cap=args.friction_cap,
                tilt_terminate_rad=args.tilt_terminate_rad,
            )
            prev_ckpt = run_dir / "checkpoints" / "final_model.zip"
            state = {"curriculum_id": cid, "scale": s, "last_ckpt": str(prev_ckpt), "stage": run_id}
            state_path.write_text(json.dumps(state, indent=2))
            continue

        retries += 1
        if retries > args.max_retries:
            write_progress(
                curr_dir,
                f"## ABORT at s={s:g}\nExceeded max_retries={args.max_retries}. last_ckpt=`{prev_ckpt}`",
            )
            state["aborted"] = True
            state_path.write_text(json.dumps(state, indent=2))
            return 2
        stage_idx += 1
        run_id = f"{cid}-C{stage_idx}-retry{retries}-tip-s{s:g}-subproc{args.num_envs}"
        write_progress(curr_dir, f"Retry {retries} at s={s:g} → `{run_id}`")
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
            notes=f"Retry tip-connect s={s} curriculum {cid}",
            friction_cap=args.friction_cap,
            tilt_terminate_rad=args.tilt_terminate_rad,
        )
        prev_ckpt = run_dir / "checkpoints" / "final_model.zip"
        state = {
            "curriculum_id": cid,
            "scale": s,
            "last_ckpt": str(prev_ckpt),
            "stage": run_id,
            "retries": retries,
        }
        state_path.write_text(json.dumps(state, indent=2))

    # Final hold / eval at s=1
    final_metrics = eval_gate(
        model=prev_ckpt,
        physics="tip_connect",
        rod_mass_scale=1.0,
        episodes=max(args.eval_episodes, 20),
        seed=args.seed + 99,
        out=curr_dir / "final_s1_eval.json",
    )
    write_progress(
        curr_dir,
        f"## DONE s=1.0\n"
        f"- ckpt=`{prev_ckpt}`\n"
        f"- eval={json.dumps({k: final_metrics.get(k) for k in ['success_rate','drop_rate','axis_rotation_deg_mean','omega_hold_satisfied_rate','termination_reasons','passed']})}",
    )
    state = {
        "curriculum_id": cid,
        "scale": 1.0,
        "last_ckpt": str(prev_ckpt),
        "stage": "complete",
        "final_eval": final_metrics,
    }
    state_path.write_text(json.dumps(state, indent=2))
    (curr_dir / "latest_ckpt.txt").write_text(str(prev_ckpt) + "\n")
    return 0 if final_metrics.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
