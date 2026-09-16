#!/usr/bin/env python3
"""Launch the T00 2×2 support-collapse ablation (history × support-aware reward).

Conditions share T00 physics, grasp, PPO hyperparameters, seed, and eval seeds.
Only observation stacking and the two support-aware reward terms change.

This script does not start T01 / lower masses and does not change tilt_terminate=1.2.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

POSE = "configs/hand_poses/my_grasp.json"
GRASP = "configs/hand_grasps/my_grasp_tip_connect_heavy.json"
PARENT_T00 = (
    ROOT
    / "runs"
    / "20260823-2040-proportional-physics-C-tip-seed0-T00-s400-mu4-lr3e5-seed0"
    / "checkpoints"
    / "final_model.zip"
)

CONDITIONS = {
    "A": {"obs_history_len": 1, "support_aware": False},
    "B": {"obs_history_len": 4, "support_aware": False},
    "C": {"obs_history_len": 1, "support_aware": True},
    "D": {"obs_history_len": 4, "support_aware": True},
}


def t00_env_args() -> list[str]:
    return [
        "--hand-model", "allegro",
        "--hand-pose-config", POSE,
        "--hand-grasp-config", GRASP,
        "--physics", "tip_connect",
        "--reward-style", "dexscrew",
        "--tip-anchor", "bottom",
        "--rod-mass-scale", "400",
        "--contact-friction-scale", "4",
        "--contact-friction-scaling-mode", "full_vector",
        "--scale-rod-joint-dynamics-from-s400",
        "--no-scale-tip-solref-with-mass",
        "--axis-stabilizer-scale", "0",
        "--contact-reward-mode", "discrete",
        "--three-contact-reward", "0.3",
        "--contact-reward-scale", "0.1",
        "--contact-window-steps", "25",
        "--contact-window-threshold", "18",
        "--three-contact-required",
        "--no-rotation-requires-three-contacts",
        "--no-contact-support-termination",
        "--success-mode", "net_angle",
        "--net-angle-success-threshold-rad", str(math.pi),
        "--tip-connect",
        "--tip-connect-solref", "0.008",
        "--dexscrew-tilt-scale", "1",
        "--tilt-terminate-rad", "1.2",
    ]


def t00_train_args() -> list[str]:
    return [
        *t00_env_args(),
        "--net-arch", "512,256,128",
        "--n-steps", "256",
        "--batch-size", "256",
        "--ent-coef", "0",
    ]


def condition_args(name: str) -> list[str]:
    spec = CONDITIONS[name]
    args = ["--obs-history-len", str(spec["obs_history_len"])]
    if spec["support_aware"]:
        args += [
            "--support-aware-reward",
            "--rotation-contact-scale-0", "0.0",
            "--rotation-contact-scale-1", "0.1",
            "--rotation-contact-scale-2plus", "1.0",
            "--low-support-wobble-scale", "0.5",
        ]
    else:
        args.append("--no-support-aware-reward")
    return args


def train_command(args: argparse.Namespace, name: str, run_id: str) -> list[str]:
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_parallel.py"),
        "--run-id", run_id,
        *t00_train_args(),
        *condition_args(name),
        "--steps", str(args.steps),
        "--num-envs", str(args.num_envs),
        "--checkpoint-freq", str(args.checkpoint_freq),
        "--learning-rate", str(args.learning_rate),
        "--device", args.device,
        "--seed", str(args.seed),
        "--notes",
        (
            f"T00 2x2 ablation condition {name}: "
            f"hist={CONDITIONS[name]['obs_history_len']} "
            f"support_aware={CONDITIONS[name]['support_aware']}"
        ),
    ]
    return command


def eval_command(args: argparse.Namespace, name: str, run_dir: Path, seed: int, label: str) -> list[str]:
    model = run_dir / "checkpoints" / "final_model.zip"
    vecnorm = run_dir / "checkpoints" / "vecnormalize.pkl"
    out = run_dir / f"eval_{label}.json"
    traces = run_dir / "traces" / label
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "eval_policy.py"),
        str(model),
        *t00_env_args(),
        *condition_args(name),
        "--episodes", str(args.eval_episodes),
        "--episode-seconds", "20",
        "--seed", str(seed),
        "--out", str(out),
        "--trace-dir", str(traces),
        "--trace-seeds", "6,10000",
        "--recontact-window-steps", "10",
        "--collapse-lookback-steps", "10",
    ]
    if vecnorm.exists():
        command += ["--vecnormalize", str(vecnorm)]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        default="A,B,C,D",
        help="Comma-separated subset of A,B,C,D",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--stamp", default=None, help="Run-id timestamp prefix YYYYMMDD-HHMM")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    from datetime import datetime

    stamp = args.stamp or datetime.now().strftime("%Y%m%d-%H%M")
    names = [part.strip().upper() for part in args.conditions.split(",") if part.strip()]
    for name in names:
        if name not in CONDITIONS:
            parser.error(f"unknown condition {name}")

    for name in names:
        hist = CONDITIONS[name]["obs_history_len"]
        rew = "rewsup" if CONDITIONS[name]["support_aware"] else "rewbase"
        run_id = f"{stamp}-t00-ablation-{name}-hist{hist}-{rew}-seed{args.seed}"
        run_dir = ROOT / "runs" / run_id
        print(f"=== condition {name}  obs={hist * 48}D  {rew}  run={run_id}", flush=True)
        if not args.eval_only:
            subprocess.check_call(train_command(args, name, run_id), cwd=ROOT)
        if args.skip_eval:
            continue
        if not (run_dir / "checkpoints" / "final_model.zip").exists():
            raise FileNotFoundError(run_dir / "checkpoints" / "final_model.zip")
        (run_dir / "logs").mkdir(exist_ok=True)
        for label, seed in (("fixed", 10_000), ("unseen", 20_000)):
            log = run_dir / "logs" / f"eval_{label}.log"
            with log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    eval_command(args, name, run_dir, seed, label),
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            print(f"wrote {run_dir / f'eval_{label}.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
