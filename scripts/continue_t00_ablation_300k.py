#!/usr/bin/env python3
"""Resume the T00 2×2 ablation from ~100k checkpoints to a cumulative 300k steps.

Writes new run directories so the original 100k artifacts stay untouched.
Uses --continue-timesteps so SB3 PPO.learn(total_timesteps=300000) is cumulative.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_t00_support_ablation import (  # noqa: E402
    CONDITIONS,
    PYTHON,
    condition_args,
    t00_train_args,
)

PARENTS = {
    "A": ROOT / "runs" / "20260914-1638-t00-ablation-A-hist1-rewbase-seed0",
    "B": ROOT / "runs" / "20260914-1638-t00-ablation-B-hist4-rewbase-seed0",
    "C": ROOT / "runs" / "20260914-1638-t00-ablation-C-hist1-rewsup-seed0",
    "D": ROOT / "runs" / "20260914-1638-t00-ablation-D-hist4-rewsup-seed0",
}


def resume_command(args: argparse.Namespace, name: str, run_id: str) -> list[str]:
    parent = PARENTS[name]
    model = parent / "checkpoints" / "final_model.zip"
    vecnorm = parent / "checkpoints" / "vecnormalize.pkl"
    if not model.exists():
        raise FileNotFoundError(model)
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_parallel.py"),
        "--run-id", run_id,
        "--resume", str(model),
        "--continue-timesteps",
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
            f"T00 2x2 continuation {name} to cumulative {args.steps}: "
            f"hist={CONDITIONS[name]['obs_history_len']} "
            f"support_aware={CONDITIONS[name]['support_aware']} "
            f"parent={parent.name}"
        ),
    ]
    if vecnorm.exists():
        command += ["--vecnormalize-path", str(vecnorm)]
    return command


def smoke_resume(args: argparse.Namespace, name: str = "A") -> None:
    """Load parent checkpoint, take two rollouts, require finite rewards."""
    from train_parallel import build_vec_env, parse_args, smoke_load_checkpoint

    parent = PARENTS[name]
    model = parent / "checkpoints" / "final_model.zip"
    vecnorm = parent / "checkpoints" / "vecnormalize.pkl"
    argv = [
        "--run-id", f"smoke-resume-{name}",
        "--resume", str(model),
        "--vecnormalize-path", str(vecnorm),
        *t00_train_args(),
        *condition_args(name),
        "--steps", "8192",
        "--num-envs", str(args.num_envs),
        "--learning-rate", str(args.learning_rate),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    parsed = parse_args(argv)
    stabilizer_range = (
        None
        if parsed.axis_stabilizer_min is None
        else (parsed.axis_stabilizer_min, parsed.axis_stabilizer_max)
    )
    smoke_load_checkpoint(model, vecnorm, parsed, stabilizer_range)
    env = build_vec_env(parsed, stabilizer_range, vecnormalize_path=str(vecnorm))
    try:
        from stable_baselines3 import PPO
        import numpy as np

        if hasattr(env, "training"):
            env.training = False
        model_obj = PPO.load(str(model), env=env, device=parsed.device)
        obs = env.reset()
        returns = []
        for _ in range(2):
            done = False
            ep_ret = 0.0
            steps = 0
            while not done and steps < 32:
                action, _ = model_obj.predict(obs, deterministic=True)
                obs, rewards, dones, _infos = env.step(action)
                if not np.isfinite(rewards).all():
                    raise RuntimeError(f"Non-finite rewards on smoke rollout: {rewards}")
                ep_ret += float(np.mean(rewards))
                done = bool(np.any(dones))
                steps += 1
            returns.append(ep_ret)
        print(f"Resume smoke rollouts OK ({name}): finite rewards, mean_step_return={returns}", flush=True)
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000, help="Cumulative SB3 timesteps.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    names = [part.strip().upper() for part in args.conditions.split(",") if part.strip()]
    for name in names:
        if name not in CONDITIONS:
            parser.error(f"unknown condition {name}")

    if args.smoke_only:
        smoke_resume(args, names[0])
        return 0

    stamp = args.stamp or datetime.now().strftime("%Y%m%d-%H%M")
    for name in names:
        hist = CONDITIONS[name]["obs_history_len"]
        rew = "rewsup" if CONDITIONS[name]["support_aware"] else "rewbase"
        run_id = f"{stamp}-t00-ablation-{name}-continue300k-seed{args.seed}"
        print(f"=== continue {name} hist={hist} {rew} run={run_id} target={args.steps}", flush=True)
        subprocess.check_call(resume_command(args, name, run_id), cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
