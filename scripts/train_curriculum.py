#!/usr/bin/env python3
"""Stage 0→1→2 curriculum with eval gates, extend budgets, and mild tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# first_steps, extend_chunk, max_extra
STAGE_BUDGETS = {
    0: (150_000, 100_000, 300_000),
    1: (250_000, 150_000, 450_000),
    2: (400_000, 200_000, 600_000),
}


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {' '.join(cmd)}", flush=True)
    with log_path.open("a") as log:
        log.write(f"\n=== {' '.join(cmd)} ===\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"  exit={proc.returncode} log={log_path}", flush=True)
    return proc.returncode


def eval_model(model: Path, stage: int, out: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_policy.py"),
        str(model),
        "--stage",
        str(stage),
        "--episodes",
        "20",
        "--out",
        str(out),
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    metrics = json.loads(out.read_text()) if out.exists() else {"passed": False}
    print(json.dumps(metrics, indent=2), flush=True)
    # eval_policy already printed; returncode indicates gate
    metrics["eval_exit"] = proc.returncode
    return metrics


def train_stage(
    stage: int,
    steps: int,
    resume: Path | None,
    log_path: Path,
    ent_coef: float | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Train via train.py; optionally override hyperparams with a small wrapper call."""
    out = ROOT / "checkpoints" / f"stage{stage}"
    out.mkdir(parents=True, exist_ok=True)

    if ent_coef is None and learning_rate is None:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--stage",
            str(stage),
            "--steps",
            str(steps),
        ]
        if resume is not None:
            cmd.extend(["--resume", str(resume)])
        code = run(cmd, log_path)
        if code != 0:
            raise RuntimeError(f"train.py failed for stage {stage} with code {code}")
        return out / "final_model.zip"

    # Tuned resume path: load, set hyperparams, learn, save.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from allegro_rod_mvp import RodRotationEnv

    def make_env():
        return Monitor(RodRotationEnv(curriculum_stage=stage))

    env = DummyVecEnv([make_env])
    assert resume is not None
    model = PPO.load(str(resume), env=env, device="cpu")
    if ent_coef is not None:
        model.ent_coef = ent_coef
    if learning_rate is not None:
        model.learning_rate = learning_rate
    callback = CheckpointCallback(save_freq=25_000, save_path=str(out), name_prefix="ppo_rod")
    with log_path.open("a") as log:
        # Redirect is imperfect for SB3; still train and append a marker.
        log.write(
            f"\n=== tuned train stage={stage} steps={steps} "
            f"ent_coef={ent_coef} lr={learning_rate} resume={resume} ===\n"
        )
    model.learn(total_timesteps=steps, callback=callback, progress_bar=True, reset_num_timesteps=False)
    final = out / "final_model.zip"
    model.save(str(final.with_suffix("")))  # SB3 adds .zip
    env.close()
    return final


def run_curriculum() -> int:
    summary: dict = {"stages": {}}
    prev_model: Path | None = None

    for stage in (0, 1, 2):
        first, chunk, max_extra = STAGE_BUDGETS[stage]
        out_dir = ROOT / "checkpoints" / f"stage{stage}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train.log"
        eval_path = out_dir / "eval_metrics.json"

        resume = prev_model
        print(f"\n===== STAGE {stage}: first train {first} steps =====", flush=True)
        model = train_stage(stage, first, resume, log_path)
        metrics = eval_model(model, stage, eval_path)

        extra_done = 0
        tune_level = 0  # 0=extend only, 1=ent_coef, 2=lr
        while not metrics.get("passed", False) and extra_done < max_extra:
            extend = min(chunk, max_extra - extra_done)
            print(
                f"\n===== STAGE {stage}: extend +{extend} "
                f"(extra_done={extra_done}/{max_extra}, tune={tune_level}) =====",
                flush=True,
            )
            ent = 0.01 if tune_level >= 1 else None
            lr = 1e-4 if tune_level >= 2 else None
            if tune_level == 0:
                model = train_stage(stage, extend, model, log_path)
            else:
                model = train_stage(stage, extend, model, log_path, ent_coef=ent, learning_rate=lr)
            extra_done += extend
            metrics = eval_model(model, stage, eval_path)
            # Escalate tuning after one failed extend chunk at current level.
            if not metrics.get("passed", False) and extra_done >= chunk * (tune_level + 1):
                tune_level = min(tune_level + 1, 2)

        summary["stages"][str(stage)] = {
            "model": str(model),
            "extra_steps": extra_done,
            "metrics": metrics,
            "passed": bool(metrics.get("passed", False)),
        }
        if not metrics.get("passed", False):
            print(f"WARNING: stage {stage} did not pass gate within budget.", flush=True)
            # Continue curriculum from best available model anyway.
        prev_model = model

    summary_path = ROOT / "checkpoints" / "curriculum_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # Overall success if stage 2 passed.
    return 0 if summary["stages"].get("2", {}).get("passed") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    sys.exit(run_curriculum())
