#!/usr/bin/env python3
"""Gated Allegro index/middle/thumb curriculum for the bottom-tip task.

The final task deliberately retains the MuJoCo point-connect constraint:
revolute -> soft assisted connect -> hard connect -> zero stabilizer ->
nominal mass/friction hard connect.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class Stage:
    name: str
    physics: str
    mass_scale: float
    stabilizer: float
    tip_solref: float | None
    final: bool = False


def curriculum_stages(start_scale: float) -> list[Stage]:
    scales: list[float] = []
    for candidate in (start_scale, 4.0, 2.0, 1.0):
        value = max(1.0, min(float(start_scale), candidate))
        if not scales or not math.isclose(scales[-1], value):
            scales.append(value)
    return [
        Stage("A0-revolute", "revolute", start_scale, 0.0, None),
        Stage("B0-soft-connect", "tip_connect", start_scale, 1.0, 0.05),
        Stage("B1-medium-connect", "tip_connect", start_scale, 0.5, 0.02),
        Stage("B2-hard-connect", "tip_connect", start_scale, 0.25, 0.008),
        Stage(
            "B3-zero-stabilizer",
            "tip_connect",
            start_scale,
            0.0,
            0.008,
            final=len(scales) == 1,
        ),
        *[
            Stage(
                f"C{index}-mass-{scale:g}",
                "tip_connect",
                scale,
                0.0,
                0.008,
                final=index == len(scales) - 1,
            )
            for index, scale in enumerate(scales[1:], start=1)
        ],
    ]


def vecnormalize_for(model: Path) -> Path:
    if model.name == "final_model.zip":
        return model.parent / "vecnormalize.pkl"
    stem = model.stem
    return model.parent / f"{stem}_vecnormalize.pkl"


def train_stage(
    stage: Stage,
    *,
    run_id: str,
    steps: int,
    num_envs: int,
    device: str,
    seed: int,
    checkpoint_freq: int,
    parent: Path | None,
    hand_pose_config: str | None,
) -> Path:
    cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_parallel.py"),
        "--run-id", run_id,
        "--hand-model", "allegro",
        "--physics", stage.physics,
        "--reward-style", "dexscrew",
        "--tip-anchor", "bottom",
        "--rod-mass-scale", str(stage.mass_scale),
        "--rod-friction-cap", "4",
        "--axis-stabilizer-scale", str(stage.stabilizer),
        "--tilt-terminate-rad", "1.2",
        "--contact-reward-mode", "discrete",
        "--three-contact-reward", "3",
        "--contact-window-steps", "25",
        "--contact-window-threshold", "18",
        "--three-contact-required",
        "--omega-success-threshold", "0.5",
        "--omega-success-hold-seconds", "10",
        "--num-envs", str(num_envs),
        "--n-steps", "256",
        "--batch-size", "256",
        "--steps", str(steps),
        "--checkpoint-freq", str(checkpoint_freq),
        "--ent-coef", "0",
        "--device", device,
        "--seed", str(seed),
        "--notes",
        (
            f"Allegro three-finger bottom-tip curriculum stage {stage.name}; "
            "final physics retains point-connect."
        ),
    ]
    if stage.physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref", str(stage.tip_solref),
            "--dexscrew-tilt-scale", "1",
        ]
    if hand_pose_config:
        cmd += ["--hand-pose-config", hand_pose_config]
    if parent is not None:
        cmd += ["--resume", str(parent)]
        parent_vecnorm = vecnormalize_for(parent)
        if parent_vecnorm.exists():
            cmd += ["--vecnormalize-path", str(parent_vecnorm)]
    subprocess.check_call(cmd, cwd=ROOT)
    return ROOT / "runs" / run_id


def evaluate_checkpoint(
    stage: Stage,
    model: Path,
    *,
    episodes: int,
    seed: int,
    output: Path,
    hand_pose_config: str | None,
) -> dict:
    cmd = [
        str(PYTHON),
        str(ROOT / "scripts" / "eval_policy.py"),
        str(model),
        "--hand-model", "allegro",
        "--physics", stage.physics,
        "--reward-style", "dexscrew",
        "--tip-anchor", "bottom",
        "--rod-mass-scale", str(stage.mass_scale),
        "--rod-friction-cap", "4",
        "--axis-stabilizer-scale", str(stage.stabilizer),
        "--tilt-terminate-rad", "1.2",
        "--contact-reward-mode", "discrete",
        "--three-contact-reward", "3",
        "--contact-window-steps", "25",
        "--contact-window-threshold", "18",
        "--three-contact-required",
        "--omega-success-threshold", "0.5",
        "--omega-success-hold-seconds", "10",
        "--episode-seconds", "20",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--out", str(output),
    ]
    if stage.physics == "revolute":
        cmd += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        cmd += [
            "--tip-connect",
            "--tip-connect-solref", str(stage.tip_solref),
            "--dexscrew-tilt-scale", "1",
        ]
    if hand_pose_config:
        cmd += ["--hand-pose-config", hand_pose_config]
    vecnorm = vecnormalize_for(model)
    if vecnorm.exists():
        cmd += ["--vecnormalize", str(vecnorm)]
    subprocess.run(cmd, cwd=ROOT, check=False)
    if not output.exists():
        raise RuntimeError(f"Evaluation did not produce {output}")
    return json.loads(output.read_text())


def checkpoint_score(metrics: dict) -> tuple[float, ...]:
    contact3 = float(
        (metrics.get("contact_count_step_distribution") or {}).get("3", 0.0)
    )
    return (
        float(metrics.get("passed", False)),
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("omega_hold_seconds_max_mean", 0.0)),
        contact3,
        -float(metrics.get("drop_rate", 1.0)),
        float(metrics.get("axis_rotation_deg_mean", 0.0)),
    )


def gate(stage: Stage, metrics: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    contact3 = float(
        (metrics.get("contact_count_step_distribution") or {}).get("3", 0.0)
    )
    if contact3 < 0.72:
        failures.append(f"three-contact occupancy {contact3:.3f} < 0.72")
    if float(metrics.get("drop_rate", 1.0)) > (0.15 if stage.final else 0.40):
        failures.append(f"drop rate {metrics.get('drop_rate')} too high")
    if stage.physics == "tip_connect" and float(metrics.get("tip_error_m_mean", 1.0)) >= 0.02:
        failures.append(f"tip error {metrics.get('tip_error_m_mean')} >= 0.02 m")
    if stage.final:
        if float(metrics.get("success_rate", 0.0)) < 0.5:
            failures.append(f"success rate {metrics.get('success_rate')} < 0.5")
        if float(metrics.get("stabilizer_torque_max_mean", 1.0)) > 1e-9:
            failures.append("final stage used nonzero stabilizer torque")
    elif (
        float(metrics.get("success_rate", 0.0)) < 0.2
        and float(metrics.get("omega_hold_seconds_max_mean", 0.0)) < 2.0
    ):
        failures.append("neither success>=0.2 nor mean max omega hold>=2 s")
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-id", default=None)
    parser.add_argument("--start-scale", type=float, default=10.0)
    parser.add_argument("--steps-per-stage", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hand-pose-config", type=str, default=None)
    args = parser.parse_args()
    if args.start_scale < 1.0:
        parser.error("--start-scale must be >= 1")
    if args.smoke:
        args.steps_per_stage = min(args.steps_per_stage, 20_000)
        args.num_envs = min(args.num_envs, 4)
        args.eval_episodes = min(args.eval_episodes, 5)
        args.checkpoint_freq = min(args.checkpoint_freq, 10_000)
        args.max_retries = 0

    curriculum_id = args.curriculum_id or (
        f"{datetime.now():%Y%m%d-%H%M}-allegro-tip-bottom-seed{args.seed}"
    )
    curriculum_dir = ROOT / "runs" / "curricula" / curriculum_id
    curriculum_dir.mkdir(parents=True, exist_ok=False)
    stages = curriculum_stages(args.start_scale)
    state: dict = {
        "curriculum_id": curriculum_id,
        "seed": args.seed,
        "smoke": args.smoke,
        "hand_pose_config": (
            str(Path(args.hand_pose_config).expanduser().resolve())
            if args.hand_pose_config
            else None
        ),
        "stages": [asdict(stage) for stage in stages],
        "completed": [],
    }
    (curriculum_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    parent: Path | None = None
    for stage_index, stage in enumerate(stages):
        retries = 0
        while True:
            run_id = (
                f"{curriculum_id}-{stage.name}-s{stage.mass_scale:g}"
                f"-retry{retries}"
            )
            run_dir = train_stage(
                stage,
                run_id=run_id,
                steps=args.steps_per_stage,
                num_envs=args.num_envs,
                device=args.device,
                seed=args.seed,
                checkpoint_freq=args.checkpoint_freq,
                parent=parent,
                hand_pose_config=args.hand_pose_config,
            )
            checkpoints = sorted(
                (run_dir / "checkpoints").glob("ppo_rod_*_steps.zip")
            )
            checkpoints.append(run_dir / "checkpoints" / "final_model.zip")
            evaluated: list[tuple[Path, dict]] = []
            for candidate_index, checkpoint in enumerate(checkpoints):
                output = curriculum_dir / (
                    f"{stage_index:02d}_{stage.name}_r{retries}_"
                    f"candidate{candidate_index:02d}.json"
                )
                metrics = evaluate_checkpoint(
                    stage,
                    checkpoint,
                    episodes=args.eval_episodes,
                    seed=args.seed + 100 * stage_index,
                    output=output,
                    hand_pose_config=args.hand_pose_config,
                )
                evaluated.append((checkpoint, metrics))
            parent, best_metrics = max(
                evaluated, key=lambda item: checkpoint_score(item[1])
            )
            passed, failures = gate(stage, best_metrics)
            record = {
                "stage": asdict(stage),
                "run_id": run_id,
                "selected_checkpoint": str(parent),
                "selected_vecnormalize": str(vecnormalize_for(parent)),
                "metrics": best_metrics,
                "passed": passed,
                "failures": failures,
                "retry": retries,
            }
            state["completed"].append(record)
            state["last_checkpoint"] = str(parent)
            (curriculum_dir / "state.json").write_text(
                json.dumps(state, indent=2) + "\n"
            )
            with (curriculum_dir / "CURRICULUM_PROGRESS.md").open("a") as handle:
                handle.write(
                    f"## {stage.name} retry {retries}\n"
                    f"- selected: `{parent}`\n"
                    f"- passed: {passed}\n"
                    f"- failures: {failures or 'none'}\n"
                    f"- success: {best_metrics.get('success_rate')}\n"
                    f"- rotation: {best_metrics.get('axis_rotation_deg_mean')}\n"
                    f"- tip error: {best_metrics.get('tip_error_m_mean')}\n"
                    f"- drop: {best_metrics.get('drop_rate')}\n\n"
                )
            if passed or args.smoke:
                break
            retries += 1
            if retries > args.max_retries:
                state["status"] = "failed"
                state["failed_stage"] = stage.name
                (curriculum_dir / "state.json").write_text(
                    json.dumps(state, indent=2) + "\n"
                )
                return 2

    state["status"] = "smoke_completed" if args.smoke else "completed"
    state["final_checkpoint"] = str(parent)
    (curriculum_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
