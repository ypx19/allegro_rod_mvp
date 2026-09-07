#!/usr/bin/env python3
"""Corrected net-angle curriculum with fully proportional contact physics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from allegro_rod_mvp.hand_pose import load_hand_pose

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SCHEDULE = (400.0, 200.0, 100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 1.0)
FIXED_SEED = 10_000
UNSEEN_SEED = 20_000


def friction_scale(mass_scale: float) -> float:
    return 4.0 * float(mass_scale) / 400.0


def vecnormalize_for(model: Path) -> Path:
    return model.parent / "vecnormalize.pkl"


def common_args(
    physics: str,
    mass_scale: float,
    pose: str,
    grasp: str,
) -> list[str]:
    args = [
        "--hand-model", "allegro",
        "--hand-pose-config", pose,
        "--hand-grasp-config", grasp,
        "--physics", physics,
        "--reward-style", "dexscrew",
        "--tip-anchor", "bottom",
        "--rod-mass-scale", str(mass_scale),
        "--contact-friction-scale", str(friction_scale(mass_scale)),
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
    ]
    if physics == "revolute":
        args += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        args += [
            "--tip-connect",
            "--tip-connect-solref", "0.008",
            "--dexscrew-tilt-scale", "1",
            "--tilt-terminate-rad", "1.2",
        ]
    return args


def evaluate(
    model: Path,
    run_dir: Path,
    physics: str,
    mass_scale: float,
    pose: str,
    grasp: str,
    episodes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = []
    for label, seed in (("fixed", FIXED_SEED), ("unseen", UNSEEN_SEED)):
        out = run_dir / f"eval_{label}.json"
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "eval_policy.py"),
            str(model),
            *common_args(physics, mass_scale, pose, grasp),
            "--episodes", str(episodes),
            "--episode-seconds", "20",
            "--seed", str(seed),
            "--out", str(out),
        ]
        vecnorm = vecnormalize_for(model)
        if vecnorm.exists():
            command += ["--vecnormalize", str(vecnorm)]
        with (run_dir / "logs" / f"eval_{label}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if not out.exists():
            raise RuntimeError(f"evaluation did not produce metrics: {out}")
        results.append(json.loads(out.read_text()))
    return results[0], results[1]


def accepted(fixed: dict[str, Any], unseen: dict[str, Any]) -> bool:
    return bool(
        fixed.get("success_mode") == "net_angle"
        and unseen.get("success_mode") == "net_angle"
        and float(fixed["success_rate"]) >= 0.5
        and float(unseen["success_rate"]) >= 0.5
    )


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": metrics["success_rate"],
        "axis_rotation_deg_mean": metrics["axis_rotation_deg_mean"],
        "tip_error_m_mean": metrics["tip_error_m_mean"],
        "tip_error_m_max": metrics["tip_error_m_max"],
        "drop_rate": metrics["drop_rate"],
        "numerical_instability_episodes": metrics["numerical_instability_episodes"],
        "contact_distribution": metrics["contact_count_step_distribution"],
        "contact_fraction_at_least_one": metrics["contact_fraction_at_least_one"],
        "contact_fraction_at_least_two": metrics["contact_fraction_at_least_two"],
        "normal_contact_force_n": metrics["normal_contact_force_n"],
        "effective_pair_friction_vectors": metrics[
            "effective_pair_friction_vectors"
        ],
        "rod_dof_damping": metrics["rod_dof_damping"],
        "rod_dof_armature": metrics["rod_dof_armature"],
        "rod_dof_frictionloss": metrics["rod_dof_frictionloss"],
    }


def write_eval_only_config(
    run_dir: Path,
    run_id: str,
    model: Path,
    fixed: dict[str, Any],
) -> None:
    config = {
        "run_id": run_id,
        "kind": "corrected_s400_parent_evaluation",
        "checkpoint": str(model),
        "mass_scale": 400.0,
        "friction_scale": 4.0,
        "contact_friction_scaling_mode": "full_vector",
        "scale_rod_joint_dynamics_from_s400": True,
        "effective_pair_friction_vectors": fixed["effective_pair_friction_vectors"],
        "hand_pose_config": fixed["hand_pose_config"],
        "hand_pose_config_sha256": fixed["hand_pose_config_sha256"],
        "hand_grasp_config": fixed["hand_grasp_config"],
        "hand_grasp_config_sha256": fixed["hand_grasp_config_sha256"],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-id", required=True)
    parser.add_argument("--hand-pose-config", required=True)
    parser.add_argument("--revolute-grasp-config", required=True)
    parser.add_argument("--tip-grasp-config")
    parser.add_argument(
        "--start-phase",
        choices=["revolute", "tip_connect"],
        default="revolute",
    )
    parser.add_argument("--initial-revolute-checkpoint", type=Path, required=True)
    parser.add_argument("--steps-per-stage", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pose_content, pose_path, pose_hash = load_hand_pose(
        args.hand_pose_config, "allegro_three_finger_rod_revolute"
    )
    curriculum_dir = ROOT / "runs" / "curricula" / args.curriculum_id
    curriculum_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "curriculum_id": args.curriculum_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "methodology": "full proportional friction; policy force diagnostic only",
        "schedule": list(SCHEDULE),
        "friction_schedule": "4*s/400",
        "pose": {
            "path": str(pose_path),
            "sha256": pose_hash,
            "content": pose_content,
        },
        "gate": "fixed and unseen net-angle success_rate >= 0.5",
        "start_phase": args.start_phase,
        "external_parent_checkpoint": str(
            args.initial_revolute_checkpoint.resolve()
        ),
        "completed": [],
        "status": "running",
    }
    state_path = curriculum_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    parent = args.initial_revolute_checkpoint.resolve()
    if not parent.exists():
        raise FileNotFoundError(parent)

    phase_specs = [
        ("R", "revolute", args.revolute_grasp_config),
        ("T", "tip_connect", args.tip_grasp_config),
    ]
    if args.start_phase == "tip_connect":
        phase_specs = phase_specs[1:]
    for phase, physics, grasp in phase_specs:
        if grasp is None:
            state["status"] = "revolute_completed_tip_grasp_required"
            state["final_revolute_checkpoint"] = str(parent)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2
        if phase == "T" and args.start_phase != "tip_connect" and (
            not state["completed"]
            or state["completed"][-1]["phase"] != "R"
            or state["completed"][-1]["mass_scale"] != 1.0
            or not state["completed"][-1]["accepted"]
        ):
            state["status"] = "blocked_before_tip_connect"
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2

        for stage_index, mass_scale in enumerate(SCHEDULE):
            run_id = (
                f"{args.curriculum_id}-{phase}{stage_index:02d}-s{mass_scale:g}"
                f"-mu{friction_scale(mass_scale):.6g}-seed{args.seed}"
            )
            run_dir = ROOT / "runs" / run_id
            if phase == "R" and stage_index == 0:
                run_dir.mkdir(parents=True, exist_ok=False)
                (run_dir / "logs").mkdir()
                model = parent
            else:
                command = [
                    str(PYTHON),
                    str(ROOT / "scripts" / "train_parallel.py"),
                    "--run-id", run_id,
                    *common_args(physics, mass_scale, str(pose_path), grasp),
                    "--resume", str(parent),
                    "--vecnormalize-path", str(vecnormalize_for(parent)),
                    "--steps", str(args.steps_per_stage),
                    "--num-envs", str(args.num_envs),
                    "--n-steps", "256",
                    "--batch-size", "256",
                    "--checkpoint-freq", str(args.checkpoint_freq),
                    "--ent-coef", "0",
                    "--device", args.device,
                    "--seed", str(args.seed),
                    "--notes",
                    (
                        f"EXP-20260823-017 corrected proportional physics; "
                        f"{physics} s={mass_scale:g}; force diagnostic only"
                    ),
                ]
                subprocess.check_call(command, cwd=ROOT)
                model = run_dir / "checkpoints" / "final_model.zip"

            fixed, unseen = evaluate(
                model,
                run_dir,
                physics,
                mass_scale,
                str(pose_path),
                grasp,
                args.eval_episodes,
            )
            if phase == "R" and stage_index == 0:
                write_eval_only_config(run_dir, run_id, model, fixed)
            stage_accepted = accepted(fixed, unseen)
            record = {
                "phase": phase,
                "physics": physics,
                "stage_index": stage_index,
                "mass_scale": mass_scale,
                "friction_scale": friction_scale(mass_scale),
                "run_id": run_id,
                "checkpoint": str(model),
                "warm_start": str(parent),
                "fixed": compact(fixed),
                "unseen": compact(unseen),
                "accepted": stage_accepted,
            }
            state["completed"].append(record)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            if not stage_accepted:
                state["status"] = f"failed_{phase}_s{mass_scale:g}"
                state["failed_record"] = record
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                return 2
            parent = model

    state["status"] = "completed"
    state["final_checkpoint"] = str(parent)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
