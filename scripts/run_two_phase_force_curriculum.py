#!/usr/bin/env python3
"""Strict revolute-then-bottom-tip curriculum with measured force calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from allegro_rod_mvp.hand_pose import load_hand_pose

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

SCHEDULE = (400.0, 200.0, 100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 1.0)
MU_SCALE_MIN = 0.10
MU_SCALE_MAX = 4.0
FORCE_REL_TOL = 0.20
FIXED_SEED = 10_000
UNSEEN_SEED = 20_000


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        return ""


def pose_snapshot(path: str) -> dict[str, Any]:
    content_r, resolved, digest = load_hand_pose(path, "allegro_three_finger_rod_revolute")
    content_t, _, digest_t = load_hand_pose(path, "allegro_three_finger_rod")
    if digest != digest_t or content_r != content_t:
        raise ValueError("hand pose resolves differently between physics modes")
    return {"path": str(resolved), "sha256": digest, "content": content_r}


def clamp_mu(value: float) -> float:
    return min(MU_SCALE_MAX, max(MU_SCALE_MIN, float(value)))


def physical_mu_guess(reference_mu: float, scale: float) -> float:
    # I∝s and τ=Iα. With τ_friction≈μNr, constant N initially implies μ∝s.
    return clamp_mu(reference_mu * float(scale) / 400.0)


def vecnormalize_for(model: Path) -> Path:
    if model.name == "final_model.zip":
        return model.parent / "vecnormalize.pkl"
    return model.parent / f"{model.stem}_vecnormalize.pkl"


def common_args(
    physics: str,
    scale: float,
    mu_scale: float,
    pose_path: str,
    grasp_path: str,
) -> list[str]:
    args = [
        "--hand-model", "allegro",
        "--hand-pose-config", pose_path,
        "--hand-grasp-config", grasp_path,
        "--physics", physics,
        "--reward-style", "dexscrew",
        "--tip-anchor", "bottom",
        "--rod-mass-scale", str(scale),
        "--contact-friction-scale", str(mu_scale),
        "--no-scale-tip-solref-with-mass",
        "--axis-stabilizer-scale", "0",
        "--contact-reward-mode", "discrete",
        "--three-contact-reward", "0.3",
        "--contact-window-steps", "25",
        "--contact-window-threshold", "18",
        "--three-contact-required",
        "--no-rotation-requires-three-contacts",
        "--no-contact-support-termination",
        "--contact-reward-scale", "0.1",
        "--omega-success-threshold", "0.5",
        "--omega-success-hold-seconds", "10",
        "--success-mode", "net_angle",
        "--net-angle-success-threshold-rad", str(math.pi),
    ]
    if physics == "revolute":
        args += ["--no-tip-connect", "--dexscrew-tilt-scale", "0"]
    else:
        args += [
            "--tip-connect", "--tip-connect-solref", "0.008",
            "--dexscrew-tilt-scale", "1", "--tilt-terminate-rad", "1.2",
        ]
    return args


def train(
    run_id: str,
    physics: str,
    scale: float,
    mu_scale: float,
    pose_path: str,
    grasp_path: str,
    parent: Path | None,
    args: argparse.Namespace,
) -> Path:
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    cmd = [
        str(PYTHON), str(ROOT / "scripts" / "train_parallel.py"),
        "--run-id", run_id,
        *common_args(physics, scale, mu_scale, pose_path, grasp_path),
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
            f"Strict phase {physics}; s={scale:g}; explicit rod+tip friction scale "
            f"{mu_scale:g}; C gait settings; net-angle success; pose={pose_path}"
        ),
    ]
    if parent is not None:
        cmd += ["--resume", str(parent)]
        vecnorm = vecnormalize_for(parent)
        if vecnorm.exists():
            cmd += ["--vecnormalize-path", str(vecnorm)]
    subprocess.check_call(cmd, cwd=ROOT)
    return run_dir / "checkpoints" / "final_model.zip"


def write_eval_artifacts(
    run_id: str,
    command: list[str],
    metrics: dict[str, Any],
    pose: dict[str, Any],
    grasp_path: str,
    parent: Path,
) -> Path:
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "plots", "videos", "images"):
        (run_dir / name).mkdir()
    (run_dir / "logs" / "eval.log").write_text(json.dumps(metrics, indent=2) + "\n")
    config = {
        "run_id": run_id,
        "kind": "deterministic_force_calibration",
        "warm_start_checkpoint": str(parent),
        "hand_pose": pose,
        "hand_grasp_config": str(Path(grasp_path).expanduser().resolve()),
        "command": " ".join(command),
        "physics": metrics["physics_mode"],
        "rod_mass_scale": metrics["rod_mass_scale"],
        "contact_friction_scale": metrics["contact_friction_scale"],
        "contact_friction_vector": metrics.get("contact_friction_vector"),
        "force_tolerance_relative": FORCE_REL_TOL,
        "mu_scale_bounds": [MU_SCALE_MIN, MU_SCALE_MAX],
    }
    import yaml

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_value("rev-parse", "--short", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "command": " ".join(command),
        "seed": metrics["seed"],
        "device": "cpu deterministic evaluation",
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "framework_versions": {},
        "baseline_run": parent.parents[1].name,
        "notes": "Normal contact force is MuJoCo contact-frame force, not actuator effort.",
        "hand_pose": pose,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    conditioned = metrics["conditioned_positive_rotation_support_force_n"]
    per_tip = conditioned["per_tip"]
    total = conditioned["total"]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed", "episodes", "success_rate", "axis_rotation_deg_mean",
                "tip_error_m_mean", "three_contact_occupancy", "drop_rate",
                "force_tip0_p95_n", "force_tip1_p95_n", "force_tip2_p95_n",
                "force_eligible_step_fraction", "force_total_mean_n",
                "force_total_median_n", "force_total_p95_n",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "seed": metrics["seed"],
            "episodes": metrics["episodes"],
            "success_rate": metrics["success_rate"],
            "axis_rotation_deg_mean": metrics["axis_rotation_deg_mean"],
            "tip_error_m_mean": metrics["tip_error_m_mean"],
            "three_contact_occupancy": metrics["contact_count_step_distribution"]["3"],
            "drop_rate": metrics["drop_rate"],
            "force_eligible_step_fraction": conditioned["eligible_step_fraction"],
            "force_tip0_p95_n": per_tip[0]["p95"],
            "force_tip1_p95_n": per_tip[1]["p95"],
            "force_tip2_p95_n": per_tip[2]["p95"],
            "force_total_mean_n": total["mean"],
            "force_total_median_n": total["median"],
            "force_total_p95_n": total["p95"],
        })
    median_text = (
        "unavailable" if total["median"] is None else f"{total['median']:.3f} N"
    )
    (run_dir / "summary.md").write_text(
        "# Deterministic force calibration\n\n"
        f"- checkpoint: `{parent}`\n"
        f"- success rate: {metrics['success_rate']:.3f}\n"
        f"- rotation: {metrics['axis_rotation_deg_mean']:.3f} deg\n"
        f"- 3-tip occupancy: {metrics['contact_count_step_distribution']['3']:.3f}\n"
        f"- drop rate: {metrics['drop_rate']:.3f}\n"
        f"- conditioned total normal force median: {median_text}\n"
        f"- force-eligible step fraction: {conditioned['eligible_step_fraction']:.3f}\n"
        f"- excluded step fraction: {conditioned['excluded_step_fraction']:.3f}\n"
        f"- condition: {conditioned['condition']}\n"
        "- force definition: measured contact-frame normal force; not actuator effort\n"
    )
    return run_dir


def evaluate(
    run_id: str,
    model: Path,
    physics: str,
    scale: float,
    mu_scale: float,
    pose: dict[str, Any],
    grasp_path: str,
    seed: int,
    episodes: int,
) -> dict[str, Any]:
    temp_out = ROOT / "runs" / "curricula" / f".{run_id}.json"
    cmd = [
        str(PYTHON), str(ROOT / "scripts" / "eval_policy.py"), str(model),
        *common_args(physics, scale, mu_scale, pose["path"], grasp_path),
        "--episodes", str(episodes), "--episode-seconds", "20",
        "--seed", str(seed), "--out", str(temp_out),
    ]
    vecnorm = vecnormalize_for(model)
    if vecnorm.exists():
        cmd += ["--vecnormalize", str(vecnorm)]
    subprocess.run(cmd, cwd=ROOT, check=False)
    metrics = json.loads(temp_out.read_text())
    temp_out.unlink()
    metrics["seed"] = seed
    write_eval_artifacts(run_id, cmd, metrics, pose, grasp_path, model)
    return metrics


def task_gate(metrics: dict[str, Any], physics: str) -> list[str]:
    failures: list[str] = []
    if metrics.get("success_mode") != "net_angle":
        failures.append("evaluation did not use net_angle success")
    if float(metrics["success_rate"]) < 0.5:
        failures.append("success_rate < 0.5")
    return failures


def force_gate(metrics: dict[str, Any], target_median: float | None) -> tuple[list[str], float | None]:
    conditioned = metrics["conditioned_positive_rotation_support_force_n"]
    measured_raw = conditioned["total"]["median"]
    if measured_raw is None or int(conditioned["eligible_steps"]) == 0:
        return ["no positive-rotation >=2-contact force samples"], None
    measured = float(measured_raw)
    if target_median is None:
        return [], measured
    low = target_median * (1.0 - FORCE_REL_TOL)
    high = target_median * (1.0 + FORCE_REL_TOL)
    return (
        []
        if low <= measured <= high
        else [f"conditioned force median {measured:.3f} outside [{low:.3f}, {high:.3f}]"],
        measured,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-id", default=None)
    parser.add_argument("--hand-pose-config", required=True)
    parser.add_argument("--revolute-grasp-config", required=True)
    parser.add_argument("--tip-grasp-config")
    parser.add_argument(
        "--initial-revolute-checkpoint",
        type=Path,
        help="Warm-start Phase R s=400 from a prior gated training attempt.",
    )
    parser.add_argument("--steps-per-stage", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-force-iterations", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.steps_per_stage = min(args.steps_per_stage, 2_048)
        args.num_envs = min(args.num_envs, 2)
        args.checkpoint_freq = min(args.checkpoint_freq, 1_024)
        args.eval_episodes = min(args.eval_episodes, 2)
    pose = pose_snapshot(args.hand_pose_config)
    curriculum_id = args.curriculum_id or f"{datetime.now():%Y%m%d-%H%M}-two-phase-force-seed{args.seed}"
    curriculum_dir = ROOT / "runs" / "curricula" / curriculum_id
    curriculum_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "curriculum_id": curriculum_id,
        "pose": pose,
        "revolute_grasp_config": str(Path(args.revolute_grasp_config).resolve()),
        "tip_grasp_config": (
            str(Path(args.tip_grasp_config).resolve()) if args.tip_grasp_config else None
        ),
        "initial_revolute_checkpoint": (
            str(args.initial_revolute_checkpoint.resolve())
            if args.initial_revolute_checkpoint
            else None
        ),
        "schedule": list(SCHEDULE),
        "mu_scale_bounds": [MU_SCALE_MIN, MU_SCALE_MAX],
        "force_statistic": (
            "median total fingertip normal force conditioned on axial_omega > "
            "0.5 rad/s and contact_count >= 2"
        ),
        "force_tolerance_relative": FORCE_REL_TOL,
        "fixed_seed": FIXED_SEED,
        "unseen_seed": UNSEEN_SEED,
        "gate": {
            "success_mode": "net_angle",
            "success_rate_min_fixed_and_unseen": 0.5,
            "per_episode": (
                "net angle >= pi, tilt <0.25 rad, tip error <0.02 m, "
                "finite/stable, no physical drop"
            ),
            "contact_metrics_are_diagnostic_only": True,
        },
        "completed": [],
        "status": "running",
    }
    state_path = curriculum_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    parent: Path | None = args.initial_revolute_checkpoint
    if parent is not None and not parent.exists():
        raise FileNotFoundError(f"initial checkpoint does not exist: {parent}")
    reference_mu = MU_SCALE_MAX
    reference_force: dict[str, float | None] = {"revolute": None, "tip_connect": None}
    if parent is not None:
        fixed = evaluate(
            f"{curriculum_id}-R00-s400-C-parent-cal-fixed",
            parent,
            "revolute",
            400.0,
            reference_mu,
            pose,
            args.revolute_grasp_config,
            FIXED_SEED,
            args.eval_episodes,
        )
        unseen = evaluate(
            f"{curriculum_id}-R00-s400-C-parent-cal-unseen",
            parent,
            "revolute",
            400.0,
            reference_mu,
            pose,
            args.revolute_grasp_config,
            UNSEEN_SEED,
            args.eval_episodes,
        )
        task_failures = task_gate(fixed, "revolute") + task_gate(unseen, "revolute")
        force_failures, measured_force = force_gate(fixed, None)
        accepted = not task_failures and not force_failures
        record = {
            "phase": "R",
            "physics": "revolute",
            "scale": 400.0,
            "mu_scale": reference_mu,
            "mu_vector": fixed.get("contact_friction_vector"),
            "iteration": "accepted_parent_evaluation",
            "run_id": "20260823-1812-finger-gait-contact-scale010-C-s400-seed0",
            "checkpoint": str(parent),
            "warm_start": None,
            "fixed_metrics": fixed,
            "unseen_metrics": unseen,
            "force_target_median_n": None,
            "force_measured_median_n": measured_force,
            "task_failures": task_failures,
            "force_failures": force_failures,
            "accepted": accepted,
        }
        state["completed"].append(record)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        if not accepted:
            state["status"] = "failed_C_parent_s400"
            state["failed_record"] = record
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2
        reference_force["revolute"] = measured_force
    for phase, physics in (("R", "revolute"), ("T", "tip_connect")):
        grasp_path = (
            args.revolute_grasp_config if phase == "R" else args.tip_grasp_config
        )
        if grasp_path is None:
            state["status"] = "blocked_missing_tip_grasp_config"
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            return 2
        if phase == "T":
            revolute_final = state["completed"][-1]
            if (
                revolute_final["phase"] != "R"
                or revolute_final["scale"] != 1.0
                or not revolute_final["accepted"]
            ):
                state["status"] = "blocked_before_tip_connect"
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                return 2
        phase_schedule = SCHEDULE[1:] if phase == "R" and parent is not None else SCHEDULE
        stage_offset = 1 if phase == "R" and parent is not None else 0
        for local_stage_index, scale in enumerate(phase_schedule):
            stage_index = local_stage_index + stage_offset
            mu_scale = physical_mu_guess(reference_mu, scale)
            for iteration in range(args.max_force_iterations + 1):
                run_id = (
                    f"{curriculum_id}-{phase}{stage_index:02d}-s{scale:g}"
                    f"-mu{mu_scale:.6g}-iter{iteration}-seed{args.seed}"
                )
                model = train(
                    run_id, physics, scale, mu_scale, pose["path"], grasp_path, parent, args
                )
                fixed = evaluate(
                    f"{run_id}-cal-fixed", model, physics, scale, mu_scale,
                    pose, grasp_path, FIXED_SEED, args.eval_episodes,
                )
                unseen = evaluate(
                    f"{run_id}-cal-unseen", model, physics, scale, mu_scale,
                    pose, grasp_path, UNSEEN_SEED, args.eval_episodes,
                )
                task_failures = task_gate(fixed, physics) + task_gate(unseen, physics)
                force_failures, measured_force = force_gate(
                    fixed, reference_force[physics]
                )
                accepted = not task_failures and not force_failures
                record = {
                    "phase": phase, "physics": physics, "scale": scale,
                    "mu_scale": mu_scale, "mu_vector": fixed.get("contact_friction_vector"),
                    "iteration": iteration, "run_id": run_id,
                    "checkpoint": str(model), "warm_start": str(parent) if parent else None,
                    "fixed_metrics": fixed, "unseen_metrics": unseen,
                    "force_target_median_n": reference_force[physics],
                    "force_measured_median_n": measured_force,
                    "task_failures": task_failures, "force_failures": force_failures,
                    "accepted": accepted,
                }
                state["completed"].append(record)
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                if accepted:
                    parent = model
                    if math.isclose(scale, 400.0):
                        reference_force[physics] = measured_force
                        reference_mu = mu_scale
                    break
                # Friction adaptation is meaningful only when the policy still rotates
                # and the sole failed criterion is force matching.
                if task_failures or reference_force[physics] is None:
                    state["status"] = f"failed_{phase}_s{scale:g}"
                    state["failed_record"] = record
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    return 2
                if measured_force is None:
                    state["status"] = f"force_unavailable_{phase}_s{scale:g}"
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    return 2
                next_mu = clamp_mu(mu_scale * measured_force / float(reference_force[physics]))
                if math.isclose(next_mu, mu_scale, rel_tol=1e-4):
                    state["status"] = f"force_saturated_{phase}_s{scale:g}"
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    return 2
                mu_scale = next_mu
            else:
                state["status"] = f"force_calibration_failed_{phase}_s{scale:g}"
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                return 2
    state["status"] = "completed"
    state["final_checkpoint"] = str(parent)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
