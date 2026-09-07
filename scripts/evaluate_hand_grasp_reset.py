#!/usr/bin/env python3
"""Deterministically audit a companion Allegro reset/grasp configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from allegro_rod_mvp import RodRotationEnv


def non_tip_rod_contacts(env: RodRotationEnv) -> int:
    tip_ids = set(env.tip_geom_ids)
    count = 0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        pair = {contact.geom1, contact.geom2}
        if env.rod_geom in pair and not pair.intersection(tip_ids):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-pose-config", type=Path, required=True)
    parser.add_argument("--hand-grasp-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for physics in ("revolute", "tip_connect"):
        for mass_scale, friction_scale in ((400.0, 4.0), (1.0, 0.1)):
            for seed in range(args.seeds):
                env = RodRotationEnv(
                    hand_model="allegro",
                    physics_mode=physics,
                    tip_anchor="bottom",
                    hand_pose_config=str(args.hand_pose_config),
                    hand_grasp_config=str(args.hand_grasp_config),
                    rod_mass_scale=mass_scale,
                    contact_friction_scale=friction_scale,
                    scale_tip_solref_with_mass=False,
                    tip_connect_enabled=physics == "tip_connect",
                    tip_connect_solref=0.008,
                    axis_stabilizer_scale=0.0,
                    contact_reward_mode="discrete",
                    three_contact_reward=0.3,
                    contact_window_steps=25,
                    contact_window_threshold=18,
                    three_contact_required=True,
                    reward_style="dexscrew",
                    dexscrew_tilt_scale=0.0 if physics == "revolute" else 1.0,
                    tilt_terminate_rad=1.2,
                )
                obs, _ = env.reset(seed=seed)
                reset_forces = env._touch()
                non_tip = non_tip_rod_contacts(env)
                three_contact_steps = 0
                force_steps: list[np.ndarray] = []
                max_tip_error = 0.0
                termination = "none"
                finite = bool(np.isfinite(obs).all())
                executed = 0
                for _ in range(args.steps):
                    obs, _, terminated, truncated, info = env.step(
                        np.zeros(12, dtype=np.float32)
                    )
                    executed += 1
                    three_contact_steps += int(info["contact_count"] == 3)
                    force_steps.append(np.asarray(info["finger_contact_forces_n"]))
                    max_tip_error = max(max_tip_error, float(info["tip_error_m"]))
                    finite = finite and bool(np.isfinite(obs).all())
                    termination = str(info["termination_reason"])
                    if terminated or truncated:
                        break
                forces = np.asarray(force_steps)
                medians = np.median(forces, axis=0)
                rows.append(
                    {
                        "physics": physics,
                        "mass_scale": mass_scale,
                        "friction_scale": friction_scale,
                        "seed": seed,
                        "steps": executed,
                        "three_contact_occupancy": three_contact_steps / executed,
                        "reset_force_f0_n": reset_forces[0],
                        "reset_force_f1_n": reset_forces[1],
                        "reset_force_f2_n": reset_forces[2],
                        "median_force_f0_n": medians[0],
                        "median_force_f1_n": medians[1],
                        "median_force_f2_n": medians[2],
                        "max_constraint_error_m": max_tip_error,
                        "non_tip_rod_contacts": non_tip,
                        "finite": finite,
                        "termination_reason": termination,
                        "passed": (
                            executed == args.steps
                            and three_contact_steps == args.steps
                            and non_tip == 0
                            and finite
                            and termination == "none"
                        ),
                    }
                )
                env.close()
    with (args.out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[str, dict[str, object]] = {}
    for physics in ("revolute", "tip_connect"):
        for mass_scale in (400.0, 1.0):
            selected = [
                row
                for row in rows
                if row["physics"] == physics and row["mass_scale"] == mass_scale
            ]
            key = f"{physics}_s{mass_scale:g}"
            groups[key] = {
                "passed_seeds": sum(bool(row["passed"]) for row in selected),
                "total_seeds": len(selected),
                "min_three_contact_occupancy": min(
                    float(row["three_contact_occupancy"]) for row in selected
                ),
                "median_forces_n": np.median(
                    [
                        [
                            row["median_force_f0_n"],
                            row["median_force_f1_n"],
                            row["median_force_f2_n"],
                        ]
                        for row in selected
                    ],
                    axis=0,
                ).tolist(),
                "max_constraint_error_m": max(
                    float(row["max_constraint_error_m"]) for row in selected
                ),
                "max_non_tip_rod_contacts": max(
                    int(row["non_tip_rod_contacts"]) for row in selected
                ),
            }
    summary = {
        "hand_pose_path": str(args.hand_pose_config.resolve()),
        "hand_pose_sha256": hashlib.sha256(args.hand_pose_config.read_bytes()).hexdigest(),
        "hand_grasp_path": str(args.hand_grasp_config.resolve()),
        "hand_grasp_sha256": hashlib.sha256(
            args.hand_grasp_config.read_bytes()
        ).hexdigest(),
        "steps_per_seed": args.steps,
        "groups": groups,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(group["passed_seeds"] == group["total_seeds"] for group in groups.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
