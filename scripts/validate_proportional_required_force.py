#!/usr/bin/env python3
"""Controlled static-trajectory preload sweep for proportional friction physics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from allegro_rod_mvp import RodRotationEnv


MASSES = (400.0, 25.0, 1.0)
PRELOAD_MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
REFERENCE_TORQUE_NM = 5.0
TRACK_SECONDS = 2.0
MAX_ANGLE_ERROR_RAD = np.deg2rad(5.0)
MAX_OMEGA_RAD_S = 0.2
MIN_TWO_CONTACT_FRACTION = 0.90
ROD_RADIUS_M = 0.01


def run_trial(
    mass_scale: float,
    preload_multiplier: float,
    pose: str,
    grasp: str,
    seed: int,
    reference_torque_nm: float,
) -> dict:
    friction_scale = 4.0 * mass_scale / 400.0
    env = RodRotationEnv(
        hand_model="allegro",
        physics_mode="revolute",
        tip_anchor="bottom",
        hand_pose_config=pose,
        hand_grasp_config=grasp,
        rod_mass_scale=mass_scale,
        contact_friction_scale=friction_scale,
        contact_friction_scaling_mode="full_vector",
        scale_rod_joint_dynamics_from_s400=True,
        scale_tip_solref_with_mass=False,
        axis_stabilizer_scale=0.0,
        reset_joint_noise=0.0,
    )
    try:
        env.reset_joint_noise = 0.0
        env.reset(seed=seed)
        contact_qpos = np.asarray(
            env.data.qpos[env.hand_qpos_adr], dtype=np.float64
        ).copy()
        target = contact_qpos + preload_multiplier * (
            env._grasp_qpos - contact_qpos
        )
        target = np.clip(
            target,
            env.model.actuator_ctrlrange[:, 0],
            env.model.actuator_ctrlrange[:, 1],
        )
        env.data.ctrl[:] = target
        for _ in range(100):
            env.data.xfrc_applied[:] = 0.0
            mujoco.mj_step(env.model, env.data)

        initial_angle = float(env.data.qpos[env.hinge_qposadr])
        torque_nm = reference_torque_nm * mass_scale / 400.0
        steps = int(round(TRACK_SECONDS / env.model.opt.timestep))
        angle_errors: list[float] = []
        omegas: list[float] = []
        total_forces: list[float] = []
        contact_counts: list[int] = []
        for _ in range(steps):
            axis = env.data.xmat[env.rod_body].reshape(3, 3)[:, 0]
            env.data.xfrc_applied[:] = 0.0
            env.data.xfrc_applied[env.rod_body, 3:] = axis * torque_nm
            mujoco.mj_step(env.model, env.data)
            angle_errors.append(
                abs(float(env.data.qpos[env.hinge_qposadr]) - initial_angle)
            )
            omegas.append(abs(float(env.data.qvel[env.hinge_dofadr])))
            forces = env._touch()
            total_forces.append(float(np.sum(forces)))
            contact_counts.append(int(np.sum(forces > 0.05)))

        effective = env._effective_pair_friction(env.tip_geom_ids[0])
        analytical_min_force = torque_nm / (
            effective[0] * ROD_RADIUS_M + effective[2]
        )
        max_angle = float(np.max(angle_errors))
        max_omega = float(np.max(omegas))
        two_contact_fraction = float(np.mean(np.asarray(contact_counts) >= 2))
        finite = bool(
            np.isfinite(angle_errors).all()
            and np.isfinite(omegas).all()
            and np.isfinite(total_forces).all()
        )
        passed = bool(
            finite
            and max_angle < MAX_ANGLE_ERROR_RAD
            and max_omega < MAX_OMEGA_RAD_S
            and two_contact_fraction >= MIN_TWO_CONTACT_FRACTION
        )
        return {
            "mass_scale": mass_scale,
            "friction_scale": friction_scale,
            "preload_multiplier": preload_multiplier,
            "applied_axial_torque_nm": torque_nm,
            "effective_pair_friction_vector": effective,
            "analytical_min_total_normal_force_n": analytical_min_force,
            "measured_total_normal_force_median_n": float(
                np.median(total_forces)
            ),
            "measured_total_normal_force_p95_n": float(
                np.percentile(total_forces, 95)
            ),
            "max_angle_error_rad": max_angle,
            "max_angle_error_deg": float(np.degrees(max_angle)),
            "max_abs_omega_rad_s": max_omega,
            "two_contact_fraction": two_contact_fraction,
            "finite": finite,
            "tracks_without_slip": passed,
            "rod_dof_damping": env.model.dof_damping[
                env.rod_dof_adrs
            ].tolist(),
            "rod_dof_armature": env.model.dof_armature[
                env.rod_dof_adrs
            ].tolist(),
            "rod_dof_frictionloss": env.model.dof_frictionloss[
                env.rod_dof_adrs
            ].tolist(),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-pose-config", required=True)
    parser.add_argument("--hand-grasp-config", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-torque-nm", type=float, default=REFERENCE_TORQUE_NM)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    trials = [
        run_trial(
            mass,
            preload,
            args.hand_pose_config,
            args.hand_grasp_config,
            args.seed,
            args.reference_torque_nm,
        )
        for mass in MASSES
        for preload in PRELOAD_MULTIPLIERS
    ]
    minima = {}
    for mass in MASSES:
        passed = [
            trial
            for trial in trials
            if trial["mass_scale"] == mass and trial["tracks_without_slip"]
        ]
        minima[str(mass)] = passed[0] if passed else None

    result = {
        "method": (
            "stationary angular trajectory under axial torque disturbance; "
            "fixed preload controls, no learned policy"
        ),
        "reference_torque_nm_at_s400": args.reference_torque_nm,
        "torque_schedule": f"tau(s)={args.reference_torque_nm:g}*s/400 N m",
        "friction_schedule": "scale(s)=4.0*s/400; full 3-vector on rod and pads",
        "preload_multipliers": list(PRELOAD_MULTIPLIERS),
        "track_seconds": TRACK_SECONDS,
        "thresholds": {
            "max_angle_error_rad": MAX_ANGLE_ERROR_RAD,
            "max_abs_omega_rad_s": MAX_OMEGA_RAD_S,
            "minimum_two_contact_fraction": MIN_TWO_CONTACT_FRACTION,
        },
        "analytical_equation": (
            "N_min ~= tau / (mu_slide*r + mu_torsion), r=0.01 m; "
            "assumes full simultaneous friction capacity and ignores contact "
            "distribution, elliptic coupling, compliance, and actuator limits"
        ),
        "minimum_passing_trials": minima,
        "trials": trials,
    }
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mass_scale",
                "friction_scale",
                "preload_multiplier",
                "applied_axial_torque_nm",
                "analytical_min_total_normal_force_n",
                "measured_total_normal_force_median_n",
                "measured_total_normal_force_p95_n",
                "max_angle_error_deg",
                "max_abs_omega_rad_s",
                "two_contact_fraction",
                "finite",
                "tracks_without_slip",
            ],
        )
        writer.writeheader()
        for trial in trials:
            writer.writerow({key: trial[key] for key in writer.fieldnames})
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
