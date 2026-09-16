#!/usr/bin/env python3
"""Headless policy evaluation with stage / DexScrew success gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any
import math

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.support_collapse_metrics import (
    DEFAULT_COLLAPSE_LOOKBACK,
    DEFAULT_RECONTACT_WINDOW,
    aggregate_support_collapse_metrics,
    episode_support_collapse_metrics,
    plot_support_collapse_trace,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    values.append(" ".join(str(float(x)) for x in np.asarray(value).reshape(-1)))
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def _rollout_diagnostic_trace(
    env: RodRotationEnv,
    model: PPO,
    vecnorm: VecNormalize | None,
    seed: int,
    out_dir: Path,
) -> dict:
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    info: dict = {}
    rows: list[dict] = []
    n_contacts: list[int] = []
    omegas: list[float] = []
    tilts: list[float] = []
    forces: list[list[float]] = []
    net_angles: list[float] = []
    axials: list[float] = []
    t = 0
    while not (terminated or truncated):
        model_obs = obs
        if vecnorm is not None:
            model_obs = vecnorm.normalize_obs(
                np.asarray(obs, dtype=np.float32).reshape(1, -1)
            )[0]
        action, _ = model.predict(model_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        t += 1
        touch = np.asarray(
            info.get("finger_contact_forces_n", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        fingers = np.asarray(info.get("finger_contacts", [0, 0, 0]), dtype=np.int64)
        n_contact = int(info.get("contact_count", 0))
        omega_perp = float(info.get("omega_perp_norm", info.get("lateral_omega", 0.0)))
        tilt = float(info.get("axis_tilt_rad", 0.0))
        n_contacts.append(n_contact)
        omegas.append(omega_perp)
        tilts.append(tilt)
        forces.append(touch.tolist())
        net_angles.append(float(info.get("axis_rotation", 0.0)))
        axials.append(float(info.get("axial_omega", 0.0)))
        row = {
            "timestep": t,
            "net_angle": float(info.get("axis_rotation", 0.0)),
            "axial_omega": float(info.get("axial_omega", 0.0)),
            "omega_perp_norm": omega_perp,
            "tilt": tilt,
            "tip_error": float(info.get("tip_error_m", 0.0)),
            "n_contact": n_contact,
            "finger0_contact": int(fingers[0]) if fingers.size else 0,
            "finger1_contact": int(fingers[1]) if fingers.size > 1 else 0,
            "finger2_contact": int(fingers[2]) if fingers.size > 2 else 0,
            "finger0_force": float(touch[0]) if touch.size else 0.0,
            "finger1_force": float(touch[1]) if touch.size > 1 else 0.0,
            "finger2_force": float(touch[2]) if touch.size > 2 else 0.0,
            "reward_rotation_before_support_gate": float(
                info.get("reward_rotation_before_support_gate", info.get("reward_rotation", 0.0))
            ),
            "reward_rotation_after_support_gate": float(info.get("reward_rotation", 0.0)),
            "reward_low_support_wobble": float(info.get("reward_low_support_wobble", 0.0)),
            "reward_total": float(info.get("reward_total", reward)),
        }
        for i, value in enumerate(action.tolist()):
            row[f"action_{i}"] = float(value)
        rows.append(row)
    summary = episode_support_collapse_metrics(
        n_contacts,
        omegas,
        tilts,
        termination_reason=str(info.get("termination_reason", "none")),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"trace_seed{seed}.csv"
    json_path = out_dir / f"trace_seed{seed}.json"
    plot_path = out_dir / f"trace_seed{seed}.png"
    _write_trace_csv(csv_path, rows)
    payload = {
        "seed": seed,
        "termination_reason": info.get("termination_reason", "none"),
        "episode_length": t,
        "axis_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
        "metrics": summary,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    plot_support_collapse_trace(
        {
            "timestep": [row["timestep"] for row in rows],
            "n_contact": n_contacts,
            "omega_perp_norm": omegas,
            "tilt": tilts,
            "axial_omega": axials,
            "net_angle": net_angles,
            "finger_force": np.asarray(forces, dtype=np.float64),
            "support_loss_steps": summary["support_loss_steps"],
            "recontact_steps": summary["recontact_steps"],
            "tilt_cross_0_25_steps": summary["tilt_cross_0_25_steps"],
        },
        str(plot_path),
        title=f"seed {seed}  {info.get('termination_reason', 'none')}  {t} steps",
        termination_step=t if (terminated or truncated) else None,
    )
    return payload


def evaluate(
    model_path: str,
    stage: int,
    episodes: int,
    seed: int,
    episode_seconds: float,
    tip_connect_solref: float | None = None,
    tip_connect_enabled: bool | None = None,
    axis_stabilizer_scale: float | None = None,
    axis_tilt_penalty_weight: float = 1.0,
    axis_tilt_recovery_scale: float = 0.0,
    axis_tilt_growth_scale: float = 0.0,
    rotation_reward_scale: float = 16.0,
    contact_reward_mode: str = "linear",
    three_contact_reward: float = 10.0,
    contact_window_steps: int = 0,
    contact_window_threshold: float = 0.0,
    three_contact_required: bool = False,
    rotation_requires_three_contacts: bool = True,
    contact_support_termination_enabled: bool = True,
    contact_reward_scale: float = 1.0,
    physics_mode: str = "tip_connect",
    reward_style: str = "stage",
    privileged_obs: bool = False,
    omega_success_threshold: float = 0.5,
    omega_success_hold_seconds: float = 10.0,
    success_mode: str = "omega_hold",
    net_angle_success_threshold_rad: float = np.pi,
    dexscrew_tilt_scale: float | None = None,
    rod_mass_scale: float = 1.0,
    rod_friction_cap: float = 4.0,
    contact_friction_scale: float | None = None,
    contact_friction_scaling_mode: str = "sliding_only",
    scale_rod_joint_dynamics_from_s400: bool = False,
    scale_tip_solref_with_mass: bool = True,
    tilt_terminate_rad: float = 0.7,
    tip_anchor: str = "top",
    dexscrew_tip_penalty_scale: float = 0.5,
    dexscrew_tip_sigma: float = 0.025,
    vecnormalize: str | None = None,
    hand_model: str = "allegro",
    hand_pose_config: str | None = None,
    hand_grasp_config: str | None = None,
    obs_history_len: int = 1,
    support_aware_reward_enabled: bool = False,
    rotation_contact_scale_0: float = 0.0,
    rotation_contact_scale_1: float = 0.1,
    rotation_contact_scale_2plus: float = 1.0,
    low_support_wobble_scale: float = 0.5,
    recontact_window_steps: int = DEFAULT_RECONTACT_WINDOW,
    collapse_lookback_steps: int = DEFAULT_COLLAPSE_LOOKBACK,
    trace_dir: str | None = None,
    trace_seeds: list[int] | None = None,
) -> dict:
    def _make_env() -> RodRotationEnv:
        return RodRotationEnv(
            render_mode=None,
            curriculum_stage=stage,
            episode_seconds=episode_seconds,
            tip_connect_solref=tip_connect_solref,
            tip_connect_enabled=tip_connect_enabled,
            axis_stabilizer_scale=axis_stabilizer_scale,
            axis_tilt_penalty_weight=axis_tilt_penalty_weight,
            axis_tilt_recovery_scale=axis_tilt_recovery_scale,
            axis_tilt_growth_scale=axis_tilt_growth_scale,
            rotation_reward_scale=rotation_reward_scale,
            contact_reward_mode=contact_reward_mode,
            three_contact_reward=three_contact_reward,
            contact_window_steps=contact_window_steps,
            contact_window_threshold=contact_window_threshold,
            three_contact_required=three_contact_required,
            rotation_requires_three_contacts=rotation_requires_three_contacts,
            contact_support_termination_enabled=contact_support_termination_enabled,
            contact_reward_scale=contact_reward_scale,
            physics_mode=physics_mode,
            reward_style=reward_style,
            privileged_obs=privileged_obs,
            omega_success_threshold=omega_success_threshold,
            omega_success_hold_seconds=omega_success_hold_seconds,
            success_mode=success_mode,
            net_angle_success_threshold_rad=net_angle_success_threshold_rad,
            dexscrew_tilt_scale=dexscrew_tilt_scale,
            dexscrew_tip_penalty_scale=dexscrew_tip_penalty_scale,
            dexscrew_tip_sigma=dexscrew_tip_sigma,
            rod_mass_scale=rod_mass_scale,
            rod_friction_cap=rod_friction_cap,
            contact_friction_scale=contact_friction_scale,
            contact_friction_scaling_mode=contact_friction_scaling_mode,
            scale_rod_joint_dynamics_from_s400=scale_rod_joint_dynamics_from_s400,
            scale_tip_solref_with_mass=scale_tip_solref_with_mass,
            tilt_terminate_rad=tilt_terminate_rad,
            tip_anchor=tip_anchor,
            hand_model=hand_model,
            hand_pose_config=hand_pose_config,
            hand_grasp_config=hand_grasp_config,
            obs_history_len=obs_history_len,
            support_aware_reward_enabled=support_aware_reward_enabled,
            rotation_contact_scale_0=rotation_contact_scale_0,
            rotation_contact_scale_1=rotation_contact_scale_1,
            rotation_contact_scale_2plus=rotation_contact_scale_2plus,
            low_support_wobble_scale=low_support_wobble_scale,
        )

    env = _make_env()
    step_seconds = float(env.frame_skip * env.model.opt.timestep)
    model = PPO.load(model_path, device="cpu")
    vecnorm: VecNormalize | None = None
    if vecnormalize:
        dummy = DummyVecEnv([_make_env])
        vecnorm = VecNormalize.load(vecnormalize, dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False

    rotations = []
    tip_errors = []
    contacts = []
    contact_step_counts: Counter[int] = Counter()
    finger_contact_steps = np.zeros(3, dtype=np.int64)
    total_contact_steps = 0
    successes = []
    legacy_omega_successes = []
    net_angle_successes = []
    drops = []
    final_axis_tilts = []
    episode_axis_tilt_maxes = []
    omega_hold_satisfied = []
    max_omega_hold_seconds = []
    min_window_three_contact_occupancy = []
    gate_failure_steps = []
    finger_loss_longest_steps = [[], [], []]
    finger_leave_return_events = [[], [], []]
    complete_loss_longest_steps = []
    below_two_contact_longest_steps = []
    cumulative_abs_rotation_deg = []
    completed_net_rotation_cycles = []
    axial_omega_values = []
    axial_slip_values = []
    episode_tip_error_maxes = []
    episode_torque_means = []
    episode_torque_maxes = []
    termination_reasons: Counter[str] = Counter()
    reward_keys = (
        "reward_rotation",
        "reward_tip_penalty",
        "reward_axis_tilt_penalty",
        "reward_axis_tilt_penalty_raw",
        "reward_axis_tilt_recovery",
        "reward_axis_tilt_growth",
        "reward_lateral_omega_penalty",
        "reward_contact_bonus",
        "reward_contact_bonus_raw",
        "reward_proximity",
        "reward_force_penalty",
        "reward_action_rate_penalty",
        "reward_rotation_before_support_gate",
        "reward_support_gating_effect",
        "reward_low_support_wobble",
        "reward_total",
    )
    reward_episode_means = {key: [] for key in reward_keys}
    normal_forces = [[], [], []]
    total_normal_forces = []
    conditioned_normal_forces = [[], [], []]
    conditioned_total_normal_forces = []
    collapse_episodes: list[dict] = []
    episode_lengths: list[int] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        info: dict = {}
        torque_values = []
        reward_values = {key: [] for key in reward_keys}
        ep_three_contact: list[int] = []
        ep_finger_contacts: list[np.ndarray] = []
        ep_gate_failures = 0
        ep_dtheta: list[float] = []
        ep_tip_errors: list[float] = []
        ep_axis_tilts: list[float] = []
        ep_n_contact: list[int] = []
        ep_omega_perp: list[float] = []
        ep_tilt_rad: list[float] = []
        ep_max_hold = 0.0
        while not (terminated or truncated):
            model_obs = obs
            if vecnorm is not None:
                model_obs = vecnorm.normalize_obs(
                    np.asarray(obs, dtype=np.float32).reshape(1, -1)
                )[0]
            action, _ = model.predict(model_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            count = int(info.get("contact_count", 0))
            step_forces = np.asarray(
                info.get("finger_contact_forces_n", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            for finger_index, force in enumerate(step_forces):
                normal_forces[finger_index].append(float(force))
            total_normal_forces.append(float(np.sum(step_forces)))
            if (
                float(info.get("axial_omega", 0.0)) > omega_success_threshold
                and count >= 2
            ):
                for finger_index, force in enumerate(step_forces):
                    conditioned_normal_forces[finger_index].append(float(force))
                conditioned_total_normal_forces.append(float(np.sum(step_forces)))
            contact_step_counts[count] += 1
            finger_bits = np.asarray(
                info.get("finger_contacts", [0, 0, 0]), dtype=np.int64
            )
            finger_contact_steps += finger_bits
            ep_three_contact.append(int(count == 3))
            ep_finger_contacts.append(finger_bits)
            ep_gate_failures += int(
                info.get("contact_gate_ready", False)
                and not info.get("contact_gate_satisfied", True)
            )
            ep_dtheta.append(float(info.get("dtheta", 0.0)))
            ep_tip_errors.append(float(info.get("tip_error_m", 0.0)))
            ep_axis_tilts.append(float(info.get("axis_tilt_deg", 0.0)))
            ep_n_contact.append(count)
            ep_omega_perp.append(
                float(info.get("omega_perp_norm", info.get("lateral_omega", 0.0)))
            )
            ep_tilt_rad.append(float(info.get("axis_tilt_rad", 0.0)))
            axial_omega_values.append(float(info.get("axial_omega", 0.0)))
            axial_slip_values.append(float(info.get("axial_slip_proxy", 0.0)))
            total_contact_steps += 1
            torque_values.append(float(info.get("stabilizer_torque_norm", 0.0)))
            ep_max_hold = max(ep_max_hold, float(info.get("omega_hold_seconds", 0.0)))
            for key in reward_keys:
                reward_values[key].append(float(info.get(key, 0.0)))

        episode_lengths.append(len(ep_n_contact))
        collapse_episodes.append(
            episode_support_collapse_metrics(
                ep_n_contact,
                ep_omega_perp,
                ep_tilt_rad,
                termination_reason=str(info.get("termination_reason", "none")),
                recontact_window=recontact_window_steps,
                collapse_lookback=collapse_lookback_steps,
            )
        )
        rotations.append(float(info.get("axis_rotation_deg", 0.0)))
        tip_errors.append(float(info.get("tip_error_m", 0.0)))
        contacts.append(float(info.get("contact_count", 0.0)))
        successes.append(bool(info.get("is_success", False)))
        legacy_omega_successes.append(bool(info.get("legacy_omega_success", False)))
        net_angle_successes.append(bool(info.get("net_angle_success", False)))
        omega_hold_satisfied.append(bool(info.get("omega_hold_satisfied", False)))
        max_omega_hold_seconds.append(ep_max_hold)
        window = max(int(contact_window_steps), 1)
        if len(ep_three_contact) >= window:
            rolling = np.convolve(
                np.asarray(ep_three_contact, dtype=np.float64),
                np.ones(window, dtype=np.float64),
                mode="valid",
            ) / window
            min_window_three_contact_occupancy.append(float(np.min(rolling)))
        else:
            min_window_three_contact_occupancy.append(float(np.mean(ep_three_contact)))
        gate_failure_steps.append(ep_gate_failures)
        ep_fingers = np.asarray(ep_finger_contacts, dtype=np.int64)
        for finger_index in range(3):
            longest = 0
            current = 0
            away_after_contact = False
            events = 0
            previous = int(ep_fingers[0, finger_index])
            for present in ep_fingers[:, finger_index]:
                current = 0 if present else current + 1
                longest = max(longest, current)
                present = int(present)
                if previous == 1 and present == 0:
                    away_after_contact = True
                elif previous == 0 and present == 1 and away_after_contact:
                    events += 1
                    away_after_contact = False
                previous = present
            finger_loss_longest_steps[finger_index].append(longest)
            finger_leave_return_events[finger_index].append(events)
        zero_contact = np.asarray(
            [int(np.sum(bits) == 0) for bits in ep_finger_contacts], dtype=np.int64
        )
        longest_zero = 0
        current_zero = 0
        for unsupported in zero_contact:
            current_zero = current_zero + 1 if unsupported else 0
            longest_zero = max(longest_zero, current_zero)
        complete_loss_longest_steps.append(longest_zero)
        below_two = np.asarray(
            [int(np.sum(bits) < 2) for bits in ep_finger_contacts], dtype=np.int64
        )
        longest_below_two = 0
        current_below_two = 0
        for under_supported in below_two:
            current_below_two = current_below_two + 1 if under_supported else 0
            longest_below_two = max(longest_below_two, current_below_two)
        below_two_contact_longest_steps.append(longest_below_two)
        cumulative_abs_rotation_deg.append(float(np.degrees(np.sum(np.abs(ep_dtheta)))))
        completed_net_rotation_cycles.append(
            int(np.floor(abs(float(info.get("axis_rotation_deg", 0.0))) / 360.0))
        )
        episode_tip_error_maxes.append(float(np.max(ep_tip_errors)))
        # Drop if terminated early for tip/rod failure (not time truncation).
        drops.append(bool(terminated))
        final_axis_tilts.append(float(info.get("axis_tilt_deg", 0.0)))
        episode_axis_tilt_maxes.append(
            float(np.max(ep_axis_tilts)) if ep_axis_tilts else 0.0
        )
        episode_torque_means.append(float(np.mean(torque_values)))
        episode_torque_maxes.append(float(np.max(torque_values)))
        termination_reasons[str(info.get("termination_reason", "none"))] += 1
        for key in reward_keys:
            reward_episode_means[key].append(float(np.mean(reward_values[key])))

    trace_manifest: list[dict] = []
    if trace_dir:
        trace_path = Path(trace_dir)
        seeds_to_trace = list(trace_seeds or [])
        if not seeds_to_trace:
            seeds_to_trace = [seed, seed + min(5, max(episodes - 1, 0))]
        for trace_seed in seeds_to_trace:
            trace_manifest.append(
                _rollout_diagnostic_trace(env, model, vecnorm, int(trace_seed), trace_path)
            )

    rotations_arr = np.asarray(rotations, dtype=np.float64)
    tip_arr = np.asarray(tip_errors, dtype=np.float64)
    contact_arr = np.asarray(contacts, dtype=np.float64)
    success_rate = float(np.mean(successes))
    drop_rate = float(np.mean(drops))

    metrics = {
        "model": model_path,
        "stage": stage,
        "episodes": episodes,
        "physics_mode": physics_mode,
        "reward_style": reward_style,
        "tip_connect_solref": tip_connect_solref,
        "tip_connect_enabled": tip_connect_enabled,
        "axis_stabilizer_scale": axis_stabilizer_scale,
        "axis_tilt_penalty_weight": axis_tilt_penalty_weight,
        "axis_tilt_recovery_scale": axis_tilt_recovery_scale,
        "axis_tilt_growth_scale": axis_tilt_growth_scale,
        "rotation_reward_scale": rotation_reward_scale,
        "contact_reward_mode": contact_reward_mode,
        "three_contact_reward": three_contact_reward,
        "rotation_requires_three_contacts": rotation_requires_three_contacts,
        "contact_support_termination_enabled": contact_support_termination_enabled,
        "contact_reward_scale": contact_reward_scale,
        "contact_window_steps": contact_window_steps,
        "contact_window_threshold": contact_window_threshold,
        "omega_success_threshold": omega_success_threshold,
        "omega_success_hold_seconds": omega_success_hold_seconds,
        "success_mode": success_mode,
        "net_angle_success_threshold_rad": net_angle_success_threshold_rad,
        "net_angle_success_threshold_deg": float(
            np.degrees(net_angle_success_threshold_rad)
        ),
        "rod_mass_scale": rod_mass_scale,
        "rod_friction_cap": rod_friction_cap,
        "contact_friction_scale": contact_friction_scale,
        "contact_friction_scaling_mode": contact_friction_scaling_mode,
        "scale_rod_joint_dynamics_from_s400": scale_rod_joint_dynamics_from_s400,
        "scale_tip_solref_with_mass": scale_tip_solref_with_mass,
        "contact_friction_vector": env.model.geom_friction[env.rod_geom].tolist(),
        "contact_friction_geom_vectors": {
            "rod": env.model.geom_friction[env.rod_geom].tolist(),
            **{
                f"tip{index}": env.model.geom_friction[geom_id].tolist()
                for index, geom_id in enumerate(env.tip_geom_ids)
            },
        },
        "effective_pair_friction_vectors": {
            f"rod_tip{index}": env._effective_pair_friction(geom_id)
            for index, geom_id in enumerate(env.tip_geom_ids)
        },
        "rod_dof_damping": env.model.dof_damping[env.rod_dof_adrs].tolist(),
        "rod_dof_armature": env.model.dof_armature[env.rod_dof_adrs].tolist(),
        "rod_dof_frictionloss": env.model.dof_frictionloss[
            env.rod_dof_adrs
        ].tolist(),
        "tilt_terminate_rad": tilt_terminate_rad,
        "tip_anchor": tip_anchor,
        "hand_model": hand_model,
        "hand_pose_config": env.hand_pose_config_path,
        "hand_pose_config_sha256": env.hand_pose_config_sha256,
        "hand_pose_config_content": env.hand_pose_config_content,
        "hand_grasp_config": env.hand_grasp_config_path,
        "hand_grasp_config_sha256": env.hand_grasp_config_sha256,
        "hand_grasp_config_content": env.hand_grasp_config_content,
        "obs_history_len": int(obs_history_len),
        "support_aware_reward_enabled": bool(support_aware_reward_enabled),
        "rotation_contact_scale_0": float(rotation_contact_scale_0),
        "rotation_contact_scale_1": float(rotation_contact_scale_1),
        "rotation_contact_scale_2plus": float(rotation_contact_scale_2plus),
        "low_support_wobble_scale": float(low_support_wobble_scale),
        "dexscrew_tip_penalty_scale": dexscrew_tip_penalty_scale,
        "vecnormalize": vecnormalize,
        "axis_rotation_deg_mean": float(rotations_arr.mean()),
        "axis_rotation_deg_std": float(rotations_arr.std()),
        "tip_error_m_mean": float(tip_arr.mean()),
        "tip_error_m_std": float(tip_arr.std()),
        "contact_count_mean": float(contact_arr.mean()),
        "contact_count_step_distribution": {
            str(count): float(contact_step_counts[count] / max(total_contact_steps, 1))
            for count in range(4)
        },
        "contact_fraction_at_least_one": float(
            1.0 - contact_step_counts[0] / max(total_contact_steps, 1)
        ),
        "contact_fraction_at_least_two": float(
            (contact_step_counts[2] + contact_step_counts[3])
            / max(total_contact_steps, 1)
        ),
        "finger_contact_step_fraction": (
            finger_contact_steps / max(total_contact_steps, 1)
        ).tolist(),
        "minimum_window_three_contact_occupancy": {
            "window_steps": int(contact_window_steps),
            "minimum_across_episodes": float(
                np.min(min_window_three_contact_occupancy)
            ),
            "mean_episode_minimum": float(
                np.mean(min_window_three_contact_occupancy)
            ),
        },
        "rolling_contact_gate_failure_steps": {
            "total": int(np.sum(gate_failure_steps)),
            "episodes_with_failure": int(np.sum(np.asarray(gate_failure_steps) > 0)),
        },
        "per_tip_contact_loss_longest": [
            {
                "mean_steps": float(np.mean(values)),
                "max_steps": int(np.max(values)),
                "mean_seconds": float(np.mean(values) * step_seconds),
                "max_seconds": float(np.max(values) * step_seconds),
            }
            for values in finger_loss_longest_steps
        ],
        "per_tip_leave_return_events": [
            {
                "mean": float(np.mean(values)),
                "minimum": int(np.min(values)),
                "maximum": int(np.max(values)),
                "total": int(np.sum(values)),
            }
            for values in finger_leave_return_events
        ],
        "complete_contact_loss_longest": {
            "mean_steps": float(np.mean(complete_loss_longest_steps)),
            "max_steps": int(np.max(complete_loss_longest_steps)),
            "mean_seconds": float(
                np.mean(complete_loss_longest_steps) * step_seconds
            ),
            "max_seconds": float(np.max(complete_loss_longest_steps) * step_seconds),
        },
        "below_two_contact_longest": {
            "mean_steps": float(np.mean(below_two_contact_longest_steps)),
            "max_steps": int(np.max(below_two_contact_longest_steps)),
            "mean_seconds": float(
                np.mean(below_two_contact_longest_steps) * step_seconds
            ),
            "max_seconds": float(
                np.max(below_two_contact_longest_steps) * step_seconds
            ),
        },
        "rotation_cycle_metrics": {
            "cumulative_absolute_rotation_deg_mean": float(
                np.mean(cumulative_abs_rotation_deg)
            ),
            "completed_net_cycles_mean": float(
                np.mean(completed_net_rotation_cycles)
            ),
            "episodes_with_at_least_two_net_cycles": int(
                np.sum(np.asarray(completed_net_rotation_cycles) >= 2)
            ),
        },
        "axial_omega_metrics": {
            "mean_rad_s": float(np.mean(axial_omega_values)),
            "p95_rad_s": float(np.percentile(axial_omega_values, 95)),
            "fraction_above_success_threshold": float(
                np.mean(
                    np.asarray(axial_omega_values) > float(omega_success_threshold)
                )
            ),
        },
        "axial_slip_proxy_metrics": {
            "mean_m_s": float(np.mean(axial_slip_values)),
            "median_m_s": float(np.median(axial_slip_values)),
            "p95_m_s": float(np.percentile(axial_slip_values, 95)),
            "max_m_s": float(np.max(axial_slip_values)),
            "definition": (
                "Absolute rod-axis component of mean contacting fingertip "
                "velocity minus rod center-of-mass velocity"
            ),
        },
        "tip_error_m_max": float(np.max(episode_tip_error_maxes)),
        "numerical_instability_episodes": int(
            termination_reasons.get("nonfinite_reward", 0)
            + termination_reasons.get("unstable", 0)
        ),
        "final_axis_tilt_deg_mean": float(np.mean(final_axis_tilts)),
        "axis_tilt_deg_max_mean": float(np.mean(episode_axis_tilt_maxes)),
        "axis_tilt_deg_max_max": float(np.max(episode_axis_tilt_maxes)),
        "omega_hold_satisfied_rate": float(np.mean(omega_hold_satisfied)),
        "omega_hold_seconds_max_mean": float(np.mean(max_omega_hold_seconds)),
        "stabilizer_torque_mean": float(np.mean(episode_torque_means)),
        "stabilizer_torque_max_mean": float(np.mean(episode_torque_maxes)),
        "termination_reasons": dict(termination_reasons),
        "reward_component_step_means": {
            key: float(np.mean(values)) for key, values in reward_episode_means.items()
        },
        "normal_contact_force_n": {
            "per_tip": [
                {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                }
                for values in normal_forces
            ],
            "total": {
                "mean": float(np.mean(total_normal_forces)),
                "median": float(np.median(total_normal_forces)),
                "p95": float(np.percentile(total_normal_forces, 95)),
            },
            "definition": "MuJoCo contact-frame normal force, not actuator effort",
        },
        "conditioned_positive_rotation_support_force_n": {
            "condition": (
                f"axial_omega > {omega_success_threshold:g} rad/s AND "
                "contact_count >= 2"
            ),
            "eligible_steps": len(conditioned_total_normal_forces),
            "eligible_step_fraction": float(
                len(conditioned_total_normal_forces) / max(1, total_contact_steps)
            ),
            "excluded_step_fraction": float(
                1.0 - len(conditioned_total_normal_forces) / max(1, total_contact_steps)
            ),
            "per_tip": [
                {
                    "mean": float(np.mean(values)) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                }
                for values in conditioned_normal_forces
            ],
            "total": {
                "mean": (
                    float(np.mean(conditioned_total_normal_forces))
                    if conditioned_total_normal_forces
                    else None
                ),
                "median": (
                    float(np.median(conditioned_total_normal_forces))
                    if conditioned_total_normal_forces
                    else None
                ),
                "p95": (
                    float(np.percentile(conditioned_total_normal_forces, 95))
                    if conditioned_total_normal_forces
                    else None
                ),
            },
            "reference_statistic": "total median",
            "definition": (
                "MuJoCo contact-frame normal force on genuine positive-rotation "
                "timesteps with at least two supporting fingertips; not actuator effort"
            ),
        },
        "success_rate": success_rate,
        "legacy_omega_success_rate": float(np.mean(legacy_omega_successes)),
        "net_angle_success_rate": float(np.mean(net_angle_successes)),
        "drop_rate": drop_rate,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_length_std": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "support_collapse": aggregate_support_collapse_metrics(
            collapse_episodes,
            recontact_window=recontact_window_steps,
            collapse_lookback=collapse_lookback_steps,
        ),
        "diagnostic_traces": trace_manifest,
        "passed": False,
    }

    if reward_style == "dexscrew":
        # Gate: sustained-ω success rate, tip, drop (angle is metric only).
        metrics["passed"] = bool(
            metrics["success_rate"] >= 0.5
            and metrics["tip_error_m_mean"] < 0.02
            and metrics["drop_rate"] <= 0.15
        )
    else:
        # Legacy gate: mean rotation > 180°, tip, drop.
        metrics["passed"] = bool(
            metrics["axis_rotation_deg_mean"] > 180.0
            and metrics["tip_error_m_mean"] < 0.02
            and metrics["drop_rate"] <= 0.15
        )
    env.close()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--tip-connect-solref", type=float, default=None)
    parser.add_argument("--tip-connect", dest="tip_connect_enabled", action="store_true")
    parser.add_argument("--no-tip-connect", dest="tip_connect_enabled", action="store_false")
    parser.set_defaults(tip_connect_enabled=None)
    parser.add_argument("--axis-stabilizer-scale", type=float, default=None)
    parser.add_argument("--axis-tilt-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--axis-tilt-recovery-scale",
        type=float,
        default=0.0,
        help=(
            "Tilt-recovery scale (default 0). DexScrew: one-sided "
            "scale*clip(prev-curr, 0, 0.05)*1[curr>0.05 rad]."
        ),
    )
    parser.add_argument(
        "--axis-tilt-growth-scale",
        type=float,
        default=0.0,
        help=(
            "DexScrew tilt-growth penalty scale (default 0). "
            "r_growth=-scale*clip(curr-prev,0,0.05)*w(curr) with "
            "w=clip((curr-0.05)/0.20, 0, 1). Experiment scale 50 is not default."
        ),
    )
    parser.add_argument("--rotation-reward-scale", type=float, default=16.0)
    parser.add_argument(
        "--contact-reward-mode",
        choices=["linear", "discrete", "gait_two_support"],
        default="linear",
    )
    parser.add_argument("--three-contact-reward", type=float, default=10.0)
    parser.add_argument("--contact-window-steps", type=int, default=0)
    parser.add_argument("--contact-window-threshold", type=float, default=0.0)
    parser.add_argument("--three-contact-required", action="store_true")
    parser.set_defaults(rotation_requires_three_contacts=True)
    parser.add_argument(
        "--rotation-requires-three-contacts",
        dest="rotation_requires_three_contacts",
        action="store_true",
    )
    parser.add_argument(
        "--no-rotation-requires-three-contacts",
        dest="rotation_requires_three_contacts",
        action="store_false",
    )
    parser.set_defaults(contact_support_termination_enabled=True)
    parser.add_argument(
        "--contact-support-termination",
        dest="contact_support_termination_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--no-contact-support-termination",
        dest="contact_support_termination_enabled",
        action="store_false",
    )
    parser.add_argument("--contact-reward-scale", type=float, default=1.0)
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--hand-model", choices=["allegro", "surrogate"], default="allegro")
    parser.add_argument("--hand-pose-config", type=str, default=None)
    parser.add_argument("--hand-grasp-config", type=str, default=None)
    parser.add_argument(
        "--obs-history-len",
        type=int,
        default=1,
        help="Must match training. 1 = 48-D, 4 = 192-D at the current 48-D frame.",
    )
    parser.add_argument(
        "--support-aware-reward",
        dest="support_aware_reward_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--no-support-aware-reward",
        dest="support_aware_reward_enabled",
        action="store_false",
    )
    parser.set_defaults(support_aware_reward_enabled=False)
    parser.add_argument("--rotation-contact-scale-0", type=float, default=0.0)
    parser.add_argument("--rotation-contact-scale-1", type=float, default=0.1)
    parser.add_argument("--rotation-contact-scale-2plus", type=float, default=1.0)
    parser.add_argument("--low-support-wobble-scale", type=float, default=0.5)
    parser.add_argument(
        "--recontact-window-steps",
        type=int,
        default=DEFAULT_RECONTACT_WINDOW,
        help="Steps allowed to recover n_contact>=2 after a support-loss event.",
    )
    parser.add_argument(
        "--collapse-lookback-steps",
        type=int,
        default=DEFAULT_COLLAPSE_LOOKBACK,
        help="Lookback window for axis_tilt preceded-by-support-loss.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="If set, save per-step CSV/JSON/PNG traces for --trace-seeds.",
    )
    parser.add_argument(
        "--trace-seeds",
        type=str,
        default="6,10000",
        help="Comma-separated env reset seeds for diagnostic traces.",
    )
    parser.add_argument("--reward-style", choices=["stage", "dexscrew"], default="stage")
    parser.add_argument("--privileged-obs", action="store_true")
    parser.add_argument("--omega-success-threshold", type=float, default=0.5)
    parser.add_argument("--omega-success-hold-seconds", type=float, default=10.0)
    parser.add_argument(
        "--success-mode",
        choices=["omega_hold", "net_angle"],
        default="omega_hold",
    )
    parser.add_argument(
        "--net-angle-success-threshold-rad",
        type=float,
        default=float(np.pi),
    )
    parser.add_argument(
        "--dexscrew-tilt-scale",
        type=float,
        default=None,
        help="Default: 1.0 for tip_connect+dexscrew, else 0.0.",
    )
    parser.add_argument("--rod-mass-scale", type=float, default=1.0)
    parser.add_argument("--rod-friction-cap", type=float, default=4.0)
    parser.add_argument("--contact-friction-scale", type=float, default=None)
    parser.add_argument(
        "--contact-friction-scaling-mode",
        choices=["sliding_only", "full_vector"],
        default="sliding_only",
    )
    parser.add_argument(
        "--scale-rod-joint-dynamics-from-s400",
        action="store_true",
    )
    parser.add_argument(
        "--scale-tip-solref-with-mass",
        dest="scale_tip_solref_with_mass",
        action="store_true",
    )
    parser.add_argument(
        "--no-scale-tip-solref-with-mass",
        dest="scale_tip_solref_with_mass",
        action="store_false",
    )
    parser.set_defaults(scale_tip_solref_with_mass=True)
    parser.add_argument("--tilt-terminate-rad", type=float, default=0.7)
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="top")
    parser.add_argument("--dexscrew-tip-penalty-scale", type=float, default=0.5)
    parser.add_argument("--dexscrew-tip-sigma", type=float, default=0.025)
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="VecNormalize .pkl from training (required for fair transfer eval).",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSON metrics path")
    args = parser.parse_args()

    metrics = evaluate(
        args.model,
        args.stage,
        args.episodes,
        args.seed,
        args.episode_seconds,
        args.tip_connect_solref,
        args.tip_connect_enabled,
        args.axis_stabilizer_scale,
        args.axis_tilt_penalty_weight,
        args.axis_tilt_recovery_scale,
        args.axis_tilt_growth_scale,
        args.rotation_reward_scale,
        args.contact_reward_mode,
        args.three_contact_reward,
        args.contact_window_steps,
        args.contact_window_threshold,
        three_contact_required=args.three_contact_required,
        rotation_requires_three_contacts=args.rotation_requires_three_contacts,
        contact_support_termination_enabled=args.contact_support_termination_enabled,
        contact_reward_scale=args.contact_reward_scale,
        physics_mode=args.physics,
        reward_style=args.reward_style,
        privileged_obs=args.privileged_obs,
        omega_success_threshold=args.omega_success_threshold,
        omega_success_hold_seconds=args.omega_success_hold_seconds,
        success_mode=args.success_mode,
        net_angle_success_threshold_rad=args.net_angle_success_threshold_rad,
        dexscrew_tilt_scale=args.dexscrew_tilt_scale,
        rod_mass_scale=args.rod_mass_scale,
        rod_friction_cap=args.rod_friction_cap,
        contact_friction_scale=args.contact_friction_scale,
        contact_friction_scaling_mode=args.contact_friction_scaling_mode,
        scale_rod_joint_dynamics_from_s400=args.scale_rod_joint_dynamics_from_s400,
        scale_tip_solref_with_mass=args.scale_tip_solref_with_mass,
        tilt_terminate_rad=args.tilt_terminate_rad,
        tip_anchor=args.tip_anchor,
        dexscrew_tip_penalty_scale=args.dexscrew_tip_penalty_scale,
        dexscrew_tip_sigma=args.dexscrew_tip_sigma,
        vecnormalize=args.vecnormalize,
        hand_model=args.hand_model,
        hand_pose_config=args.hand_pose_config,
        hand_grasp_config=args.hand_grasp_config,
        obs_history_len=args.obs_history_len,
        support_aware_reward_enabled=args.support_aware_reward_enabled,
        rotation_contact_scale_0=args.rotation_contact_scale_0,
        rotation_contact_scale_1=args.rotation_contact_scale_1,
        rotation_contact_scale_2plus=args.rotation_contact_scale_2plus,
        low_support_wobble_scale=args.low_support_wobble_scale,
        recontact_window_steps=args.recontact_window_steps,
        collapse_lookback_steps=args.collapse_lookback_steps,
        trace_dir=args.trace_dir,
        trace_seeds=[
            int(part) for part in str(args.trace_seeds).split(",") if part.strip()
        ]
        if args.trace_dir
        else None,
    )
    print(json.dumps(_jsonable(metrics), indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_jsonable(metrics), indent=2))

    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
