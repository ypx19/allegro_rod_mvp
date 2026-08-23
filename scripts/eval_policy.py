#!/usr/bin/env python3
"""Headless policy evaluation with stage / DexScrew success gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv


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
    rotation_reward_scale: float = 16.0,
    contact_reward_mode: str = "linear",
    three_contact_reward: float = 10.0,
    contact_window_steps: int = 0,
    contact_window_threshold: float = 0.0,
    three_contact_required: bool = False,
    physics_mode: str = "tip_connect",
    reward_style: str = "stage",
    privileged_obs: bool = False,
    omega_success_threshold: float = 0.5,
    omega_success_hold_seconds: float = 10.0,
    dexscrew_tilt_scale: float | None = None,
    rod_mass_scale: float = 1.0,
    rod_friction_cap: float = 4.0,
    tilt_terminate_rad: float = 0.7,
    tip_anchor: str = "top",
    dexscrew_tip_penalty_scale: float = 0.5,
    dexscrew_tip_sigma: float = 0.025,
    vecnormalize: str | None = None,
    hand_model: str = "allegro",
    hand_pose_config: str | None = None,
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
            rotation_reward_scale=rotation_reward_scale,
            contact_reward_mode=contact_reward_mode,
            three_contact_reward=three_contact_reward,
            contact_window_steps=contact_window_steps,
            contact_window_threshold=contact_window_threshold,
            three_contact_required=three_contact_required,
            physics_mode=physics_mode,
            reward_style=reward_style,
            privileged_obs=privileged_obs,
            omega_success_threshold=omega_success_threshold,
            omega_success_hold_seconds=omega_success_hold_seconds,
            dexscrew_tilt_scale=dexscrew_tilt_scale,
            dexscrew_tip_penalty_scale=dexscrew_tip_penalty_scale,
            dexscrew_tip_sigma=dexscrew_tip_sigma,
            rod_mass_scale=rod_mass_scale,
            rod_friction_cap=rod_friction_cap,
            tilt_terminate_rad=tilt_terminate_rad,
            tip_anchor=tip_anchor,
            hand_model=hand_model,
            hand_pose_config=hand_pose_config,
        )

    env = _make_env()
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
    drops = []
    final_axis_tilts = []
    omega_hold_satisfied = []
    max_omega_hold_seconds = []
    episode_torque_means = []
    episode_torque_maxes = []
    termination_reasons: Counter[str] = Counter()
    reward_keys = (
        "reward_rotation",
        "reward_tip_penalty",
        "reward_axis_tilt_penalty",
        "reward_axis_tilt_penalty_raw",
        "reward_axis_tilt_recovery",
        "reward_lateral_omega_penalty",
        "reward_contact_bonus",
        "reward_proximity",
        "reward_force_penalty",
        "reward_action_rate_penalty",
    )
    reward_episode_means = {key: [] for key in reward_keys}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        info: dict = {}
        torque_values = []
        reward_values = {key: [] for key in reward_keys}
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
            contact_step_counts[count] += 1
            finger_contact_steps += np.asarray(info.get("finger_contacts", [0, 0, 0]), dtype=np.int64)
            total_contact_steps += 1
            torque_values.append(float(info.get("stabilizer_torque_norm", 0.0)))
            ep_max_hold = max(ep_max_hold, float(info.get("omega_hold_seconds", 0.0)))
            for key in reward_keys:
                reward_values[key].append(float(info.get(key, 0.0)))

        rotations.append(float(info.get("axis_rotation_deg", 0.0)))
        tip_errors.append(float(info.get("tip_error_m", 0.0)))
        contacts.append(float(info.get("contact_count", 0.0)))
        successes.append(bool(info.get("is_success", False)))
        omega_hold_satisfied.append(bool(info.get("omega_hold_satisfied", False)))
        max_omega_hold_seconds.append(ep_max_hold)
        # Drop if terminated early for tip/rod failure (not time truncation).
        drops.append(bool(terminated))
        final_axis_tilts.append(float(info.get("axis_tilt_deg", 0.0)))
        episode_torque_means.append(float(np.mean(torque_values)))
        episode_torque_maxes.append(float(np.max(torque_values)))
        termination_reasons[str(info.get("termination_reason", "none"))] += 1
        for key in reward_keys:
            reward_episode_means[key].append(float(np.mean(reward_values[key])))

    env.close()

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
        "rotation_reward_scale": rotation_reward_scale,
        "contact_reward_mode": contact_reward_mode,
        "three_contact_reward": three_contact_reward,
        "contact_window_steps": contact_window_steps,
        "contact_window_threshold": contact_window_threshold,
        "omega_success_threshold": omega_success_threshold,
        "omega_success_hold_seconds": omega_success_hold_seconds,
        "rod_mass_scale": rod_mass_scale,
        "rod_friction_cap": rod_friction_cap,
        "tilt_terminate_rad": tilt_terminate_rad,
        "tip_anchor": tip_anchor,
        "hand_model": hand_model,
        "hand_pose_config": env.hand_pose_config_path,
        "hand_pose_config_sha256": env.hand_pose_config_sha256,
        "hand_pose_config_content": env.hand_pose_config_content,
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
        "finger_contact_step_fraction": (
            finger_contact_steps / max(total_contact_steps, 1)
        ).tolist(),
        "final_axis_tilt_deg_mean": float(np.mean(final_axis_tilts)),
        "omega_hold_satisfied_rate": float(np.mean(omega_hold_satisfied)),
        "omega_hold_seconds_max_mean": float(np.mean(max_omega_hold_seconds)),
        "stabilizer_torque_mean": float(np.mean(episode_torque_means)),
        "stabilizer_torque_max_mean": float(np.mean(episode_torque_maxes)),
        "termination_reasons": dict(termination_reasons),
        "reward_component_step_means": {
            key: float(np.mean(values)) for key, values in reward_episode_means.items()
        },
        "success_rate": success_rate,
        "drop_rate": drop_rate,
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
    parser.add_argument("--axis-tilt-recovery-scale", type=float, default=0.0)
    parser.add_argument("--rotation-reward-scale", type=float, default=16.0)
    parser.add_argument(
        "--contact-reward-mode",
        choices=["linear", "discrete"],
        default="linear",
    )
    parser.add_argument("--three-contact-reward", type=float, default=10.0)
    parser.add_argument("--contact-window-steps", type=int, default=0)
    parser.add_argument("--contact-window-threshold", type=float, default=0.0)
    parser.add_argument("--three-contact-required", action="store_true")
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--hand-model", choices=["allegro", "surrogate"], default="allegro")
    parser.add_argument("--hand-pose-config", type=str, default=None)
    parser.add_argument("--reward-style", choices=["stage", "dexscrew"], default="stage")
    parser.add_argument("--privileged-obs", action="store_true")
    parser.add_argument("--omega-success-threshold", type=float, default=0.5)
    parser.add_argument("--omega-success-hold-seconds", type=float, default=10.0)
    parser.add_argument(
        "--dexscrew-tilt-scale",
        type=float,
        default=None,
        help="Default: 1.0 for tip_connect+dexscrew, else 0.0.",
    )
    parser.add_argument("--rod-mass-scale", type=float, default=1.0)
    parser.add_argument("--rod-friction-cap", type=float, default=4.0)
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
        args.rotation_reward_scale,
        args.contact_reward_mode,
        args.three_contact_reward,
        args.contact_window_steps,
        args.contact_window_threshold,
        three_contact_required=args.three_contact_required,
        physics_mode=args.physics,
        reward_style=args.reward_style,
        privileged_obs=args.privileged_obs,
        omega_success_threshold=args.omega_success_threshold,
        omega_success_hold_seconds=args.omega_success_hold_seconds,
        dexscrew_tilt_scale=args.dexscrew_tilt_scale,
        rod_mass_scale=args.rod_mass_scale,
        rod_friction_cap=args.rod_friction_cap,
        tilt_terminate_rad=args.tilt_terminate_rad,
        tip_anchor=args.tip_anchor,
        dexscrew_tip_penalty_scale=args.dexscrew_tip_penalty_scale,
        dexscrew_tip_sigma=args.dexscrew_tip_sigma,
        vecnormalize=args.vecnormalize,
        hand_model=args.hand_model,
        hand_pose_config=args.hand_pose_config,
    )
    print(json.dumps(metrics, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))

    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
