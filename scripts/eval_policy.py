#!/usr/bin/env python3
"""Headless policy evaluation with stage success gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

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
) -> dict:
    env = RodRotationEnv(
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
    )
    model = PPO.load(model_path, device="cpu")

    rotations = []
    tip_errors = []
    contacts = []
    contact_step_counts: Counter[int] = Counter()
    finger_contact_steps = np.zeros(3, dtype=np.int64)
    total_contact_steps = 0
    successes = []
    drops = []
    final_axis_tilts = []
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
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            count = int(info.get("contact_count", 0))
            contact_step_counts[count] += 1
            finger_contact_steps += np.asarray(info.get("finger_contacts", [0, 0, 0]), dtype=np.int64)
            total_contact_steps += 1
            torque_values.append(float(info.get("stabilizer_torque_norm", 0.0)))
            for key in reward_keys:
                reward_values[key].append(float(info.get(key, 0.0)))

        rotations.append(float(info.get("axis_rotation_deg", 0.0)))
        tip_errors.append(float(info.get("tip_error_m", 0.0)))
        contacts.append(float(info.get("contact_count", 0.0)))
        successes.append(bool(info.get("is_success", False)))
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
        "tip_connect_solref": tip_connect_solref,
        "tip_connect_enabled": tip_connect_enabled,
        "axis_stabilizer_scale": axis_stabilizer_scale,
        "axis_tilt_penalty_weight": axis_tilt_penalty_weight,
        "axis_tilt_recovery_scale": axis_tilt_recovery_scale,
        "rotation_reward_scale": rotation_reward_scale,
        "contact_reward_mode": contact_reward_mode,
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

    # Gate: mean rotation > 180°, mean tip error < 0.02 m, drop rate near 0.
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
    parser.add_argument("--episode-seconds", type=float, default=12.0)
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
    )
    print(json.dumps(metrics, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))

    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
