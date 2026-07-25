#!/usr/bin/env python3
"""Expand a PPO MLP policy while preserving its initial deterministic function."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from allegro_rod_mvp import RodRotationEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--tip-connect-solref", type=float, default=0.10)
    parser.add_argument("--axis-stabilizer-scale", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def make_env():
        return Monitor(
            RodRotationEnv(
                curriculum_stage=args.stage,
                tip_connect_solref=args.tip_connect_solref,
                axis_stabilizer_scale=args.axis_stabilizer_scale,
            )
        )

    env = DummyVecEnv([make_env])
    source = PPO.load(args.source, device="cpu")
    expanded = PPO(
        "MlpPolicy",
        env,
        learning_rate=source.learning_rate,
        n_steps=source.n_steps,
        batch_size=source.batch_size,
        n_epochs=source.n_epochs,
        gamma=source.gamma,
        gae_lambda=source.gae_lambda,
        clip_range=source.clip_range,
        ent_coef=source.ent_coef,
        vf_coef=source.vf_coef,
        max_grad_norm=source.max_grad_norm,
        policy_kwargs={
            "net_arch": {
                "pi": [args.width, args.width],
                "vf": [args.width, args.width],
            }
        },
        seed=args.seed,
        device="cpu",
        verbose=0,
    )

    old_state = source.policy.state_dict()
    new_state = expanded.policy.state_dict()
    for name, destination in new_state.items():
        if name not in old_state:
            continue
        source_tensor = old_state[name]
        if destination.ndim != source_tensor.ndim:
            raise ValueError(f"rank mismatch for {name}: {source_tensor.shape} -> {destination.shape}")
        destination.zero_()
        slices = tuple(slice(0, size) for size in source_tensor.shape)
        destination[slices].copy_(source_tensor)
    expanded.policy.load_state_dict(new_state)

    observations = []
    raw_env = env.envs[0].unwrapped
    for seed in range(10):
        obs, _ = raw_env.reset(seed=seed)
        observations.append(obs)
    observations_array = np.asarray(observations, dtype=np.float32)
    old_actions, _ = source.predict(observations_array, deterministic=True)
    new_actions, _ = expanded.predict(observations_array, deterministic=True)
    max_action_error = float(np.max(np.abs(old_actions - new_actions)))
    if max_action_error > 1e-6:
        raise RuntimeError(f"expanded policy changed initial actions: max error={max_action_error}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    expanded.save(out)
    old_parameters = sum(parameter.numel() for parameter in source.policy.parameters())
    new_parameters = sum(parameter.numel() for parameter in expanded.policy.parameters())
    print(f"source_parameters={old_parameters}")
    print(f"expanded_parameters={new_parameters}")
    print(f"max_deterministic_action_error={max_action_error:.3e}")
    print(f"saved={out.with_suffix('.zip')}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
