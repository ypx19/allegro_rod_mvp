#!/usr/bin/env python3
"""Bounded joint-space search for fingertip-to-rod geometric reachability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from allegro_rod_mvp import RodRotationEnv


def geom_distances(env: RodRotationEnv) -> np.ndarray:
    """Exact signed capsule-to-capsule distances; negative means penetration."""
    return np.asarray(
        [
            mujoco.mj_geomDistance(
                env.model,
                env.data,
                tip_geom,
                env.rod_geom,
                1.0,
                None,
            )
            for tip_geom in env.tip_geom_ids
        ],
        dtype=np.float64,
    )


def search(seed: int, samples: int) -> dict:
    env = RodRotationEnv(
        curriculum_stage=2,
        tip_connect_enabled=True,
        tip_connect_solref=0.10,
        axis_stabilizer_scale=0.10,
    )
    env.reset(seed=seed)
    rod_qpos = env.data.qpos[env.nu :].copy()
    joint_ranges = env.model.jnt_range[: env.nu]
    rng = np.random.default_rng(seed)

    initial_distances = geom_distances(env)
    initial_forces = env._touch()
    best_individual = [
        {"distance_m": float(initial_distances[i]), "qpos": env.data.qpos[: env.nu].tolist()}
        for i in range(3)
    ]
    best_simultaneous = {
        "max_distance_m": float(np.max(initial_distances)),
        "distances_m": initial_distances.tolist(),
        "qpos": env.data.qpos[: env.nu].tolist(),
    }
    reachable_counts = np.zeros(4, dtype=np.int64)

    for _ in range(samples):
        q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        env.data.qpos[: env.nu] = q
        env.data.qpos[env.nu :] = rod_qpos
        env.data.qvel[:] = 0.0
        env.data.ctrl[:] = q
        mujoco.mj_forward(env.model, env.data)
        distances = geom_distances(env)
        contact_count = int(np.sum(distances <= 0.0))
        reachable_counts[contact_count] += 1
        for i in range(3):
            if distances[i] < best_individual[i]["distance_m"]:
                best_individual[i] = {
                    "distance_m": float(distances[i]),
                    "qpos": q.tolist(),
                }
        max_distance = float(np.max(distances))
        if max_distance < best_simultaneous["max_distance_m"]:
            best_simultaneous = {
                "max_distance_m": max_distance,
                "distances_m": distances.tolist(),
                "qpos": q.tolist(),
            }

    # Replay the closest simultaneous configuration through forward dynamics.
    best_q = np.asarray(best_simultaneous["qpos"], dtype=np.float64)
    env.data.qpos[: env.nu] = best_q
    env.data.qpos[env.nu :] = rod_qpos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = best_q
    mujoco.mj_forward(env.model, env.data)
    static_distances = geom_distances(env)
    for _ in range(100):
        env._apply_axis_stabilizer()
        mujoco.mj_step(env.model, env.data)
    settled_forces = env._touch()
    settled_distances = geom_distances(env)

    result = {
        "seed": seed,
        "samples": samples,
        "initial_distances_m": initial_distances.tolist(),
        "initial_forces_n": initial_forces.tolist(),
        "best_individual": best_individual,
        "best_simultaneous": best_simultaneous,
        "static_replay_distances_m": static_distances.tolist(),
        "settled_distances_m": settled_distances.tolist(),
        "settled_forces_n": settled_forces.tolist(),
        "settled_contact_count": int(np.sum(settled_forces > 0.05)),
        "sample_contact_count_histogram": {
            str(count): int(reachable_counts[count]) for count in range(4)
        },
    }
    env.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [search(seed, args.samples) for seed in range(args.seeds)]
    (args.out_dir / "reachability.json").write_text(json.dumps(results, indent=2))
    with (args.out_dir / "metrics.csv").open("w", newline="") as handle:
        fieldnames = [
            "seed",
            "samples",
            "finger0_min_distance_m",
            "finger1_min_distance_m",
            "finger2_min_distance_m",
            "best_simultaneous_max_distance_m",
            "best_simultaneous_finger0_distance_m",
            "best_simultaneous_finger1_distance_m",
            "best_simultaneous_finger2_distance_m",
            "settled_contact_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "seed": result["seed"],
                    "samples": result["samples"],
                    "finger0_min_distance_m": result["best_individual"][0]["distance_m"],
                    "finger1_min_distance_m": result["best_individual"][1]["distance_m"],
                    "finger2_min_distance_m": result["best_individual"][2]["distance_m"],
                    "best_simultaneous_max_distance_m": result["best_simultaneous"][
                        "max_distance_m"
                    ],
                    "best_simultaneous_finger0_distance_m": result["best_simultaneous"][
                        "distances_m"
                    ][0],
                    "best_simultaneous_finger1_distance_m": result["best_simultaneous"][
                        "distances_m"
                    ][1],
                    "best_simultaneous_finger2_distance_m": result["best_simultaneous"][
                        "distances_m"
                    ][2],
                    "settled_contact_count": result["settled_contact_count"],
                }
            )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
