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


def search(
    seed: int,
    samples: int,
    *,
    hand_model: str = "allegro",
    physics_mode: str = "tip_connect",
    tip_anchor: str = "bottom",
) -> dict:
    env = RodRotationEnv(
        hand_model=hand_model,
        physics_mode=physics_mode,
        tip_anchor=tip_anchor,
        curriculum_stage=2,
        tip_connect_enabled=True if physics_mode == "tip_connect" else None,
        tip_connect_solref=0.10,
        axis_stabilizer_scale=1.0 if physics_mode == "tip_connect" else 0.0,
    )
    env.reset(seed=seed)
    settled_qpos = env.data.qpos.copy()
    joint_ranges = env.model.jnt_range[env.hand_joint_ids]
    rng = np.random.default_rng(seed)

    initial_distances = geom_distances(env)
    initial_forces = env._touch()
    initial_qpos = env.data.qpos[env.hand_qpos_adr].copy()
    best_individual = [
        {
            "distance_m": float(initial_distances[i]),
            "qpos": initial_qpos.tolist(),
        }
        for i in range(3)
    ]
    best_simultaneous = {
        "max_distance_m": float(np.max(initial_distances)),
        "distances_m": initial_distances.tolist(),
        "qpos": initial_qpos.tolist(),
    }
    best_shallow_three_contact: dict | None = None
    three_contact_candidates: list[dict] = []
    reachable_counts = np.zeros(4, dtype=np.int64)

    for _ in range(samples):
        q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        env.data.qpos[:] = settled_qpos
        env.data.qpos[env.hand_qpos_adr] = q
        env.data.qvel[:] = 0.0
        env.data.ctrl[:] = q
        mujoco.mj_forward(env.model, env.data)
        distances = geom_distances(env)
        contact_count = int(np.sum(distances <= 0.0))
        reachable_counts[contact_count] += 1
        if contact_count == 3:
            # Prefer a physically plausible just-touching grasp over deep interpenetration.
            target_penetration = -0.001
            shallow_score = float(np.sum((distances - target_penetration) ** 2))
            if (
                best_shallow_three_contact is None
                or shallow_score < best_shallow_three_contact["score"]
            ):
                best_shallow_three_contact = {
                    "score": shallow_score,
                    "distances_m": distances.tolist(),
                    "qpos": q.tolist(),
                }
            three_contact_candidates.append(
                {"distances_m": distances.tolist(), "qpos": q.tolist()}
            )
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

    # The post-reset state is already dynamically settled by the environment;
    # retain it as the baseline instead of replaying its measured qpos with a
    # different control target.
    initial_count = int(np.sum(initial_forces > 0.05))
    initial_score = (
        initial_count,
        float(np.min(initial_forces)),
        float(np.sum(np.minimum(initial_forces, 20.0))),
    )
    best_dynamic: dict | None = {
        "score": list(initial_score),
        "qpos": initial_qpos.tolist(),
        "static_distances_m": initial_distances.tolist(),
        "settled_distances_m": initial_distances.tolist(),
        "settled_forces_n": initial_forces.tolist(),
        "settled_contact_count": initial_count,
        "source": "environment_reset",
    }
    # Replay random geometric three-contact candidates; if none were sampled,
    # replay the geometric minimax state as a diagnostic only.
    replay_candidates = three_contact_candidates or [best_simultaneous]
    for candidate in replay_candidates:
        q = np.asarray(candidate["qpos"], dtype=np.float64)
        env.data.qpos[:] = settled_qpos
        env.data.qpos[env.hand_qpos_adr] = q
        env.data.qvel[:] = 0.0
        env.data.ctrl[:] = q
        mujoco.mj_forward(env.model, env.data)
        static_distances = geom_distances(env)
        for _ in range(100):
            env._apply_axis_stabilizer()
            mujoco.mj_step(env.model, env.data)
        settled_forces = env._touch()
        settled_distances = geom_distances(env)
        settled_count = int(np.sum(settled_forces > 0.05))
        score = (
            settled_count,
            float(np.min(settled_forces)),
            float(np.sum(np.minimum(settled_forces, 20.0))),
        )
        if best_dynamic is None or score > tuple(best_dynamic["score"]):
            best_dynamic = {
                "score": list(score),
                "qpos": q.tolist(),
                "static_distances_m": static_distances.tolist(),
                "settled_distances_m": settled_distances.tolist(),
                "settled_forces_n": settled_forces.tolist(),
                "settled_contact_count": settled_count,
                "source": "random_search_replay",
            }
    assert best_dynamic is not None

    result = {
        "seed": seed,
        "samples": samples,
        "hand_model": hand_model,
        "physics_mode": physics_mode,
        "tip_anchor": tip_anchor,
        "initial_distances_m": initial_distances.tolist(),
        "initial_forces_n": initial_forces.tolist(),
        "best_individual": best_individual,
        "best_simultaneous": best_simultaneous,
        "best_shallow_three_contact": best_shallow_three_contact,
        "dynamic_candidates_replayed": len(replay_candidates),
        "best_dynamic_replay": best_dynamic,
        "static_replay_distances_m": best_dynamic["static_distances_m"],
        "settled_distances_m": best_dynamic["settled_distances_m"],
        "settled_forces_n": best_dynamic["settled_forces_n"],
        "settled_contact_count": best_dynamic["settled_contact_count"],
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
    parser.add_argument("--hand-model", choices=["allegro", "surrogate"], default="allegro")
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        search(
            seed,
            args.samples,
            hand_model=args.hand_model,
            physics_mode=args.physics,
            tip_anchor=args.tip_anchor,
        )
        for seed in range(args.seeds)
    ]
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
