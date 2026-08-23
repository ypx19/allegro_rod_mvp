#!/usr/bin/env python3
"""Export one state-driven MuJoCo policy video per curriculum stage."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv


def _environment_kwargs(record: dict[str, Any], episode_seconds: float) -> dict[str, Any]:
    """Reconstruct the environment used by the Allegro bottom-tip driver."""
    stage = record["stage"]
    metrics = record["metrics"]
    physics = stage["physics"]
    return {
        "curriculum_stage": int(metrics.get("stage", 0)),
        "episode_seconds": float(episode_seconds),
        "tip_connect_solref": stage["tip_solref"],
        "tip_connect_enabled": physics == "tip_connect",
        "axis_stabilizer_scale": float(stage["stabilizer"]),
        "axis_tilt_penalty_weight": float(metrics["axis_tilt_penalty_weight"]),
        "axis_tilt_recovery_scale": float(metrics["axis_tilt_recovery_scale"]),
        "rotation_reward_scale": float(metrics["rotation_reward_scale"]),
        "contact_reward_mode": metrics["contact_reward_mode"],
        "three_contact_reward": float(metrics["three_contact_reward"]),
        "contact_window_steps": int(metrics["contact_window_steps"]),
        "contact_window_threshold": float(metrics["contact_window_threshold"]),
        # The curriculum driver always trained and evaluated with this flag.
        "three_contact_required": True,
        "physics_mode": physics,
        "reward_style": metrics["reward_style"],
        "omega_success_threshold": float(metrics["omega_success_threshold"]),
        "omega_success_hold_seconds": float(metrics["omega_success_hold_seconds"]),
        "dexscrew_tilt_scale": 0.0 if physics == "revolute" else 1.0,
        "dexscrew_tip_penalty_scale": float(metrics["dexscrew_tip_penalty_scale"]),
        "rod_mass_scale": float(stage["mass_scale"]),
        "rod_friction_cap": float(metrics["rod_friction_cap"]),
        "tilt_terminate_rad": float(metrics["tilt_terminate_rad"]),
        "tip_anchor": metrics["tip_anchor"],
        "hand_model": metrics["hand_model"],
        "hand_pose_config": metrics.get("hand_pose_config"),
    }


def _normalized_observation(
    observation: np.ndarray, vecnormalize: VecNormalize
) -> np.ndarray:
    return vecnormalize.normalize_obs(
        np.asarray(observation, dtype=np.float32).reshape(1, -1)
    )[0]


def _load_policy(
    checkpoint: Path,
    vecnormalize_path: Path,
    environment_kwargs: dict[str, Any],
) -> tuple[PPO, VecNormalize]:
    model = PPO.load(checkpoint, device="cpu")
    dummy = DummyVecEnv(
        [lambda: RodRotationEnv(render_mode=None, **environment_kwargs)]
    )
    vecnormalize = VecNormalize.load(vecnormalize_path, dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return model, vecnormalize


def _rollout_metrics(
    model: PPO,
    vecnormalize: VecNormalize,
    environment_kwargs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    env = RodRotationEnv(render_mode=None, **environment_kwargs)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info: dict[str, Any] = {}
    contact_counts: list[int] = []
    episode_return = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(
            _normalized_observation(observation, vecnormalize), deterministic=True
        )
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        contact_counts.append(int(info.get("contact_count", 0)))
    env.close()
    return _final_metrics(
        seed, info, terminated, truncated, contact_counts, episode_return
    )


def _final_metrics(
    seed: int,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    contact_counts: list[int],
    episode_return: float,
) -> dict[str, Any]:
    counts = np.asarray(contact_counts, dtype=np.int64)
    return {
        "seed": seed,
        "num_steps": len(contact_counts),
        "duration_seconds": len(contact_counts) / 25.0,
        "episode_return": episode_return,
        "axis_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
        "tip_error_m": float(info.get("tip_error_m", 0.0)),
        "contact_count": int(info.get("contact_count", 0)),
        "contact_count_mean": float(counts.mean()) if len(counts) else 0.0,
        "three_contact_fraction": float(np.mean(counts == 3)) if len(counts) else 0.0,
        "axis_tilt_deg": float(info.get("axis_tilt_deg", 0.0)),
        "is_success": bool(info.get("is_success", False)),
        "omega_hold_satisfied": bool(info.get("omega_hold_satisfied", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(info.get("termination_reason", "none")),
    }


def _choose_seed(
    model: PPO,
    vecnormalize: VecNormalize,
    environment_kwargs: dict[str, Any],
    preferred_seed: int,
    alternate_seed_count: int,
    immediate_failure_seconds: float,
) -> tuple[int, list[dict[str, Any]], str]:
    preferred = _rollout_metrics(
        model, vecnormalize, environment_kwargs, preferred_seed
    )
    trials = [preferred]
    if preferred["duration_seconds"] >= immediate_failure_seconds:
        return preferred_seed, trials, "preferred seed completed several seconds"

    for seed in range(preferred_seed + 1, preferred_seed + 1 + alternate_seed_count):
        trials.append(_rollout_metrics(model, vecnormalize, environment_kwargs, seed))
    longest = max(trials, key=lambda item: item["num_steps"])
    if (
        longest["duration_seconds"] >= immediate_failure_seconds
        and longest["num_steps"] >= 1.25 * preferred["num_steps"]
    ):
        return (
            int(longest["seed"]),
            trials,
            "preferred seed failed immediately; selected longest fixed alternate",
        )
    return (
        preferred_seed,
        trials,
        "all checked seeds failed similarly; retained representative preferred seed",
    )


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        return ImageFont.load_default()


def _overlay(
    frame: np.ndarray,
    *,
    stage_name: str,
    checkpoint_name: str,
    seed: int,
    step: int,
    info: dict[str, Any],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    lines = [
        f"{stage_name} | seed {seed} | step {step} | {checkpoint_name}",
        (
            f"rotation {float(info.get('axis_rotation_deg', 0.0)):7.2f} deg | "
            f"tip {1000.0 * float(info.get('tip_error_m', 0.0)):6.2f} mm | "
            f"contacts {int(info.get('contact_count', 0))}/3 | "
            f"tilt {float(info.get('axis_tilt_deg', 0.0)):5.2f} deg"
        ),
        (
            f"success {bool(info.get('is_success', False))} | "
            f"termination {str(info.get('termination_reason', 'none'))}"
        ),
    ]
    font = _font()
    line_height = 22
    draw.rectangle((5, 5, 635, 5 + line_height * len(lines) + 8), fill=(0, 0, 0, 180))
    for index, line in enumerate(lines):
        draw.text((12, 10 + index * line_height), line, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def _render_video(
    output: Path,
    model: PPO,
    vecnormalize: VecNormalize,
    environment_kwargs: dict[str, Any],
    *,
    stage_name: str,
    checkpoint_name: str,
    seed: int,
    fps: int,
) -> dict[str, Any]:
    env = RodRotationEnv(render_mode="rgb_array", **environment_kwargs)
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    info: dict[str, Any] = {}
    contact_counts: list[int] = []
    episode_return = 0.0
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        initial = env.render()
        if initial is not None:
            writer.append_data(
                _overlay(
                    np.asarray(initial),
                    stage_name=stage_name,
                    checkpoint_name=checkpoint_name,
                    seed=seed,
                    step=0,
                    info=info,
                )
            )
        while not (terminated or truncated):
            action, _ = model.predict(
                _normalized_observation(observation, vecnormalize),
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            contact_counts.append(int(info.get("contact_count", 0)))
            frame = env.render()
            if frame is not None:
                writer.append_data(
                    _overlay(
                        np.asarray(frame),
                        stage_name=stage_name,
                        checkpoint_name=checkpoint_name,
                        seed=seed,
                        step=len(contact_counts),
                        info=info,
                    )
                )
    finally:
        writer.close()
        env.close()
    return _final_metrics(
        seed, info, terminated, truncated, contact_counts, episode_return
    )


def _write_index(output_dir: Path, state_path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Curriculum Stage Videos",
        "",
        f"- Source state: `{state_path}`",
        "- Policy: deterministic PPO",
        "- Preferred evaluation seed: 0",
        "- Episode limit: 20 seconds",
        "- Each video uses the selected checkpoint and selected VecNormalize file recorded in state.json.",
        "",
    ]
    for entry in entries:
        metrics = entry["final_metrics"]
        lines.extend(
            [
                f"## {entry['stage']}",
                f"- Video: `{entry['video']}`",
                f"- Checkpoint: `{entry['checkpoint']}`",
                f"- VecNormalize: `{entry['vecnormalize']}`",
                f"- Seed: {metrics['seed']} ({entry['seed_selection_reason']})",
                (
                    f"- Final: rotation {metrics['axis_rotation_deg']:.2f}°, "
                    f"tip error {metrics['tip_error_m'] * 1000.0:.2f} mm, "
                    f"contacts {metrics['contact_count']}/3 "
                    f"(mean {metrics['contact_count_mean']:.3f}), "
                    f"tilt {metrics['axis_tilt_deg']:.2f}°, "
                    f"success {metrics['is_success']}, "
                    f"termination `{metrics['termination_reason']}`, "
                    f"duration {metrics['duration_seconds']:.2f} s"
                ),
                "",
            ]
        )
    (output_dir / "INDEX.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--alternate-seeds", type=int, default=9)
    parser.add_argument("--immediate-failure-seconds", type=float, default=3.0)
    args = parser.parse_args()

    state_path = args.state.resolve()
    state = json.loads(state_path.read_text())
    completed = state.get("completed", [])
    if len(completed) != 8:
        raise ValueError(f"Expected 8 completed stages, found {len(completed)}")
    output_dir = args.out_dir or (
        state_path.parent / f"stage_videos_{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, Any]] = []
    for index, record in enumerate(completed):
        stage_name = str(record["stage"]["name"])
        checkpoint = Path(record["selected_checkpoint"]).resolve()
        vecnormalize_path = Path(record["selected_vecnormalize"]).resolve()
        if not checkpoint.is_file() or not vecnormalize_path.is_file():
            raise FileNotFoundError(
                f"{stage_name}: missing checkpoint or VecNormalize file"
            )
        environment_kwargs = _environment_kwargs(record, args.episode_seconds)
        model, vecnormalize = _load_policy(
            checkpoint, vecnormalize_path, environment_kwargs
        )
        selected_seed, seed_trials, reason = _choose_seed(
            model,
            vecnormalize,
            environment_kwargs,
            args.seed,
            args.alternate_seeds,
            args.immediate_failure_seconds,
        )
        checkpoint_tag = checkpoint.stem.replace("final_model", "final")
        video = output_dir / (
            f"{index:02d}_{stage_name}_seed{selected_seed}_{checkpoint_tag}.mp4"
        )
        final_metrics = _render_video(
            video,
            model,
            vecnormalize,
            environment_kwargs,
            stage_name=stage_name,
            checkpoint_name=checkpoint.name,
            seed=selected_seed,
            fps=args.fps,
        )
        vecnormalize.close()
        if not video.is_file() or video.stat().st_size == 0:
            raise RuntimeError(f"Video was not written: {video}")
        entries.append(
            {
                "stage": stage_name,
                "run_id": record["run_id"],
                "video": str(video.resolve()),
                "checkpoint": str(checkpoint),
                "vecnormalize": str(vecnormalize_path),
                "environment": environment_kwargs,
                "seed_selection_reason": reason,
                "seed_trials": seed_trials,
                "final_metrics": final_metrics,
                "file_size_bytes": video.stat().st_size,
            }
        )
        print(
            f"{stage_name}: wrote {video} "
            f"({final_metrics['duration_seconds']:.2f}s, seed {selected_seed})",
            flush=True,
        )

    metadata = {
        "curriculum_id": state["curriculum_id"],
        "source_state": str(state_path),
        "generated_at": datetime.now().astimezone().isoformat(),
        "deterministic": True,
        "entries": entries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_index(output_dir, state_path, entries)
    print(f"metadata: {output_dir / 'metadata.json'}")
    print(f"index: {output_dir / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
