#!/usr/bin/env python3
"""Roll out a policy and export MP4 videos of successful / best trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from allegro_rod_mvp import RodRotationEnv


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    # imageio-ffmpeg backend preferred; pillow gif fallback if mp4 fails.
    try:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
    except Exception:
        gif_path = path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"mp4 encode failed; wrote {gif_path} instead")


def rollout_episode(model: PPO, env: RodRotationEnv, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    infos: list[dict] = []
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        infos.append(info)
    # Final frame.
    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame))
    return {
        "seed": seed,
        "frames": frames,
        "info": info,
        "axis_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
        "tip_error_m": float(info.get("tip_error_m", 0.0)),
        "is_success": bool(info.get("is_success", False)),
        "dropped": bool(terminated),
        "peak_rotation_deg": float(max((i.get("axis_rotation_deg", 0.0) for i in infos), default=0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="checkpoints/stage0/final_model.zip")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--search-episodes", type=int, default=80)
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
    parser.add_argument("--three-contact-reward", type=float, default=10.0)
    parser.add_argument("--contact-window-steps", type=int, default=0)
    parser.add_argument("--contact-window-threshold", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--min-rotation-deg",
        type=float,
        default=90.0,
        help="If fewer than num-videos strict successes, keep top rollouts above this rotation.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = RodRotationEnv(
        render_mode="rgb_array",
        curriculum_stage=args.stage,
        episode_seconds=args.episode_seconds,
        tip_connect_solref=args.tip_connect_solref,
        tip_connect_enabled=args.tip_connect_enabled,
        axis_stabilizer_scale=args.axis_stabilizer_scale,
        axis_tilt_penalty_weight=args.axis_tilt_penalty_weight,
        axis_tilt_recovery_scale=args.axis_tilt_recovery_scale,
        rotation_reward_scale=args.rotation_reward_scale,
        contact_reward_mode=args.contact_reward_mode,
        three_contact_reward=args.three_contact_reward,
        contact_window_steps=args.contact_window_steps,
        contact_window_threshold=args.contact_window_threshold,
    )
    model = PPO.load(args.model, device="cpu")

    successes: list[dict] = []
    candidates: list[dict] = []

    for i in range(args.search_episodes):
        seed = args.seed + i
        ep = rollout_episode(model, env, seed)
        # Drop heavy frame payload from candidate ranking copies later.
        meta = {k: v for k, v in ep.items() if k != "frames"}
        print(
            f"seed={seed:4d} rot={meta['axis_rotation_deg']:7.1f}° "
            f"peak={meta['peak_rotation_deg']:7.1f}° tip={meta['tip_error_m']:.4f} "
            f"success={meta['is_success']} dropped={meta['dropped']}",
            flush=True,
        )
        if meta["is_success"]:
            successes.append(ep)
            if len(successes) >= args.num_videos:
                break
        elif (
            not meta["dropped"]
            and meta["axis_rotation_deg"] >= args.min_rotation_deg
            and meta["tip_error_m"] < 0.02
        ):
            candidates.append(ep)

    env.close()

    selected = successes[: args.num_videos]
    if len(selected) < args.num_videos:
        candidates.sort(key=lambda e: e["axis_rotation_deg"], reverse=True)
        for ep in candidates:
            if len(selected) >= args.num_videos:
                break
            selected.append(ep)

    if not selected:
        # Absolute fallback: best rotations regardless of tip threshold.
        print("No success/near-success found; exporting top rotations instead.", flush=True)
        # Re-roll a small set keeping frames — search already discarded non-selected frames.
        # Re-run ranked by a fresh sweep storing only top-k frames.
        env = RodRotationEnv(
            render_mode="rgb_array",
            curriculum_stage=args.stage,
            episode_seconds=args.episode_seconds,
            tip_connect_solref=args.tip_connect_solref,
            tip_connect_enabled=args.tip_connect_enabled,
            axis_stabilizer_scale=args.axis_stabilizer_scale,
            axis_tilt_penalty_weight=args.axis_tilt_penalty_weight,
            axis_tilt_recovery_scale=args.axis_tilt_recovery_scale,
            rotation_reward_scale=args.rotation_reward_scale,
            contact_reward_mode=args.contact_reward_mode,
            three_contact_reward=args.three_contact_reward,
            contact_window_steps=args.contact_window_steps,
            contact_window_threshold=args.contact_window_threshold,
        )
        ranked: list[dict] = []
        for i in range(min(args.search_episodes, 40)):
            ep = rollout_episode(model, env, args.seed + i)
            ranked.append(ep)
        env.close()
        ranked.sort(key=lambda e: e["axis_rotation_deg"], reverse=True)
        selected = ranked[: args.num_videos]

    manifest = []
    for idx, ep in enumerate(selected):
        tag = "success" if ep["is_success"] else "best"
        fname = f"stage{args.stage}_{tag}_{idx:02d}_seed{ep['seed']}_rot{ep['axis_rotation_deg']:.0f}deg.mp4"
        path = out_dir / fname
        _write_mp4(path, ep["frames"], fps=args.fps)
        entry = {
            "file": str(path),
            "seed": ep["seed"],
            "axis_rotation_deg": ep["axis_rotation_deg"],
            "peak_rotation_deg": ep["peak_rotation_deg"],
            "tip_error_m": ep["tip_error_m"],
            "is_success": ep["is_success"],
            "dropped": ep["dropped"],
            "num_frames": len(ep["frames"]),
        }
        manifest.append(entry)
        print(f"wrote {path} ({entry['num_frames']} frames)", flush=True)

    manifest_path = out_dir / f"stage{args.stage}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {manifest_path}", flush=True)
    return 0 if any(m["is_success"] for m in manifest) or manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
