#!/usr/bin/env python3
"""Export T00 A/B/C/D demo videos from the 300k continue checkpoints.

Uses the ~306k snapshot (nearest saved step to the reported 303,104 row).
Does not overwrite runs/20260914-1638-t00-ablation-videos/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv

ROOT = Path(__file__).resolve().parents[1]
POSE = "configs/hand_poses/my_grasp.json"
GRASP = "configs/hand_grasps/my_grasp_tip_connect_heavy.json"
RUNS = {
    "A": ROOT / "runs/20260914-1712-t00-ablation-A-continue300k-seed0",
    "B": ROOT / "runs/20260914-1712-t00-ablation-B-continue300k-seed0",
    "C": ROOT / "runs/20260914-1712-t00-ablation-C-continue300k-seed0",
    "D": ROOT / "runs/20260914-1712-t00-ablation-D-continue300k-seed0",
}
SPECS = {
    "A": {"hist": 1, "support": False, "title": "A  48-D  baseline"},
    "B": {"hist": 4, "support": False, "title": "B  192-D  baseline"},
    "C": {"hist": 1, "support": True, "title": "C  48-D  support-aware"},
    "D": {"hist": 4, "support": True, "title": "D  192-D  support-aware"},
}
# 640x480 panels → 1280x960 2x2 grid.
WIDTH, HEIGHT, FPS = 640, 480, 25
DEFAULT_CKPT_STEP = 306432
DEFAULT_SEEDS = (6, 10000)
OVERLAY_HOLD = 12


def t00_kwargs(hist: int, support: bool) -> dict:
    return dict(
        render_mode="rgb_array",
        curriculum_stage=0,
        episode_seconds=20.0,
        tip_connect_solref=0.008,
        tip_connect_enabled=True,
        axis_stabilizer_scale=0.0,
        physics_mode="tip_connect",
        reward_style="dexscrew",
        tip_anchor="bottom",
        rod_mass_scale=400.0,
        contact_friction_scale=4.0,
        contact_friction_scaling_mode="full_vector",
        scale_rod_joint_dynamics_from_s400=True,
        scale_tip_solref_with_mass=False,
        tilt_terminate_rad=1.2,
        contact_reward_mode="discrete",
        three_contact_reward=0.3,
        contact_reward_scale=0.1,
        contact_window_steps=25,
        contact_window_threshold=18.0,
        three_contact_required=True,
        rotation_requires_three_contacts=False,
        contact_support_termination_enabled=False,
        success_mode="net_angle",
        dexscrew_tilt_scale=1.0,
        hand_pose_config=POSE,
        hand_grasp_config=GRASP,
        obs_history_len=hist,
        support_aware_reward_enabled=support,
        rotation_contact_scale_0=0.0,
        rotation_contact_scale_1=0.1,
        rotation_contact_scale_2plus=1.0,
        low_support_wobble_scale=0.5,
    )


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def checkpoint_paths(run: Path, prefer_step: int) -> tuple[Path, Path, int]:
    snap_zip = run / "checkpoints" / f"ppo_rod_{prefer_step}_steps.zip"
    snap_vec = run / "checkpoints" / f"ppo_rod_{prefer_step}_steps_vecnormalize.pkl"
    if snap_zip.is_file() and snap_vec.is_file():
        return snap_zip, snap_vec, prefer_step
    final_zip = run / "checkpoints" / "final_model.zip"
    final_vec = run / "checkpoints" / "vecnormalize.pkl"
    if not final_zip.is_file() or not final_vec.is_file():
        raise FileNotFoundError(f"missing checkpoint in {run / 'checkpoints'}")
    return final_zip, final_vec, -1


def ensure_renderer(env: RodRotationEnv) -> None:
    # Do not call env.render() first: that builds a default 640x480 Renderer
    # and update_scene without the free camera.
    if env.renderer is None:
        env.renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.00, -0.05, -0.02], dtype=np.float64)
    cam.distance = 0.40
    cam.azimuth = 125.0
    cam.elevation = 10.0
    env.renderer.update_scene(env.data, camera=cam)


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    bar_h = 8 + 18 * len(lines)
    draw.rectangle((0, 0, img.width, bar_h), fill=(0, 0, 0, 165))
    fnt = font(14)
    y = 4
    for line in lines:
        draw.text((8, y), line, font=fnt, fill=(255, 255, 255, 255))
        y += 18
    return np.asarray(img)


def overlay_lines(
    spec_title: str,
    run_id: str,
    ckpt_step: int,
    seed: int,
    step: int,
    ep_return: float,
    success: bool,
    contact_count: int,
    rot: float,
    tilt: float,
    reason: str,
) -> list[str]:
    return [
        f"{spec_title}   seed {seed}   ckpt {ckpt_step}",
        run_id,
        (
            f"t={step:03d}  ret={ep_return:+.1f}  success={int(success)}  "
            f"len={step}  n={contact_count}"
        ),
        f"rot={rot:.0f}deg  tilt={tilt:.1f}deg  {reason}",
    ]


def rollout(name: str, seed: int, prefer_step: int) -> dict:
    spec = SPECS[name]
    run = RUNS[name]
    zip_path, vec_path, labeled_step = checkpoint_paths(run, prefer_step)
    env = RodRotationEnv(**t00_kwargs(spec["hist"], spec["support"]))
    model = PPO.load(str(zip_path), device="cpu")
    dummy = DummyVecEnv([lambda: RodRotationEnv(**t00_kwargs(spec["hist"], spec["support"]))])
    vecnorm = VecNormalize.load(str(vec_path), dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    ckpt_step = int(getattr(model, "num_timesteps", 0) or labeled_step)
    if ckpt_step <= 0:
        ckpt_step = labeled_step if labeled_step > 0 else prefer_step
    run_id = run.name
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    info: dict = {}
    terminated = truncated = False
    step = 0
    ep_return = 0.0
    while not (terminated or truncated):
        ensure_renderer(env)
        raw = env.renderer.render()
        model_obs = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        ep_return += float(reward)
        frames.append(
            overlay(
                raw,
                overlay_lines(
                    spec["title"],
                    run_id,
                    ckpt_step,
                    seed,
                    step,
                    ep_return,
                    bool(info.get("is_success", False)),
                    int(info.get("contact_count", 0)),
                    float(info.get("axis_rotation_deg", 0.0)),
                    float(info.get("axis_tilt_deg", 0.0)),
                    str(info.get("termination_reason", "none")),
                ),
            )
        )
    if frames:
        last = frames[-1]
        frames.extend([last] * OVERLAY_HOLD)
    env.close()
    dummy.close()
    return {
        "name": name,
        "seed": seed,
        "frames": frames,
        "steps": step,
        "return": ep_return,
        "success": bool(info.get("is_success", False)),
        "rot": float(info.get("axis_rotation_deg", 0.0)),
        "tilt": float(info.get("axis_tilt_deg", 0.0)),
        "n_contact": int(info.get("contact_count", 0)),
        "reason": str(info.get("termination_reason", "none")),
        "ckpt_step": ckpt_step,
        "run_id": run_id,
        "zip_path": str(zip_path),
        "vec_path": str(vec_path),
    }


def write_mp4(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=FPS, codec="libx264", quality=8)


def grid4(clips: list[dict]) -> list[np.ndarray]:
    n = max(len(c["frames"]) for c in clips)
    panels = []
    for clip in clips:
        frames = list(clip["frames"])
        if not frames:
            continue
        last = frames[-1]
        while len(frames) < n:
            frames.append(last)
        panels.append(frames)
    out = []
    for t in range(n):
        top = np.concatenate([panels[0][t], panels[1][t]], axis=1)
        bot = np.concatenate([panels[2][t], panels[3][t]], axis=1)
        out.append(np.concatenate([top, bot], axis=0))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: runs/<timestamp>-t00-ablation-videos-300k",
    )
    p.add_argument("--ckpt-step", type=int, default=DEFAULT_CKPT_STEP)
    p.add_argument("--seeds", default="6,10000")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if args.out_dir:
        out = Path(args.out_dir)
        if not out.is_absolute():
            out = ROOT / out
    else:
        out = ROOT / "runs" / "20260914-1748-t00-ablation-videos-300k"
    out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for seed in seeds:
        clips = []
        for name in ("A", "B", "C", "D"):
            print(f"rollout {name} seed={seed}", flush=True)
            clip = rollout(name, seed, args.ckpt_step)
            clips.append(clip)
            path = out / (
                f"{name}_seed{seed}_rot{clip['rot']:.0f}_tilt{clip['tilt']:.0f}_{clip['reason']}.mp4"
            )
            write_mp4(path, clip["frames"])
            rec = {k: v for k, v in clip.items() if k != "frames"}
            rec["path"] = str(path)
            rec["bytes"] = path.stat().st_size
            summary.append(rec)
            print(
                f"wrote {path} steps={clip['steps']} ret={clip['return']:+.1f} "
                f"rot={clip['rot']:.1f} tilt={clip['tilt']:.1f} n={clip['n_contact']} "
                f"{clip['reason']} ckpt={clip['ckpt_step']}",
                flush=True,
            )
        grid_path = out / f"ABCD_grid_seed{seed}.mp4"
        write_mp4(grid_path, grid4(clips))
        summary.append(
            {
                "name": "ABCD",
                "seed": seed,
                "path": str(grid_path),
                "bytes": grid_path.stat().st_size,
                "ckpt_step": clips[0]["ckpt_step"],
                "steps": max(c["steps"] for c in clips),
            }
        )
        print(f"wrote {grid_path}", flush=True)
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary {out / 'export_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
