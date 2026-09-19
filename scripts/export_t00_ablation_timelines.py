#!/usr/bin/env python3
"""Replay T00 A/B/C/D 300k policies on seed 6 and dump seed-6-style timelines.

Writes per-step JSON + 25 Hz JPEG frames. Does not retune λ and does not
overwrite the original seed-6 collapse page.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.support_collapse_metrics import episode_support_collapse_metrics

import importlib.util

_VID = Path(__file__).resolve().parent / "export_t00_ablation_videos.py"
_spec = importlib.util.spec_from_file_location("export_t00_ablation_videos", _VID)
_vid = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vid)
DEFAULT_CKPT_STEP = _vid.DEFAULT_CKPT_STEP
FPS = _vid.FPS
ROOT = _vid.ROOT
RUNS = _vid.RUNS
SPECS = _vid.SPECS
checkpoint_paths = _vid.checkpoint_paths
ensure_renderer = _vid.ensure_renderer
t00_kwargs = _vid.t00_kwargs

FINGER_NAMES = ("index/tip0", "middle/tip1", "thumb/tip2")
EVAL_TILT_RAD = 1.2
VIDEO_DIR = ROOT / "runs/20260914-1748-t00-ablation-videos-300k"
VIDEO_NAMES = {
    "A": "A_seed6_rot457_tilt75_axis_tilt.mp4",
    "B": "B_seed6_rot410_tilt77_axis_tilt.mp4",
    "C": "C_seed6_rot368_tilt71_axis_tilt.mp4",
    "D": "D_seed6_rot387_tilt75_axis_tilt.mp4",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _axis_world(env: RodRotationEnv) -> np.ndarray:
    axis = np.asarray(env.data.xmat[env.rod_body].reshape(3, 3)[:, 0], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n > 1e-8:
        axis = axis / n
    return axis


def _finger_records(env: RodRotationEnv) -> list[dict]:
    touch = env._touch()
    dists = env._tip_rod_distances()
    centers = env._contact_centers().reshape(3, 2)
    n_per = [0, 0, 0]
    force6 = [np.zeros(6, dtype=np.float64) for _ in range(3)]
    world_pos: list[list[float] | None] = [None, None, None]
    buf = env._contact_force_buf
    for k in range(env.data.ncon):
        con = env.data.contact[k]
        for i, tip_geom in enumerate(env.tip_geom_ids):
            if {con.geom1, con.geom2} == {tip_geom, env.rod_geom}:
                n_per[i] += 1
                mujoco.mj_contactForce(env.model, env.data, k, buf)
                force6[i] = np.asarray(buf, dtype=np.float64).copy()
                body_id = env.model.geom_bodyid[tip_geom]
                world_pos[i] = env.data.xpos[body_id].tolist()
    rows = []
    for i, name in enumerate(FINGER_NAMES):
        in_contact = bool(touch[i] > 0.05)
        rows.append(
            {
                "name": name,
                "force_n": float(touch[i]),
                "in_contact": in_contact,
                "axis_dist_m": float(dists[i]),
                "local_xy_m": centers[i].tolist() if in_contact else [0.0, 0.0],
                "n_contacts": int(n_per[i]),
                "world_pos_m": world_pos[i] if in_contact else None,
                "force6_n": force6[i].tolist() if in_contact else [0.0] * 6,
            }
        )
    return rows


def _reset_kinematics(env: RodRotationEnv) -> dict:
    axis = _axis_world(env)
    align = abs(float(np.dot(axis, env.target_axis)))
    tilt_rad = float(np.arccos(np.clip(align, 0.0, 1.0)))
    omega = np.asarray(env.data.cvel[env.rod_body, :3], dtype=np.float64)
    body_axial = float(np.dot(omega, axis))
    return {
        "axis_tilt_rad": tilt_rad,
        "axis_tilt_deg": float(np.degrees(tilt_rad)),
        "lateral_omega_rad_s": float(np.linalg.norm(omega - body_axial * axis)),
        "axial_omega_rad_s": float(env._axial_omega()),
        "axis_rotation_deg": float(np.degrees(env.unwrapped_angle)),
        "tip_error_m": float(np.linalg.norm(env.data.site_xpos[env.tip_site] - env.target_tip)),
        "contact_count": int(np.sum(env._touch() > 0.05)),
        "ncon": int(env.data.ncon),
        "rod_axis_world": axis.tolist(),
        "rod_pos_m": env.data.xpos[env.rod_body].tolist(),
        "reward": None,
        "reward_rotation": 0.0,
        "reward_rotation_before_support_gate": 0.0,
        "reward_support_gating_effect": 0.0,
        "reward_low_support_wobble": 0.0,
        "reward_axis_tilt_penalty": 0.0,
        "termination_reason": "none",
        "would_eval_axis_tilt_kill": tilt_rad > EVAL_TILT_RAD,
        "terminated": False,
        "truncated": False,
    }


def _step_row(env: RodRotationEnv, info: dict, reward: float) -> dict:
    axis = _axis_world(env)
    tilt_rad = float(info.get("axis_tilt_rad", 0.0))
    return {
        "axis_tilt_rad": tilt_rad,
        "axis_tilt_deg": float(info.get("axis_tilt_deg", np.degrees(tilt_rad))),
        "lateral_omega_rad_s": float(info.get("omega_perp_norm", info.get("lateral_omega", 0.0))),
        "axial_omega_rad_s": float(info.get("axial_omega", 0.0)),
        "axis_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
        "tip_error_m": float(info.get("tip_error_m", 0.0)),
        "contact_count": int(info.get("contact_count", 0)),
        "ncon": int(env.data.ncon),
        "rod_axis_world": axis.tolist(),
        "rod_pos_m": env.data.xpos[env.rod_body].tolist(),
        "reward": float(info.get("reward_total", reward)),
        "reward_rotation": float(info.get("reward_rotation", 0.0)),
        "reward_rotation_before_support_gate": float(
            info.get("reward_rotation_before_support_gate", info.get("reward_rotation", 0.0))
        ),
        "reward_support_gating_effect": float(info.get("reward_support_gating_effect", 0.0)),
        "reward_low_support_wobble": float(info.get("reward_low_support_wobble", 0.0)),
        "reward_axis_tilt_penalty": float(info.get("reward_axis_tilt_penalty", 0.0)),
        "termination_reason": str(info.get("termination_reason", "none")),
        "would_eval_axis_tilt_kill": tilt_rad > EVAL_TILT_RAD,
        "terminated": False,
        "truncated": False,
    }


def _save_frame(env: RodRotationEnv, path: Path) -> None:
    ensure_renderer(env)
    rgb = env.renderer.render()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="JPEG", quality=86, optimize=True)


def _annotate_velocities(steps: list[dict], dt: float) -> None:
    prev = None
    for row in steps:
        if prev is None:
            row["tilt_vel_deg_s"] = 0.0
            row["tilt_vel_rad_s"] = 0.0
        else:
            row["tilt_vel_deg_s"] = (row["axis_tilt_deg"] - prev["axis_tilt_deg"]) / dt
            row["tilt_vel_rad_s"] = (row["axis_tilt_rad"] - prev["axis_tilt_rad"]) / dt
        row["axial_omega_deg_s"] = float(np.degrees(row["axial_omega_rad_s"]))
        row["lateral_omega_deg_s"] = float(np.degrees(row["lateral_omega_rad_s"]))
        row["contact_force_total_n"] = float(sum(f["force_n"] for f in row.get("fingers", [])))
        prev = row


def _event_summary(steps: list[dict], collapse: dict) -> dict:
    first_loss = None
    first_zero = None
    first_kill = None
    recontact_after_loss = False
    for row in steps:
        n = int(row["contact_count"])
        if first_loss is None and row["step"] > 0:
            prev_n = int(steps[row["step"] - 1]["contact_count"])
            if prev_n >= 2 and n <= 1:
                first_loss = int(row["step"])
        if first_zero is None and n == 0 and row["step"] > 0:
            first_zero = int(row["step"])
        if first_kill is None and row["would_eval_axis_tilt_kill"]:
            first_kill = int(row["step"])
    if first_loss is not None:
        for row in steps:
            if row["step"] > first_loss and int(row["contact_count"]) >= 2:
                recontact_after_loss = True
                break
    kill = first_kill if first_kill is not None else (
        int(steps[-1]["step"]) if steps[-1]["termination_reason"] == "axis_tilt" else None
    )
    steps_to_tilt = None
    if first_loss is not None and kill is not None:
        steps_to_tilt = int(kill - first_loss)
    return {
        "first_support_loss_step": first_loss,
        "first_n0_step": first_zero,
        "first_tilt_kill_step": kill,
        "steps_to_tilt_after_loss": steps_to_tilt,
        "recontact_after_first_loss": recontact_after_loss,
        "n_support_loss_events": int(collapse.get("n_support_loss_events", 0)),
        "n_recontact_events": int(collapse.get("n_recontact_events", 0)),
        "support_loss_steps": list(collapse.get("support_loss_steps", [])),
        "recontact_steps": list(collapse.get("recontact_steps", [])),
    }


def rollout_condition(name: str, seed: int, prefer_step: int, out_dir: Path) -> dict:
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

    cond_dir = out_dir / name
    frames_dir = cond_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    dt = 1.0 / float(env.policy_hz)
    steps: list[dict] = []
    ep_return = 0.0

    reset = _reset_kinematics(env)
    reset.update(
        {
            "step": 0,
            "t_s": 0.0,
            "fingers": _finger_records(env),
            "frame": f"{name}/frames/step_0000.jpg",
        }
    )
    _save_frame(env, frames_dir / "step_0000.jpg")
    steps.append(reset)

    terminated = truncated = False
    info: dict = {}
    reward = 0.0
    t = 0
    while not (terminated or truncated):
        model_obs = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        t += 1
        ep_return += float(reward)
        row = _step_row(env, info, float(reward))
        row.update(
            {
                "step": t,
                "t_s": t * dt,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "fingers": _finger_records(env),
                "frame": f"{name}/frames/step_{t:04d}.jpg",
            }
        )
        _save_frame(env, frames_dir / f"step_{t:04d}.jpg")
        steps.append(row)

    policy_hz = int(getattr(env, "policy_hz", FPS))
    env.close()
    dummy.close()
    _annotate_velocities(steps, dt)

    n_contacts = [int(s["contact_count"]) for s in steps[1:]]
    omegas = [float(s["lateral_omega_rad_s"]) for s in steps[1:]]
    tilts = [float(s["axis_tilt_rad"]) for s in steps[1:]]
    collapse = episode_support_collapse_metrics(
        n_contacts,
        omegas,
        tilts,
        termination_reason=str(info.get("termination_reason", "none")),
    )
    # Metrics are 0-indexed over post-reset steps; map to recorded step numbers.
    collapse["support_loss_steps"] = [int(i) + 1 for i in collapse.get("support_loss_steps", [])]
    collapse["recontact_steps"] = [int(i) + 1 for i in collapse.get("recontact_steps", [])]
    events = _event_summary(steps, collapse)

    video_name = VIDEO_NAMES[name]
    dest_video = cond_dir / video_name
    src_video = VIDEO_DIR / video_name
    if src_video.is_file():
        shutil.copy2(src_video, dest_video)

    payload = {
        "meta": {
            "condition": name,
            "title": spec["title"],
            "hist": spec["hist"],
            "support_aware": spec["support"],
            "obs_dim": 48 * spec["hist"],
            "source_run": run.name,
            "checkpoint": str(zip_path),
            "vecnormalize": str(vec_path),
            "ckpt_step": ckpt_step,
            "seed": seed,
            "dt_s": dt,
            "policy_hz": policy_hz,
            "tilt_terminate_rad_logged": EVAL_TILT_RAD,
            "eval_tilt_terminate_rad": EVAL_TILT_RAD,
            "eval_kill_step": events["first_tilt_kill_step"],
            "n_steps": int(steps[-1]["step"]),
            "n_frames": len(steps),
            "episode_return": float(ep_return),
            "termination_reason": str(info.get("termination_reason", "none")),
            "final_tilt_deg": float(steps[-1]["axis_tilt_deg"]),
            "final_rotation_deg": float(steps[-1]["axis_rotation_deg"]),
            "final_contact_count": int(steps[-1]["contact_count"]),
            "video": f"{name}/{video_name}",
            "tilt_vel_source": (
                "finite difference of tilt angle / dt (dt=0.04 s at 25 Hz); "
                "not ω_perp"
            ),
            "axial_omega_source": "ω_axial = −ω·â, same sign as unwrap",
            "omega_perp_source": "ω_perp = ||ω − (ω·â)â|| lateral angular rate",
            "contact_force_source": (
                "_touch() abs(mj_contactForce[0]) summed per tip, threshold 0.05 N"
            ),
            "highlight_steps": [
                s
                for s in (
                    events["first_support_loss_step"],
                    events["first_n0_step"],
                    events["first_tilt_kill_step"],
                )
                if s is not None
            ],
        },
        "events": events,
        "collapse": _jsonable(collapse),
        "steps": steps,
    }
    (cond_dir / "timeline.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{name} steps={t} ret={ep_return:+.1f} rot={steps[-1]['axis_rotation_deg']:.1f} "
        f"tilt={steps[-1]['axis_tilt_deg']:.1f} n={steps[-1]['contact_count']} "
        f"loss={events['first_support_loss_step']} n0={events['first_n0_step']} "
        f"kill={events['first_tilt_kill_step']} recontact={events['recontact_after_first_loss']} "
        f"{info.get('termination_reason')}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="runs/20260914-1805-t00-ablation-demos-300k",
        help="Analysis run directory (new dir; does not overwrite seed-6 page).",
    )
    p.add_argument("--ckpt-step", type=int, default=DEFAULT_CKPT_STEP)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--conditions", default="A,B,C,D")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    names = [part.strip() for part in args.conditions.split(",") if part.strip()]
    timelines = {}
    comparison = []
    for name in names:
        payload = rollout_condition(name, args.seed, args.ckpt_step, out)
        timelines[name] = payload
        comparison.append(
            {
                "condition": name,
                "title": payload["meta"]["title"],
                "hist": payload["meta"]["hist"],
                "support_aware": payload["meta"]["support_aware"],
                "length": payload["meta"]["n_steps"],
                "return": payload["meta"]["episode_return"],
                "rotation_deg": payload["meta"]["final_rotation_deg"],
                "tilt_deg": payload["meta"]["final_tilt_deg"],
                "termination_reason": payload["meta"]["termination_reason"],
                **payload["events"],
            }
        )
    compact = {
        name: {
            "meta": payload["meta"],
            "events": payload["events"],
            "collapse": payload["collapse"],
            "steps": [
                {
                    "step": s["step"],
                    "t_s": s["t_s"],
                    "axis_tilt_deg": s["axis_tilt_deg"],
                    "axis_tilt_rad": s["axis_tilt_rad"],
                    "tilt_vel_deg_s": s["tilt_vel_deg_s"],
                    "tilt_vel_rad_s": s["tilt_vel_rad_s"],
                    "lateral_omega_rad_s": s["lateral_omega_rad_s"],
                    "lateral_omega_deg_s": s["lateral_omega_deg_s"],
                    "axial_omega_rad_s": s["axial_omega_rad_s"],
                    "axial_omega_deg_s": s["axial_omega_deg_s"],
                    "contact_force_total_n": s["contact_force_total_n"],
                    "axis_rotation_deg": s["axis_rotation_deg"],
                    "tip_error_m": s["tip_error_m"],
                    "contact_count": s["contact_count"],
                    "ncon": s["ncon"],
                    "fingers": [
                        {
                            "name": f["name"],
                            "force_n": f["force_n"],
                            "in_contact": f["in_contact"],
                            "axis_dist_m": f["axis_dist_m"],
                        }
                        for f in s["fingers"]
                    ],
                    "rod_axis_world": s["rod_axis_world"],
                    "reward": s["reward"],
                    "reward_rotation": s["reward_rotation"],
                    "reward_rotation_before_support_gate": s["reward_rotation_before_support_gate"],
                    "reward_low_support_wobble": s["reward_low_support_wobble"],
                    "termination_reason": s["termination_reason"],
                    "would_eval_axis_tilt_kill": s["would_eval_axis_tilt_kill"],
                    "frame": s["frame"],
                }
                for s in payload["steps"]
            ],
        }
        for name, payload in timelines.items()
    }
    (out / "timelines.json").write_text(json.dumps(_jsonable(compact), indent=2) + "\n")
    (out / "comparison.json").write_text(json.dumps(_jsonable(comparison), indent=2) + "\n")
    print(f"wrote {out / 'timelines.json'}", flush=True)


if __name__ == "__main__":
    main()
