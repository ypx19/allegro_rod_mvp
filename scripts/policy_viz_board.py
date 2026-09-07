#!/usr/bin/env python3
"""Headless local Web UI for PPO policy debugging (Phase T tilt / contacts).

Launch
------
From the repo root (use the project venv)::

    .venv/bin/python scripts/policy_viz_board.py \\
      --model runs/<run>/checkpoints/final_model.zip \\
      --vecnormalize runs/<run>/checkpoints/vecnormalize.pkl \\
      --physics tip_connect --tip-anchor bottom --rod-mass-scale 400 \\
      --hand-pose-config configs/hand_poses/my_grasp.json \\
      --host 127.0.0.1 --port 8770

If ``--vecnormalize`` is omitted, the board auto-loads ``vecnormalize.pkl`` next
to the checkpoint when present.

UI features
-----------
- Live EGL MuJoCo viewport with pause / single-step / play / reset
- Seed, physics mode (tip_connect | revolute), tip-anchor, mass scale
- Load an arbitrary checkpoint (+ optional VecNormalize) at runtime
- Fingertip-local XY contact-center heatmap (3 fingers) + live scatter
  (same frame as obs ``centers``; zeros when out of contact)
- Time series: tilt, tip error, axial ω, unwrapped rotation, contact forces,
  contact count, episode return
- 12-DoF joint-angle bars
- Termination reason and key info fields from the env

Artifacts (optional) are written under a unique ``runs/viz-...`` directory so
prior experiment runs are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import webbrowser
from urllib.parse import urlparse
import zlib

# Select a headless backend before importing MuJoCo.
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv

MAX_BODY_BYTES = 256 * 1024
HISTORY_LEN = 1500
HEATMAP_RES = 48
CONTACT_XY_HALF_M = 0.03  # matches obs center scaling denominator
FINGER_COLORS = ((0.95, 0.35, 0.25), (0.25, 0.85, 0.45), (0.3, 0.55, 1.0))
FINGER_NAMES = ("index", "middle", "thumb")


def _encode_rgb_png(frame: np.ndarray) -> bytes:
    """Encode uint8 RGB using only the Python standard library."""
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rendered frame must have shape (height, width, 3)")
    height, width, _ = image.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\0" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines, level=3))
        + chunk(b"IEND", b"")
    )


def _safe_repo_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path under the repo root (or absolute path that stays under it)."""
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must be inside {ROOT}") from exc
    if must_exist and not resolved.is_file():
        raise ValueError(f"file does not exist: {resolved}")
    return resolved


def _find_vecnormalize(model_path: Path, explicit: str | None) -> Path | None:
    if explicit:
        return _safe_repo_path(explicit, must_exist=True)
    sibling = model_path.parent / "vecnormalize.pkl"
    if sibling.is_file():
        return sibling
    stem = model_path.stem
    paired = model_path.parent / f"{stem}_vecnormalize.pkl"
    if paired.is_file():
        return paired
    return None


def _new_artifact_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = ROOT / "runs" / f"viz-{stamp}-policy-board"
    path.mkdir(parents=True, exist_ok=False)
    (path / "images").mkdir()
    (path / "logs").mkdir()
    return path


class PolicyVizBoard:
    """Thread-safe PPO rollout + MuJoCo render backend for the HTTP API."""

    def __init__(
        self,
        model_path: Path,
        *,
        vecnormalize: Path | None = None,
        physics: str = "tip_connect",
        tip_anchor: str = "bottom",
        rod_mass_scale: float = 400.0,
        seed: int = 0,
        episode_seconds: float = 20.0,
        hand_pose_config: str | None = None,
        hand_grasp_config: str | None = None,
        reward_style: str = "stage",
        success_mode: str = "net_angle",
        width: int = 640,
        height: int = 480,
        artifact_dir: Path | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.width = width
        self.height = height
        self.episode_seconds = float(episode_seconds)
        self.reward_style = reward_style
        self.success_mode = success_mode
        self.hand_pose_config = hand_pose_config
        self.hand_grasp_config = hand_grasp_config
        self.physics = physics
        self.tip_anchor = tip_anchor
        self.rod_mass_scale = float(rod_mass_scale)
        self.seed = int(seed)
        self.playing = False
        self.deterministic = True
        self.renderer: mujoco.Renderer | None = None
        self.camera = {
            "azimuth": 135.0,
            "elevation": -18.0,
            "distance": 0.42,
            "lookat": [0.0, -0.05, 0.0],
        }
        self.artifact_dir = artifact_dir
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize
        self.env: RodRotationEnv | None = None
        self.policy: PPO | None = None
        self.vecnorm: VecNormalize | None = None
        self.obs: np.ndarray | None = None
        self.last_info: dict[str, object] = {}
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.episode_index = 0
        self.done = False
        self.status = "init"
        self._history: dict[str, list[float]] = {k: [] for k in (
            "step",
            "tilt_rad",
            "tilt_deg",
            "tip_error_m",
            "axial_omega",
            "axis_rotation_deg",
            "contact_count",
            "force0",
            "force1",
            "force2",
            "reward",
            "episode_return",
        )}
        self._joint_history: list[list[float]] = []
        self._heatmap = np.zeros((3, HEATMAP_RES, HEATMAP_RES), dtype=np.float64)
        self._latest_centers = np.zeros((3, 2), dtype=np.float64)
        self._latest_joints = np.zeros(12, dtype=np.float64)
        self._build_env_and_policy()
        self.reset(seed=self.seed)

    def _make_env(self) -> RodRotationEnv:
        return RodRotationEnv(
            render_mode=None,
            hand_model="allegro",
            physics_mode=self.physics,
            tip_anchor=self.tip_anchor,
            rod_mass_scale=self.rod_mass_scale,
            episode_seconds=self.episode_seconds,
            hand_pose_config=self.hand_pose_config,
            hand_grasp_config=self.hand_grasp_config,
            reward_style=self.reward_style,
            success_mode=self.success_mode,
            reset_joint_noise=0.0,
        )

    def _build_env_and_policy(self) -> None:
        if self.env is not None:
            self.env.close()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.env = self._make_env()
        self.model = self.env.model
        self.data = self.env.data
        self.policy = PPO.load(str(self.model_path), device="cpu")
        self.vecnorm = None
        if self.vecnormalize_path is not None:
            dummy = DummyVecEnv([self._make_env])
            self.vecnorm = VecNormalize.load(str(self.vecnormalize_path), dummy)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
        self.status = (
            f"loaded {self.model_path.name}"
            + (f" + {self.vecnormalize_path.name}" if self.vecnormalize_path else "")
        )

    def close(self) -> None:
        with self.lock:
            if self.renderer is not None:
                self.renderer.close()
            if self.env is not None:
                self.env.close()
            if self.vecnorm is not None and hasattr(self.vecnorm, "venv"):
                self.vecnorm.venv.close()

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return obs
        return self.vecnorm.normalize_obs(obs)

    def _clear_episode_buffers(self) -> None:
        for key in self._history:
            self._history[key].clear()
        self._joint_history.clear()
        self._heatmap.fill(0.0)
        self.episode_return = 0.0
        self.done = False

    def _record(self, info: dict[str, object], reward: float) -> None:
        assert self.env is not None
        centers = self.env._contact_centers().reshape(3, 2)
        self._latest_centers = centers.copy()
        joints = np.asarray(self.env.data.qpos[self.env.hand_qpos_adr], dtype=np.float64)
        self._latest_joints = joints.copy()
        forces = info.get("finger_contact_forces_n", [0.0, 0.0, 0.0])
        contacts = info.get("finger_contacts", [0, 0, 0])
        for i in range(3):
            if int(contacts[i]) or float(forces[i]) > 0.05:
                x = centers[i, 0]
                y = centers[i, 1]
                ix = int((x + CONTACT_XY_HALF_M) / (2 * CONTACT_XY_HALF_M) * (HEATMAP_RES - 1))
                iy = int((y + CONTACT_XY_HALF_M) / (2 * CONTACT_XY_HALF_M) * (HEATMAP_RES - 1))
                if 0 <= ix < HEATMAP_RES and 0 <= iy < HEATMAP_RES:
                    self._heatmap[i, iy, ix] += 1.0

        self.episode_return += float(reward)
        step = float(self.env.step_count)
        row = {
            "step": step,
            "tilt_rad": float(info.get("axis_tilt_rad", 0.0)),
            "tilt_deg": float(info.get("axis_tilt_deg", 0.0)),
            "tip_error_m": float(info.get("tip_error_m", 0.0)),
            "axial_omega": float(info.get("axial_omega", 0.0)),
            "axis_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
            "contact_count": float(info.get("contact_count", 0)),
            "force0": float(forces[0]),
            "force1": float(forces[1]),
            "force2": float(forces[2]),
            "reward": float(reward),
            "episode_return": float(self.episode_return),
        }
        for key, value in row.items():
            hist = self._history[key]
            hist.append(float(value) if np.isfinite(value) else 0.0)
            if len(hist) > HISTORY_LEN:
                del hist[: len(hist) - HISTORY_LEN]
        self._joint_history.append(joints.tolist())
        if len(self._joint_history) > HISTORY_LEN:
            del self._joint_history[: len(self._joint_history) - HISTORY_LEN]

    def reset(self, seed: int | None = None) -> dict[str, object]:
        with self.lock:
            assert self.env is not None
            if seed is not None:
                self.seed = int(seed)
            self._clear_episode_buffers()
            self.obs, info = self.env.reset(seed=self.seed)
            self.last_info = dict(info)
            self.last_reward = 0.0
            self.done = False
            self.episode_index += 1
            # Record post-reset diagnostics without advancing physics.
            self._record(self.last_info, 0.0)
            self.status = f"reset seed={self.seed} episode={self.episode_index}"
            return self.state()

    def step(self, n: int = 1) -> dict[str, object]:
        with self.lock:
            assert self.env is not None and self.policy is not None and self.obs is not None
            n = max(1, min(int(n), 50))
            for _ in range(n):
                if self.done:
                    break
                obs_in = self._normalize_obs(self.obs)
                action, _ = self.policy.predict(obs_in, deterministic=self.deterministic)
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                self.last_reward = float(reward)
                self.last_info = dict(info)
                self._record(info, float(reward))
                if terminated or truncated:
                    self.done = True
                    reason = info.get("termination_reason", "done")
                    self.status = f"episode ended: {reason}"
                    self.playing = False
            return self.state()

    def tick(self) -> dict[str, object]:
        with self.lock:
            if self.playing and not self.done:
                return self.step(1)
            return self.state()

    def set_playing(self, playing: bool) -> dict[str, object]:
        with self.lock:
            self.playing = bool(playing) and not self.done
            self.status = "playing" if self.playing else "paused"
            return self.state()

    def load_policy(self, model: str, vecnormalize: str | None = None) -> dict[str, object]:
        with self.lock:
            model_path = _safe_repo_path(model, must_exist=True)
            vec_path = _find_vecnormalize(model_path, vecnormalize)
            self.model_path = model_path
            self.vecnormalize_path = vec_path
            self._build_env_and_policy()
            return self.reset(seed=self.seed)

    def update_config(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            rebuild = False
            if "physics" in payload:
                physics = str(payload["physics"])
                if physics not in ("tip_connect", "revolute"):
                    raise ValueError("physics must be tip_connect or revolute")
                if physics != self.physics:
                    self.physics = physics
                    rebuild = True
            if "tip_anchor" in payload:
                tip_anchor = str(payload["tip_anchor"])
                if tip_anchor not in ("top", "bottom"):
                    raise ValueError("tip_anchor must be top or bottom")
                if tip_anchor != self.tip_anchor:
                    self.tip_anchor = tip_anchor
                    rebuild = True
            if "rod_mass_scale" in payload:
                scale = float(payload["rod_mass_scale"])
                if scale <= 0:
                    raise ValueError("rod_mass_scale must be > 0")
                if abs(scale - self.rod_mass_scale) > 1e-12:
                    self.rod_mass_scale = scale
                    rebuild = True
            if "seed" in payload:
                self.seed = int(payload["seed"])
            if "deterministic" in payload:
                self.deterministic = bool(payload["deterministic"])
            if "camera" in payload and isinstance(payload["camera"], dict):
                cam = payload["camera"]
                for key in ("azimuth", "elevation", "distance"):
                    if key in cam:
                        self.camera[key] = float(cam[key])
                if "lookat" in cam:
                    look = list(cam["lookat"])
                    if len(look) != 3:
                        raise ValueError("lookat must have 3 values")
                    self.camera["lookat"] = [float(x) for x in look]
            if rebuild:
                self._build_env_and_policy()
            return self.reset(seed=self.seed)

    def save_snapshot(self) -> dict[str, object]:
        with self.lock:
            if self.artifact_dir is None:
                self.artifact_dir = _new_artifact_dir()
            out_dir = self.artifact_dir
            png = self.render_png()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            image_path = out_dir / "images" / f"frame_{stamp}_ep{self.episode_index}.png"
            image_path.write_bytes(png)
            metrics_path = out_dir / "logs" / f"metrics_{stamp}_ep{self.episode_index}.json"
            payload = {
                "timestamp": stamp,
                "model": str(self.model_path),
                "vecnormalize": str(self.vecnormalize_path) if self.vecnormalize_path else None,
                "physics": self.physics,
                "tip_anchor": self.tip_anchor,
                "rod_mass_scale": self.rod_mass_scale,
                "seed": self.seed,
                "episode_index": self.episode_index,
                "episode_return": self.episode_return,
                "done": self.done,
                "last_info": self._jsonable_info(self.last_info),
                "history": {k: list(v) for k, v in self._history.items()},
                "joint_names": list(self.env.hand_joint_names) if self.env else [],
                "latest_joints": self._latest_joints.tolist(),
                "latest_contact_centers_m": self._latest_centers.tolist(),
            }
            metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
            summary = out_dir / "summary.md"
            if not summary.exists():
                summary.write_text(
                    "# Policy viz board artifacts\n\n"
                    f"- Created: {stamp}\n"
                    f"- Model: `{self.model_path}`\n"
                    "- Contact heatmap uses fingertip-local XY (obs centers).\n"
                )
            self.status = f"saved {image_path.relative_to(ROOT)}"
            return {
                **self.state(),
                "saved_image": str(image_path.relative_to(ROOT)),
                "saved_metrics": str(metrics_path.relative_to(ROOT)),
                "artifact_dir": str(out_dir.relative_to(ROOT)),
            }

    @staticmethod
    def _json_number(value: object) -> float | None:
        number = float(value)
        if not np.isfinite(number):
            return None
        return number

    @staticmethod
    def _jsonable_info(info: dict[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in info.items():
            if isinstance(value, (np.floating, float)):
                out[key] = PolicyVizBoard._json_number(value)
            elif isinstance(value, (np.integer, int, bool, str)) or value is None:
                out[key] = value.item() if isinstance(value, np.generic) else value
            elif isinstance(value, (list, tuple)):
                converted = []
                for item in value:
                    if isinstance(item, (np.floating, float)):
                        converted.append(PolicyVizBoard._json_number(item))
                    elif isinstance(item, np.generic):
                        converted.append(item.item())
                    else:
                        converted.append(item)
                out[key] = converted
            elif isinstance(value, dict):
                out[key] = PolicyVizBoard._jsonable_info(value)
            else:
                out[key] = str(value)
        return out

    def _heatmap_payload(self) -> list[dict[str, object]]:
        panels = []
        for i, name in enumerate(FINGER_NAMES):
            grid = self._heatmap[i]
            peak = float(grid.max()) if grid.size else 0.0
            panels.append(
                {
                    "finger": name,
                    "peak": peak,
                    "grid": grid.tolist(),
                    "center_xy_m": self._latest_centers[i].tolist(),
                    "in_contact": bool(
                        abs(self._latest_centers[i, 0]) > 1e-9
                        or abs(self._latest_centers[i, 1]) > 1e-9
                    ),
                    "color": list(FINGER_COLORS[i]),
                }
            )
        return panels

    def state(self) -> dict[str, object]:
        assert self.env is not None
        info = self._jsonable_info(self.last_info)
        return {
            "status": self.status,
            "playing": self.playing,
            "done": self.done,
            "deterministic": self.deterministic,
            "physics": self.physics,
            "tip_anchor": self.tip_anchor,
            "rod_mass_scale": self.rod_mass_scale,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "step_count": int(self.env.step_count),
            "max_steps": int(self.env.max_steps),
            "episode_return": float(self.episode_return),
            "last_reward": float(self.last_reward),
            "model_path": str(self.model_path.relative_to(ROOT)),
            "vecnormalize_path": (
                str(self.vecnormalize_path.relative_to(ROOT))
                if self.vecnormalize_path
                else None
            ),
            "joint_names": list(self.env.hand_joint_names),
            "joints_rad": self._latest_joints.tolist(),
            "contact_centers_m": self._latest_centers.tolist(),
            "contact_heatmap": {
                "resolution": HEATMAP_RES,
                "half_extent_m": CONTACT_XY_HALF_M,
                "frame": "fingertip_local_xy",
                "note": (
                    "Accumulated contact centers in each fingertip's local XY "
                    "while that finger contacts the rod (same as obs centers)."
                ),
                "panels": self._heatmap_payload(),
            },
            "history": {k: list(v) for k, v in self._history.items()},
            "info": info,
            "camera": {
                "azimuth": self.camera["azimuth"],
                "elevation": self.camera["elevation"],
                "distance": self.camera["distance"],
                "lookat": list(self.camera["lookat"]),
            },
            "artifact_dir": (
                str(self.artifact_dir.relative_to(ROOT)) if self.artifact_dir else None
            ),
            "server_time": time.time(),
        }

    def render_png(self) -> bytes:
        with self.lock:
            assert self.env is not None
            if self.renderer is None:
                self.renderer = mujoco.Renderer(
                    self.model, height=self.height, width=self.width
                )
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = float(self.camera["azimuth"])
            camera.elevation = float(self.camera["elevation"])
            camera.distance = float(self.camera["distance"])
            camera.lookat[:] = np.asarray(self.camera["lookat"], dtype=np.float64)
            self.renderer.update_scene(self.data, camera=camera)
            frame = self.renderer.render().copy()
            return _encode_rgb_png(frame)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Allegro Policy Viz Board</title>
<style>
:root{color-scheme:dark;--bg:#0b0e13;--panel:#141923;--line:#283142;--text:#eef3fa;--muted:#9aa8bb;--accent:#66d9a6;--warn:#ffcf66;--danger:#ff8e8e}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#1a2436 0,#0b0e13 50%);font:13px "IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;color:var(--text)}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:18px;margin:0;font-family:"IBM Plex Serif",Georgia,serif}header .meta{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(420px,1.2fr) minmax(420px,1fr);gap:14px;padding:14px;max-width:1600px;margin:auto}
.card{background:rgba(20,25,35,.94);border:1px solid var(--line);border-radius:12px;padding:12px;box-shadow:0 16px 40px #0005}
.viewport img{width:100%;display:block;background:#050608;border-radius:8px;aspect-ratio:4/3;object-fit:contain}
.metrics{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;color:var(--muted)}.metrics b{color:var(--text);font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
label{display:block;color:var(--muted);margin-bottom:4px;font-size:11px;letter-spacing:.04em;text-transform:uppercase}
input,select{width:100%;background:#0c1119;color:var(--text);border:1px solid #344056;border-radius:7px;padding:7px 8px}
.buttonrow{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
button{background:#273247;color:var(--text);border:1px solid #3a4961;border-radius:8px;padding:8px 11px;font-weight:650;cursor:pointer}
button.primary{background:#176b50;border-color:#299873}button.warn{background:#5a4820;border-color:#8a6d2e}button:hover{filter:brightness(1.12)}
.status{min-height:34px;padding:8px 10px;border-radius:8px;background:#0c1119;color:var(--muted);white-space:pre-wrap}.status.ok{color:#82e8bb}.status.error{color:var(--danger)}
h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
canvas.chart{width:100%;height:120px;background:#0c1119;border-radius:8px;border:1px solid var(--line)}
canvas.heat{width:100%;aspect-ratio:1;background:#0c1119;border-radius:8px;border:1px solid var(--line)}
.jointbar{display:grid;grid-template-columns:72px 1fr 52px;gap:6px;align-items:center;margin:3px 0}
.jointbar .track{height:8px;background:#1a2230;border-radius:99px;overflow:hidden}.jointbar .fill{height:100%;background:var(--accent)}
.jointbar span{font-variant-numeric:tabular-nums;color:var(--muted);font-size:11px}
.note{color:var(--muted);font-size:12px;line-height:1.4;margin:6px 0 0}
@media(max-width:980px){main{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>Allegro Policy Viz Board</h1>
  <div class="meta" id="headerMeta">connecting…</div>
</header>
<main>
<section class="card">
  <img id="render" alt="MuJoCo scene">
  <div class="metrics">
    <span>Tilt <b id="mTilt">—</b></span>
    <span>Tip err <b id="mTip">—</b></span>
    <span>ω <b id="mOmega">—</b></span>
    <span>Rotation <b id="mRot">—</b></span>
    <span>Contacts <b id="mContacts">—</b></span>
    <span>Return <b id="mReturn">—</b></span>
    <span>Term <b id="mTerm">—</b></span>
  </div>
  <div class="buttonrow">
    <button class="primary" id="btnPlay">Play</button>
    <button id="btnPause">Pause</button>
    <button id="btnStep">Step</button>
    <button id="btnReset">Reset</button>
    <button class="warn" id="btnSave">Save snapshot</button>
  </div>
  <div class="grid2">
    <div><label for="seed">Seed</label><input id="seed" type="number" step="1"></div>
    <div><label for="mass">Rod mass scale</label><input id="mass" type="number" step="any"></div>
    <div><label for="physics">Physics</label><select id="physics"><option value="tip_connect">tip_connect</option><option value="revolute">revolute</option></select></div>
    <div><label for="anchor">Tip anchor</label><select id="anchor"><option value="bottom">bottom</option><option value="top">top</option></select></div>
  </div>
  <div class="buttonrow"><button id="btnApply">Apply env config</button>
    <label style="display:flex;align-items:center;gap:6px;text-transform:none;letter-spacing:0;font-size:13px;color:var(--text)"><input id="deterministic" type="checkbox" checked style="width:auto">Deterministic policy</label>
  </div>
  <div><label for="modelPath">Checkpoint (.zip)</label><input id="modelPath" type="text"></div>
  <div style="margin-top:8px"><label for="vecPath">VecNormalize (.pkl, optional)</label><input id="vecPath" type="text" placeholder="auto: vecnormalize.pkl beside checkpoint"></div>
  <div class="buttonrow"><button id="btnLoad">Load checkpoint</button></div>
  <div id="status" class="status">Connecting…</div>
  <p class="note">Contact heatmap: fingertip-local XY while in contact (obs centers). Spatial extent ±30 mm.</p>
</section>
<section class="card">
  <h2>Contact center heatmaps</h2>
  <div class="grid3" id="heatPanels"></div>
  <h2 style="margin-top:14px">Tilt / tip / rotation</h2>
  <canvas class="chart" id="chartMain" width="640" height="120"></canvas>
  <h2 style="margin-top:14px">Contact forces (N)</h2>
  <canvas class="chart" id="chartForce" width="640" height="120"></canvas>
  <h2 style="margin-top:14px">Joint angles (rad)</h2>
  <div id="joints"></div>
</section>
</main>
<script>
const FINGER_NAMES=['index','middle','thumb'];
const FINGER_COLORS=['#f25840','#40d873','#4d8cff'];
let state=null, pollTimer=null, renderSeq=0, busy=false;

function status(msg, error=false){
  const el=document.getElementById('status');
  el.textContent=msg; el.className='status '+(error?'error':'ok');
}
async function api(path, body){
  const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const text=await r.text(); let j={};
  try{j=text?JSON.parse(text):{}}catch(_){throw new Error(`Invalid JSON (${r.status})`)}
  if(!r.ok)throw new Error(j.error||`HTTP ${r.status}`);
  return j;
}
function drawSeries(canvas, series, colors, yHint){
  const ctx=canvas.getContext('2d');
  const w=canvas.width, h=canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle='#0c1119'; ctx.fillRect(0,0,w,h);
  let ymin=Infinity, ymax=-Infinity;
  series.forEach(s=>s.forEach(v=>{if(Number.isFinite(v)){ymin=Math.min(ymin,v); ymax=Math.max(ymax,v)}}));
  if(!Number.isFinite(ymin)){ymin=0; ymax=1}
  if(ymax-ymin<1e-9){ymax=ymin+1}
  if(yHint){ymin=Math.min(ymin,yHint[0]); ymax=Math.max(ymax,yHint[1])}
  const pad=ymin===0&&ymax>0?0: (ymax-ymin)*0.08;
  ymin-=pad; ymax+=pad;
  ctx.strokeStyle='#283142'; ctx.beginPath();
  const zero=h-(0-ymin)/(ymax-ymin)*h;
  if(zero>=0&&zero<=h){ctx.moveTo(0,zero); ctx.lineTo(w,zero); ctx.stroke()}
  series.forEach((vals,i)=>{
    if(!vals.length)return;
    ctx.strokeStyle=colors[i]; ctx.lineWidth=1.5; ctx.beginPath();
    vals.forEach((v,idx)=>{
      const x=idx/(Math.max(vals.length-1,1))*w;
      const y=h-(v-ymin)/(ymax-ymin)*h;
      if(idx===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
  });
}
function drawHeat(canvas, panel){
  const ctx=canvas.getContext('2d');
  const n=panel.grid.length;
  const img=ctx.createImageData(n,n);
  const peak=Math.max(panel.peak,1);
  for(let y=0;y<n;y++){
    for(let x=0;x<n;x++){
      const v=panel.grid[y][x]/peak;
      const i=(y*n+x)*4;
      img.data[i]=Math.floor(20+panel.color[0]*255*v);
      img.data[i+1]=Math.floor(20+panel.color[1]*255*v);
      img.data[i+2]=Math.floor(20+panel.color[2]*255*v);
      img.data[i+3]=255;
    }
  }
  // Upscale via temporary canvas
  const tmp=document.createElement('canvas'); tmp.width=n; tmp.height=n;
  tmp.getContext('2d').putImageData(img,0,0);
  ctx.imageSmoothingEnabled=false;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
  // live center crosshair
  const half=state.contact_heatmap.half_extent_m;
  const cx=(panel.center_xy_m[0]+half)/(2*half)*canvas.width;
  const cy=(panel.center_xy_m[1]+half)/(2*half)*canvas.height;
  if(panel.in_contact){
    ctx.strokeStyle='#fff'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cx-5,cy); ctx.lineTo(cx+5,cy); ctx.moveTo(cx,cy-5); ctx.lineTo(cx,cy+5); ctx.stroke();
  }
}
function ensureHeatPanels(){
  const host=document.getElementById('heatPanels');
  if(host.children.length)return;
  FINGER_NAMES.forEach((name,i)=>{
    host.insertAdjacentHTML('beforeend',`<div><div style="color:${FINGER_COLORS[i]};margin-bottom:4px">${name}</div><canvas class="heat" id="heat${i}" width="160" height="160"></canvas></div>`);
  });
}
function renderJoints(joints, names){
  const host=document.getElementById('joints');
  if(host.children.length!==joints.length){
    host.innerHTML='';
    joints.forEach((_,i)=>{
      host.insertAdjacentHTML('beforeend',`<div class="jointbar"><span>${names[i]||('j'+i)}</span><div class="track"><div class="fill" id="jf${i}"></div></div><span id="jv${i}">0</span></div>`);
    });
  }
  joints.forEach((v,i)=>{
    const lo=-0.2, hi=1.6;
    const pct=Math.max(0,Math.min(100,(v-lo)/(hi-lo)*100));
    document.getElementById('jf'+i).style.width=pct+'%';
    document.getElementById('jv'+i).textContent=v.toFixed(3);
  });
}
function refresh(s, syncInputs=false){
  state=s;
  document.getElementById('headerMeta').textContent=`${s.physics} · ${s.tip_anchor} · s=${s.rod_mass_scale} · ep ${s.episode_index} step ${s.step_count}/${s.max_steps}`;
  document.getElementById('mTilt').textContent=(s.info.axis_tilt_deg??0).toFixed(2)+'°';
  document.getElementById('mTip').textContent=((s.info.tip_error_m??0)*1000).toFixed(2)+' mm';
  document.getElementById('mOmega').textContent=(s.info.axial_omega??0).toFixed(3)+' rad/s';
  document.getElementById('mRot').textContent=(s.info.axis_rotation_deg??0).toFixed(1)+'°';
  document.getElementById('mContacts').textContent=`${s.info.contact_count??0}/3`;
  document.getElementById('mReturn').textContent=s.episode_return.toFixed(2);
  document.getElementById('mTerm').textContent=s.info.termination_reason|| (s.done?'done':'none');
  if(syncInputs){
    document.getElementById('seed').value=s.seed;
    document.getElementById('mass').value=s.rod_mass_scale;
    document.getElementById('physics').value=s.physics;
    document.getElementById('anchor').value=s.tip_anchor;
    document.getElementById('modelPath').value=s.model_path;
    document.getElementById('vecPath').value=s.vecnormalize_path||'';
    document.getElementById('deterministic').checked=!!s.deterministic;
  }
  ensureHeatPanels();
  s.contact_heatmap.panels.forEach((p,i)=>drawHeat(document.getElementById('heat'+i), p));
  const h=s.history;
  drawSeries(document.getElementById('chartMain'),[h.tilt_deg, h.tip_error_m.map(x=>x*1000), h.axis_rotation_deg],['#ffcf66','#66d9a6','#4d8cff']);
  drawSeries(document.getElementById('chartForce'),[h.force0,h.force1,h.force2],FINGER_COLORS,[0,1]);
  renderJoints(s.joints_rad, s.joint_names);
  document.getElementById('render').src='/api/render.png?v='+(++renderSeq);
  document.getElementById('btnPlay').textContent=s.playing?'Playing…':'Play';
}
async function call(path, body, syncInputs=false){
  if(busy)return; busy=true;
  try{const s=await api(path,body); refresh(s,syncInputs); status(s.status||'ok')}
  catch(e){status(e.message,true)}
  finally{busy=false}
}
function startPoll(){
  clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    if(busy)return;
    if(!state||(!state.playing&&document.hidden))return;
    busy=true;
    try{const s=await api('/api/tick',{}); refresh(s,false); if(s.status)status(s.status)}
    catch(e){status(e.message,true)}
    finally{busy=false}
  }, 50);
}
document.getElementById('btnPlay').onclick=()=>call('/api/play',{playing:true});
document.getElementById('btnPause').onclick=()=>call('/api/play',{playing:false});
document.getElementById('btnStep').onclick=()=>call('/api/step',{n:1});
document.getElementById('btnReset').onclick=()=>call('/api/reset',{seed:Number(document.getElementById('seed').value)},true);
document.getElementById('btnSave').onclick=()=>call('/api/save',{});
document.getElementById('btnApply').onclick=()=>call('/api/config',{
  seed:Number(document.getElementById('seed').value),
  rod_mass_scale:Number(document.getElementById('mass').value),
  physics:document.getElementById('physics').value,
  tip_anchor:document.getElementById('anchor').value,
  deterministic:document.getElementById('deterministic').checked,
},true);
document.getElementById('btnLoad').onclick=()=>call('/api/load_policy',{
  model:document.getElementById('modelPath').value,
  vecnormalize:document.getElementById('vecPath').value||null,
},true);
(async()=>{
  try{refresh(await api('/api/state'),true); status('Ready.'); startPoll()}
  catch(e){status(e.message,true)}
})();
</script></body></html>"""


def make_handler(board: PolicyVizBoard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AllegroPolicyViz/1"

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, status: int, payload: object) -> None:
            # allow_nan=False keeps the API strict; board code maps non-finite
            # floats to null via _json_number / _jsonable_info.
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, exc: Exception | str) -> None:
            self._json(status, {"error": str(exc)})

        def _body(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                return {}
            length = int(raw_length)
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be an object")
            return payload

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._json(HTTPStatus.OK, board.state())
                elif path == "/api/render.png":
                    body = board.render_png()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/tick":
                    result = board.tick()
                elif path == "/api/step":
                    result = board.step(int(body.get("n", 1)))
                elif path == "/api/play":
                    result = board.set_playing(bool(body.get("playing", True)))
                elif path == "/api/reset":
                    seed = body.get("seed")
                    result = board.reset(None if seed is None else int(seed))
                elif path == "/api/config":
                    result = board.update_config(body)
                elif path == "/api/load_policy":
                    model = str(body.get("model", ""))
                    vec = body.get("vecnormalize")
                    result = board.load_policy(
                        model, None if vec in (None, "", "null") else str(vec)
                    )
                elif path == "/api/save":
                    result = board.save_snapshot()
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                self._json(HTTPStatus.OK, result)
            except (ValueError, FileNotFoundError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    return Handler


def _port_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Debug PPO tip-connect / revolute policies in a local Web viz board "
            "(contact heatmaps, tilt, joints, tip error)."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="PPO .zip checkpoint")
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=None,
        help="VecNormalize .pkl (default: auto-detect beside checkpoint)",
    )
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--rod-mass-scale", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--hand-pose-config", type=str, default=None)
    parser.add_argument("--hand-grasp-config", type=str, default=None)
    parser.add_argument("--reward-style", choices=["stage", "dexscrew"], default="stage")
    parser.add_argument("--success-mode", choices=["omega_hold", "net_angle"], default="net_angle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770, help="Use 0 for OS-selected port")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional unique runs/viz-* dir for snapshots (created on first save if omitted)",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="If >0, step this many times headlessly and exit (no server)",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("this local board binds only to 127.0.0.1/localhost")

    try:
        model_path = _safe_repo_path(args.model, must_exist=True)
        vec_path = _find_vecnormalize(
            model_path, str(args.vecnormalize) if args.vecnormalize else None
        )
        artifact_dir = None
        if args.artifact_dir is not None:
            artifact_dir = _safe_repo_path(args.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "images").mkdir(exist_ok=True)
            (artifact_dir / "logs").mkdir(exist_ok=True)
    except ValueError as exc:
        parser.error(str(exc))

    board = PolicyVizBoard(
        model_path,
        vecnormalize=vec_path,
        physics=args.physics,
        tip_anchor=args.tip_anchor,
        rod_mass_scale=args.rod_mass_scale,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        hand_pose_config=args.hand_pose_config,
        hand_grasp_config=args.hand_grasp_config,
        reward_style=args.reward_style,
        success_mode=args.success_mode,
        artifact_dir=artifact_dir,
    )

    if args.smoke_steps > 0:
        print(f"Smoke: stepping {args.smoke_steps} with {model_path}", flush=True)
        result = board.step(args.smoke_steps)
        snap = board.save_snapshot()
        print(
            json.dumps(
                {
                    "steps": result["step_count"],
                    "tilt_deg": result["info"].get("axis_tilt_deg"),
                    "tip_error_m": result["info"].get("tip_error_m"),
                    "contact_count": result["info"].get("contact_count"),
                    "termination_reason": result["info"].get("termination_reason"),
                    "episode_return": result["episode_return"],
                    "saved_image": snap.get("saved_image"),
                    "artifact_dir": snap.get("artifact_dir"),
                },
                indent=2,
            ),
            flush=True,
        )
        board.close()
        return 0

    port = args.port
    if port < 0 or port > 65535:
        parser.error("--port must be between 0 and 65535")
    if port and not _port_available("127.0.0.1", port):
        parser.error(f"port {port} is unavailable; pass --port 0 to choose a free port")

    server = HTTPServer(("127.0.0.1", port), make_handler(board))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Allegro policy viz board: {url}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"VecNormalize: {vec_path}", flush=True)
    print(
        f"Physics: {args.physics}; tip_anchor={args.tip_anchor}; "
        f"rod_mass_scale={args.rod_mass_scale}",
        flush=True,
    )
    print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL', 'default')}", flush=True)
    print(
        "Contact heatmap frame: fingertip-local XY (obs centers), ±30 mm",
        flush=True,
    )
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
