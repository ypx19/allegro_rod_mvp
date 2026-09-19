#!/usr/bin/env python3
"""Local Web UI: compare my_grasp vs palm_down palm translations.

Launch from the repo root::

    .venv/bin/python scripts/palm_translation_compare_web.py --port 8768

The page is a live EGL MuJoCo view (shared free camera) plus each setup's
most successful demo video. Camera: drag orbit, scroll zoom, Shift/right-drag pan.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import sys
import threading
import webbrowser
from urllib.parse import urlparse

if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.hand_pose import load_hand_pose, quat_normalize
from scripts.edit_hand_pose_web import _add_marker, _encode_rgb_png

PAGE_DIR = ROOT / "docs" / "pages" / "palm-translation-compare"
POSE_PATH = ROOT / "configs" / "hand_poses" / "my_grasp.json"
GRASP_PATH = ROOT / "configs" / "hand_grasps" / "my_grasp_tip_connect_heavy.json"
PALM_DOWN_XML = (
    ROOT / "palm_down_screwdriver" / "models" / "screwdriver_palm_down_h0.015.xml"
)
MAX_BODY_BYTES = 64 * 1024
PANEL_W = 640
PANEL_H = 480

MY_GRASP_VIDEO = (
    ROOT
    / "runs"
    / "20260823-2015-proportional-physics-C-seed0-R09-s1-mu0.01-seed0"
    / "videos"
    / "revolute_success_00_seed10000_rot11271deg_tilt0deg_steps500_none.mp4"
)
MY_GRASP_T00_VIDEO = (
    ROOT
    / "runs"
    / "20260912-cinematic-t00-flipped-seed6"
    / "cinematic-allegro-tip-bottom.mp4"
)
PALM_DOWN_60S = ROOT / "docs" / "media" / "palm-down-screwdriver-60s-seed4000.mp4"
PALM_DOWN_20S = ROOT / "docs" / "media" / "palm-down-screwdriver-20s-seed7000.mp4"


def _encode_rgb_jpeg(frame: np.ndarray, quality: int = 72) -> bytes:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=False)
    return buf.getvalue()


def _add_connector(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    rgba: tuple[float, float, float, float],
    width: float = 0.0025,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _default_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, -0.05, 0.02], dtype=np.float64)
    camera.distance = 0.46
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


class SceneSlot:
    """One MuJoCo environment plus a lazy EGL renderer."""

    def __init__(self, name: str, env: RodRotationEnv, native_pos: np.ndarray) -> None:
        self.name = name
        self.env = env
        self.model = env.model
        self.data = env.data
        self.palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        if self.palm_id < 0:
            raise RuntimeError(f"{name} scene has no palm body")
        self.native_pos = np.asarray(native_pos, dtype=np.float64)
        self.renderer: mujoco.Renderer | None = None

    def set_translation(self, translation_m: np.ndarray) -> None:
        self.model.body_pos[self.palm_id] = np.asarray(translation_m, dtype=np.float64)
        self.env.reset(seed=0)
        mujoco.mj_forward(self.model, self.data)

    def translation_m(self) -> np.ndarray:
        return self.model.body_pos[self.palm_id].copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.env.close()


class PalmCompareBoard:
    """Thread-safe two-scene palm-translation comparison."""

    def __init__(self, width: int = PANEL_W, height: int = PANEL_H) -> None:
        self.lock = threading.RLock()
        self.width = int(width)
        self.height = int(height)
        if not POSE_PATH.is_file():
            raise FileNotFoundError(f"missing pose config: {POSE_PATH}")
        if not PALM_DOWN_XML.is_file():
            raise FileNotFoundError(f"missing palm_down XML: {PALM_DOWN_XML}")
        pose, _, pose_hash = load_hand_pose(POSE_PATH, "allegro_three_finger_rod")
        self.my_grasp_native = np.asarray(pose["translation"], dtype=np.float64)
        self.my_grasp_quat = quat_normalize(
            np.asarray(pose["quaternion_wxyz"], dtype=np.float64)
        )
        self.my_grasp_sha256 = pose_hash
        xml_model = mujoco.MjModel.from_xml_path(str(PALM_DOWN_XML))
        xml_palm = mujoco.mj_name2id(xml_model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.palm_down_native = xml_model.body_pos[xml_palm].copy()
        self.palm_down_quat = quat_normalize(xml_model.body_quat[xml_palm].copy())
        my_env = RodRotationEnv(
            hand_model="allegro",
            physics_mode="tip_connect",
            tip_anchor="bottom",
            hand_pose_config=str(POSE_PATH),
            hand_grasp_config=str(GRASP_PATH) if GRASP_PATH.is_file() else None,
            rod_mass_scale=1.0,
            axis_stabilizer_scale=0.0,
            reset_joint_noise=0.0,
            grasp_ramp_steps=1,
            grasp_hold_steps=0,
            contact_support_termination_enabled=False,
            rotation_requires_three_contacts=False,
        )
        pd_env = RodRotationEnv(
            xml_path=str(PALM_DOWN_XML),
            hand_model="allegro",
            physics_mode="tip_connect",
            tip_anchor="bottom",
            rod_mass_scale=1.0,
            axis_stabilizer_scale=0.0,
            tip_connect_enabled=True,
            reset_joint_noise=0.0,
            grasp_ramp_steps=1,
            grasp_hold_steps=0,
            contact_support_termination_enabled=False,
            rotation_requires_three_contacts=False,
        )
        my_env.reset_joint_noise = 0.0
        pd_env.reset_joint_noise = 0.0
        self.scenes = {
            "my_grasp": SceneSlot("my_grasp", my_env, self.my_grasp_native),
            "palm_down": SceneSlot("palm_down", pd_env, self.palm_down_native),
        }
        self.camera = _default_camera()
        self.blend = 0.0
        self._media = self._media_catalog()
        self._media_bytes: dict[str, bytes] = {}
        self.reset_native()
        self.settle(40)

    def _media_catalog(self) -> dict[str, dict[str, object]]:
        videos = {
            "my_grasp": [
                {
                    "id": "my_grasp_revolute_s1",
                    "url": "/media/my_grasp_revolute_s1.mp4",
                    "path": MY_GRASP_VIDEO,
                    "label": "Revolute s=1 · 11,271° (best my_grasp success)",
                    "tag": "success · revolute hinge · seed 10000",
                    "success": True,
                    "caption": (
                        "Best verified my_grasp clip: proportional-physics C at "
                        "mass-scale 1, net-angle success. Hinge, not free-tilt "
                        "tip-connect. T00 at s=400 still fails."
                    ),
                },
                {
                    "id": "my_grasp_t00",
                    "url": "/media/my_grasp_t00.mp4",
                    "path": MY_GRASP_T00_VIDEO,
                    "label": "T00 tip-connect cinematic (190° prefix, not a pass)",
                    "tag": "not a T00 pass · seed 6 loop",
                    "success": False,
                    "caption": (
                        "my_grasp tip-connect at s=400: cinematic of the non-tilt "
                        "prefix (~190°, tilt ≤7.4°), then the policy dies axis_tilt."
                    ),
                },
            ],
            "palm_down": [
                {
                    "id": "palm_down_60s",
                    "url": "/media/palm_down_60s.mp4",
                    "path": PALM_DOWN_60S,
                    "label": "Bounded 60 s · 10.36 turns (best palm_down)",
                    "tag": "success · tip-connect s=1 · seed 4000",
                    "success": True,
                    "caption": (
                        "Independent EXP-20260918-001: obs[37] clipped to 3 turns, "
                        "10.36 mean turns, max tilt 2.97°, tip <1 mm, 2-contact gait."
                    ),
                },
                {
                    "id": "palm_down_20s",
                    "url": "/media/palm_down_20s.mp4",
                    "path": PALM_DOWN_20S,
                    "label": "Original 20 s · 3.28 turns",
                    "tag": "success · seed 7000",
                    "success": True,
                    "caption": (
                        "Independent 20 s original protocol, 8/8 on new seeds. "
                        "Same 300k CPU policy, no observation clip."
                    ),
                },
            ],
        }
        by_url = {}
        for group in videos.values():
            for item in group:
                by_url[str(item["url"])] = item
        return {"videos": videos, "by_url": by_url}

    def close(self) -> None:
        with self.lock:
            for slot in self.scenes.values():
                slot.close()

    @staticmethod
    def _finite_vector(value: object, name: str, length: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {length} finite numbers")
        return array

    def _contact_count(self, slot: SceneSlot) -> int:
        forces = slot.env._touch()
        return int(np.count_nonzero(np.asarray(forces) > 0.05))

    def _scene_state(self, name: str) -> dict[str, object]:
        slot = self.scenes[name]
        pos = slot.translation_m()
        return {
            "translation_mm": (pos * 1000.0).tolist(),
            "native_mm": (slot.native_pos * 1000.0).tolist(),
            "contact_count": self._contact_count(slot),
        }

    def state(self) -> dict[str, object]:
        with self.lock:
            my = self.scenes["my_grasp"].translation_m()
            pd = self.scenes["palm_down"].translation_m()
            delta = (my - pd) * 1000.0
            videos = {
                key: [
                    {k: item[k] for k in ("id", "url", "label", "tag", "success", "caption")}
                    for item in group
                ]
                for key, group in self._media["videos"].items()
            }
            return {
                "scenes": {
                    "my_grasp": self._scene_state("my_grasp"),
                    "palm_down": self._scene_state("palm_down"),
                },
                "native_delta_mm": (
                    (self.my_grasp_native - self.palm_down_native) * 1000.0
                ).tolist(),
                "native_delta_norm_mm": float(
                    np.linalg.norm(self.my_grasp_native - self.palm_down_native) * 1000.0
                ),
                "live_delta_mm": delta.tolist(),
                "live_delta_norm_mm": float(np.linalg.norm(delta)),
                "blend": float(self.blend),
                "camera": {
                    "azimuth": float(self.camera.azimuth),
                    "elevation": float(self.camera.elevation),
                    "distance": float(self.camera.distance),
                    "lookat": self.camera.lookat.tolist(),
                },
                "videos": videos,
                "my_grasp_sha256": self.my_grasp_sha256,
            }

    def set_translation(self, scene: str, translation_mm: object) -> dict[str, object]:
        if scene not in self.scenes:
            raise ValueError("scene must be 'my_grasp' or 'palm_down'")
        pos = self._finite_vector(translation_mm, "translation_mm", 3) / 1000.0
        if np.any(np.abs(pos) > 2.0):
            raise ValueError("translation must stay within ±2000 mm")
        with self.lock:
            self.scenes[scene].set_translation(pos)
            if scene == "palm_down":
                span = self.my_grasp_native - self.palm_down_native
                denom = float(np.dot(span, span))
                if denom > 1e-12:
                    self.blend = float(np.clip(np.dot(pos - self.palm_down_native, span) / denom, 0.0, 1.0))
            return self.state()

    def reset_native(self) -> dict[str, object]:
        with self.lock:
            for slot in self.scenes.values():
                slot.set_translation(slot.native_pos)
            self.blend = 0.0
            return self.state()

    def copy_translation(self, source: str, dest: str) -> dict[str, object]:
        if source not in self.scenes or dest not in self.scenes:
            raise ValueError("from/to must be 'my_grasp' or 'palm_down'")
        with self.lock:
            pos = self.scenes[source].translation_m()
            self.scenes[dest].set_translation(pos)
            if dest == "palm_down":
                span = self.my_grasp_native - self.palm_down_native
                denom = float(np.dot(span, span))
                if denom > 1e-12:
                    self.blend = float(
                        np.clip(np.dot(pos - self.palm_down_native, span) / denom, 0.0, 1.0)
                    )
            return self.state()

    def set_blend(self, t: object) -> dict[str, object]:
        value = float(t)
        if not np.isfinite(value):
            raise ValueError("blend t must be finite")
        value = float(np.clip(value, 0.0, 1.0))
        pos = (1.0 - value) * self.palm_down_native + value * self.my_grasp_native
        with self.lock:
            self.blend = value
            self.scenes["palm_down"].set_translation(pos)
            return self.state()

    def settle(self, steps: int = 80) -> dict[str, object]:
        if steps < 1 or steps > 5000:
            raise ValueError("settle steps must be between 1 and 5000")
        with self.lock:
            for slot in self.scenes.values():
                slot.env.reset(seed=0)
                slot.data.ctrl[:] = slot.env._grasp_qpos
                for _ in range(steps):
                    mujoco.mj_step(slot.model, slot.data)
            return self.state()

    def move_camera(self, action: str, dx: object, dy: object) -> dict[str, object]:
        mapping = {
            "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
            "pan": mujoco.mjtMouse.mjMOUSE_MOVE_H,
            "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
        }
        if action not in mapping:
            raise ValueError("camera action must be rotate, pan, or zoom")
        reldx = float(dx)
        reldy = float(dy)
        if not np.isfinite(reldx) or not np.isfinite(reldy):
            raise ValueError("camera deltas must be finite")
        with self.lock:
            mujoco.mjv_moveCamera(
                self.scenes["my_grasp"].model,
                mapping[action],
                reldx,
                reldy,
                self.camera,
            )
            self.camera.elevation = float(np.clip(self.camera.elevation, -89.0, 89.0))
            self.camera.distance = float(np.clip(self.camera.distance, 0.08, 3.0))
            return self.state()

    def reset_camera(self) -> dict[str, object]:
        with self.lock:
            self.camera = _default_camera()
            return self.state()

    def _ensure_renderer(self, slot: SceneSlot) -> mujoco.Renderer:
        if slot.renderer is None:
            slot.renderer = mujoco.Renderer(
                slot.model, height=self.height, width=self.width
            )
        return slot.renderer

    def _overlay_markers(self, slot: SceneSlot) -> None:
        scene = slot.renderer.scene
        my = self.scenes["my_grasp"].model.body_pos[self.scenes["my_grasp"].palm_id]
        pd = self.scenes["palm_down"].model.body_pos[self.scenes["palm_down"].palm_id]
        _add_marker(scene, my, (0.91, 0.65, 0.29, 1.0), radius=0.008)
        _add_marker(scene, pd, (0.36, 0.72, 0.63, 1.0), radius=0.008)
        _add_connector(scene, my, pd, (0.91, 0.78, 0.45, 0.85), width=0.002)
        current = slot.model.body_pos[slot.palm_id]
        _add_marker(scene, current, (1.0, 1.0, 1.0, 0.55), radius=0.012)
        rod = slot.env.rod_body
        axis = slot.data.xmat[rod].reshape(3, 3)[:, 0]
        bottom = slot.data.xpos[rod] - 0.07 * axis
        top = slot.data.xpos[rod] + 0.07 * axis
        _add_marker(scene, top, (0.12, 0.95, 0.28, 1.0), radius=0.005)
        _add_marker(scene, bottom, (0.15, 0.55, 1.0, 1.0), radius=0.005)

    def _render_panel(self, name: str) -> np.ndarray:
        slot = self.scenes[name]
        renderer = self._ensure_renderer(slot)
        renderer.update_scene(slot.data, camera=self.camera)
        self._overlay_markers(slot)
        return renderer.render().copy()

    def render_frame(self) -> np.ndarray:
        with self.lock:
            left = self._render_panel("my_grasp")
            right = self._render_panel("palm_down")
            divider = np.full((self.height, 4, 3), (232, 165, 75), dtype=np.uint8)
            return np.concatenate([left, divider, right], axis=1)

    def render_jpeg(self) -> bytes:
        return _encode_rgb_jpeg(self.render_frame())

    def render_png(self) -> bytes:
        return _encode_rgb_png(self.render_frame())

    def media_bytes(self, url_path: str) -> tuple[bytes, str]:
        item = self._media["by_url"].get(url_path)
        if item is None:
            raise FileNotFoundError(url_path)
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if url_path not in self._media_bytes:
            self._media_bytes[url_path] = path.read_bytes()
        return self._media_bytes[url_path], "video/mp4"

    def page_html(self) -> bytes:
        path = PAGE_DIR / "index.html"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path.read_bytes()


def _send_bytes(
    handler: BaseHTTPRequestHandler,
    payload: bytes,
    content_type: str,
    *,
    status: int = HTTPStatus.OK,
    extra: dict[str, str] | None = None,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Accept-Ranges", "bytes")
    if extra:
        for key, value in extra.items():
            handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(payload)


def _send_video(handler: BaseHTTPRequestHandler, data: bytes) -> None:
    length = len(data)
    range_header = handler.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        spec = range_header.split("=", 1)[1].split(",", 1)[0]
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else length - 1
        if start < 0 or start >= length or end < start:
            handler.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        end = min(end, length - 1)
        chunk = data[start : end + 1]
        _send_bytes(
            handler,
            chunk,
            "video/mp4",
            status=HTTPStatus.PARTIAL_CONTENT,
            extra={"Content-Range": f"bytes {start}-{end}/{length}"},
        )
        return
    _send_bytes(handler, data, "video/mp4")


def make_handler(board: PalmCompareBoard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PalmCompareWeb/1"

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            _send_bytes(self, body, "application/json", status=status)

        def _error(self, status: int, exc: Exception | str) -> None:
            self._json(status, {"error": str(exc)})

        def _body(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("request body is too large")
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
                    _send_bytes(self, board.page_html(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(HTTPStatus.OK, board.state())
                elif path == "/api/render.jpg":
                    _send_bytes(self, board.render_jpeg(), "image/jpeg")
                elif path == "/api/render.png":
                    _send_bytes(self, board.render_png(), "image/png")
                elif path.startswith("/media/"):
                    data, _ = board.media_bytes(path)
                    _send_video(self, data)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
            except FileNotFoundError as exc:
                self._error(HTTPStatus.NOT_FOUND, exc)
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/translation":
                    result = board.set_translation(
                        str(body.get("scene", "")), body.get("translation_mm")
                    )
                elif path == "/api/native":
                    result = board.reset_native()
                elif path == "/api/copy":
                    result = board.copy_translation(
                        str(body.get("from", "")), str(body.get("to", ""))
                    )
                elif path == "/api/blend":
                    result = board.set_blend(body.get("t", 0.0))
                elif path == "/api/settle":
                    result = board.settle(int(body.get("steps", 80)))
                elif path == "/api/camera/move":
                    result = board.move_camera(
                        str(body.get("action", "")),
                        body.get("dx", 0.0),
                        body.get("dy", 0.0),
                    )
                elif path == "/api/camera/reset":
                    result = board.reset_camera()
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                self._json(HTTPStatus.OK, result)
            except (ValueError, TypeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, exc)
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    return Handler


def _port_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare my_grasp vs palm_down palm translations in a local Web UI"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768, help="Use 0 for an OS-selected port")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("this local viewer binds only to 127.0.0.1/localhost")
    port = args.port
    if port < 0 or port > 65535:
        parser.error("--port must be between 0 and 65535")
    if port and not _port_available("127.0.0.1", port):
        parser.error(f"port {port} is unavailable; pass --port 0 to choose a free port")
    board = PalmCompareBoard()
    server = HTTPServer(("127.0.0.1", port), make_handler(board))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Palm translation compare UI: {url}", flush=True)
    print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL', 'default')}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
