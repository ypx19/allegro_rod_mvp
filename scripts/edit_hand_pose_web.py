#!/usr/bin/env python3
"""Headless local Web UI for editing the rigid Allegro palm-root pose."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import sys
import threading
import webbrowser
from urllib.parse import urlparse
import zlib

# Select a headless backend before importing MuJoCo. Users may override this.
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.hand_pose import (
    apply_hand_pose,
    euler_xyz_degrees_to_quat,
    load_hand_pose,
    make_hand_pose,
    model_variant_for_physics,
    quat_normalize,
    quat_to_euler_xyz_degrees,
    write_hand_pose,
)


POSE_ROOT = (ROOT / "configs" / "hand_poses").resolve()
MAX_BODY_BYTES = 64 * 1024
ROD_HALF_LENGTH_M = 0.07


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


def _safe_pose_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a JSON pose path while confining it to configs/hand_poses."""
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve()
    try:
        resolved.relative_to(POSE_ROOT)
    except ValueError as exc:
        raise ValueError(f"pose path must be inside {POSE_ROOT}") from exc
    if resolved.suffix.lower() != ".json":
        raise ValueError("pose path must end in .json")
    if must_exist and not resolved.is_file():
        raise ValueError(f"pose file does not exist: {resolved}")
    return resolved


def _add_marker(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rgba: tuple[float, float, float, float],
    radius: float = 0.006,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


class PoseEditor:
    """Thread-safe state and MuJoCo rendering backend used by the HTTP API."""

    def __init__(
        self,
        physics: str,
        tip_anchor: str = "bottom",
        load_path: Path | None = None,
        output_path: Path | None = None,
        width: int = 640,
        height: int = 480,
        notes: str = "",
    ) -> None:
        self.lock = threading.RLock()
        self.physics = physics
        self.tip_anchor = tip_anchor
        self.model_variant = model_variant_for_physics(physics)
        self.output_path = _safe_pose_path(
            output_path or POSE_ROOT / "hand_pose_web.json"
        )
        self.notes = notes
        self.env = RodRotationEnv(
            hand_model="allegro",
            physics_mode=physics,
            tip_anchor=tip_anchor,
            hand_pose_config=str(load_path) if load_path else None,
            reset_joint_noise=0.0,
            grasp_ramp_steps=1,
            grasp_hold_steps=0,
        )
        self.model = self.env.model
        self.data = self.env.data
        self.palm_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "palm"
        )
        if self.palm_id < 0:
            raise RuntimeError("scene has no palm body")
        self.model_default_pos, self.model_default_quat = self._model_default_pose()
        if load_path:
            loaded, resolved, digest = load_hand_pose(load_path, self.model_variant)
            self.source_pose: dict[str, object] = {
                "type": "loaded_config",
                "path": str(resolved),
                "sha256": digest,
                "translation": loaded["translation"],
                "quaternion_wxyz": loaded["quaternion_wxyz"],
                "model_default": {
                    "translation": self.model_default_pos.tolist(),
                    "quaternion_wxyz": self.model_default_quat.tolist(),
                },
            }
        else:
            self.source_pose = {
                "type": "model_default",
                "model": f"models/{self.model_variant}.xml",
                "translation": self.model_default_pos.tolist(),
                "quaternion_wxyz": self.model_default_quat.tolist(),
            }
        self.start_pos = self.model.body_pos[self.palm_id].copy()
        self.start_quat = quat_normalize(self.model.body_quat[self.palm_id].copy())
        self.camera = {
            "azimuth": 135.0,
            "elevation": -18.0,
            "distance": 0.42,
            "lookat": [0.0, -0.05, 0.0],
        }
        # EGL contexts are thread-affine. Build the renderer lazily on the
        # server/request thread (or the caller thread in direct backend tests).
        self.renderer: mujoco.Renderer | None = None
        self.width = width
        self.height = height
        self.last_settle_steps = 0
        self._reset_dynamics()

    def _model_default_pose(self) -> tuple[np.ndarray, np.ndarray]:
        xml_path = ROOT / "models" / f"{self.model_variant}.xml"
        baseline = mujoco.MjModel.from_xml_path(str(xml_path))
        palm = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_BODY, "palm")
        return baseline.body_pos[palm].copy(), quat_normalize(
            baseline.body_quat[palm].copy()
        )

    def _reset_dynamics(self) -> None:
        self.env.reset(seed=0)
        mujoco.mj_forward(self.model, self.data)
        self.last_settle_steps = 0

    def close(self) -> None:
        with self.lock:
            if self.renderer is not None:
                self.renderer.close()
            self.env.close()

    def _rod_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rod = self.env.rod_body
        axis = self.data.xmat[rod].reshape(3, 3)[:, 0]
        endpoints = (
            self.data.xpos[rod] - ROD_HALF_LENGTH_M * axis,
            self.data.xpos[rod] + ROD_HALF_LENGTH_M * axis,
        )
        bottom, top = sorted(endpoints, key=lambda point: float(point[2]))
        if self.physics == "revolute":
            anchor = self.data.xanchor[self.env.hinge_id].copy()
        else:
            anchor = np.array(
                [0.0, -0.05, 0.07 if self.tip_anchor == "top" else -0.07],
                dtype=np.float64,
            )
        return np.asarray(bottom), np.asarray(top), anchor

    def _metrics(self) -> dict[str, object]:
        bottom, top, anchor = self._rod_points()
        palm_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision"
        )
        rotation = self.data.geom_xmat[palm_geom].reshape(3, 3)
        palm_bottom_z = float(
            self.data.geom_xpos[palm_geom, 2]
            - np.sum(np.abs(rotation[2, :]) * self.model.geom_size[palm_geom])
        )
        thumb = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "finger2"
        )
        rod_xy = self.data.xpos[self.env.rod_body, :2]
        thumb_radial = float(np.linalg.norm(self.data.xpos[thumb, :2] - rod_xy))
        forces = self.env._touch()  # same contact calculation used by observations
        return {
            "palm_clearance_mm": (palm_bottom_z - float(top[2])) * 1000.0,
            "thumb_root_radial_mm": thumb_radial * 1000.0,
            "contact_count": int(np.count_nonzero(forces > 0.05)),
            "contact_forces_n": forces.tolist(),
            "rod_top_m": top.tolist(),
            "rod_bottom_m": bottom.tolist(),
            "anchor_m": anchor.tolist(),
            "settle_steps": self.last_settle_steps,
        }

    def state(self) -> dict[str, object]:
        with self.lock:
            pos = self.model.body_pos[self.palm_id].copy()
            quat = quat_normalize(self.model.body_quat[self.palm_id].copy())
            return {
                "physics": self.physics,
                "tip_anchor": self.tip_anchor,
                "model_variant": self.model_variant,
                "translation_mm": (pos * 1000.0).tolist(),
                "euler_deg": quat_to_euler_xyz_degrees(quat).tolist(),
                "quaternion_wxyz": quat.tolist(),
                "camera": dict(self.camera),
                "metrics": self._metrics(),
                "output_path": str(self.output_path.relative_to(ROOT)),
                "pose_root": str(POSE_ROOT),
            }

    @staticmethod
    def _finite_vector(value: object, name: str, length: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {length} finite numbers")
        return array

    def update_pose(
        self, translation_mm: object, euler_deg: object
    ) -> dict[str, object]:
        with self.lock:
            pos = self._finite_vector(translation_mm, "translation_mm", 3) / 1000.0
            euler = self._finite_vector(euler_deg, "euler_deg", 3)
            if np.any(np.abs(pos) > 2.0):
                raise ValueError("translation must stay within ±2000 mm")
            if np.any(np.abs(euler) > 3600.0):
                raise ValueError("Euler angles must stay within ±3600 degrees")
            content = make_hand_pose(
                pos,
                euler_xyz_degrees_to_quat(euler),
                model_variant=self.model_variant,
                source_pose=self.source_pose,
                notes=self.notes,
            )
            apply_hand_pose(self.model, content)
            self._reset_dynamics()
            return self.state()

    def update_camera(self, payload: dict[str, object]) -> dict[str, object]:
        with self.lock:
            for key in ("azimuth", "elevation", "distance"):
                if key in payload:
                    value = float(payload[key])
                    if not np.isfinite(value):
                        raise ValueError(f"camera {key} must be finite")
                    self.camera[key] = value
            if "lookat" in payload:
                self.camera["lookat"] = self._finite_vector(
                    payload["lookat"], "camera lookat", 3
                ).tolist()
            self.camera["elevation"] = float(
                np.clip(float(self.camera["elevation"]), -89.0, 89.0)
            )
            self.camera["distance"] = float(
                np.clip(float(self.camera["distance"]), 0.05, 4.0)
            )
            return self.state()

    def reset(self) -> dict[str, object]:
        with self.lock:
            content = make_hand_pose(
                self.start_pos,
                self.start_quat,
                model_variant=self.model_variant,
                source_pose=self.source_pose,
                notes=self.notes,
            )
            apply_hand_pose(self.model, content)
            self._reset_dynamics()
            return self.state()

    def settle(self, steps: int = 100) -> dict[str, object]:
        if steps < 1 or steps > 5000:
            raise ValueError("settle steps must be between 1 and 5000")
        with self.lock:
            self._reset_dynamics()
            self.data.ctrl[:] = self.env._grasp_qpos
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            self.last_settle_steps = steps
            return self.state()

    def load(self, value: str) -> dict[str, object]:
        path = _safe_pose_path(value, must_exist=True)
        loaded, resolved, digest = load_hand_pose(path, self.model_variant)
        with self.lock:
            apply_hand_pose(self.model, loaded)
            self.start_pos = np.asarray(loaded["translation"], dtype=np.float64)
            self.start_quat = np.asarray(loaded["quaternion_wxyz"], dtype=np.float64)
            self.source_pose = {
                "type": "loaded_config",
                "path": str(resolved),
                "sha256": digest,
                "translation": loaded["translation"],
                "quaternion_wxyz": loaded["quaternion_wxyz"],
                "model_default": {
                    "translation": self.model_default_pos.tolist(),
                    "quaternion_wxyz": self.model_default_quat.tolist(),
                },
            }
            self._reset_dynamics()
            return self.state()

    def save(self, value: str, overwrite: bool = False) -> dict[str, object]:
        path = _safe_pose_path(value)
        with self.lock:
            content = make_hand_pose(
                self.model.body_pos[self.palm_id],
                self.model.body_quat[self.palm_id],
                model_variant=self.model_variant,
                source_pose=self.source_pose,
                notes=self.notes,
            )
            saved = write_hand_pose(path, content, overwrite=overwrite)
            _, _, digest = load_hand_pose(saved, self.model_variant)
            return {
                "saved_path": str(saved),
                "sha256": digest,
                "content": content,
            }

    def render_png(self) -> bytes:
        with self.lock:
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
            bottom, top, anchor = self._rod_points()
            _add_marker(self.renderer.scene, top, (0.1, 0.95, 0.2, 1.0))
            _add_marker(self.renderer.scene, bottom, (0.1, 0.7, 1.0, 1.0))
            anchor_color = (
                (1.0, 0.85, 0.05, 1.0)
                if self.physics == "revolute"
                else (1.0, 0.1, 0.8, 1.0)
            )
            _add_marker(self.renderer.scene, anchor, anchor_color, radius=0.007)
            frame = self.renderer.render().copy()
            return _encode_rgb_png(frame)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Allegro Hand Pose Studio</title>
<style>
:root{color-scheme:dark;--bg:#0b0e13;--panel:#141923;--line:#283142;--text:#eef3fa;--muted:#9aa8bb;--accent:#66d9a6;--warn:#ffcf66}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#172131 0,#0b0e13 45%);font:14px Inter,system-ui,sans-serif;color:var(--text)}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}h1{font-size:19px;margin:0}header span{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(480px,1.45fr) minmax(370px,1fr);gap:16px;padding:16px;max-width:1500px;margin:auto}.card{background:rgba(20,25,35,.94);border:1px solid var(--line);border-radius:14px;box-shadow:0 18px 50px #0005;overflow:hidden}
.viewport{padding:12px}.viewport img{width:100%;display:block;background:#050608;border-radius:9px;aspect-ratio:4/3;object-fit:contain}.legend,.metrics{display:flex;gap:12px;flex-wrap:wrap;padding:11px 3px 0;color:var(--muted)}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px}.metrics b{color:var(--text);font-variant-numeric:tabular-nums}
.controls{padding:16px;display:grid;gap:16px}.section{border-top:1px solid var(--line);padding-top:14px}.section:first-child{border:0;padding-top:0}.section h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.row{display:grid;grid-template-columns:66px 1fr 94px;gap:9px;align-items:center;margin:7px 0}.row label{font-weight:650}.row input[type=range]{accent-color:var(--accent);width:100%}input[type=number],input[type=text],select{width:100%;background:#0c1119;color:var(--text);border:1px solid #344056;border-radius:7px;padding:7px 9px}
.buttonrow{display:flex;gap:8px;flex-wrap:wrap}button{background:#273247;color:var(--text);border:1px solid #3a4961;border-radius:8px;padding:8px 12px;font-weight:650;cursor:pointer}button.primary{background:#176b50;border-color:#299873}button.danger{background:#5b3131;border-color:#915050}button:hover{filter:brightness(1.15)}
.fileline{display:grid;grid-template-columns:1fr auto;gap:8px;margin-top:8px}.status{min-height:38px;padding:9px 11px;border-radius:8px;background:#0c1119;color:var(--muted);white-space:pre-wrap}.status.ok{color:#82e8bb}.status.error{color:#ff8e8e}
.step{display:flex;align-items:center;gap:8px}.pill{background:#0c1119;border:1px solid var(--line);padding:5px 8px;border-radius:99px;color:var(--muted)}small{color:var(--muted)}
@media(max-width:950px){main{grid-template-columns:1fr}.controls{grid-row:2}}
</style></head><body>
<header><h1>Allegro Hand Pose Studio</h1><span id="scene"></span></header>
<main><section class="card viewport"><img id="render" alt="Live MuJoCo scene">
<div class="legend"><span><i class="dot" style="background:#20ef50"></i>Rod top</span><span><i class="dot" style="background:#1ab5ff"></i>Rod bottom</span><span id="anchorLegend"></span></div>
<div class="metrics"><span>Clearance <b id="clearance">—</b></span><span>Thumb radial <b id="thumb">—</b></span><span>Tips <b id="contacts">—</b></span><span>Forces <b id="forces">—</b></span></div></section>
<section class="card controls">
<div class="section"><h2>Palm root · world coordinates</h2><div class="step">Increment mode <select id="stepMode"><option value="fine">Fine · 1 mm / 1°</option><option value="coarse">Coarse · 10 mm / 10°</option></select><span class="pill">saved as m + wxyz</span></div><div id="poseRows"></div><div class="buttonrow"><button id="reset">Reset to loaded/start pose</button><button id="settle">Settle + recompute contacts</button></div></div>
<div class="section"><h2>Camera · view only</h2><div id="cameraRows"></div><small>Camera changes are never written to the hand-pose JSON.</small></div>
<div class="section"><h2>Load / save pose</h2><div class="fileline"><input id="path" type="text"><button id="load">Load existing</button></div><div class="buttonrow" style="margin-top:8px"><button class="primary" id="save">Save as new</button><button class="danger" id="overwrite">Overwrite with confirmation</button></div><small>Paths are restricted to configs/hand_poses/*.json.</small></div>
<div id="status" class="status">Connecting…</div></section></main>
<script>
const poseDefs=[['X','translation_mm',0,-500,500,'mm'],['Y','translation_mm',1,-500,500,'mm'],['Z','translation_mm',2,-500,500,'mm'],['Roll','euler_deg',0,-360,360,'°'],['Pitch','euler_deg',1,-360,360,'°'],['Yaw','euler_deg',2,-360,360,'°']];
const camDefs=[['Azimuth','azimuth',-360,720,'°'],['Elevation','elevation',-89,89,'°'],['Distance','distance',.05,1.5,'m'],['Look X','lookat.0',-.5,.5,'m'],['Look Y','lookat.1',-.5,.5,'m'],['Look Z','lookat.2',-.5,.5,'m']];
let state, timer, renderSeq=0;
function row(def,parent,prefix){const [label,key,min,max,unit]=def,id=prefix+key.replace('.','-');parent.insertAdjacentHTML('beforeend',`<div class="row"><label>${label} <small>${unit}</small></label><input id="${id}-r" type="range" min="${min}" max="${max}"><input id="${id}-n" type="number"></div>`);return [document.getElementById(id+'-r'),document.getElementById(id+'-n')]}
const poseInputs=poseDefs.map(d=>[d,...row(d,document.getElementById('poseRows'),'p-')]);const camInputs=camDefs.map(d=>[d,...row(d,document.getElementById('cameraRows'),'c-')]);
function status(message,error=false){const el=document.getElementById('status');el.textContent=message;el.className='status '+(error?'error':'ok')}
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function valueFor(obj,key){return key.includes('.')?obj[key.split('.')[0]][+key.split('.')[1]]:obj[key]}
function refresh(s,inputs=true){state=s;document.getElementById('scene').textContent=`${s.physics} · ${s.tip_anchor} anchor`;document.getElementById('path').value=s.output_path;document.getElementById('anchorLegend').innerHTML=`<i class="dot" style="background:${s.physics==='revolute'?'#ffd90d':'#ff1acc'}"></i>${s.physics==='revolute'?'Revolute anchor':'Point-connect anchor'}`;const m=s.metrics;document.getElementById('clearance').textContent=m.palm_clearance_mm.toFixed(2)+' mm';document.getElementById('thumb').textContent=m.thumb_root_radial_mm.toFixed(2)+' mm';document.getElementById('contacts').textContent=m.contact_count+'/3';document.getElementById('forces').textContent=m.contact_forces_n.map(x=>x.toFixed(2)).join(' / ')+' N';if(inputs){poseInputs.forEach(([d,r,n])=>{const v=s[d[1]][d[2]];r.value=n.value=v;r.step=n.step=d[1]==='translation_mm'?1:1});camInputs.forEach(([d,r,n])=>{const v=valueFor(s.camera,d[1]);r.value=n.value=v;r.step=n.step=d[1]==='distance'?0.01:1})}document.getElementById('render').src='/api/render.png?v='+(++renderSeq)}
function posePayload(){return {translation_mm:state.translation_mm.map((v,i)=>+poseInputs[i][2].value),euler_deg:state.euler_deg.map((v,i)=>+poseInputs[i+3][2].value)}}
function debounce(fn){clearTimeout(timer);timer=setTimeout(fn,90)}
poseInputs.forEach(([d,r,n],i)=>{for(const el of [r,n])el.addEventListener('input',()=>{if(el===r)n.value=r.value;else r.value=n.value;debounce(async()=>{try{refresh(await api('/api/pose',posePayload()),false)}catch(e){status(e.message,true)}})});});
camInputs.forEach(([d,r,n])=>{for(const el of [r,n])el.addEventListener('input',()=>{if(el===r)n.value=r.value;else r.value=n.value;debounce(async()=>{try{const body={};if(d[1].startsWith('lookat')){body.lookat=state.camera.lookat.slice();body.lookat[+d[1].split('.')[1]]=+n.value}else body[d[1]]=+n.value;refresh(await api('/api/camera',body),false)}catch(e){status(e.message,true)}})});});
document.getElementById('stepMode').onchange=e=>{const coarse=e.target.value==='coarse';poseInputs.forEach(([d,r,n])=>r.step=n.step=(d[1]==='translation_mm'?(coarse?10:1):(coarse?10:1)))};
document.getElementById('reset').onclick=async()=>{try{refresh(await api('/api/reset',{}));status('Reset to the editor starting pose.')}catch(e){status(e.message,true)}};
document.getElementById('settle').onclick=async()=>{try{refresh(await api('/api/settle',{steps:100}));status('Settled 100 simulation steps and recomputed contact forces.')}catch(e){status(e.message,true)}};
document.getElementById('load').onclick=async()=>{try{refresh(await api('/api/load',{path:document.getElementById('path').value}));status('Loaded pose; reset now returns to this pose.')}catch(e){status(e.message,true)}};
document.getElementById('save').onclick=async()=>{try{const x=await api('/api/save',{path:document.getElementById('path').value,overwrite:false});status(`Saved ${x.saved_path}\nSHA-256 ${x.sha256}`)}catch(e){status(e.message,true)}};
document.getElementById('overwrite').onclick=async()=>{if(!confirm('Explicitly overwrite this existing pose file? This cannot be undone.'))return;try{const x=await api('/api/save',{path:document.getElementById('path').value,overwrite:true});status(`Overwrote ${x.saved_path}\nSHA-256 ${x.sha256}`)}catch(e){status(e.message,true)}};
(async()=>{try{refresh(await api('/api/state'));status('Ready. Slider updates are debounced by 90 ms.')}catch(e){status(e.message,true)}})();
</script></body></html>"""


def make_handler(editor: PoseEditor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AllegroPoseWeb/1"

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, status: int, payload: object) -> None:
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
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._json(HTTPStatus.OK, editor.state())
                elif path == "/api/render.png":
                    body = editor.render_png()
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
                if path == "/api/pose":
                    result = editor.update_pose(
                        body.get("translation_mm"), body.get("euler_deg")
                    )
                elif path == "/api/camera":
                    result = editor.update_camera(body)
                elif path == "/api/reset":
                    result = editor.reset()
                elif path == "/api/settle":
                    result = editor.settle(int(body.get("steps", 100)))
                elif path == "/api/load":
                    result = editor.load(str(body.get("path", "")))
                elif path == "/api/save":
                    result = editor.save(
                        str(body.get("path", "")),
                        overwrite=body.get("overwrite") is True,
                    )
                else:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                self._json(HTTPStatus.OK, result)
            except (ValueError, FileExistsError, TypeError) as exc:
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
        description="Edit the complete Allegro palm subtree in a headless local Web UI"
    )
    parser.add_argument("--physics", choices=["revolute", "tip_connect"], default="revolute")
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--load", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/hand_poses/hand_pose_web.json")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Use 0 for an OS-selected port")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.host != "127.0.0.1" and args.host != "localhost":
        parser.error("this local editor binds only to 127.0.0.1/localhost")
    try:
        output = _safe_pose_path(args.output)
        load_path = _safe_pose_path(args.load, must_exist=True) if args.load else None
    except ValueError as exc:
        parser.error(str(exc))
    port = args.port
    if port < 0 or port > 65535:
        parser.error("--port must be between 0 and 65535")
    if port and not _port_available("127.0.0.1", port):
        parser.error(f"port {port} is unavailable; pass --port 0 to choose a free port")
    editor = PoseEditor(
        args.physics,
        args.tip_anchor,
        load_path,
        output,
        notes=args.notes,
    )
    # A single request thread is intentional: MuJoCo's EGL context is
    # thread-affine, while the UI debounce prevents a request backlog.
    server = HTTPServer(("127.0.0.1", port), make_handler(editor))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Allegro hand-pose Web UI: {url}", flush=True)
    print(f"Physics: {args.physics}; output: {output}", flush=True)
    print(f"MuJoCo GL backend: {os.environ.get('MUJOCO_GL', 'default')}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()
        editor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
