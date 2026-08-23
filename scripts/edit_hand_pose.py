#!/usr/bin/env python3
"""Interactively edit and save the rigid Allegro palm-root transform."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from allegro_rod_mvp.hand_pose import (
    load_hand_pose,
    make_hand_pose,
    model_variant_for_physics,
    quat_multiply,
    quat_normalize,
    quat_to_euler_xyz_degrees,
    write_hand_pose,
)
from allegro_rod_mvp import RodRotationEnv


CONTROLS = """
Rigid Allegro hand pose controls (viewer window must have focus)
  W / S   world X + / -       A / D   world Y + / -
  E / C   world Z + / -       I / K   world roll + / -
  J / L   world pitch + / -   U / O   world yaw + / -
  F       toggle fine/coarse increments
  R       reset to editor starting pose
  P       print current transform
  V       save JSON                 H       print this help
  Q / Esc quit
Mouse controls remain MuJoCo defaults (orbit, pan, zoom).
"""


def _display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _delta_quaternion(axis: int, radians: float) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    quat[0] = np.cos(radians / 2.0)
    quat[axis + 1] = np.sin(radians / 2.0)
    return quat


def _print_transform(pos: np.ndarray, quat: np.ndarray, mode: str) -> None:
    euler = quat_to_euler_xyz_degrees(quat)
    print(
        "\n"
        f"  mode={mode}\n"
        f"  translation_m=[{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}]\n"
        f"  quaternion_wxyz=[{quat[0]:+.8f}, {quat[1]:+.8f}, "
        f"{quat[2]:+.8f}, {quat[3]:+.8f}]\n"
        f"  euler_xyz_deg=[{euler[0]:+.3f}, {euler[1]:+.3f}, {euler[2]:+.3f}]",
        flush=True,
    )


def _add_marker(scene: mujoco.MjvScene, pos: np.ndarray, rgba: tuple[float, ...]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.006, 0.006, 0.006]),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_markers(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    physics: str,
    tip_anchor: str,
) -> None:
    scene.ngeom = 0
    rod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod")
    axis = data.xmat[rod].reshape(3, 3)[:, 0]
    endpoints = [data.xpos[rod] - 0.07 * axis, data.xpos[rod] + 0.07 * axis]
    bottom, top = sorted(endpoints, key=lambda point: point[2])
    _add_marker(scene, top, (0.1, 0.9, 0.2, 1.0))
    _add_marker(scene, bottom, (0.1, 0.7, 1.0, 1.0))
    if physics == "revolute":
        hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rod_hinge")
        anchor = data.xanchor[hinge]
        color = (1.0, 0.85, 0.05, 1.0)
    else:
        anchor = np.array([0.0, -0.05, 0.07 if tip_anchor == "top" else -0.07])
        color = (1.0, 0.1, 0.8, 1.0)
    _add_marker(scene, anchor, color)


def _default_output() -> Path:
    return ROOT / "configs" / "hand_poses" / time.strftime("hand_pose_%Y%m%d-%H%M%S.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Edit the complete Allegro palm subtree rigidly in MuJoCo"
    )
    parser.add_argument("--physics", choices=["revolute", "tip_connect"], default="revolute")
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--load", type=Path, default=None, help="Existing pose JSON to edit")
    parser.add_argument("--output", type=Path, default=None, help="New pose JSON path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fine-translation", type=float, default=0.001, metavar="METERS")
    parser.add_argument("--coarse-translation", type=float, default=0.01, metavar="METERS")
    parser.add_argument("--fine-rotation", type=float, default=1.0, metavar="DEGREES")
    parser.add_argument("--coarse-rotation", type=float, default=10.0, metavar="DEGREES")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if not _display_available():
        parser.error(
            "no graphical display detected. Use SSH X11 forwarding (`ssh -X host`), "
            "run inside a desktop session, or use "
            "`xvfb-run -a python scripts/edit_hand_pose.py ...`."
        )
    for name in (
        "fine_translation",
        "coarse_translation",
        "fine_rotation",
        "coarse_rotation",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    model_variant = model_variant_for_physics(args.physics)
    xml_path = ROOT / "models" / f"{model_variant}.xml"
    baseline_model = mujoco.MjModel.from_xml_path(str(xml_path))
    palm = mujoco.mj_name2id(baseline_model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if palm < 0:
        raise RuntimeError("loaded scene has no palm body")
    default_pos = baseline_model.body_pos[palm].copy()
    default_quat = quat_normalize(baseline_model.body_quat[palm].copy())
    source_pose: dict[str, object] = {
        "type": "model_default",
        "model": str(xml_path.relative_to(ROOT)),
        "translation": default_pos.tolist(),
        "quaternion_wxyz": default_quat.tolist(),
    }
    if args.load is not None:
        loaded, loaded_path, loaded_hash = load_hand_pose(args.load, model_variant)
        source_pose = {
            "type": "loaded_config",
            "path": str(loaded_path),
            "sha256": loaded_hash,
            "translation": loaded["translation"],
            "quaternion_wxyz": loaded["quaternion_wxyz"],
            "model_default": {
                "translation": default_pos.tolist(),
                "quaternion_wxyz": default_quat.tolist(),
            },
        }

    env = RodRotationEnv(
        hand_model="allegro",
        physics_mode=args.physics,
        tip_anchor=args.tip_anchor,
        hand_pose_config=str(args.load) if args.load is not None else None,
        reset_joint_noise=0.0,
        grasp_ramp_steps=1,
        grasp_hold_steps=0,
    )
    model = env.model
    data = env.data
    start_pos = model.body_pos[palm].copy()
    start_quat = quat_normalize(model.body_quat[palm].copy())
    mujoco.mj_forward(model, data)
    commands: queue.SimpleQueue[int] = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        commands.put(keycode)

    output = args.output or _default_output()
    coarse = False
    should_quit = False
    print(CONTROLS)
    print("Markers: green=rod top, cyan=rod bottom, yellow=revolute anchor, magenta=connect anchor")
    print(f"Scene: {model_variant}; tip anchor: {args.tip_anchor}; output: {output}")
    _print_transform(model.body_pos[palm], model.body_quat[palm], "fine")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not should_quit:
            changed = False
            while not commands.empty():
                key = commands.get()
                char = chr(key).lower() if 0 <= key < 128 else ""
                translation_step = (
                    args.coarse_translation if coarse else args.fine_translation
                )
                rotation_step = np.radians(
                    args.coarse_rotation if coarse else args.fine_rotation
                )
                translations = {
                    "w": (0, 1.0), "s": (0, -1.0),
                    "a": (1, 1.0), "d": (1, -1.0),
                    "e": (2, 1.0), "c": (2, -1.0),
                }
                rotations = {
                    "i": (0, 1.0), "k": (0, -1.0),
                    "j": (1, 1.0), "l": (1, -1.0),
                    "u": (2, 1.0), "o": (2, -1.0),
                }
                if char in translations:
                    axis, sign = translations[char]
                    model.body_pos[palm, axis] += sign * translation_step
                    changed = True
                elif char in rotations:
                    axis, sign = rotations[char]
                    delta = _delta_quaternion(axis, sign * rotation_step)
                    model.body_quat[palm] = quat_normalize(
                        quat_multiply(delta, model.body_quat[palm])
                    )
                    changed = True
                elif char == "f":
                    coarse = not coarse
                    print(f"\nIncrement mode: {'COARSE' if coarse else 'fine'}", flush=True)
                elif char == "r":
                    model.body_pos[palm] = start_pos
                    model.body_quat[palm] = start_quat
                    changed = True
                    print("\nReset to editor starting pose.", flush=True)
                elif char == "p":
                    _print_transform(
                        model.body_pos[palm],
                        model.body_quat[palm],
                        "coarse" if coarse else "fine",
                    )
                elif char == "h":
                    print(CONTROLS)
                elif char == "v":
                    content = make_hand_pose(
                        model.body_pos[palm],
                        model.body_quat[palm],
                        model_variant=model_variant,
                        source_pose=source_pose,
                        notes=args.notes,
                    )
                    try:
                        saved = write_hand_pose(output, content, overwrite=args.overwrite)
                    except FileExistsError as exc:
                        print(f"\nSAVE REFUSED: {exc}", file=sys.stderr, flush=True)
                    else:
                        print(f"\nSaved hand pose: {saved}", flush=True)
                elif char == "q" or key == 256:
                    should_quit = True
            if changed:
                mujoco.mj_forward(model, data)
                _print_transform(
                    model.body_pos[palm],
                    model.body_quat[palm],
                    "coarse" if coarse else "fine",
                )
            with viewer.lock():
                _update_markers(
                    viewer.user_scn, model, data, args.physics, args.tip_anchor
                )
            viewer.sync()
            time.sleep(0.01)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
