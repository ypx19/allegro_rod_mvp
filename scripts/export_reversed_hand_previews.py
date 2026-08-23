#!/usr/bin/env python3
"""Export static, reversible previews of the Allegro hand flipped about world X."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PIVOT = np.array([0.0, -0.05, 0.0], dtype=np.float64)
WORLD_X_180 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
ROD_HALF_LENGTH_M = 0.07
GRASP_QPOS = np.array(
    [
        -0.14000000, 0.81227255, 0.83463606, 0.71621797,
         0.17242762, 0.93530435, 0.88562056, 0.71457097,
         0.93765753, 0.51544830, 0.98571988, 0.80006279,
    ],
    dtype=np.float64,
)


def _numbers(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _format(values: np.ndarray) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotate_x_180(vector: np.ndarray) -> np.ndarray:
    return np.array([vector[0], -vector[1], -vector[2]], dtype=np.float64)


def _add_world_axes(worldbody: ET.Element) -> None:
    origin = np.array([-0.115, -0.115, -0.115])
    axes = (
        ("x", np.array([0.055, 0.0, 0.0]), "1 0.15 0.15 1"),
        ("y", np.array([0.0, 0.055, 0.0]), "0.15 1 0.15 1"),
        ("z", np.array([0.0, 0.0, 0.055]), "0.15 0.35 1 1"),
    )
    for name, delta, rgba in axes:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"preview_world_{name}",
                "type": "capsule",
                "fromto": f"{_format(origin)} {_format(origin + delta)}",
                "size": "0.0025",
                "rgba": rgba,
                "density": "0",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _add_anchor_marker(worldbody: ET.Element, mode: str) -> None:
    anchor = np.array([0.0, -0.05, -0.07])
    if mode == "revolute":
        color = "1 1 0.02 1"
        prefix = "revolute"
        offsets = (
            np.array([0.016, 0.0, 0.0]),
            np.array([-0.016, 0.0, 0.0]),
            np.array([0.0, 0.016, 0.0]),
            np.array([0.0, -0.016, 0.0]),
        )
    else:
        color = "1 0.02 1 1"
        prefix = "point_connect"
        offsets = (
            np.array([0.014, 0.0, 0.014]),
            np.array([-0.014, 0.0, 0.014]),
            np.array([0.014, 0.0, -0.014]),
            np.array([-0.014, 0.0, -0.014]),
        )
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": f"preview_{prefix}_anchor",
            "type": "sphere",
            "pos": _format(anchor),
            "size": "0.011",
            "rgba": color,
        },
    )
    for index, offset in enumerate(offsets):
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": f"preview_{prefix}_satellite_{index}",
                "type": "sphere",
                "pos": _format(anchor + offset),
                "size": "0.0045",
                "rgba": color,
            },
        )


def _write_preview_model(
    source: Path,
    output: Path,
    mode: str,
    requested_clearance_m: float,
    horizontal_translation_xy_m: np.ndarray,
) -> dict[str, object]:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{source}: missing worldbody")
    palm = worldbody.find("./body[@name='palm']")
    if palm is None:
        raise ValueError(f"{source}: missing palm root")

    old_pos = _numbers(palm.attrib["pos"])
    old_quat = _numbers(palm.attrib["quat"])
    old_quat /= np.linalg.norm(old_quat)
    flipped_pos = PIVOT + _rotate_x_180(old_pos - PIVOT)
    new_quat = _quat_mul(WORLD_X_180, old_quat)
    new_quat /= np.linalg.norm(new_quat)
    palm_geom = palm.find("./geom[@name='palm_collision']")
    if palm_geom is None or palm_geom.attrib.get("type") != "box":
        raise ValueError(f"{source}: palm_collision must be a direct box geom")
    geom_pos = _numbers(palm_geom.attrib.get("pos", "0 0 0"))
    half_size = _numbers(palm_geom.attrib["size"])
    rotation = _quat_to_matrix(new_quat)
    flipped_geom_center = flipped_pos + rotation @ geom_pos
    flipped_bottom_z = float(
        flipped_geom_center[2] - np.sum(np.abs(rotation[2, :]) * half_size)
    )
    rod_top_z = ROD_HALF_LENGTH_M
    target_palm_bottom_z = rod_top_z + requested_clearance_m
    delta_z = target_palm_bottom_z - flipped_bottom_z
    new_pos = flipped_pos + np.array(
        [
            horizontal_translation_xy_m[0],
            horizontal_translation_xy_m[1],
            delta_z,
        ]
    )
    palm.set("pos", _format(new_pos))
    palm.set("quat", _format(new_quat))

    rod_tip = worldbody.find(".//site[@name='rod_tip']")
    if rod_tip is None:
        raise ValueError(f"{source}: missing rod_tip")
    rod_tip.set("pos", "0.07 0 0")
    tip_target = worldbody.find("./site[@name='tip_target']")
    if tip_target is not None:
        tip_target.set("pos", "0 -0.05 -0.07")
    if mode == "revolute":
        rod_mount = worldbody.find("./body[@name='rod_mount']")
        hinge = worldbody.find(".//joint[@name='rod_hinge']")
        if rod_mount is None or hinge is None:
            raise ValueError(f"{source}: missing revolute mount")
        rod_mount.set("pos", "0 -0.05 0")
        hinge.set("pos", "0.07 0 0")
    else:
        connect = root.find("./equality/connect[@name='tip_anchor']")
        if connect is None:
            raise ValueError(f"{source}: missing point-connect equality")
        connect.set("anchor", "0.07 0 0")

    _add_world_axes(worldbody)
    _add_anchor_marker(worldbody, mode)
    tree.write(output, encoding="unicode", xml_declaration=False)
    return {
        "source_palm_pos": old_pos.tolist(),
        "source_palm_quat_normalized_wxyz": old_quat.tolist(),
        "flipped_palm_pos_before_clearance": flipped_pos.tolist(),
        "preview_palm_pos": new_pos.tolist(),
        "preview_palm_quat_wxyz": new_quat.tolist(),
        "additional_world_z_translation_m": float(delta_z),
        "additional_world_xy_translation_m": horizontal_translation_xy_m.tolist(),
        "additional_world_translation_m": [
            float(horizontal_translation_xy_m[0]),
            float(horizontal_translation_xy_m[1]),
            float(delta_z),
        ],
        "calculated_flipped_palm_bottom_z_m": flipped_bottom_z,
        "target_palm_bottom_z_m": target_palm_bottom_z,
        "rod_top_tip_z_m": rod_top_z,
        "requested_clearance_m": requested_clearance_m,
    }


def _joint_addresses(model: mujoco.MjModel) -> np.ndarray:
    addresses = []
    for finger in range(3):
        for joint in range(4):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"f{finger}_j{joint}"
            )
            if joint_id < 0:
                raise ValueError("Preview model does not contain the expected 12 hand joints")
            addresses.append(model.jnt_qposadr[joint_id])
    return np.asarray(addresses, dtype=np.int32)


def _contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    rod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom")
    tips = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip{finger}")
        for finger in range(3)
    ]
    distances = []
    for tip in tips:
        distances.append(float(mujoco.mj_geomDistance(model, data, tip, rod, 1.0, None)))
    forces = np.zeros(3, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        for finger, tip in enumerate(tips):
            if {contact.geom1, contact.geom2} == {tip, rod}:
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                forces[finger] += abs(float(contact_force[0]))
    return {
        "signed_tip_to_rod_distance_m": distances,
        "geometric_contact_count": int(np.sum(np.asarray(distances) <= 0.0)),
        "normal_contact_force_n": forces.tolist(),
        "force_contact_count": int(np.sum(forces > 0.05)),
    }


def _load_pose(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = _joint_addresses(model)
    data.qpos[addresses] = GRASP_QPOS
    data.ctrl[:] = GRASP_QPOS
    mujoco.mj_forward(model, data)
    return model, data, addresses


def _clearance_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    palm_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "palm_collision"
    )
    rod_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod")
    if palm_geom < 0 or rod_body < 0:
        raise ValueError("Preview model is missing palm_collision or rod")
    palm_rotation = data.geom_xmat[palm_geom].reshape(3, 3)
    palm_half_size = model.geom_size[palm_geom]
    palm_bottom_z = float(
        data.geom_xpos[palm_geom, 2]
        - np.sum(np.abs(palm_rotation[2, :]) * palm_half_size)
    )
    rod_local_x_world = data.xmat[rod_body].reshape(3, 3)[:, 0]
    endpoint_a_z = float(data.xpos[rod_body, 2] + ROD_HALF_LENGTH_M * rod_local_x_world[2])
    endpoint_b_z = float(data.xpos[rod_body, 2] - ROD_HALF_LENGTH_M * rod_local_x_world[2])
    rod_top_z = max(endpoint_a_z, endpoint_b_z)
    return {
        "palm_bottom_z_m": palm_bottom_z,
        "rod_top_tip_z_m": rod_top_z,
        "clearance_m": palm_bottom_z - rod_top_z,
    }


def _thumb_root_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, object]:
    thumb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger2")
    rod = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rod")
    if thumb < 0 or rod < 0:
        raise ValueError("Preview model is missing finger2 or rod")
    thumb_xy = np.asarray(data.xpos[thumb, :2], dtype=np.float64)
    rod_xy = np.asarray(data.xpos[rod, :2], dtype=np.float64)
    radial = thumb_xy - rod_xy
    return {
        "thumb_root_world_xy_m": thumb_xy.tolist(),
        "rod_axis_world_xy_m": rod_xy.tolist(),
        "thumb_root_to_rod_axis_xy_vector_m": radial.tolist(),
        "thumb_root_to_rod_axis_distance_m": float(np.linalg.norm(radial)),
    }


def _horizontal_translation_from_reference(
    reference_model_path: Path,
    closer_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    model, data, _ = _load_pose(reference_model_path)
    before = _thumb_root_metrics(model, data)
    radial = np.asarray(before["thumb_root_to_rod_axis_xy_vector_m"], dtype=np.float64)
    distance = float(before["thumb_root_to_rod_axis_distance_m"])
    if closer_m < 0.0 or closer_m >= distance:
        raise ValueError(
            f"thumb-root closer distance must be in [0, {distance:.6f}) m"
        )
    translation = -closer_m * radial / distance
    return translation, before


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _overlay(
    frame: np.ndarray,
    mode: str,
    view_name: str,
    clearance: dict[str, float],
    thumb_root: dict[str, object],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    constraint = "BOTTOM REVOLUTE" if mode == "revolute" else "BOTTOM POINT CONNECT"
    lines = [
        "ORIENTATION / CONSTRAINT PREVIEW — NO POLICY",
        "180 deg about world X",
        f"{constraint} | view: {view_name}",
        (
            f"palm bottom Z {clearance['palm_bottom_z_m']:.3f} m | "
            f"rod top Z {clearance['rod_top_tip_z_m']:.3f} m"
        ),
        f"clearance {clearance['clearance_m'] * 1000.0:.1f} mm",
        (
            "thumb-root to rod axis "
            f"{float(thumb_root['thumb_root_to_rod_axis_distance_m']) * 1000.0:.1f} mm"
        ),
    ]
    draw.rectangle((5, 5, 635, 163), fill=(0, 0, 0, 190))
    for index, line in enumerate(lines):
        draw.text((12, 10 + 25 * index), line, font=_font(18), fill=(255, 255, 255, 255))
    return np.asarray(image)


def _render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output: Path,
    mode: str,
    fps: int,
    seconds: float,
    clearance: dict[str, float],
    thumb_root: dict[str, object],
) -> dict[str, object]:
    renderer = mujoco.Renderer(model, height=480, width=640)
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    frame_count = int(round(fps * seconds))
    marker_pixels: list[int] = []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, -0.05, -0.005])
    camera.distance = 0.42
    try:
        for frame_index in range(frame_count):
            phase = frame_index / max(frame_count - 1, 1)
            camera.azimuth = 35.0 + 300.0 * phase
            camera.elevation = -18.0 + 22.0 * np.sin(2.0 * np.pi * phase)
            view_name = (
                "front/side orbit" if phase < 0.5 else "rear/side orbit"
            )
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            if mode == "revolute":
                mask = (
                    (frame[:, :, 0] > 150)
                    & (frame[:, :, 1] > 150)
                    & (frame[:, :, 2] < 100)
                )
            else:
                mask = (
                    (frame[:, :, 0] > 150)
                    & (frame[:, :, 1] < 100)
                    & (frame[:, :, 2] > 150)
                )
            marker_pixels.append(int(np.sum(mask)))
            writer.append_data(
                _overlay(frame, mode, view_name, clearance, thumb_root)
            )
    finally:
        writer.close()
        renderer.close()
    return {
        "frame_count": frame_count,
        "marker_color_pixels_min": min(marker_pixels),
        "marker_color_pixels_max": max(marker_pixels),
        "marker_visible_every_frame": bool(min(marker_pixels) > 0),
    }


def _verify_video(path: Path) -> dict[str, object]:
    reader = imageio.get_reader(path)
    count = 0
    shape = None
    try:
        metadata = reader.get_meta_data()
        for frame in reader:
            count += 1
            shape = list(frame.shape)
    finally:
        reader.close()
    return {
        "decode_ok": count > 0,
        "decoded_frames": count,
        "frame_shape": shape,
        "fps": float(metadata["fps"]),
        "dimensions": [int(metadata["size"][0]), int(metadata["size"][1])],
        "file_size_bytes": path.stat().st_size,
    }


def _settled_metrics(model_path: Path, steps: int = 100) -> dict[str, object]:
    model, data, addresses = _load_pose(model_path)
    static = _contact_metrics(model, data)
    static_clearance = _clearance_metrics(model, data)
    static_thumb_root = _thumb_root_metrics(model, data)
    for _ in range(steps):
        data.ctrl[:] = GRASP_QPOS
        mujoco.mj_step(model, data)
    settled = _contact_metrics(model, data)
    return {
        "static": static,
        "static_clearance": static_clearance,
        "static_thumb_root": static_thumb_root,
        "after_0_2_seconds": {
            **settled,
            "clearance": _clearance_metrics(model, data),
            "thumb_root": _thumb_root_metrics(model, data),
            "dynamic_state_finite": bool(
                np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            ),
        },
        "hand_qpos": data.qpos[addresses].tolist(),
    }


def _write_index(
    output_dir: Path,
    entries: list[dict[str, object]],
    requested_clearance_m: float,
    horizontal_reference: dict[str, object],
    horizontal_translation_xy_m: np.ndarray,
    thumb_root_closer_m: float,
) -> None:
    lines = [
        "# Reversed Allegro Hand Geometry Previews",
        "",
        "- Purpose: orientation and bottom-constraint inspection only; no policy was loaded.",
        "- Transform: rigid 180° rotation about world X, applied only at the `palm` subtree root.",
        (
            "- Additional transform: rigid world-Z translation of the same subtree "
            f"to request {requested_clearance_m * 1000.0:.1f} mm palm-to-rod-top clearance."
        ),
        (
            "- Horizontal transform: rigid world-XY translation "
            f"`{horizontal_translation_xy_m.tolist()}` m, computed from MuJoCo body positions "
            f"to move the thumb root {thumb_root_closer_m * 1000.0:.1f} mm closer to the rod axis."
        ),
        (
            "- Reference thumb-root distance: "
            f"`{float(horizontal_reference['thumb_root_to_rod_axis_distance_m']) * 1000.0:.6f}` mm."
        ),
        f"- Pivot: `{PIVOT.tolist()}` m (rod center / grasp region).",
        "- Joint frames and all palm-relative child transforms are unchanged.",
        "- World-axis indicator: red X, green Y, blue Z.",
        "",
    ]
    for entry in entries:
        contact = entry["contact_metrics"]
        lines.extend(
            [
                f"## {entry['label']}",
                f"- Video: `{entry['video']}`",
                f"- Preview MJCF: `{entry['preview_model']}`",
                f"- Marker: {entry['marker_description']}",
                (
                    "- Placement: palm-bottom "
                    f"`{contact['static_clearance']['palm_bottom_z_m']:.9f}` m; "
                    f"rod-top `{contact['static_clearance']['rod_top_tip_z_m']:.9f}` m; "
                    f"clearance `{contact['static_clearance']['clearance_m'] * 1000.0:.6f}` mm; "
                    f"applied ΔZ `{entry['transform']['additional_world_z_translation_m']:.9f}` m."
                ),
                (
                    "- Thumb root: before "
                    f"`{float(horizontal_reference['thumb_root_to_rod_axis_distance_m']) * 1000.0:.6f}` mm; "
                    f"after `{float(entry['initial_thumb_root']['thumb_root_to_rod_axis_distance_m']) * 1000.0:.6f}` mm; "
                    f"change `{(float(entry['initial_thumb_root']['thumb_root_to_rod_axis_distance_m']) - float(horizontal_reference['thumb_root_to_rod_axis_distance_m'])) * 1000.0:.6f}` mm."
                ),
                (
                    "- Static signed fingertip distances to rod (m): "
                    f"`{contact['static']['signed_tip_to_rod_distance_m']}`; "
                    f"contacts: {contact['static']['geometric_contact_count']}/3."
                ),
                (
                    "- After 0.2 s signed distances (m): "
                    f"`{contact['after_0_2_seconds']['signed_tip_to_rod_distance_m']}`; "
                    f"geometric contacts: "
                    f"{contact['after_0_2_seconds']['geometric_contact_count']}/3; "
                    f"normal forces (N): "
                    f"`{contact['after_0_2_seconds']['normal_contact_force_n']}`; "
                    f"force contacts: {contact['after_0_2_seconds']['force_contact_count']}/3."
                ),
                "",
            ]
        )
    (output_dir / "INDEX.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--palm-clearance-m", type=float, default=0.010)
    parser.add_argument("--thumb-root-closer-m", type=float, default=0.0)
    parser.add_argument("--horizontal-reference-model", type=Path, default=None)
    args = parser.parse_args()
    if args.palm_clearance_m < 0.0:
        raise ValueError("--palm-clearance-m must be non-negative")
    if args.thumb_root_closer_m < 0.0:
        raise ValueError("--thumb-root-closer-m must be non-negative")
    if args.thumb_root_closer_m > 0.0 and args.horizontal_reference_model is None:
        raise ValueError(
            "--horizontal-reference-model is required when moving the thumb root"
        )
    if args.horizontal_reference_model is not None:
        reference_model_path = args.horizontal_reference_model.resolve()
        horizontal_translation_xy, horizontal_reference = (
            _horizontal_translation_from_reference(
                reference_model_path, args.thumb_root_closer_m
            )
        )
    else:
        reference_model_path = None
        horizontal_translation_xy = np.zeros(2, dtype=np.float64)
        horizontal_reference = {
            "thumb_root_to_rod_axis_distance_m": 0.0,
            "thumb_root_world_xy_m": [],
            "rod_axis_world_xy_m": [],
            "thumb_root_to_rod_axis_xy_vector_m": [],
        }
    clearance_tag = int(round(args.palm_clearance_m * 1000.0))
    closer_tag = int(round(args.thumb_root_closer_m * 1000.0))
    output_dir = args.out_dir or (
        ROOT
        / "runs"
        / "previews"
        / (
            f"reversed_world_x_180_clearance{clearance_tag}mm_"
            f"thumb{closer_tag}mmcloser_{datetime.now():%Y%m%d-%H%M%S}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    specifications = (
        (
            "revolute",
            "A0 reversed — bottom revolute",
            ROOT / "models" / "allegro_three_finger_rod_revolute.xml",
            "reversed_A0_bottom_revolute.xml",
            "reversed_A0_bottom_revolute.mp4",
            "yellow center sphere plus four cardinal satellites",
        ),
        (
            "point_connect",
            "C3 reversed — bottom point connect",
            ROOT / "models" / "allegro_three_finger_rod.xml",
            "reversed_C3_bottom_point_connect.xml",
            "reversed_C3_bottom_point_connect.mp4",
            "magenta center sphere plus four diagonal satellites",
        ),
    )
    entries: list[dict[str, object]] = []
    for mode, label, source, model_name, video_name, marker in specifications:
        preview_model = output_dir / model_name
        transform = _write_preview_model(
            source,
            preview_model,
            mode,
            args.palm_clearance_m,
            horizontal_translation_xy,
        )
        model, data, _ = _load_pose(preview_model)
        clearance = _clearance_metrics(model, data)
        thumb_root = _thumb_root_metrics(model, data)
        video = output_dir / video_name
        visibility = _render_video(
            model,
            data,
            video,
            mode,
            args.fps,
            args.seconds,
            clearance,
            thumb_root,
        )
        entry = {
            "mode": mode,
            "label": label,
            "source_model": str(source.resolve()),
            "preview_model": str(preview_model.resolve()),
            "video": str(video.resolve()),
            "policy": None,
            "pose_source": "validated Allegro grasp qpos; no action rollout",
            "constraint_physics": (
                "A0 bottom revolute"
                if mode == "revolute"
                else "C3 nominal-mass final point-connect parameters"
            ),
            "marker_description": marker,
            "transform": transform,
            "initial_clearance": clearance,
            "initial_thumb_root": thumb_root,
            "contact_metrics": _settled_metrics(preview_model),
            "render_visibility": visibility,
            "video_verification": _verify_video(video),
        }
        if not entry["render_visibility"]["marker_visible_every_frame"]:
            raise RuntimeError(f"Anchor marker was not visible in every frame: {video}")
        entries.append(entry)

    metadata = {
        "artifact_type": "orientation_and_constraint_preview",
        "generated_at": datetime.now().astimezone().isoformat(),
        "claim_policy_success": False,
        "world_transform": {
            "description": "180 deg about world X",
            "quaternion_wxyz": WORLD_X_180.tolist(),
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            "pivot_world_m": PIVOT.tolist(),
            "applied_to": "palm root (complete palm + index + middle + thumb subtree)",
            "unchanged": "all palm-relative child body transforms and joint frames",
            "additional_translation_world_m": [
                float(horizontal_translation_xy[0]),
                float(horizontal_translation_xy[1]),
                entries[0]["transform"]["additional_world_z_translation_m"],
            ],
            "requested_palm_to_rod_top_clearance_m": args.palm_clearance_m,
            "horizontal_reference_model": (
                str(reference_model_path) if reference_model_path is not None else None
            ),
            "horizontal_reference_measurement": horizontal_reference,
            "requested_thumb_root_closer_m": args.thumb_root_closer_m,
            "horizontal_translation_world_xy_m": horizontal_translation_xy.tolist(),
        },
        "render": {
            "fps": args.fps,
            "seconds": args.seconds,
            "dimensions": [640, 480],
            "camera": "300-degree free-camera orbit with varying elevation",
        },
        "entries": entries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_index(
        output_dir,
        entries,
        args.palm_clearance_m,
        horizontal_reference,
        horizontal_translation_xy,
        args.thumb_root_closer_m,
    )
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
