"""Validated, reproducible Allegro palm-root pose configuration."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


SCHEMA = "allegro_rod_mvp.hand_pose"
SCHEMA_VERSION = 1
MODEL_VARIANTS = (
    "allegro_three_finger_rod",
    "allegro_three_finger_rod_revolute",
)


def model_variant_for_physics(physics_mode: str) -> str:
    return (
        "allegro_three_finger_rod_revolute"
        if physics_mode == "revolute"
        else "allegro_three_finger_rod"
    )


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must have finite, non-zero norm")
    return quat / norm


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo wxyz quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def euler_xyz_degrees_to_quat(euler: np.ndarray) -> np.ndarray:
    """World-axis extrinsic XYZ Euler angles in degrees, returned as wxyz."""
    roll, pitch, yaw = np.radians(np.asarray(euler, dtype=np.float64)) / 2.0
    qx = np.array([np.cos(roll), np.sin(roll), 0.0, 0.0])
    qy = np.array([np.cos(pitch), 0.0, np.sin(pitch), 0.0])
    qz = np.array([np.cos(yaw), 0.0, 0.0, np.sin(yaw)])
    return quat_normalize(quat_multiply(qz, quat_multiply(qy, qx)))


def quat_to_euler_xyz_degrees(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to world-axis extrinsic XYZ Euler degrees."""
    w, x, y, z = quat_normalize(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def _vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be an array of {size} finite numbers")
    return array


def validate_hand_pose(
    content: dict[str, Any], expected_model_variant: str | None = None
) -> dict[str, Any]:
    if not isinstance(content, dict):
        raise ValueError("hand pose config root must be a JSON object")
    if content.get("schema") != SCHEMA or content.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported hand pose schema; expected {SCHEMA!r} version {SCHEMA_VERSION}"
        )
    if content.get("transform_space") != "model":
        raise ValueError("transform_space must be 'model'")
    if content.get("body") != "palm":
        raise ValueError("hand pose body must be 'palm'")
    translation = _vector(content.get("translation"), "translation", 3)
    quaternion = _vector(content.get("quaternion_wxyz"), "quaternion_wxyz", 4)
    _vector(content.get("euler_xyz_degrees"), "euler_xyz_degrees", 3)
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"quaternion_wxyz must be normalized (norm={norm:.9g})")
    compatible = content.get("compatible_model_variants")
    if not isinstance(compatible, list) or not compatible or not all(
        isinstance(item, str) for item in compatible
    ):
        raise ValueError("compatible_model_variants must be a non-empty string array")
    model_variant = content.get("model_variant")
    if not isinstance(model_variant, str) or model_variant not in compatible:
        raise ValueError("model_variant must name one compatible_model_variants entry")
    if not isinstance(content.get("source_pose"), dict):
        raise ValueError("source_pose must be a JSON object")
    if content.get("pivot") != "palm_body_origin":
        raise ValueError("pivot must be 'palm_body_origin'")
    if not isinstance(content.get("rotation_convention"), str):
        raise ValueError("rotation_convention must be a string")
    if not isinstance(content.get("created_at"), str):
        raise ValueError("created_at must be a timestamp string")
    if not isinstance(content.get("notes"), str):
        raise ValueError("notes must be a string")
    if expected_model_variant is not None and expected_model_variant not in compatible:
        raise ValueError(
            f"hand pose is incompatible with model variant {expected_model_variant!r}; "
            f"compatible variants: {compatible}"
        )
    validated = dict(content)
    validated["translation"] = translation.tolist()
    validated["quaternion_wxyz"] = quat_normalize(quaternion).tolist()
    validated["euler_xyz_degrees"] = quat_to_euler_xyz_degrees(quaternion).tolist()
    return validated


def load_hand_pose(
    path: str | Path, expected_model_variant: str | None = None
) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read hand pose config {resolved}: {exc}") from exc
    try:
        content = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid hand pose JSON {resolved}: {exc}") from exc
    validated = validate_hand_pose(content, expected_model_variant)
    return validated, resolved, hashlib.sha256(raw).hexdigest()


def apply_hand_pose(model: mujoco.MjModel, content: dict[str, Any]) -> int:
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if palm_id < 0:
        raise ValueError("model has no body named 'palm'")
    model.body_pos[palm_id] = np.asarray(content["translation"], dtype=np.float64)
    model.body_quat[palm_id] = quat_normalize(
        np.asarray(content["quaternion_wxyz"], dtype=np.float64)
    )
    return int(palm_id)


def make_hand_pose(
    translation: np.ndarray,
    quaternion: np.ndarray,
    *,
    model_variant: str,
    source_pose: dict[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    quat = quat_normalize(quaternion)
    content = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "transform_space": "model",
        "body": "palm",
        "translation": _vector(translation, "translation", 3).tolist(),
        "quaternion_wxyz": quat.tolist(),
        "euler_xyz_degrees": quat_to_euler_xyz_degrees(quat).tolist(),
        "source_pose": source_pose,
        "pivot": "palm_body_origin",
        "rotation_convention": (
            "world-axis incremental rotations pre-multiplied onto the palm "
            "orientation; Euler display is extrinsic XYZ in degrees"
        ),
        "model_variant": model_variant,
        "compatible_model_variants": list(MODEL_VARIANTS),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": notes,
    }
    return validate_hand_pose(content, model_variant)


def write_hand_pose(path: str | Path, content: dict[str, Any], overwrite: bool = False) -> Path:
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing pose {output}; pass --overwrite explicitly"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_hand_pose(content)
    output.write_text(json.dumps(validated, indent=2) + "\n", encoding="utf-8")
    return output
