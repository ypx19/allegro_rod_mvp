"""Validated companion Allegro joint-reset configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "allegro_rod_mvp.hand_grasp"
SCHEMA_VERSION = 1


def load_hand_grasp(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    content = json.loads(raw)
    if not isinstance(content, dict):
        raise ValueError("hand grasp config root must be a JSON object")
    if content.get("schema") != SCHEMA or content.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported hand grasp schema; expected {SCHEMA!r} version 1")
    for key in ("reset_qpos", "grasp_qpos"):
        values = np.asarray(content.get(key), dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain 12 finite joint positions")
    pose_hash = content.get("hand_pose_sha256")
    if not isinstance(pose_hash, str) or len(pose_hash) != 64:
        raise ValueError("hand_pose_sha256 must be a SHA-256 hex digest")
    for key, minimum in (
        ("reset_joint_noise", 0.0),
        ("grasp_ramp_steps", 1),
        ("grasp_hold_steps", 0),
    ):
        value = content.get(key)
        if not isinstance(value, (int, float)) or value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
    validated = dict(content)
    validated["reset_qpos"] = np.asarray(content["reset_qpos"], dtype=np.float64).tolist()
    validated["grasp_qpos"] = np.asarray(content["grasp_qpos"], dtype=np.float64).tolist()
    return validated, resolved, hashlib.sha256(raw).hexdigest()
