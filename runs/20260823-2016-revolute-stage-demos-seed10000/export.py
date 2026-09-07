#!/usr/bin/env python3
"""Export a fixed-seed comparison video for accepted revolute stages through s=25."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml

from allegro_rod_mvp import RodRotationEnv


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "runs/curricula/20260823-2015-proportional-physics-C-seed0/state.json"
BASE_CONFIG = (
    ROOT
    / "runs/20260823-2015-proportional-physics-C-seed0-R04-s25-mu0.25-seed0/config.yaml"
)
SEED = 10000
FPS = 25
EPISODE_SECONDS = 20.0
EXPECTED = [
    (400.0, 4.0, "revolute_s400.mp4"),
    (200.0, 2.0, "revolute_s200.mp4"),
    (100.0, 1.0, "revolute_s100.mp4"),
    (50.0, 0.5, "revolute_s50.mp4"),
    (25.0, 0.25, "revolute_s25.mp4"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def env_kwargs(config: dict[str, Any], mass: float, friction: float) -> dict[str, Any]:
    return {
        "render_mode": "rgb_array",
        "curriculum_stage": int(config["stage"]),
        "episode_seconds": EPISODE_SECONDS,
        "tip_connect_solref": config["tip_connect_solref"],
        "tip_connect_enabled": bool(config["tip_connect_enabled"]),
        "axis_stabilizer_scale": float(config["axis_stabilizer_scale"]),
        "axis_tilt_penalty_weight": float(config["axis_tilt_penalty_weight"]),
        "axis_tilt_recovery_scale": float(config["axis_tilt_recovery_scale"]),
        "rotation_reward_scale": float(config["rotation_reward_scale"]),
        "contact_reward_mode": config["contact_reward_mode"],
        "three_contact_reward": float(config["three_contact_reward"]),
        "contact_window_steps": int(config["contact_window_steps"]),
        "contact_window_threshold": float(config["contact_window_threshold"]),
        "three_contact_required": bool(config["three_contact_required"]),
        "rotation_requires_three_contacts": bool(config["rotation_requires_three_contacts"]),
        "contact_support_termination_enabled": bool(
            config["contact_support_termination_enabled"]
        ),
        "contact_reward_scale": float(config["contact_reward_scale"]),
        "physics_mode": config["physics"],
        "reward_style": config["reward_style"],
        "omega_success_threshold": float(config["omega_success_threshold"]),
        "omega_success_hold_seconds": float(config["omega_success_hold_seconds"]),
        "success_mode": config["success_mode"],
        "net_angle_success_threshold_rad": float(
            config["net_angle_success_threshold_rad"]
        ),
        "dexscrew_tilt_scale": float(config["dexscrew_tilt_scale"]),
        "dexscrew_tip_penalty_scale": float(config["dexscrew_tip_penalty_scale"]),
        "dexscrew_tip_sigma": float(config["dexscrew_tip_sigma"]),
        "rod_mass_scale": mass,
        "rod_friction_cap": float(config["rod_friction_cap"]),
        "contact_friction_scale": friction,
        "contact_friction_scaling_mode": config["contact_friction_scaling_mode"],
        "scale_rod_joint_dynamics_from_s400": bool(
            config["scale_rod_joint_dynamics_from_s400"]
        ),
        "scale_tip_solref_with_mass": bool(config["scale_tip_solref_with_mass"]),
        "tilt_terminate_rad": float(config["tilt_terminate_rad"]),
        "tip_anchor": config["tip_anchor"],
        "hand_model": config["hand_model"],
        "hand_pose_config": config["hand_pose_config"],
        "hand_grasp_config": config["hand_grasp_config"],
    }


def overlay(
    frame: np.ndarray,
    *,
    mass: float,
    friction: list[float],
    step: int,
    info: dict[str, Any],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    lines = [
        (
            f"revolute s={mass:g} | pair friction "
            f"[{', '.join(f'{value:g}' for value in friction)}]"
        ),
        (
            f"seed {SEED} | step {step:03d}/500 | net rotation "
            f"{float(info.get('axis_rotation_deg', 0.0)):8.2f} deg"
        ),
        (
            f"tilt {float(info.get('axis_tilt_deg', 0.0)):5.2f} deg | tip error "
            f"{1000.0 * float(info.get('tip_error_m', 0.0)):7.4f} mm"
        ),
        (
            f"success {bool(info.get('is_success', False))} | contacts "
            f"{int(info.get('contact_count', 0))}/3 | "
            f"termination {info.get('termination_reason', 'none')}"
        ),
    ]
    draw.rectangle((5, 5, 635, 101), fill=(0, 0, 0, 190))
    text_font = font(16)
    for index, line in enumerate(lines):
        draw.text(
            (12, 10 + 22 * index),
            line,
            font=text_font,
            fill=(255, 255, 255, 255),
        )
    return np.asarray(image)


def render_stage(
    *,
    record: dict[str, Any],
    config: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    mass = float(record["mass_scale"])
    friction_scale = float(record["friction_scale"])
    checkpoint = Path(record["checkpoint"])
    vecnormalize_path = checkpoint.parent / "vecnormalize.pkl"
    missing = [
        str(path)
        for path in (checkpoint, vecnormalize_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"s={mass:g} missing required files: {missing}")

    kwargs = env_kwargs(config, mass, friction_scale)
    dummy = DummyVecEnv([lambda: RodRotationEnv(**{**kwargs, "render_mode": None})])
    vecnormalize = VecNormalize.load(vecnormalize_path, dummy)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    model = PPO.load(checkpoint, device="cpu")
    env = RodRotationEnv(**kwargs)
    observation, reset_info = env.reset(seed=SEED)
    info: dict[str, Any] = dict(reset_info)
    contacts: list[int] = []
    episode_return = 0.0
    terminated = truncated = False
    output = OUT / filename
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    pair_friction = list(
        record["fixed"]["effective_pair_friction_vectors"]["rod_tip0"]
    )
    writer = imageio.get_writer(
        output, fps=FPS, codec="libx264", quality=8, macro_block_size=None
    )
    try:
        initial = env.render()
        if initial is None:
            raise RuntimeError(f"s={mass:g}: initial render returned None")
        writer.append_data(
            overlay(
                np.asarray(initial),
                mass=mass,
                friction=pair_friction,
                step=0,
                info=info,
            )
        )
        while not (terminated or truncated):
            normalized = vecnormalize.normalize_obs(
                np.asarray(observation, dtype=np.float32).reshape(1, -1)
            )[0]
            action, _ = model.predict(normalized, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            contacts.append(int(info.get("contact_count", 0)))
            episode_return += float(reward)
            frame = env.render()
            if frame is None:
                raise RuntimeError(f"s={mass:g}: render returned None")
            writer.append_data(
                overlay(
                    np.asarray(frame),
                    mass=mass,
                    friction=pair_friction,
                    step=len(contacts),
                    info=info,
                )
            )
    finally:
        writer.close()
        env.close()
        vecnormalize.close()

    counts = np.asarray(contacts, dtype=np.int64)
    return {
        "stage": f"s={mass:g}",
        "mass_scale": mass,
        "friction_scale": friction_scale,
        "effective_pair_friction_vector": pair_friction,
        "run_id": record["run_id"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "vecnormalize": str(vecnormalize_path.resolve()),
        "vecnormalize_sha256": sha256(vecnormalize_path),
        "video": str(output.resolve()),
        "seed": SEED,
        "steps": len(contacts),
        "episode_return": episode_return,
        "net_rotation_deg": float(info.get("axis_rotation_deg", 0.0)),
        "tilt_deg": float(info.get("axis_tilt_deg", 0.0)),
        "tip_error_m": float(info.get("tip_error_m", 0.0)),
        "success": bool(info.get("is_success", False)),
        "contact_count_final": int(info.get("contact_count", 0)),
        "contact_count_mean": float(counts.mean()),
        "contact_count_distribution": {
            str(value): float(np.mean(counts == value)) for value in range(4)
        },
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(info.get("termination_reason", "none")),
    }


def decode_check(entry: dict[str, Any]) -> None:
    path = Path(entry["video"])
    reader = imageio.get_reader(path)
    frame_count = 0
    resolution: list[int] | None = None
    try:
        for frame in reader:
            shape = list(np.asarray(frame).shape)
            current = [shape[1], shape[0]]
            if resolution is None:
                resolution = current
            elif current != resolution:
                raise RuntimeError(f"{path}: varying frame resolution")
            frame_count += 1
    finally:
        reader.close()
    if frame_count != entry["steps"] + 1:
        raise RuntimeError(
            f"{path}: decoded {frame_count}, expected {entry['steps'] + 1}"
        )
    if resolution != [640, 480]:
        raise RuntimeError(f"{path}: resolution {resolution}, expected [640, 480]")
    entry["integrity"] = {
        "full_decode_passed": True,
        "decoded_frame_count": frame_count,
        "resolution": resolution,
        "fps": FPS,
        "duration_seconds": frame_count / FPS,
        "file_size_bytes": path.stat().st_size,
        "video_sha256": sha256(path),
    }


def write_index(metadata: dict[str, Any]) -> None:
    lines = [
        "# Accepted Revolute Mass-Stage Demo Videos",
        "",
        f"- Artifact directory: `{OUT}`",
        f"- Curriculum state snapshot: `{STATE_PATH}`",
        f"- Protocol: deterministic PPO, fixed seed {SEED}, 20.0 s / 500 steps, 25 FPS",
        "- Physics: proportional full-vector friction and proportional rod-joint dynamics",
        (
            f"- Saved palm pose: `{metadata['protocol']['hand_pose_config']}` "
            f"(SHA-256 `{metadata['protocol']['hand_pose_sha256']}`)"
        ),
        (
            f"- Recovered grasp: `{metadata['protocol']['hand_grasp_config']}` "
            f"(SHA-256 `{metadata['protocol']['hand_grasp_sha256']}`)"
        ),
        "- Success gate: positive net unwrapped angle >= 180 deg, tilt/tip/stability/drop checks; contact is diagnostic.",
        "- Every MP4 was decoded from first through final frame after encoding.",
        "",
        "## Checkpoint provenance and results",
        "",
    ]
    for entry in metadata["entries"]:
        lines.extend(
            [
                f"### {entry['stage']}",
                f"- Video: [{Path(entry['video']).name}]({Path(entry['video']).name})",
                f"- Exact path: `{entry['video']}`",
                f"- Accepted run record: `{entry['run_id']}`",
                f"- Checkpoint: `{entry['checkpoint']}`",
                f"- VecNormalize: `{entry['vecnormalize']}`",
                (
                    f"- Friction scale/vector: {entry['friction_scale']:g} / "
                    f"`{entry['effective_pair_friction_vector']}`"
                ),
                (
                    f"- Seed-{entry['seed']} result: net rotation "
                    f"{entry['net_rotation_deg']:.3f} deg; tilt "
                    f"{entry['tilt_deg']:.3f} deg; tip error "
                    f"{entry['tip_error_m'] * 1000.0:.6f} mm; success "
                    f"{entry['success']}; final/mean contacts "
                    f"{entry['contact_count_final']}/3 / "
                    f"{entry['contact_count_mean']:.3f}"
                ),
                (
                    f"- Integrity: {entry['integrity']['decoded_frame_count']} "
                    f"decoded frames; {entry['integrity']['resolution'][0]}x"
                    f"{entry['integrity']['resolution'][1]}; "
                    f"{entry['integrity']['fps']} FPS; "
                    f"{entry['integrity']['duration_seconds']:.2f} s"
                ),
                "",
            ]
        )
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    for name in ("INDEX.md", "metadata.json"):
        if (OUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite {OUT / name}")
    state = json.loads(STATE_PATH.read_text())
    config = yaml.safe_load(BASE_CONFIG.read_text())
    completed = state.get("completed", [])
    if len(completed) < len(EXPECTED):
        raise RuntimeError(f"Only {len(completed)} completed stages in state")
    selected = completed[: len(EXPECTED)]
    for record, (mass, friction, _) in zip(selected, EXPECTED, strict=True):
        actual = (float(record["mass_scale"]), float(record["friction_scale"]))
        if actual != (mass, friction) or not record.get("accepted", False):
            raise RuntimeError(
                f"Expected accepted {(mass, friction)}, found {actual}, "
                f"accepted={record.get('accepted')}"
            )

    entries = []
    for record, (_, _, filename) in zip(selected, EXPECTED, strict=True):
        entry = render_stage(record=record, config=config, filename=filename)
        decode_check(entry)
        entries.append(entry)
        print(
            f"{entry['stage']}: {filename}, {entry['net_rotation_deg']:.3f} deg, "
            f"{entry['integrity']['decoded_frame_count']} decoded frames",
            flush=True,
        )

    metadata = {
        "artifact_id": OUT.name,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_curriculum_id": state["curriculum_id"],
        "source_state": str(STATE_PATH.resolve()),
        "source_state_sha256_after_export": sha256(STATE_PATH),
        "artifact_scope": "export only; no training or policy changes",
        "protocol": {
            "deterministic": True,
            "seed": SEED,
            "episode_seconds": EPISODE_SECONDS,
            "steps": 500,
            "fps": FPS,
            "success_mode": "net_angle",
            "hand_pose_config": config["hand_pose_config"],
            "hand_pose_sha256": config["hand_pose"]["sha256"],
            "hand_grasp_config": config["hand_grasp_config"],
            "hand_grasp_sha256": config["hand_grasp"]["sha256"],
            "contact_friction_scaling_mode": "full_vector",
            "scale_rod_joint_dynamics_from_s400": True,
        },
        "provenance_note": (
            "s=400 uses the accepted C checkpoint. s=200/100/50 use the accepted "
            "net-angle proportional-curriculum checkpoints. s=25 uses the canonical "
            "proportional-friction scale-0.25 checkpoint; no low-friction diagnostic "
            "checkpoint was substituted."
        ),
        "entries": entries,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_index(metadata)
    print(f"index: {OUT / 'INDEX.md'}", flush=True)
    print(f"metadata: {OUT / 'metadata.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
