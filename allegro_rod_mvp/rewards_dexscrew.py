"""DexScrew-style reward terms (numpy), shared by Arm A (revolute) and Arm B (tip-connect).

Scales default near DexScrew screwdriver yaml: rotate≈2.5, prox≈2.0, pose≈0.1.
Tilt term is Arm B only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DexScrewRewardConfig:
    rotate_scale: float = 2.5
    prox_scale: float = 2.0
    pose_scale: float = 0.1
    energy_scale: float = 0.05
    excess_omega_scale: float = 0.1
    omega_clip: float = 4.0
    omega_excess_thresh: float = 4.0
    prox_dist_thresh: float = 0.04
    tilt_scale: float = 0.0  # >0 enables Arm B tilt punishment
    tilt_sigma: float = 0.15  # rad
    tip_penalty_scale: float = 0.5  # mild tip stability (not primary)
    tip_sigma: float = 0.025


def compute_dexscrew_reward(
    *,
    axial_omega: float,
    fingertip_dists: np.ndarray,
    q_hand: np.ndarray,
    q0_hand: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    tip_error: float,
    axis_tilt: float,
    cfg: DexScrewRewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return (reward, component dict with raw + weighted terms)."""
    omega_clipped = float(np.clip(axial_omega, -cfg.omega_clip, cfg.omega_clip))
    rotate_raw = omega_clipped
    rotate_w = cfg.rotate_scale * rotate_raw

    mean_dist = float(np.mean(np.asarray(fingertip_dists, dtype=np.float64)))
    prox_raw = float(np.clip(1.0 - mean_dist / max(cfg.prox_dist_thresh, 1e-6), 0.0, 1.0))
    prox_w = cfg.prox_scale * prox_raw

    dq = np.asarray(q_hand, dtype=np.float64) - np.asarray(q0_hand, dtype=np.float64)
    pose_raw = -float(np.dot(dq, dq))
    pose_w = cfg.pose_scale * pose_raw

    act = np.asarray(action, dtype=np.float64)
    prev = np.asarray(last_action, dtype=np.float64)
    energy_raw = -float(np.mean((act - prev) ** 2))
    energy_w = cfg.energy_scale * energy_raw

    excess_raw = -float(max(0.0, abs(axial_omega) - cfg.omega_excess_thresh))
    excess_w = cfg.excess_omega_scale * excess_raw

    tip_raw = -float(np.clip((tip_error / max(cfg.tip_sigma, 1e-6)) ** 2, 0.0, 25.0))
    tip_w = cfg.tip_penalty_scale * tip_raw

    if cfg.tilt_scale > 0.0:
        tilt_raw = -float(np.clip((axis_tilt / max(cfg.tilt_sigma, 1e-6)) ** 2, 0.0, 25.0))
        tilt_w = cfg.tilt_scale * tilt_raw
    else:
        tilt_raw = 0.0
        tilt_w = 0.0

    reward = float(rotate_w + prox_w + pose_w + energy_w + excess_w + tip_w + tilt_w)
    components = {
        "reward_rotation": rotate_w,
        "reward_rotation_raw": rotate_raw,
        "reward_proximity": prox_w,
        "reward_proximity_raw": prox_raw,
        "reward_pose_anchor": pose_w,
        "reward_pose_anchor_raw": pose_raw,
        "reward_energy": energy_w,
        "reward_energy_raw": energy_raw,
        "reward_excess_omega": excess_w,
        "reward_excess_omega_raw": excess_raw,
        "reward_tip_penalty": tip_w,
        "reward_tip_penalty_raw": tip_raw,
        "reward_axis_tilt_penalty": tilt_w,
        "reward_axis_tilt_penalty_raw": tilt_raw,
        "axial_omega": float(axial_omega),
    }
    return reward, components
