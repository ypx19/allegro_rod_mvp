"""DexScrew-style reward terms (numpy), shared by Arm A (revolute) and Arm B (tip-connect).

Scales default near DexScrew screwdriver yaml: rotate≈2.5, prox≈2.0, pose≈0.1.
Tilt term is Arm B only.
Tilt recovery and tilt-growth are default-off (scale=0) so historical DexScrew
totals are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# One-sided DexScrew back-to-balance. Scale default 0; these shape constants
# only matter when tilt_recovery_scale > 0.
DEXSCREW_TILT_RECOVERY_DEADZONE_RAD = 0.05
DEXSCREW_TILT_RECOVERY_CLIP_RAD = 0.05

# One-sided DexScrew tilt-growth penalty. Scale default 0. Weight is zero in a
# small deadzone, ramps through the 0.25 rad success gate, and saturates at 1
# above the gate. Experiment scale 50 is not the default.
DEXSCREW_TILT_GROWTH_DEADZONE_RAD = 0.05
DEXSCREW_TILT_GROWTH_GATE_RAD = 0.25
DEXSCREW_TILT_GROWTH_CLIP_RAD = 0.05
DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE = 50.0


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
    tilt_recovery_scale: float = 0.0  # >0 enables one-sided back-to-balance
    tilt_recovery_deadzone: float = DEXSCREW_TILT_RECOVERY_DEADZONE_RAD
    tilt_recovery_clip: float = DEXSCREW_TILT_RECOVERY_CLIP_RAD
    tilt_growth_scale: float = 0.0  # >0 enables one-sided growth penalty
    tilt_growth_deadzone: float = DEXSCREW_TILT_GROWTH_DEADZONE_RAD
    tilt_growth_gate: float = DEXSCREW_TILT_GROWTH_GATE_RAD
    tilt_growth_clip: float = DEXSCREW_TILT_GROWTH_CLIP_RAD


def compute_dexscrew_tilt_recovery_reward(
    previous_tilt: float,
    current_tilt: float,
    *,
    scale: float,
    deadzone: float = DEXSCREW_TILT_RECOVERY_DEADZONE_RAD,
    clip_max: float = DEXSCREW_TILT_RECOVERY_CLIP_RAD,
) -> float:
    """Reward decreasing tilt toward upright; do not reward holding upright.

    r_recovery = scale * clip(prev - current, 0, clip_max) * 1[current > deadzone]

    Zero when scale is 0, tilt increases or is unchanged, or current tilt is
    already inside the deadzone. Non-finite inputs return 0 so the total stays
    finite.
    """
    scale_f = float(scale)
    if scale_f == 0.0:
        return 0.0
    previous = float(previous_tilt)
    current = float(current_tilt)
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0
    if current <= float(deadzone):
        return 0.0
    delta = float(np.clip(previous - current, 0.0, float(clip_max)))
    value = scale_f * delta
    return float(value) if np.isfinite(value) else 0.0


def compute_dexscrew_tilt_growth_weight(
    current_tilt: float,
    *,
    deadzone: float = DEXSCREW_TILT_GROWTH_DEADZONE_RAD,
    gate: float = DEXSCREW_TILT_GROWTH_GATE_RAD,
) -> float:
    """Linear weight that is 0 at/below the deadzone and 1 at/above the gate.

    w(θ) = clip((θ - 0.05) / (0.25 - 0.05), 0, 1)
    """
    current = float(current_tilt)
    if not np.isfinite(current):
        return 0.0
    span = float(gate) - float(deadzone)
    if span <= 0.0:
        return 1.0 if current > float(deadzone) else 0.0
    weight = (current - float(deadzone)) / span
    weight = float(np.clip(weight, 0.0, 1.0))
    return weight if np.isfinite(weight) else 0.0


def compute_dexscrew_tilt_growth_penalty(
    previous_tilt: float,
    current_tilt: float,
    *,
    scale: float,
    deadzone: float = DEXSCREW_TILT_GROWTH_DEADZONE_RAD,
    gate: float = DEXSCREW_TILT_GROWTH_GATE_RAD,
    clip_max: float = DEXSCREW_TILT_GROWTH_CLIP_RAD,
) -> float:
    """Penalize increasing tilt, weighted toward/above the 0.25 rad success gate.

    r_growth = -scale * clip(curr - prev, 0, clip_max) * w(curr)

    Zero when scale is 0, tilt decreases or is unchanged, or current tilt is
    inside the deadzone (w=0). Non-finite inputs return 0 so the total stays
    finite. The existing quadratic state penalty is unchanged.
    """
    scale_f = float(scale)
    if scale_f == 0.0:
        return 0.0
    previous = float(previous_tilt)
    current = float(current_tilt)
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0
    delta = float(np.clip(current - previous, 0.0, float(clip_max)))
    if delta == 0.0:
        return 0.0
    weight = compute_dexscrew_tilt_growth_weight(
        current, deadzone=deadzone, gate=gate
    )
    value = -scale_f * delta * weight
    return float(value) if np.isfinite(value) else 0.0


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
    prev_axis_tilt: float | None = None,
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

    if prev_axis_tilt is None:
        recovery_w = 0.0
        growth_w = 0.0
    else:
        recovery_w = compute_dexscrew_tilt_recovery_reward(
            prev_axis_tilt,
            axis_tilt,
            scale=cfg.tilt_recovery_scale,
            deadzone=cfg.tilt_recovery_deadzone,
            clip_max=cfg.tilt_recovery_clip,
        )
        growth_w = compute_dexscrew_tilt_growth_penalty(
            prev_axis_tilt,
            axis_tilt,
            scale=cfg.tilt_growth_scale,
            deadzone=cfg.tilt_growth_deadzone,
            gate=cfg.tilt_growth_gate,
            clip_max=cfg.tilt_growth_clip,
        )

    reward = float(
        rotate_w
        + prox_w
        + pose_w
        + energy_w
        + excess_w
        + tip_w
        + tilt_w
        + recovery_w
        + growth_w
    )
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
        "reward_axis_tilt_recovery": recovery_w,
        "reward_axis_tilt_growth": growth_w,
        "axial_omega": float(axial_omega),
    }
    return reward, components
