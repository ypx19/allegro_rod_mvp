"""Palm-down screwdriver tracker reward.

Matches `palm_down_screwdriver/experiment_palm_down_fixedtip_cpu.py` ScrewEnv:
speed tracking peaked at 1 rad/s, quadratic tilt wobble, action smoothness,
fingertip proximity, and a 2 mm tip-error penalty. No contact bonus.

The wrapper discards the inner DexScrew reward, subtracts 20 on terminate,
and clips to [-30, 5]. ω used in the track term is dθ / (frame_skip * dt),
not body-twist axial ω.
"""

from __future__ import annotations

import numpy as np

PALM_DOWN_TILT_TERMINATE_RAD = 0.35
PALM_DOWN_REWARD_CLIP = (-30.0, 5.0)
PALM_DOWN_TERMINAL_PENALTY = 20.0
PALM_DOWN_TARGET_OMEGA = 1.0
PALM_DOWN_TIP_PENALTY_SCALE = 0.2
PALM_DOWN_TIP_SIGMA_M = 0.002


def compute_palm_down_living_reward(
    *,
    omega_dtheta: float,
    axis_tilt: float,
    action: np.ndarray,
    last_action: np.ndarray,
    fingertip_dists: np.ndarray,
    tip_error: float,
    tip_penalty_scale: float = PALM_DOWN_TIP_PENALTY_SCALE,
    tip_sigma_m: float = PALM_DOWN_TIP_SIGMA_M,
) -> tuple[float, dict[str, float]]:
    """Return the unclipped living reward and named components."""
    w = float(omega_dtheta)
    tilt = float(axis_tilt)
    action_arr = np.asarray(action, dtype=np.float64)
    last_arr = np.asarray(last_action, dtype=np.float64)
    dists = np.asarray(fingertip_dists, dtype=np.float64)
    track = float(
        2.0 * np.clip(w, -2.0, 1.0)
        - 2.0 * (np.clip(w, -4.0, 4.0) - 1.0) ** 2
        + 2.0
    )
    wobble = float(-3.0 * (tilt / 0.20) ** 2)
    smooth = float(-0.02 * np.mean((action_arr - last_arr) ** 2))
    near = float(-0.3 * np.mean(np.maximum(dists - 0.025, 0.0) / 0.025))
    sigma = float(tip_sigma_m) if float(tip_sigma_m) > 0.0 else PALM_DOWN_TIP_SIGMA_M
    tip_term = float(-float(tip_penalty_scale) * (float(tip_error) / sigma) ** 2)
    alive = 1.0
    reward = float(alive + track + wobble + smooth + near + tip_term)
    components = {
        "omega_dtheta": w,
        "reward_track": track,
        "reward_wobble": wobble,
        "reward_smooth": smooth,
        "reward_near": near,
        "reward_tip": tip_term,
        "reward_alive": alive,
        "reward_palm_down_unclipped": reward,
        "tip_penalty_scale": float(tip_penalty_scale),
    }
    return reward, components


def finalize_palm_down_reward(living_reward: float, terminated: bool) -> float:
    reward = float(living_reward)
    if terminated:
        reward -= PALM_DOWN_TERMINAL_PENALTY
    lo, hi = PALM_DOWN_REWARD_CLIP
    return float(np.clip(reward, lo, hi))
