"""Support-aware reward terms for the T00 collapse ablation.

These helpers are independent of MuJoCo so scales and ω_perp can be tested
without constructing the environment.
"""

from __future__ import annotations

import numpy as np


def rotation_contact_support_scale(
    n_contact: int,
    *,
    scale_0: float = 0.0,
    scale_1: float = 0.1,
    scale_2plus: float = 1.0,
) -> float:
    """Multiply axial rotation reward by a support factor.

    n_contact >= 2 keeps full rotation credit. A brief 1-contact step is
    allowed but only 10% credit by default. Zero contact earns no rotation.
    """
    count = int(n_contact)
    if count <= 0:
        return float(scale_0)
    if count == 1:
        return float(scale_1)
    return float(scale_2plus)


def omega_perp_norm(omega_world: np.ndarray, axis_hat: np.ndarray) -> float:
    """Angular-velocity magnitude perpendicular to the rod long axis.

    omega_axial_vec = dot(omega, axis_hat) * axis_hat
    omega_perp = omega - omega_axial_vec
    """
    omega = np.asarray(omega_world, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_hat, dtype=np.float64).reshape(3)
    axis_n = float(np.linalg.norm(axis))
    if axis_n <= 1e-8 or not np.isfinite(omega).all() or not np.isfinite(axis).all():
        return 0.0
    axis = axis / axis_n
    axial = float(np.dot(omega, axis))
    perp = omega - axial * axis
    value = float(np.linalg.norm(perp))
    return value if np.isfinite(value) else 0.0


def low_support_wobble_penalty(
    n_contact: int,
    omega_perp: float,
    *,
    scale: float,
) -> float:
    """Return -scale * ||ω_perp||^2 when n_contact < 2, else 0.

    The term is zero whenever support is at least two fingers, including
    when scale is 0.
    """
    scale_f = float(scale)
    if scale_f == 0.0 or int(n_contact) >= 2:
        return 0.0
    omega = float(omega_perp)
    if not np.isfinite(omega):
        return 0.0
    value = -scale_f * omega * omega
    return float(value) if np.isfinite(value) else 0.0


def apply_support_aware_rotation(
    rotation_reward: float,
    n_contact: int,
    omega_perp: float,
    *,
    enabled: bool,
    scale_0: float,
    scale_1: float,
    scale_2plus: float,
    wobble_scale: float,
) -> tuple[float, float, float]:
    """Return (gated_rotation, gating_effect, wobble_penalty).

    When disabled, gated_rotation equals the input and both extras are 0.
    """
    rotation = float(rotation_reward)
    if not enabled:
        return rotation, 0.0, 0.0
    factor = rotation_contact_support_scale(
        n_contact, scale_0=scale_0, scale_1=scale_1, scale_2plus=scale_2plus
    )
    gated = rotation * factor
    if not np.isfinite(gated):
        gated = 0.0
    effect = gated - rotation
    wobble = low_support_wobble_penalty(
        n_contact, omega_perp, scale=wobble_scale
    )
    return float(gated), float(effect), float(wobble)
