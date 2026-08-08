"""Online EMA rebalancing of DexScrew rotation vs tilt reward mass."""

from __future__ import annotations

from dataclasses import dataclass

from allegro_rod_mvp.rewards_dexscrew import DexScrewRewardConfig

_COMPONENT_KEYS = (
    "reward_rotation",
    "reward_proximity",
    "reward_pose_anchor",
    "reward_energy",
    "reward_excess_omega",
    "reward_tip_penalty",
    "reward_axis_tilt_penalty",
)


@dataclass
class AdaptiveMassConfig:
    enabled: bool = False
    target_rot: float = 0.45
    target_tilt: float = 0.45
    ema_tau_steps: float = 2000.0
    kappa: float = 0.08
    rotate_scale_min: float = 0.5
    rotate_scale_max: float = 8.0
    tilt_scale_min: float = 0.5
    tilt_scale_max: float = 8.0
    eps: float = 1e-6
    warmup_steps: int = 200
    max_scale_factor: float = 1.02  # per-step multiplicative clamp


class AdaptiveMassBalancer:
    """Adjust rotate/tilt scales so |rot| and |tilt| approach target fractions of Σ|terms|.

    Updates scales for the *next* step from this step's measured absolute masses.
    """

    def __init__(self, cfg: AdaptiveMassConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveMassConfig()
        self.ema_rot = 0.0
        self.ema_tilt = 0.0
        self.ema_total = 0.0
        self.n_updates = 0
        self.mass_rot = 0.0
        self.mass_tilt = 0.0

    def _alpha(self) -> float:
        tau = max(float(self.cfg.ema_tau_steps), 1.0)
        return 1.0 / tau

    @staticmethod
    def _clip_factor(ratio: float, kappa: float, max_factor: float) -> float:
        raw = float(ratio) ** float(kappa)
        lo = 1.0 / max(float(max_factor), 1.0001)
        hi = float(max_factor)
        return min(hi, max(lo, raw))

    def update(self, components: dict[str, float], reward_cfg: DexScrewRewardConfig) -> dict[str, float]:
        """Update EMAs from weighted components; adapt reward_cfg scales in-place."""
        if not self.cfg.enabled:
            return {
                "mass_rot": self.mass_rot,
                "mass_tilt": self.mass_tilt,
                "ema_rot": self.ema_rot,
                "ema_tilt": self.ema_tilt,
                "ema_total": self.ema_total,
                "rotate_scale": float(reward_cfg.rotate_scale),
                "tilt_scale": float(reward_cfg.tilt_scale),
            }

        abs_rot = abs(float(components.get("reward_rotation", 0.0)))
        abs_tilt = abs(float(components.get("reward_axis_tilt_penalty", 0.0)))
        abs_total = 0.0
        for key in _COMPONENT_KEYS:
            abs_total += abs(float(components.get(key, 0.0)))

        a = self._alpha()
        if self.n_updates == 0:
            self.ema_rot = abs_rot
            self.ema_tilt = abs_tilt
            self.ema_total = max(abs_total, self.cfg.eps)
        else:
            self.ema_rot = (1.0 - a) * self.ema_rot + a * abs_rot
            self.ema_tilt = (1.0 - a) * self.ema_tilt + a * abs_tilt
            self.ema_total = (1.0 - a) * self.ema_total + a * max(abs_total, self.cfg.eps)
        self.n_updates += 1

        denom = max(self.ema_total, self.cfg.eps)
        self.mass_rot = self.ema_rot / denom
        self.mass_tilt = self.ema_tilt / denom

        if self.n_updates > self.cfg.warmup_steps and self.ema_total > self.cfg.eps:
            rot_factor = self._clip_factor(
                self.cfg.target_rot / (self.mass_rot + self.cfg.eps),
                self.cfg.kappa,
                self.cfg.max_scale_factor,
            )
            tilt_factor = self._clip_factor(
                self.cfg.target_tilt / (self.mass_tilt + self.cfg.eps),
                self.cfg.kappa,
                self.cfg.max_scale_factor,
            )
            reward_cfg.rotate_scale = float(
                min(
                    self.cfg.rotate_scale_max,
                    max(self.cfg.rotate_scale_min, reward_cfg.rotate_scale * rot_factor),
                )
            )
            reward_cfg.tilt_scale = float(
                min(
                    self.cfg.tilt_scale_max,
                    max(self.cfg.tilt_scale_min, reward_cfg.tilt_scale * tilt_factor),
                )
            )

        return {
            "mass_rot": float(self.mass_rot),
            "mass_tilt": float(self.mass_tilt),
            "ema_rot": float(self.ema_rot),
            "ema_tilt": float(self.ema_tilt),
            "ema_total": float(self.ema_total),
            "rotate_scale": float(reward_cfg.rotate_scale),
            "tilt_scale": float(reward_cfg.tilt_scale),
        }
