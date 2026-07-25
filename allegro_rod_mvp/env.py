from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class RodRotationEnv(gym.Env):
    """Small CPU-friendly MVP for rod axial rotation with low-dimensional tactile feedback.

    Curriculum stages:
      0: MuJoCo point equality active (tip anchored).
      1: equality disabled; strong tip-position penalty.
      2: equality disabled; tighter free-rod task and randomized physics.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Contacting two-finger grasp for hanging-tip scene (low gravity tilt).
    _GRASP_QPOS = np.array(
        [0.4830, -0.5181, 0.8549, -0.7273, 0.9422, 0.7105, 0.0298, 0.6038, -0.7947],
        dtype=np.float64,
    )
    # World-frame hanging equilibrium for the rod long axis (local +x -> world -Z).
    _VERTICAL_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def __init__(
        self,
        xml_path: str | None = None,
        render_mode: str | None = None,
        curriculum_stage: int = 0,
        episode_seconds: float = 12.0,
        policy_hz: int = 25,
        tip_connect_solref: float | None = None,
        tip_connect_enabled: bool | None = None,
        axis_stabilizer_scale: float | None = None,
        axis_stabilizer_scale_range: tuple[float, float] | None = None,
        axis_tilt_penalty_weight: float = 1.0,
        axis_tilt_recovery_scale: float = 0.0,
        rotation_reward_scale: float = 16.0,
        contact_reward_mode: str = "linear",
        three_contact_reward: float = 10.0,
    ) -> None:
        super().__init__()
        root = Path(__file__).resolve().parents[1]
        self.xml_path = Path(xml_path) if xml_path else root / "models" / "three_finger_rod.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.curriculum_stage = int(curriculum_stage)
        self.tip_connect_solref = tip_connect_solref
        self.tip_connect_enabled = tip_connect_enabled
        self.axis_stabilizer_scale_override = axis_stabilizer_scale
        self.axis_stabilizer_scale_range = axis_stabilizer_scale_range
        self.current_axis_stabilizer_scale = axis_stabilizer_scale
        self.axis_tilt_penalty_weight = float(axis_tilt_penalty_weight)
        self.axis_tilt_recovery_scale = float(axis_tilt_recovery_scale)
        self.rotation_reward_scale = float(rotation_reward_scale)
        if contact_reward_mode not in {"linear", "discrete"}:
            raise ValueError("contact_reward_mode must be 'linear' or 'discrete'")
        self.contact_reward_mode = contact_reward_mode
        self.three_contact_reward = float(three_contact_reward)
        self.frame_skip = max(1, round(1.0 / (policy_hz * self.model.opt.timestep)))
        self.max_steps = int(episode_seconds * policy_hz)
        self.step_count = 0
        self.viewer = None
        self.renderer = None

        self.nu = self.model.nu
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        # qpos, qvel, 3 touch forces, 3 tip local offsets (x,y), tip error, rod omega, sin/cos angle
        obs_dim = self.model.nq + self.model.nv + 3 + 6 + 3 + 3 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "tip_anchor")
        self.rod_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rod")
        self.rod_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom")
        self.tip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rod_tip")
        self.tip_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"tip{i}") for i in range(3)]
        self.touch_sensor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"touch{i}") for i in range(3)]
        self.baseline_rod_mass = float(self.model.body_mass[self.rod_body])
        self.baseline_rod_inertia = self.model.body_inertia[self.rod_body].copy()
        self.baseline_rod_friction = float(self.model.geom_friction[self.rod_geom, 0])
        self.ctrl_center = np.zeros(self.nu, dtype=np.float64)
        self.ctrl_scale = np.full(self.nu, 0.25, dtype=np.float64)
        self.target_tip = np.zeros(3)
        self.target_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.prev_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.prev_axis_tilt = 0.0
        self.unwrapped_angle = 0.0
        self.last_action = np.zeros(self.nu)
        self.last_stabilizer_torque_norm = 0.0
        self._contact_force_buf = np.zeros(6, dtype=np.float64)

    def _set_curriculum(self) -> None:
        # Stage 0: firm tip hang. Stage 1: softer tip spring (assist fade). Stage 2: free tip.
        if self.curriculum_stage == 0:
            self.data.eq_active[self.eq_id] = 1
            self.model.eq_solref[self.eq_id, 0] = 0.008
            self.model.eq_solref[self.eq_id, 1] = 1.0
        elif self.curriculum_stage == 1:
            self.data.eq_active[self.eq_id] = 1
            self.model.eq_solref[self.eq_id, 0] = (
                0.05 if self.tip_connect_solref is None else self.tip_connect_solref
            )
            self.model.eq_solref[self.eq_id, 1] = 1.0
        else:
            self.data.eq_active[self.eq_id] = 0
        if self.tip_connect_enabled is not None:
            self.data.eq_active[self.eq_id] = int(self.tip_connect_enabled)
            if self.tip_connect_enabled and self.tip_connect_solref is not None:
                self.model.eq_solref[self.eq_id, 0] = self.tip_connect_solref
                self.model.eq_solref[self.eq_id, 1] = 1.0

    def _randomize(self) -> None:
        # Always restore baselines so Stage 2 mass/friction cannot accumulate across resets.
        self.model.body_mass[self.rod_body] = self.baseline_rod_mass
        self.model.body_inertia[self.rod_body] = self.baseline_rod_inertia
        self.model.geom_friction[self.rod_geom, 0] = self.baseline_rod_friction
        if self.curriculum_stage < 2:
            return
        self.model.geom_friction[self.rod_geom, 0] = self.np_random.uniform(0.8, 1.5)
        mass_scale = float(self.np_random.uniform(0.85, 1.15))
        self.model.body_mass[self.rod_body] = self.baseline_rod_mass * mass_scale
        self.model.body_inertia[self.rod_body] = self.baseline_rod_inertia * mass_scale

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([
            a[0]*b[0] - np.dot(a[1:], b[1:]),
            a[0]*b[1] + b[0]*a[1] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] + b[0]*a[2] + a[3]*b[1] - a[1]*b[3],
            a[0]*b[3] + b[0]*a[3] + a[1]*b[2] - a[2]*b[1],
        ])

    @staticmethod
    def _axis_tilt_recovery_reward(previous: float, current: float, scale: float) -> float:
        """Reward reducing tilt, penalize increasing tilt, and bound outliers."""
        return float(np.clip((previous - current) * scale, -2.0, 2.0))

    @staticmethod
    def _contact_reward(
        contact_count: int,
        mode: str,
        three_contact_reward: float = 10.0,
    ) -> float:
        """Return the legacy linear reward or the EXP-20260724-003 discrete ladder."""
        if contact_count not in (0, 1, 2, 3):
            raise ValueError("contact_count must be between 0 and 3")
        if mode == "linear":
            return 0.25 * contact_count + (0.2 if contact_count >= 2 else 0.0)
        if mode == "discrete":
            return (-10.0, -1.0, 0.1, float(three_contact_reward))[contact_count]
        raise ValueError("mode must be 'linear' or 'discrete'")

    def _axis_rotation_increment(self) -> float:
        # Body quaternion uses wxyz. Relative quaternion is expressed in previous rod frame.
        quat = self.data.xquat[self.rod_body].copy()
        dq = self._quat_mul(self._quat_conj(self.prev_quat), quat)
        if dq[0] < 0:
            dq = -dq
        vnorm = np.linalg.norm(dq[1:])
        if vnorm < 1e-9:
            inc = 0.0
        else:
            angle = 2.0 * np.arctan2(vnorm, np.clip(dq[0], -1.0, 1.0))
            rotvec = angle * dq[1:] / vnorm
            # Negate so the natural fingertip rolling direction accumulates as +axial progress
            # after switching to a top-hanging tip anchor.
            inc = -float(rotvec[0])  # local rod long axis is +x
        self.prev_quat = quat
        self.unwrapped_angle += inc
        return inc

    def _contact_centers(self) -> np.ndarray:
        # Per fingertip: contact center in fingertip local x/y. Zero when no rod contact.
        out = np.zeros((3, 2), dtype=np.float64)
        counts = np.zeros(3, dtype=np.float64)
        for k in range(self.data.ncon):
            con = self.data.contact[k]
            for i, tip_geom in enumerate(self.tip_geom_ids):
                if {con.geom1, con.geom2} == {tip_geom, self.rod_geom}:
                    body_id = self.model.geom_bodyid[tip_geom]
                    rel_world = con.pos - self.data.xpos[body_id]
                    local = self.data.xmat[body_id].reshape(3, 3).T @ rel_world
                    out[i] += local[:2]
                    counts[i] += 1
        for i in range(3):
            if counts[i] > 0:
                out[i] /= counts[i]
        return out.reshape(-1)

    def _touch(self) -> np.ndarray:
        """Normal contact force per fingertip against the rod (site touch sensors are unreliable here)."""
        values = np.zeros(3, dtype=np.float64)
        for k in range(self.data.ncon):
            con = self.data.contact[k]
            for i, tip_geom in enumerate(self.tip_geom_ids):
                if {con.geom1, con.geom2} == {tip_geom, self.rod_geom}:
                    mujoco.mj_contactForce(self.model, self.data, k, self._contact_force_buf)
                    values[i] += abs(float(self._contact_force_buf[0]))
        return values

    def _tip_rod_distances(self) -> np.ndarray:
        """XY distance from each fingertip to the rod axis (world Z for this scene)."""
        rod_xy = self.data.xpos[self.rod_body, :2]
        dists = np.zeros(3, dtype=np.float64)
        for i, tip_geom in enumerate(self.tip_geom_ids):
            body_id = self.model.geom_bodyid[tip_geom]
            tip_xy = self.data.xpos[body_id, :2]
            dists[i] = float(np.linalg.norm(tip_xy - rod_xy))
        return dists

    def _get_obs(self) -> np.ndarray:
        tip = self.data.site_xpos[self.tip_site]
        tip_error = tip - self.target_tip
        # rod angular velocity is world-frame; project on current world rod axis
        axis_world = self.data.xmat[self.rod_body].reshape(3, 3)[:, 0]
        axial_omega = float(np.dot(self.data.cvel[self.rod_body, :3], axis_world))
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel,
            np.clip(self._touch(), 0, 20) / 20.0,
            self._contact_centers() / 0.03,
            tip_error / 0.05,
            np.array([axial_omega / 10.0, self.unwrapped_angle / (2*np.pi), self.curriculum_stage], dtype=np.float64),
            np.array([np.sin(self.unwrapped_angle), np.cos(self.unwrapped_angle)]),
        ])
        return obs.astype(np.float32)

    def _get_obs_safe(self) -> np.ndarray:
        obs = self._get_obs()
        if not np.isfinite(obs).all():
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_curriculum()
        self._randomize()
        if self.axis_stabilizer_scale_range is not None:
            low, high = self.axis_stabilizer_scale_range
            self.current_axis_stabilizer_scale = float(self.np_random.uniform(low, high))
        else:
            self.current_axis_stabilizer_scale = self.axis_stabilizer_scale_override
        # Start from a verified contacting grasp, plus small randomization.
        noise = self.np_random.uniform(-0.05, 0.05, size=self.nu)
        q = np.clip(self._GRASP_QPOS + noise, -1.5, 1.5)
        self.data.qpos[:self.nu] = q
        self.data.ctrl[:] = q
        # Randomize rod axial phase by rotating free-joint quaternion about local x.
        angle = self.np_random.uniform(-np.pi, np.pi)
        qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rod_free")]
        q_base = self.data.qpos[qadr + 3:qadr + 7].copy()
        q_axial = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        self.data.qpos[qadr + 3:qadr + 7] = self._quat_mul(q_base, q_axial)
        mujoco.mj_forward(self.model, self.data)
        # Settle contacts briefly so the first observation sees touch.
        for _ in range(40):
            self._apply_axis_stabilizer()
            mujoco.mj_step(self.model, self.data)
        self.target_tip = self.data.site_xpos[self.tip_site].copy()
        # Reward / Stage-0 stabilizer track the hanging vertical axis, not a tilted settle pose.
        self.target_axis = self._VERTICAL_AXIS.copy()
        axis_world = self.data.xmat[self.rod_body].reshape(3, 3)[:, 0]
        axis_align = abs(float(np.dot(axis_world, self.target_axis)))
        self.prev_axis_tilt = float(np.arccos(np.clip(axis_align, 0.0, 1.0)))
        self.prev_quat = self.data.xquat[self.rod_body].copy()
        self.unwrapped_angle = 0.0
        self.last_action.fill(0)
        self.last_stabilizer_torque_norm = 0.0
        self.step_count = 0
        self.data.xfrc_applied[:] = 0.0
        return self._get_obs(), {"curriculum_stage": self.curriculum_stage}

    def _axis_stabilizer_scale(self) -> float:
        # Stage1: mild orientation assist paired with soft tip-connect (see EXP-003/004).
        if self.current_axis_stabilizer_scale is not None:
            return self.current_axis_stabilizer_scale
        return [1.0, 0.25, 0.0][min(self.curriculum_stage, 2)]

    def _apply_axis_stabilizer(self) -> None:
        """Soft orientation spring: restore rod axis toward vertical, leave axial spin free.

        Full strength in Stage 0. Mild (0.25) in Stage 1 with soft tip-connect. Off in Stage 2.
        """
        self.data.xfrc_applied[self.rod_body] = 0.0
        self.last_stabilizer_torque_norm = 0.0
        scale = self._axis_stabilizer_scale()
        if scale <= 0.0:
            return
        axis = self.data.xmat[self.rod_body].reshape(3, 3)[:, 0].copy()
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            return
        axis /= n
        # Prefer the ±axis closest to the vertical target.
        if float(np.dot(axis, self.target_axis)) < 0.0:
            axis = -axis
        align_torque = np.cross(axis, self.target_axis)  # rotates axis toward target
        omega = self.data.cvel[self.rod_body, :3]
        axial_omega = float(np.dot(omega, axis))
        lateral_omega = omega - axial_omega * axis
        k_spring = 2.0 * scale
        k_damp = 0.12 * scale
        torque = k_spring * align_torque - k_damp * lateral_omega
        self.data.xfrc_applied[self.rod_body, 3:] = torque
        self.last_stabilizer_torque_norm = float(np.linalg.norm(torque))

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = np.clip(self.data.ctrl + self.ctrl_scale * action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        self.data.ctrl[:] = target
        for _ in range(self.frame_skip):
            self._apply_axis_stabilizer()
            mujoco.mj_step(self.model, self.data)
            qvel_ok = np.isfinite(self.data.qvel).all() and float(np.max(np.abs(self.data.qvel))) < 80.0
            qacc_ok = np.isfinite(self.data.qacc).all() and float(np.max(np.abs(self.data.qacc))) < 5e4
            qpos_ok = np.isfinite(self.data.qpos).all()
            if not (qpos_ok and qvel_ok and qacc_ok):
                # Numerical blow-up on the free joint / stiff contacts: end episode.
                tip_error = (
                    float(np.linalg.norm(self.data.site_xpos[self.tip_site] - self.target_tip))
                    if np.isfinite(self.data.site_xpos[self.tip_site]).all()
                    else 1.0
                )
                self.step_count += 1
                info = {
                    "axis_rotation": self.unwrapped_angle,
                    "axis_rotation_deg": np.degrees(self.unwrapped_angle),
                    "dtheta": 0.0,
                    "tip_error_m": tip_error,
                    "axis_tilt_rad": 1.0,
                    "axis_tilt_deg": 57.3,
                    "lateral_omega": 0.0,
                    "contact_count": 0,
                    "axial_slip_proxy": 0.0,
                    "stabilizer_torque_norm": self.last_stabilizer_torque_norm,
                    "termination_reason": "unstable",
                    "unstable": True,
                    "is_success": False,
                }
                return self._get_obs_safe(), -15.0, True, False, info

        dtheta = self._axis_rotation_increment()
        tip_error = float(np.linalg.norm(self.data.site_xpos[self.tip_site] - self.target_tip))
        touch = self._touch()
        contact_count = int(np.sum(touch > 0.05))
        axis_world = self.data.xmat[self.rod_body].reshape(3, 3)[:, 0]
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm > 1e-8:
            axis_world = axis_world / axis_norm
        # Rod is axially symmetric: treat ±axis as the same physical axis.
        axis_align = abs(float(np.dot(axis_world, self.target_axis)))
        axis_tilt = float(np.arccos(np.clip(axis_align, 0.0, 1.0)))
        omega_world = self.data.cvel[self.rod_body, :3]
        axial_omega = float(np.dot(omega_world, axis_world))
        lateral_omega = float(np.linalg.norm(omega_world - axial_omega * axis_world))
        axial_slip = abs(float(np.dot(self.data.cvel[self.rod_body, 3:], axis_world)))

        # Positive axial progress dominates; tip drift and unstable actions are penalties.
        # Asymmetric rotation reward: only credit progress in the +local-x direction.
        rotation_reward = np.clip(dtheta, 0.0, 0.25) * self.rotation_reward_scale
        tip_sigma = [0.035, 0.022, 0.012][min(self.curriculum_stage, 2)]
        # Clip penalties so contact spikes / rare physics glitches cannot dominate PPO returns.
        tip_penalty = float(np.clip((tip_error / tip_sigma) ** 2, 0.0, 25.0))
        # Keep the stick axis from tilting / wobbling (allow spin about the axis only).
        # Stage 0 sigma a bit looser so stiff tip-anchor + spin remains learnable.
        axis_tilt_sigma = [0.20, 0.10, 0.06][min(self.curriculum_stage, 2)]  # rad
        axis_tilt_penalty_raw = float(np.clip((axis_tilt / axis_tilt_sigma) ** 2, 0.0, 25.0))
        axis_tilt_penalty = self.axis_tilt_penalty_weight * axis_tilt_penalty_raw
        axis_tilt_recovery_reward = self._axis_tilt_recovery_reward(
            self.prev_axis_tilt,
            axis_tilt,
            self.axis_tilt_recovery_scale,
        )
        self.prev_axis_tilt = axis_tilt
        lateral_omega_penalty = float(np.clip(0.03 * lateral_omega ** 2, 0.0, 10.0))
        contact_bonus = self._contact_reward(
            contact_count,
            self.contact_reward_mode,
            self.three_contact_reward,
        )
        # Encourage staying near the rod when contact is lost.
        dists = self._tip_rod_distances()
        proximity = float(np.clip(0.04 - float(np.mean(dists[:2])), 0.0, 0.04)) * 8.0
        touch_for_penalty = np.clip(touch, 0.0, 50.0)
        force_penalty = 0.001 * float(np.sum(np.maximum(touch_for_penalty - 12.0, 0.0) ** 2))
        action_rate_penalty = 0.005 * float(np.mean((action - self.last_action) ** 2))
        reward = (
            rotation_reward
            - tip_penalty
            - axis_tilt_penalty
            + axis_tilt_recovery_reward
            - lateral_omega_penalty
            + contact_bonus
            + proximity
            - force_penalty
            - action_rate_penalty
        )
        self.last_action = action.copy()

        self.step_count += 1
        dropped = (
            self.data.xpos[self.rod_body, 2] < -0.12
            or tip_error > 0.12
            or axis_tilt > 0.7
            or not np.isfinite(reward)
        )
        terminated = bool(dropped)
        termination_reason = "none"
        if dropped:
            if not np.isfinite(reward):
                termination_reason = "nonfinite_reward"
            elif axis_tilt > 0.7:
                termination_reason = "axis_tilt"
            elif tip_error > 0.12:
                termination_reason = "tip_error"
            elif self.data.xpos[self.rod_body, 2] < -0.12:
                termination_reason = "rod_height"
        truncated = self.step_count >= self.max_steps
        if dropped:
            reward = -15.0 if not np.isfinite(reward) else reward - 15.0
        reward = float(np.clip(reward, -30.0, 30.0))

        info = {
            "axis_rotation": self.unwrapped_angle,
            "axis_rotation_deg": np.degrees(self.unwrapped_angle),
            "dtheta": dtheta,
            "tip_error_m": tip_error,
            "axis_tilt_rad": axis_tilt,
            "axis_tilt_deg": np.degrees(axis_tilt),
            "lateral_omega": lateral_omega,
            "contact_count": contact_count,
            "finger_contacts": (touch > 0.05).astype(int).tolist(),
            "axial_slip_proxy": axial_slip,
            "stabilizer_torque_norm": self.last_stabilizer_torque_norm,
            "reward_rotation": float(rotation_reward),
            "reward_tip_penalty": float(-tip_penalty),
            "reward_axis_tilt_penalty": float(-axis_tilt_penalty),
            "reward_axis_tilt_penalty_raw": float(-axis_tilt_penalty_raw),
            "reward_axis_tilt_recovery": axis_tilt_recovery_reward,
            "reward_lateral_omega_penalty": float(-lateral_omega_penalty),
            "reward_contact_bonus": float(contact_bonus),
            "contact_reward_mode": self.contact_reward_mode,
            "reward_proximity": float(proximity),
            "reward_force_penalty": float(-force_penalty),
            "reward_action_rate_penalty": float(-action_rate_penalty),
            "termination_reason": termination_reason,
            "unstable": False,
            "is_success": bool(
                self.unwrapped_angle > np.pi
                and tip_error < 0.02
                and axis_tilt < 0.25
                and not dropped
            ),
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer as mj_viewer
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
