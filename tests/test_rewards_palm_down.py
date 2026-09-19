import math
import unittest

import numpy as np

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.rewards_palm_down import (
    PALM_DOWN_REWARD_CLIP,
    PALM_DOWN_TERMINAL_PENALTY,
    PALM_DOWN_TILT_TERMINATE_RAD,
    compute_palm_down_living_reward,
    finalize_palm_down_reward,
)


def _zeros(**overrides):
    kwargs = dict(
        omega_dtheta=1.0,
        axis_tilt=0.0,
        action=np.zeros(12, dtype=np.float64),
        last_action=np.zeros(12, dtype=np.float64),
        fingertip_dists=np.zeros(3, dtype=np.float64),
        tip_error=0.0,
    )
    kwargs.update(overrides)
    return kwargs


class PalmDownFormulaTest(unittest.TestCase):
    def test_target_speed_upright_hits_clip_ceiling(self):
        living, parts = compute_palm_down_living_reward(**_zeros())
        self.assertAlmostEqual(parts["reward_track"], 4.0)
        self.assertAlmostEqual(parts["reward_wobble"], 0.0)
        self.assertAlmostEqual(living, 5.0)
        self.assertAlmostEqual(finalize_palm_down_reward(living, False), 5.0)

    def test_track_peaks_at_one_rad_per_sec(self):
        at_target, _ = compute_palm_down_living_reward(**_zeros(omega_dtheta=1.0))
        below, _ = compute_palm_down_living_reward(**_zeros(omega_dtheta=0.0))
        above, _ = compute_palm_down_living_reward(**_zeros(omega_dtheta=2.0))
        reverse, _ = compute_palm_down_living_reward(**_zeros(omega_dtheta=-1.0))
        self.assertGreater(at_target, below)
        self.assertGreater(at_target, above)
        self.assertGreater(at_target, reverse)

    def test_wobble_at_palm_down_kill_angle(self):
        _, parts = compute_palm_down_living_reward(
            **_zeros(omega_dtheta=0.0, axis_tilt=PALM_DOWN_TILT_TERMINATE_RAD)
        )
        expected = -3.0 * (PALM_DOWN_TILT_TERMINATE_RAD / 0.20) ** 2
        self.assertAlmostEqual(parts["reward_wobble"], expected)

    def test_terminal_penalty_then_clip_matches_wrapper(self):
        living, _ = compute_palm_down_living_reward(**_zeros())
        finalized = finalize_palm_down_reward(living, True)
        self.assertAlmostEqual(living - PALM_DOWN_TERMINAL_PENALTY, -15.0)
        self.assertAlmostEqual(finalized, -15.0)
        self.assertGreaterEqual(finalized, PALM_DOWN_REWARD_CLIP[0])
        self.assertLessEqual(finalized, PALM_DOWN_REWARD_CLIP[1])

    def test_large_tip_error_hits_floor_after_terminal(self):
        living, _ = compute_palm_down_living_reward(**_zeros(tip_error=0.05))
        finalized = finalize_palm_down_reward(living, True)
        self.assertEqual(finalized, PALM_DOWN_REWARD_CLIP[0])


class PalmDownEnvWiringTest(unittest.TestCase):
    def test_rejects_unknown_style(self):
        with self.assertRaises(ValueError):
            RodRotationEnv(reward_style="not_a_style")

    def test_env_uses_tracker_not_contact_bonus(self):
        env = RodRotationEnv(
            reward_style="palm_down",
            physics_mode="tip_connect",
            tip_anchor="bottom",
            tilt_terminate_rad=PALM_DOWN_TILT_TERMINATE_RAD,
            contact_reward_mode="discrete",
            three_contact_reward=0.3,
            contact_reward_scale=0.1,
            rotation_requires_three_contacts=False,
            contact_support_termination_enabled=False,
            hand_pose_config="configs/hand_poses/palm_down_xml_translation.json",
            hand_grasp_config="configs/hand_grasps/palm_down_xml_translation_default.json",
            rod_mass_scale=400.0,
            contact_friction_scale=4.0,
            contact_friction_scaling_mode="full_vector",
            scale_rod_joint_dynamics_from_s400=True,
            scale_tip_solref_with_mass=False,
            tip_connect_solref=0.008,
            axis_stabilizer_scale=0.0,
        )
        try:
            self.assertEqual(env.reward_style, "palm_down")
            self.assertEqual(env.tilt_terminate_rad, 0.35)
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (48,))
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.isfinite(obs).all())
            self.assertLessEqual(reward, 5.0)
            self.assertGreaterEqual(reward, -30.0)
            self.assertIn("reward_track", info)
            self.assertIn("omega_dtheta", info)
            # Discrete contact bonus is logged but not added to the tracker return.
            self.assertAlmostEqual(
                reward if not terminated else reward,
                finalize_palm_down_reward(
                    info["reward_alive"]
                    + info["reward_track"]
                    + info["reward_wobble"]
                    + info["reward_smooth"]
                    + info["reward_near"]
                    + info["reward_tip"],
                    bool(terminated),
                ),
                places=5,
            )
            dt = env.frame_skip * float(env.model.opt.timestep)
            self.assertAlmostEqual(info["omega_dtheta"], info["dtheta"] / dt, places=6)
            self.assertIn(terminated, (True, False))
            self.assertIn(truncated, (True, False))
        finally:
            env.close()

    def test_zero_action_hold_does_not_immediately_tilt_kill(self):
        env = RodRotationEnv(
            reward_style="palm_down",
            physics_mode="tip_connect",
            tip_anchor="bottom",
            tilt_terminate_rad=PALM_DOWN_TILT_TERMINATE_RAD,
            contact_support_termination_enabled=False,
            rotation_requires_three_contacts=False,
            hand_pose_config="configs/hand_poses/palm_down_xml_translation.json",
            hand_grasp_config="configs/hand_grasps/palm_down_xml_translation_default.json",
            rod_mass_scale=400.0,
            contact_friction_scale=4.0,
            contact_friction_scaling_mode="full_vector",
            scale_rod_joint_dynamics_from_s400=True,
            scale_tip_solref_with_mass=False,
            tip_connect_solref=0.008,
            axis_stabilizer_scale=0.0,
        )
        try:
            env.reset(seed=0)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            reasons = []
            for _ in range(25):
                _, reward, terminated, _, info = env.step(action)
                self.assertTrue(math.isfinite(reward))
                if terminated:
                    reasons.append(info["termination_reason"])
                    break
            self.assertNotIn("axis_tilt", reasons)
        finally:
            env.close()
