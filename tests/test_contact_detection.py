import unittest
from pathlib import Path

import numpy as np
import mujoco

from allegro_rod_mvp import RodRotationEnv


class ContactDetectionTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_allegro_joint_axes_ranges_and_dimensions(self):
        env = RodRotationEnv(hand_model="allegro")
        try:
            self.assertEqual(env.nu, 12)
            self.assertEqual(env.observation_space.shape, (48,))
            expected_axes = [
                [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                [-1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0],
            ]
            expected_ranges = [
                [-0.47, 0.47], [-0.196, 1.61], [-0.174, 1.709], [-0.227, 1.618],
                [-0.47, 0.47], [-0.196, 1.61], [-0.174, 1.709], [-0.227, 1.618],
                [0.263, 1.396], [-0.105, 1.163], [-0.189, 1.644], [-0.162, 1.719],
            ]
            np.testing.assert_allclose(
                env.model.jnt_axis[env.hand_joint_ids], expected_axes, atol=1e-8
            )
            np.testing.assert_allclose(
                env.model.jnt_range[env.hand_joint_ids], expected_ranges, atol=1e-8
            )
        finally:
            env.close()

    def test_allegro_bottom_reset_detects_three_contacts(self):
        for physics in ("revolute", "tip_connect"):
            env = RodRotationEnv(
                hand_model="allegro",
                physics_mode=physics,
                tip_anchor="bottom",
                tip_connect_enabled=True if physics == "tip_connect" else None,
                axis_stabilizer_scale=1.0 if physics == "tip_connect" else 0.0,
            )
            try:
                env.reset(seed=0)
                forces = env._touch()
                self.assertTrue(np.isfinite(forces).all())
                self.assertTrue(
                    np.all(forces > 0.05),
                    msg=f"{physics} reset forces were {forces}",
                )
            finally:
                env.close()

    def test_revolute_and_tip_connect_observation_layouts_match(self):
        revolute = RodRotationEnv(hand_model="allegro", physics_mode="revolute")
        connected = RodRotationEnv(hand_model="allegro", physics_mode="tip_connect")
        try:
            self.assertEqual(revolute.action_space.shape, connected.action_space.shape)
            self.assertEqual(
                revolute.observation_space.shape, connected.observation_space.shape
            )
            self.assertEqual(revolute.hand_joint_names, connected.hand_joint_names)
        finally:
            revolute.close()
            connected.close()

    def test_explicit_contact_friction_scales_rod_and_all_pads_only(self):
        for physics in ("revolute", "tip_connect"):
            env = RodRotationEnv(
                hand_model="allegro",
                physics_mode=physics,
                rod_mass_scale=25.0,
                contact_friction_scale=0.25,
            )
            try:
                baseline = env.baseline_contact_friction.copy()
                env.reset(seed=0)
                actual = env.model.geom_friction[env.friction_geom_ids]
                np.testing.assert_allclose(actual[:, 0], baseline[:, 0] * 0.25)
                np.testing.assert_allclose(actual[:, 1:], baseline[:, 1:])
                self.assertAlmostEqual(
                    env.model.body_mass[env.rod_body],
                    env.baseline_rod_mass * 25.0,
                )
                np.testing.assert_allclose(
                    env.model.body_inertia[env.rod_body],
                    env.baseline_rod_inertia * 25.0,
                )
            finally:
                env.close()

    def test_full_vector_friction_and_rod_dynamics_scale_from_s400(self):
        env = RodRotationEnv(
            hand_model="allegro",
            physics_mode="revolute",
            rod_mass_scale=25.0,
            contact_friction_scale=0.25,
            contact_friction_scaling_mode="full_vector",
            scale_rod_joint_dynamics_from_s400=True,
        )
        try:
            baseline_friction = env.baseline_contact_friction.copy()
            baseline_damping = env.baseline_rod_dof_damping.copy()
            baseline_armature = env.baseline_rod_dof_armature.copy()
            baseline_frictionloss = env.baseline_rod_dof_frictionloss.copy()
            env.reset(seed=0)
            np.testing.assert_allclose(
                env.model.geom_friction[env.friction_geom_ids],
                baseline_friction * 0.25,
            )
            expected_pair = [
                baseline_friction[0, 0] * 0.25,
                baseline_friction[0, 0] * 0.25,
                baseline_friction[0, 1] * 0.25,
                baseline_friction[0, 2] * 0.25,
                baseline_friction[0, 2] * 0.25,
            ]
            for tip_geom_id in env.tip_geom_ids:
                np.testing.assert_allclose(
                    env._effective_pair_friction(tip_geom_id), expected_pair
                )
            ratio = 25.0 / 400.0
            np.testing.assert_allclose(
                env.model.dof_damping[env.rod_dof_adrs],
                baseline_damping * ratio,
            )
            np.testing.assert_allclose(
                env.model.dof_armature[env.rod_dof_adrs],
                baseline_armature * ratio,
            )
            np.testing.assert_allclose(
                env.model.dof_frictionloss[env.rod_dof_adrs],
                baseline_frictionloss * ratio,
            )
        finally:
            env.close()

    def test_proportional_physics_flags_are_default_preserving(self):
        env = RodRotationEnv(
            hand_model="allegro",
            physics_mode="revolute",
            rod_mass_scale=25.0,
            contact_friction_scale=0.25,
        )
        try:
            baseline_damping = env.baseline_rod_dof_damping.copy()
            baseline_armature = env.baseline_rod_dof_armature.copy()
            env.reset(seed=0)
            self.assertEqual(env.contact_friction_scaling_mode, "sliding_only")
            self.assertFalse(env.scale_rod_joint_dynamics_from_s400)
            np.testing.assert_allclose(
                env.model.dof_damping[env.rod_dof_adrs], baseline_damping
            )
            np.testing.assert_allclose(
                env.model.dof_armature[env.rod_dof_adrs], baseline_armature
            )
        finally:
            env.close()

    def test_saved_pose_revolute_grasp_is_robust_at_mass_endpoints(self):
        pose = self.ROOT / "configs" / "hand_poses" / "my_grasp.json"
        grasp = (
            self.ROOT
            / "configs"
            / "hand_grasps"
            / "my_grasp_revolute_shared.json"
        )
        for mass_scale, friction_scale in ((400.0, 4.0), (1.0, 0.1)):
            for seed in (0, 4, 9):
                env = RodRotationEnv(
                    hand_model="allegro",
                    physics_mode="revolute",
                    tip_anchor="bottom",
                    hand_pose_config=str(pose),
                    hand_grasp_config=str(grasp),
                    rod_mass_scale=mass_scale,
                    contact_friction_scale=friction_scale,
                    contact_reward_mode="discrete",
                    three_contact_reward=0.3,
                    contact_window_steps=25,
                    contact_window_threshold=18,
                    three_contact_required=True,
                    reward_style="dexscrew",
                    axis_stabilizer_scale=0.0,
                )
                try:
                    obs, _ = env.reset(seed=seed)
                    self.assertTrue(np.isfinite(obs).all())
                    self.assertTrue(np.all(env._touch() > 0.05))
                    for _ in range(100):
                        obs, _, terminated, truncated, info = env.step(
                            np.zeros(12, dtype=np.float32)
                        )
                        self.assertTrue(np.isfinite(obs).all())
                        self.assertEqual(info["contact_count"], 3)
                        self.assertNotEqual(info["termination_reason"], "contact_support")
                        self.assertFalse(terminated)
                        self.assertFalse(truncated)
                finally:
                    env.close()

    def test_legacy_surrogate_reset_remains_reproducible(self):
        env = RodRotationEnv(
            hand_model="surrogate",
            curriculum_stage=2,
            tip_connect_enabled=True,
            tip_connect_solref=0.10,
            axis_stabilizer_scale=0.10,
        )
        try:
            env.reset(seed=0)
            forces = env._touch()
            self.assertTrue(np.isfinite(forces).all())
            self.assertGreater(forces[0], 0.05)
            self.assertGreater(forces[1], 0.05)
            self.assertLessEqual(forces[2], 0.05)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
