import unittest

import numpy as np
import mujoco

from allegro_rod_mvp import RodRotationEnv


class ContactDetectionTest(unittest.TestCase):
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
