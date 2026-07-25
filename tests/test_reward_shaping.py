import unittest

from allegro_rod_mvp import RodRotationEnv


class AxisTiltRecoveryRewardTest(unittest.TestCase):
    def test_rewards_recovery(self):
        self.assertAlmostEqual(
            RodRotationEnv._axis_tilt_recovery_reward(0.20, 0.15, 40.0),
            2.0,
        )

    def test_penalizes_worsening(self):
        self.assertAlmostEqual(
            RodRotationEnv._axis_tilt_recovery_reward(0.10, 0.12, 40.0),
            -0.8,
        )

    def test_zero_scale_preserves_baseline(self):
        self.assertEqual(
            RodRotationEnv._axis_tilt_recovery_reward(0.20, 0.10, 0.0),
            0.0,
        )

    def test_clips_outliers(self):
        self.assertEqual(
            RodRotationEnv._axis_tilt_recovery_reward(0.70, 0.0, 40.0),
            2.0,
        )
        self.assertEqual(
            RodRotationEnv._axis_tilt_recovery_reward(0.0, 0.70, 40.0),
            -2.0,
        )


class ContactRewardTest(unittest.TestCase):
    def test_discrete_contact_reward_ladder(self):
        expected = {0: -10.0, 1: -1.0, 2: 0.1, 3: 10.0}
        for count, reward in expected.items():
            self.assertEqual(RodRotationEnv._contact_reward(count, "discrete"), reward)

    def test_linear_mode_preserves_baseline(self):
        expected = {0: 0.0, 1: 0.25, 2: 0.7, 3: 0.95}
        for count, reward in expected.items():
            self.assertAlmostEqual(RodRotationEnv._contact_reward(count, "linear"), reward)

    def test_rejects_invalid_contact_count(self):
        with self.assertRaises(ValueError):
            RodRotationEnv._contact_reward(4, "discrete")

    def test_configurable_three_contact_reward(self):
        self.assertEqual(
            RodRotationEnv._contact_reward(3, "discrete", three_contact_reward=30.0),
            30.0,
        )


if __name__ == "__main__":
    unittest.main()
