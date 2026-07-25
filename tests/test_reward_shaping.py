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


if __name__ == "__main__":
    unittest.main()
