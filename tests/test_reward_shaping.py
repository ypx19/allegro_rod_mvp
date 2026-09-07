import inspect
import unittest

import numpy as np

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

    def test_two_support_gait_lattice(self):
        expected = {0: -2.0, 1: -0.5, 2: 0.0, 3: 0.0}
        for count, reward in expected.items():
            self.assertEqual(
                RodRotationEnv._contact_reward(count, "gait_two_support"),
                reward,
            )

    def test_contact_reward_mode_default_is_unchanged(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "contact_reward_mode"
        ].default
        self.assertEqual(default, "linear")

    def test_rejects_invalid_contact_count(self):
        with self.assertRaises(ValueError):
            RodRotationEnv._contact_reward(4, "discrete")

    def test_configurable_three_contact_reward(self):
        self.assertEqual(
            RodRotationEnv._contact_reward(3, "discrete", three_contact_reward=30.0),
            30.0,
        )

    def test_contact_gate_waits_for_full_window(self):
        self.assertEqual(
            RodRotationEnv._contact_gate_status([-10.0] * 19, 20, 5.0),
            (False, True, -190.0),
        )

    def test_contact_gate_rejects_no_simultaneous_contact(self):
        ready, satisfied, total = RodRotationEnv._contact_gate_status(
            [0.1] * 20,
            20,
            5.0,
        )
        self.assertTrue(ready)
        self.assertFalse(satisfied)
        self.assertAlmostEqual(total, 2.0)

    def test_contact_gate_accepts_one_three_contact_step(self):
        ready, satisfied, total = RodRotationEnv._contact_gate_status(
            [30.0] + [-1.0] * 19,
            20,
            5.0,
        )
        self.assertTrue(ready)
        self.assertTrue(satisfied)
        self.assertEqual(total, 11.0)


class RotationRewardCreditTest(unittest.TestCase):
    def test_default_requires_three_contacts(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "rotation_requires_three_contacts"
        ].default
        self.assertIs(default, True)
        self.assertEqual(
            RodRotationEnv._rotation_reward_credit(1.25, 2, default),
            0.0,
        )

    def test_ablation_credits_genuine_rotation_below_three_contacts(self):
        self.assertEqual(
            RodRotationEnv._rotation_reward_credit(1.25, 2, False),
            1.25,
        )

    def test_three_contact_rotation_component_is_identical(self):
        baseline = RodRotationEnv._rotation_reward_credit(1.25, 3, True)
        ablation = RodRotationEnv._rotation_reward_credit(1.25, 3, False)
        self.assertEqual(baseline, 1.25)
        self.assertEqual(ablation, baseline)

    def test_rotation_credit_flag_does_not_change_contact_gate(self):
        history = [1.0] * 17 + [0.0] * 8
        expected = RodRotationEnv._contact_gate_status(history, 25, 18.0)
        self.assertEqual(expected, (True, False, 17.0))
        for _rotation_requires_three_contacts in (True, False):
            self.assertEqual(
                RodRotationEnv._contact_gate_status(history, 25, 18.0),
                expected,
            )

    def test_contact_support_termination_defaults_enabled(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "contact_support_termination_enabled"
        ].default
        self.assertIs(default, True)

    def test_contact_reward_scale_for_each_contact_count(self):
        expected = {
            0: (-10.0, -1.0),
            1: (-1.0, -0.1),
            2: (0.1, 0.01),
            3: (0.3, 0.03),
        }
        for contact_count, rewards in expected.items():
            raw, scaled = RodRotationEnv._scaled_contact_reward(
                contact_count, "discrete", 0.3, 0.1
            )
            self.assertAlmostEqual(raw, rewards[0])
            self.assertAlmostEqual(scaled, rewards[1])

    def test_contact_reward_scale_default_preserves_reward(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "contact_reward_scale"
        ].default
        self.assertEqual(default, 1.0)
        for contact_count in range(4):
            raw, scaled = RodRotationEnv._scaled_contact_reward(
                contact_count, "discrete", 0.3, default
            )
            self.assertEqual(scaled, raw)

    def test_contact_reward_scale_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            RodRotationEnv._scaled_contact_reward(2, "discrete", 0.3, -0.1)


class SuccessModeTest(unittest.TestCase):
    def test_success_mode_default_preserves_omega_hold(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "success_mode"
        ].default
        self.assertEqual(default, "omega_hold")

    def test_net_angle_reaches_threshold(self):
        self.assertTrue(
            RodRotationEnv._net_angle_success(np.pi, np.pi, 0.1, 0.01, True, False)
        )

    def test_net_angle_just_below_threshold_fails(self):
        self.assertFalse(
            RodRotationEnv._net_angle_success(
                np.nextafter(np.pi, 0.0), np.pi, 0.1, 0.01, True, False
            )
        )

    def test_negative_and_oscillatory_zero_net_fail(self):
        for net_angle in (-np.pi, 0.0):
            self.assertFalse(
                RodRotationEnv._net_angle_success(
                    net_angle, np.pi, 0.1, 0.01, True, False
                )
            )

    def test_net_angle_rejects_tilt_tip_drop_and_nonfinite(self):
        cases = (
            (np.pi, np.pi, 0.25, 0.01, True, False),
            (np.pi, np.pi, 0.1, 0.02, True, False),
            (np.pi, np.pi, 0.1, 0.01, True, True),
            (np.pi, np.pi, 0.1, 0.01, False, False),
        )
        for args in cases:
            self.assertFalse(RodRotationEnv._net_angle_success(*args))


if __name__ == "__main__":
    unittest.main()
