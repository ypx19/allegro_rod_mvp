"""Tests for the T00 history × support-aware-reward ablation."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.obs_history import ObservationHistoryBuffer
from allegro_rod_mvp.support_aware_reward import (
    apply_support_aware_rotation,
    low_support_wobble_penalty,
    omega_perp_norm,
    rotation_contact_support_scale,
)
from allegro_rod_mvp.support_collapse_metrics import (
    episode_support_collapse_metrics,
)


class ObservationHistoryBufferTest(unittest.TestCase):
    def test_history_len_one_is_identity(self):
        buf = ObservationHistoryBuffer(1, 3)
        first = buf.reset(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(first, [1.0, 2.0, 3.0])
        stacked = buf.push(np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(stacked, [4.0, 5.0, 6.0])

    def test_reset_repeats_initial_frame(self):
        buf = ObservationHistoryBuffer(4, 2)
        stacked = buf.reset(np.array([7.0, 8.0]))
        np.testing.assert_array_equal(stacked, [7.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 8.0])

    def test_newest_frame_is_first(self):
        buf = ObservationHistoryBuffer(3, 1)
        buf.reset(np.array([1.0]))
        buf.push(np.array([2.0]))
        stacked = buf.push(np.array([3.0]))
        np.testing.assert_array_equal(stacked, [3.0, 2.0, 1.0])

    def test_reset_clears_prior_episode(self):
        buf = ObservationHistoryBuffer(4, 1)
        buf.reset(np.array([1.0]))
        buf.push(np.array([2.0]))
        buf.push(np.array([3.0]))
        stacked = buf.reset(np.array([9.0]))
        np.testing.assert_array_equal(stacked, [9.0, 9.0, 9.0, 9.0])


class SupportAwareRewardTest(unittest.TestCase):
    def test_gating_scales(self):
        self.assertEqual(rotation_contact_support_scale(0), 0.0)
        self.assertEqual(rotation_contact_support_scale(1), 0.1)
        self.assertEqual(rotation_contact_support_scale(2), 1.0)
        self.assertEqual(rotation_contact_support_scale(3), 1.0)

    def test_omega_perp_orthogonal_and_axial(self):
        axis = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(omega_perp_norm([2.0, 0.0, 0.0], axis), 0.0)
        self.assertAlmostEqual(omega_perp_norm([0.0, 3.0, 4.0], axis), 5.0)

    def test_wobble_zero_when_supported_or_scale_zero(self):
        self.assertEqual(low_support_wobble_penalty(2, 1.5, scale=0.5), 0.0)
        self.assertEqual(low_support_wobble_penalty(1, 1.5, scale=0.0), 0.0)

    def test_wobble_quadratic_when_unsupported(self):
        self.assertAlmostEqual(
            low_support_wobble_penalty(1, 2.0, scale=0.5),
            -0.5 * 4.0,
        )
        self.assertAlmostEqual(
            low_support_wobble_penalty(0, 2.0, scale=0.5),
            -0.5 * 4.0,
        )

    def test_disabled_leaves_rotation_unchanged(self):
        gated, effect, wobble = apply_support_aware_rotation(
            4.0,
            1,
            2.0,
            enabled=False,
            scale_0=0.0,
            scale_1=0.1,
            scale_2plus=1.0,
            wobble_scale=0.5,
        )
        self.assertEqual(gated, 4.0)
        self.assertEqual(effect, 0.0)
        self.assertEqual(wobble, 0.0)

    def test_one_contact_gates_rotation(self):
        gated, effect, wobble = apply_support_aware_rotation(
            4.0,
            1,
            0.0,
            enabled=True,
            scale_0=0.0,
            scale_1=0.1,
            scale_2plus=1.0,
            wobble_scale=0.5,
        )
        self.assertAlmostEqual(gated, 0.4)
        self.assertAlmostEqual(effect, -3.6)
        self.assertEqual(wobble, 0.0)


class SupportCollapseMetricTest(unittest.TestCase):
    def test_support_loss_recontact_and_causality(self):
        n_contact = [2, 2, 1, 1, 2, 1, 0, 0]
        omega = [0.1, 0.1, 0.2, 0.8, 0.2, 0.3, 1.5, 2.0]
        tilt = [0.05, 0.05, 0.08, 0.12, 0.10, 0.20, 0.30, 1.3]
        metrics = episode_support_collapse_metrics(
            n_contact,
            omega,
            tilt,
            termination_reason="axis_tilt",
            recontact_window=10,
            collapse_lookback=10,
            post_loss_horizon=5,
        )
        self.assertEqual(metrics["n_2plus_to_1_transitions"], 2)
        self.assertEqual(metrics["n_1_to_2plus_transitions"], 1)
        self.assertEqual(metrics["time_steps_1_contact"], 3)
        self.assertEqual(metrics["n_support_loss_recovered"], 1)
        self.assertAlmostEqual(metrics["recontact_success_rate"], 0.5)
        self.assertTrue(metrics["axis_tilt_preceded_by_support_loss"])
        self.assertEqual(metrics["n_tilt_upward_cross_0_25"], 1)
        self.assertEqual(metrics["n_tilt_excursions_recovered"], 0)

    def test_tilt_recovery_crosses_down_through_0_20(self):
        n_contact = [2] * 8
        omega = [0.1] * 8
        tilt = [0.05, 0.10, 0.26, 0.30, 0.22, 0.19, 0.10, 0.08]
        metrics = episode_support_collapse_metrics(
            n_contact, omega, tilt, termination_reason="none"
        )
        self.assertEqual(metrics["n_tilt_excursions"], 1)
        self.assertEqual(metrics["n_tilt_excursions_recovered"], 1)
        self.assertAlmostEqual(metrics["tilt_recovery_rate"], 1.0)


class EnvAblationIntegrationTest(unittest.TestCase):
    def _t00_kwargs(self, **overrides):
        kwargs = dict(
            physics_mode="tip_connect",
            reward_style="dexscrew",
            tip_anchor="bottom",
            tip_connect_solref=0.008,
            axis_stabilizer_scale=0.0,
            rod_mass_scale=400.0,
            contact_friction_scale=4.0,
            contact_friction_scaling_mode="full_vector",
            scale_rod_joint_dynamics_from_s400=True,
            scale_tip_solref_with_mass=False,
            tilt_terminate_rad=1.2,
            contact_reward_mode="discrete",
            three_contact_reward=0.3,
            contact_reward_scale=0.1,
            three_contact_required=True,
            rotation_requires_three_contacts=False,
            contact_support_termination_enabled=False,
            success_mode="net_angle",
            hand_pose_config="configs/hand_poses/my_grasp.json",
            hand_grasp_config="configs/hand_grasps/my_grasp_tip_connect_heavy.json",
        )
        kwargs.update(overrides)
        return kwargs

    def test_defaults_keep_48d_and_support_aware_off(self):
        params = inspect.signature(RodRotationEnv.__init__).parameters
        self.assertEqual(params["obs_history_len"].default, 1)
        self.assertEqual(params["support_aware_reward_enabled"].default, False)

    def test_observation_dimension_and_reset_repeat(self):
        env = RodRotationEnv(**self._t00_kwargs(obs_history_len=1))
        obs, _ = env.reset(seed=0)
        self.assertEqual(env.observation_space.shape, (48,))
        self.assertEqual(obs.shape, (48,))
        env.close()

        stacked_env = RodRotationEnv(**self._t00_kwargs(obs_history_len=4))
        stacked, _ = stacked_env.reset(seed=0)
        self.assertEqual(stacked_env.observation_space.shape, (192,))
        self.assertEqual(stacked.shape, (192,))
        np.testing.assert_allclose(stacked[0:48], stacked[48:96], atol=1e-6)
        np.testing.assert_allclose(stacked[0:48], stacked[96:144], atol=1e-6)
        np.testing.assert_allclose(stacked[0:48], stacked[144:192], atol=1e-6)
        stacked_env.close()

    def test_history_does_not_leak_across_reset(self):
        env = RodRotationEnv(**self._t00_kwargs(obs_history_len=4))
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        late = None
        for _ in range(8):
            late, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        self.assertIsNotNone(late)
        fresh, _ = env.reset(seed=1)
        self.assertEqual(fresh.shape, (192,))
        np.testing.assert_allclose(fresh[0:48], fresh[48:96], atol=1e-6)
        # A second env with the same seed must match the post-reset stack.
        other = RodRotationEnv(**self._t00_kwargs(obs_history_len=4))
        other_obs, _ = other.reset(seed=1)
        np.testing.assert_allclose(fresh, other_obs, atol=1e-5)
        env.close()
        other.close()

    def test_two_envs_keep_separate_histories(self):
        kwargs = self._t00_kwargs(obs_history_len=4)
        env_a = RodRotationEnv(**kwargs)
        env_b = RodRotationEnv(**kwargs)
        obs_a, _ = env_a.reset(seed=0)
        obs_b, _ = env_b.reset(seed=1)
        action = np.zeros(env_a.action_space.shape, dtype=np.float32)
        for _ in range(5):
            obs_a, _, term_a, trunc_a, _ = env_a.step(action)
            if term_a or trunc_a:
                break
        self.assertEqual(obs_a.shape, (192,))
        self.assertEqual(obs_b.shape, (192,))
        # env_b has not stepped; its stack must still be a repeated reset frame.
        np.testing.assert_allclose(obs_b[0:48], obs_b[48:96], atol=1e-6)
        env_a.close()
        env_b.close()

    def test_support_aware_off_matches_baseline_reward(self):
        rng = np.random.default_rng(0)
        baseline = RodRotationEnv(**self._t00_kwargs())
        gated = RodRotationEnv(
            **self._t00_kwargs(
                support_aware_reward_enabled=True,
                rotation_contact_scale_0=1.0,
                rotation_contact_scale_1=1.0,
                rotation_contact_scale_2plus=1.0,
                low_support_wobble_scale=0.0,
            )
        )
        obs_b, _ = baseline.reset(seed=3)
        obs_g, _ = gated.reset(seed=3)
        np.testing.assert_allclose(obs_b, obs_g, atol=1e-5)
        rewards_b = []
        rewards_g = []
        for _ in range(12):
            action = rng.uniform(-0.2, 0.2, size=baseline.action_space.shape).astype(
                np.float32
            )
            obs_b, rew_b, term_b, trunc_b, info_b = baseline.step(action)
            obs_g, rew_g, term_g, trunc_g, info_g = gated.step(action)
            rewards_b.append(rew_b)
            rewards_g.append(rew_g)
            self.assertEqual(info_b["contact_count"], info_g["contact_count"])
            self.assertAlmostEqual(
                info_g["reward_support_gating_effect"], 0.0, places=6
            )
            self.assertAlmostEqual(info_g["reward_low_support_wobble"], 0.0, places=6)
            if term_b or term_g or trunc_b or trunc_g:
                break
        np.testing.assert_allclose(rewards_b, rewards_g, atol=1e-5)
        baseline.close()
        gated.close()

    def test_support_aware_on_scales_logged_rotation(self):
        env = RodRotationEnv(**self._t00_kwargs(support_aware_reward_enabled=True))
        env.reset(seed=6)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(action)
            before = float(info["reward_rotation_before_support_gate"])
            after = float(info["reward_rotation"])
            n_contact = int(info["contact_count"])
            scale = rotation_contact_support_scale(n_contact)
            self.assertAlmostEqual(after, before * scale, places=5)
            if n_contact == 1:
                self.assertAlmostEqual(after, 0.1 * before, places=5)
            if n_contact == 0:
                self.assertEqual(after, 0.0)
            else:
                self.assertAlmostEqual(info["reward_low_support_wobble"], 0.0, places=6)
            if terminated or truncated:
                break
        env.close()

    def test_baseline_a_flags_match_current_t00_signature(self):
        env = RodRotationEnv(**self._t00_kwargs())
        self.assertEqual(env.obs_history_len, 1)
        self.assertFalse(env.support_aware_reward_enabled)
        self.assertEqual(env.observation_space.shape, (48,))
        self.assertEqual(env.tilt_terminate_rad, 1.2)
        self.assertEqual(env.rod_mass_scale, 400.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
