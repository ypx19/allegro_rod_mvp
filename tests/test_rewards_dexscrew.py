import inspect
import math
import unittest

import numpy as np

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.rewards_dexscrew import (
    DEXSCREW_TILT_GROWTH_CLIP_RAD,
    DEXSCREW_TILT_GROWTH_DEADZONE_RAD,
    DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE,
    DEXSCREW_TILT_GROWTH_GATE_RAD,
    DEXSCREW_TILT_RECOVERY_CLIP_RAD,
    DEXSCREW_TILT_RECOVERY_DEADZONE_RAD,
    DexScrewRewardConfig,
    compute_dexscrew_reward,
    compute_dexscrew_tilt_growth_penalty,
    compute_dexscrew_tilt_growth_weight,
    compute_dexscrew_tilt_recovery_reward,
)


def _fixture_kwargs() -> dict:
    return dict(
        axial_omega=1.0,
        fingertip_dists=np.array([0.01, 0.01, 0.01], dtype=np.float64),
        q_hand=np.zeros(12, dtype=np.float64),
        q0_hand=np.zeros(12, dtype=np.float64),
        action=np.zeros(12, dtype=np.float64),
        last_action=np.zeros(12, dtype=np.float64),
        tip_error=0.01,
        axis_tilt=0.20,
        prev_axis_tilt=0.30,
    )


class DexScrewTiltRecoveryFormulaTest(unittest.TestCase):
    def test_zero_when_tilt_increases(self):
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.20, 0.30, scale=50.0),
            0.0,
        )

    def test_zero_when_tilt_unchanged(self):
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.30, 0.30, scale=50.0),
            0.0,
        )

    def test_zero_below_or_at_deadzone(self):
        dead = DEXSCREW_TILT_RECOVERY_DEADZONE_RAD
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.20, dead, scale=50.0),
            0.0,
        )
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.20, 0.0, scale=50.0),
            0.0,
        )

    def test_positive_when_tilt_decreases_above_deadzone(self):
        value = compute_dexscrew_tilt_recovery_reward(0.30, 0.25, scale=50.0)
        self.assertAlmostEqual(value, 50.0 * 0.05)
        self.assertGreater(value, 0.0)

    def test_clips_large_single_step_recovery(self):
        value = compute_dexscrew_tilt_recovery_reward(0.80, 0.20, scale=50.0)
        self.assertAlmostEqual(value, 50.0 * DEXSCREW_TILT_RECOVERY_CLIP_RAD)

    def test_default_scale_is_zero(self):
        self.assertEqual(DexScrewRewardConfig().tilt_recovery_scale, 0.0)
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.40, 0.20, scale=0.0),
            0.0,
        )

    def test_nonfinite_inputs_return_zero(self):
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(float("nan"), 0.20, scale=50.0),
            0.0,
        )
        self.assertEqual(
            compute_dexscrew_tilt_recovery_reward(0.40, float("inf"), scale=50.0),
            0.0,
        )


class DexScrewDefaultTotalsTest(unittest.TestCase):
    """Historical DexScrew totals with default recovery scale 0 must not change."""

    def test_default_total_matches_pre_recovery_formula(self):
        cfg = DexScrewRewardConfig()
        reward, components = compute_dexscrew_reward(cfg=cfg, **_fixture_kwargs())
        rotate = 2.5 * 1.0
        prox = 2.0 * (1.0 - 0.01 / 0.04)
        tip = 0.5 * (-((0.01 / 0.025) ** 2))
        expected = rotate + prox + tip
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(components["reward_axis_tilt_recovery"], 0.0)
        self.assertEqual(components["reward_axis_tilt_growth"], 0.0)
        self.assertEqual(components["reward_axis_tilt_penalty"], 0.0)

    def test_tip_connect_penalty_total_unchanged_when_recovery_scale_is_zero(self):
        cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=0.0)
        reward, components = compute_dexscrew_reward(cfg=cfg, **_fixture_kwargs())
        rotate = 2.5 * 1.0
        prox = 2.0 * (1.0 - 0.01 / 0.04)
        tip = 0.5 * (-((0.01 / 0.025) ** 2))
        tilt = -((0.20 / 0.15) ** 2)
        expected = rotate + prox + tip + tilt
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(components["reward_axis_tilt_recovery"], 0.0)
        self.assertEqual(components["reward_axis_tilt_growth"], 0.0)
        self.assertAlmostEqual(components["reward_axis_tilt_penalty"], tilt)

    def test_omitting_prev_tilt_keeps_historical_total(self):
        cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=50.0)
        kwargs = _fixture_kwargs()
        kwargs.pop("prev_axis_tilt")
        reward_with, _ = compute_dexscrew_reward(cfg=cfg, prev_axis_tilt=None, **kwargs)
        reward_hist, _ = compute_dexscrew_reward(
            cfg=DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=0.0),
            **kwargs,
        )
        self.assertAlmostEqual(reward_with, reward_hist)

    def test_rotation_proximity_and_penalty_unchanged_when_recovery_is_added(self):
        kwargs = _fixture_kwargs()
        baseline_cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=0.0)
        recovery_cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=50.0)
        _, base = compute_dexscrew_reward(cfg=baseline_cfg, **kwargs)
        reward, rec = compute_dexscrew_reward(cfg=recovery_cfg, **kwargs)
        for key in (
            "reward_rotation",
            "reward_proximity",
            "reward_pose_anchor",
            "reward_energy",
            "reward_excess_omega",
            "reward_tip_penalty",
            "reward_axis_tilt_penalty",
        ):
            self.assertEqual(base[key], rec[key])
        self.assertGreater(rec["reward_axis_tilt_recovery"], 0.0)
        self.assertAlmostEqual(
            reward,
            sum(
                rec[key]
                for key in (
                    "reward_rotation",
                    "reward_proximity",
                    "reward_pose_anchor",
                    "reward_energy",
                    "reward_excess_omega",
                    "reward_tip_penalty",
                    "reward_axis_tilt_penalty",
                    "reward_axis_tilt_recovery",
                    "reward_axis_tilt_growth",
                )
            ),
        )

    def test_no_nans_in_components(self):
        cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_recovery_scale=50.0)
        reward, components = compute_dexscrew_reward(cfg=cfg, **_fixture_kwargs())
        self.assertTrue(math.isfinite(reward))
        for value in components.values():
            self.assertTrue(math.isfinite(float(value)))

    def test_predeclared_penalty_and_recovery_table(self):
        """Documented per-step values at 0, 0.15, 0.25, 0.5, 0.75 rad."""
        sigma = 0.15
        penalty_scale = 1.0
        recovery_scale = 50.0
        expected_penalty = {
            0.00: 0.0,
            0.15: -1.0,
            0.25: -((0.25 / sigma) ** 2),
            0.50: -((0.50 / sigma) ** 2),
            0.75: -25.0,
        }
        for tilt, expected in expected_penalty.items():
            raw = -float(np.clip((tilt / sigma) ** 2, 0.0, 25.0))
            self.assertAlmostEqual(penalty_scale * raw, expected, places=10)

        # Max-credited 0.05 rad decrease, except at/below the 0.05 rad deadzone.
        expected_recovery_max_step = {
            0.00: 0.0,
            0.15: 2.5,
            0.25: 2.5,
            0.50: 2.5,
            0.75: 2.5,
        }
        for current, expected in expected_recovery_max_step.items():
            previous = current + DEXSCREW_TILT_RECOVERY_CLIP_RAD
            value = compute_dexscrew_tilt_recovery_reward(
                previous, current, scale=recovery_scale
            )
            self.assertAlmostEqual(value, expected)


class DexScrewEnvRecoveryWiringTest(unittest.TestCase):
    def test_default_dexscrew_recovery_scale_is_zero(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "axis_tilt_recovery_scale"
        ].default
        self.assertEqual(default, 0.0)

    def test_default_env_keeps_recovery_zero_and_48d_obs(self):
        env = RodRotationEnv(
            reward_style="dexscrew",
            physics_mode="tip_connect",
            tip_anchor="bottom",
            dexscrew_tilt_scale=1.0,
        )
        try:
            self.assertEqual(env.dexscrew_cfg.tilt_recovery_scale, 0.0)
            self.assertEqual(env.dexscrew_cfg.tilt_growth_scale, 0.0)
            self.assertEqual(env.observation_space.shape, (48,))
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (48,))
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertEqual(obs.shape, (48,))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.isfinite(obs).all())
            self.assertEqual(info["reward_axis_tilt_recovery"], 0.0)
            self.assertEqual(info["reward_axis_tilt_growth"], 0.0)
            self.assertIn(terminated, (True, False))
            self.assertIn(truncated, (True, False))
        finally:
            env.close()

    def test_recovery_does_not_change_rotation_contact_or_penalty(self):
        def make_env(scale: float) -> RodRotationEnv:
            return RodRotationEnv(
                reward_style="dexscrew",
                physics_mode="tip_connect",
                tip_anchor="bottom",
                dexscrew_tilt_scale=1.0,
                axis_tilt_recovery_scale=scale,
                contact_reward_mode="discrete",
                three_contact_reward=0.3,
                contact_reward_scale=0.1,
                rotation_requires_three_contacts=False,
                contact_support_termination_enabled=False,
            )

        baseline = make_env(0.0)
        recovery = make_env(50.0)
        try:
            baseline.reset(seed=0)
            recovery.reset(seed=0)
            action = np.zeros(baseline.action_space.shape, dtype=np.float32)
            _, reward0, _, _, info0 = baseline.step(action)
            _, reward1, _, _, info1 = recovery.step(action)
            self.assertEqual(info0["reward_rotation"], info1["reward_rotation"])
            self.assertEqual(info0["reward_contact_bonus"], info1["reward_contact_bonus"])
            self.assertEqual(
                info0["reward_axis_tilt_penalty"], info1["reward_axis_tilt_penalty"]
            )
            self.assertEqual(info0["reward_axis_tilt_recovery"], 0.0)
            self.assertGreaterEqual(info1["reward_axis_tilt_recovery"], 0.0)
            self.assertTrue(math.isfinite(reward0))
            self.assertTrue(math.isfinite(reward1))
            self.assertAlmostEqual(
                reward1 - reward0,
                info1["reward_axis_tilt_recovery"] - info0["reward_axis_tilt_recovery"],
                places=6,
            )
        finally:
            baseline.close()
            recovery.close()


class DexScrewTiltGrowthFormulaTest(unittest.TestCase):
    def test_zero_when_tilt_decreases(self):
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                0.30, 0.20, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )

    def test_zero_when_tilt_unchanged(self):
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                0.25, 0.25, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )

    def test_zero_at_or_below_deadzone(self):
        dead = DEXSCREW_TILT_GROWTH_DEADZONE_RAD
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                0.0, dead, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                0.0, 0.0, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )

    def test_negative_when_tilt_increases_near_gate(self):
        value = compute_dexscrew_tilt_growth_penalty(
            0.20, 0.25, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
        )
        weight = compute_dexscrew_tilt_growth_weight(0.25)
        self.assertAlmostEqual(weight, 1.0)
        self.assertAlmostEqual(value, -50.0 * 0.05 * 1.0)
        self.assertLess(value, 0.0)

    def test_clips_large_single_step_increase(self):
        value = compute_dexscrew_tilt_growth_penalty(
            0.20, 0.80, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
        )
        self.assertAlmostEqual(
            value, -50.0 * DEXSCREW_TILT_GROWTH_CLIP_RAD * 1.0
        )

    def test_default_scale_is_zero(self):
        self.assertEqual(DexScrewRewardConfig().tilt_growth_scale, 0.0)
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(0.20, 0.30, scale=0.0),
            0.0,
        )

    def test_nonfinite_inputs_return_zero(self):
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                float("nan"), 0.30, scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )
        self.assertEqual(
            compute_dexscrew_tilt_growth_penalty(
                0.20, float("inf"), scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
            ),
            0.0,
        )

    def test_weight_ramps_through_gate(self):
        self.assertAlmostEqual(compute_dexscrew_tilt_growth_weight(0.00), 0.0)
        self.assertAlmostEqual(
            compute_dexscrew_tilt_growth_weight(0.15),
            (0.15 - DEXSCREW_TILT_GROWTH_DEADZONE_RAD)
            / (DEXSCREW_TILT_GROWTH_GATE_RAD - DEXSCREW_TILT_GROWTH_DEADZONE_RAD),
        )
        self.assertAlmostEqual(compute_dexscrew_tilt_growth_weight(0.25), 1.0)
        self.assertAlmostEqual(compute_dexscrew_tilt_growth_weight(0.50), 1.0)
        self.assertAlmostEqual(compute_dexscrew_tilt_growth_weight(0.75), 1.0)


class DexScrewTiltGrowthDefaultTotalsTest(unittest.TestCase):
    def test_default_total_unchanged_when_growth_scale_is_zero(self):
        cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_growth_scale=0.0)
        kwargs = _fixture_kwargs()
        kwargs["axis_tilt"] = 0.30
        kwargs["prev_axis_tilt"] = 0.20
        reward, components = compute_dexscrew_reward(cfg=cfg, **kwargs)
        rotate = 2.5 * 1.0
        prox = 2.0 * (1.0 - 0.01 / 0.04)
        tip = 0.5 * (-((0.01 / 0.025) ** 2))
        tilt = -((0.30 / 0.15) ** 2)
        expected = rotate + prox + tip + tilt
        self.assertAlmostEqual(reward, expected)
        self.assertEqual(components["reward_axis_tilt_growth"], 0.0)
        self.assertEqual(components["reward_axis_tilt_recovery"], 0.0)

    def test_rotation_proximity_penalty_recovery_unchanged_when_growth_is_added(self):
        kwargs = _fixture_kwargs()
        kwargs["axis_tilt"] = 0.30
        kwargs["prev_axis_tilt"] = 0.20
        baseline_cfg = DexScrewRewardConfig(tilt_scale=1.0, tilt_growth_scale=0.0)
        growth_cfg = DexScrewRewardConfig(
            tilt_scale=1.0,
            tilt_growth_scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE,
        )
        _, base = compute_dexscrew_reward(cfg=baseline_cfg, **kwargs)
        reward, grown = compute_dexscrew_reward(cfg=growth_cfg, **kwargs)
        for key in (
            "reward_rotation",
            "reward_proximity",
            "reward_pose_anchor",
            "reward_energy",
            "reward_excess_omega",
            "reward_tip_penalty",
            "reward_axis_tilt_penalty",
            "reward_axis_tilt_recovery",
        ):
            self.assertEqual(base[key], grown[key])
        self.assertLess(grown["reward_axis_tilt_growth"], 0.0)
        self.assertAlmostEqual(
            reward,
            sum(
                grown[key]
                for key in (
                    "reward_rotation",
                    "reward_proximity",
                    "reward_pose_anchor",
                    "reward_energy",
                    "reward_excess_omega",
                    "reward_tip_penalty",
                    "reward_axis_tilt_penalty",
                    "reward_axis_tilt_recovery",
                    "reward_axis_tilt_growth",
                )
            ),
        )

    def test_no_nans_in_components(self):
        cfg = DexScrewRewardConfig(
            tilt_scale=1.0,
            tilt_growth_scale=DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE,
        )
        kwargs = _fixture_kwargs()
        kwargs["axis_tilt"] = 0.30
        kwargs["prev_axis_tilt"] = 0.20
        reward, components = compute_dexscrew_reward(cfg=cfg, **kwargs)
        self.assertTrue(math.isfinite(reward))
        for value in components.values():
            self.assertTrue(math.isfinite(float(value)))

    def test_predeclared_growth_table(self):
        """Documented per-step growth at current tilt 0, 0.15, 0.25, 0.50, 0.75."""
        scale = DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE
        expected_w = {
            0.00: 0.0,
            0.15: 0.5,
            0.25: 1.0,
            0.50: 1.0,
            0.75: 1.0,
        }
        expected_d01 = {
            0.00: 0.0,
            0.15: -0.25,
            0.25: -0.50,
            0.50: -0.50,
            0.75: -0.50,
        }
        expected_d05 = {
            0.00: 0.0,
            0.15: -1.25,
            0.25: -2.50,
            0.50: -2.50,
            0.75: -2.50,
        }
        for current, weight in expected_w.items():
            self.assertAlmostEqual(
                compute_dexscrew_tilt_growth_weight(current), weight
            )
            for delta, expected in (
                (0.01, expected_d01[current]),
                (0.05, expected_d05[current]),
            ):
                previous = current - delta
                value = compute_dexscrew_tilt_growth_penalty(
                    previous, current, scale=scale
                )
                self.assertAlmostEqual(value, expected, places=10)


class DexScrewEnvGrowthWiringTest(unittest.TestCase):
    def test_default_dexscrew_growth_scale_is_zero(self):
        default = inspect.signature(RodRotationEnv.__init__).parameters[
            "axis_tilt_growth_scale"
        ].default
        self.assertEqual(default, 0.0)

    def test_growth_does_not_change_rotation_contact_or_penalty(self):
        def make_env(scale: float) -> RodRotationEnv:
            return RodRotationEnv(
                reward_style="dexscrew",
                physics_mode="tip_connect",
                tip_anchor="bottom",
                dexscrew_tilt_scale=1.0,
                axis_tilt_growth_scale=scale,
                axis_tilt_recovery_scale=0.0,
                contact_reward_mode="discrete",
                three_contact_reward=0.3,
                contact_reward_scale=0.1,
                rotation_requires_three_contacts=False,
                contact_support_termination_enabled=False,
            )

        baseline = make_env(0.0)
        growth = make_env(DEXSCREW_TILT_GROWTH_EXPERIMENT_SCALE)
        try:
            baseline.reset(seed=0)
            growth.reset(seed=0)
            action = np.zeros(baseline.action_space.shape, dtype=np.float32)
            _, reward0, _, _, info0 = baseline.step(action)
            _, reward1, _, _, info1 = growth.step(action)
            self.assertEqual(info0["reward_rotation"], info1["reward_rotation"])
            self.assertEqual(info0["reward_contact_bonus"], info1["reward_contact_bonus"])
            self.assertEqual(
                info0["reward_axis_tilt_penalty"], info1["reward_axis_tilt_penalty"]
            )
            self.assertEqual(info0["reward_axis_tilt_recovery"], 0.0)
            self.assertEqual(info1["reward_axis_tilt_recovery"], 0.0)
            self.assertEqual(info0["reward_axis_tilt_growth"], 0.0)
            self.assertLessEqual(info1["reward_axis_tilt_growth"], 0.0)
            self.assertTrue(math.isfinite(reward0))
            self.assertTrue(math.isfinite(reward1))
            self.assertAlmostEqual(
                reward1 - reward0,
                info1["reward_axis_tilt_growth"] - info0["reward_axis_tilt_growth"],
                places=6,
            )
            self.assertEqual(baseline.observation_space.shape, (48,))
            self.assertEqual(growth.observation_space.shape, (48,))
        finally:
            baseline.close()
            growth.close()


if __name__ == "__main__":
    unittest.main()
