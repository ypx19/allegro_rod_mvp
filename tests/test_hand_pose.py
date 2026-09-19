import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from allegro_rod_mvp import RodRotationEnv
from allegro_rod_mvp.hand_pose import make_hand_pose, write_hand_pose


class HandPoseConfigTest(unittest.TestCase):
    def _write_pose(self, directory: Path, **updates) -> Path:
        content = make_hand_pose(
            np.array([0.012, -0.034, 0.056]),
            np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
            model_variant="allegro_three_finger_rod",
            source_pose={"type": "test_fixture"},
            notes="deterministic unit-test pose",
        )
        content.update(updates)
        path = directory / "pose.json"
        path.write_text(json.dumps(content))
        return path

    def test_pose_applies_identically_without_changing_relative_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_pose(Path(tmp))
            environments = []
            try:
                for physics in ("revolute", "tip_connect"):
                    baseline = RodRotationEnv(
                        hand_model="allegro", physics_mode=physics
                    )
                    posed = RodRotationEnv(
                        hand_model="allegro",
                        physics_mode=physics,
                        hand_pose_config=str(path),
                    )
                    environments.extend((baseline, posed))
                    palm = mujoco.mj_name2id(
                        posed.model, mujoco.mjtObj.mjOBJ_BODY, "palm"
                    )
                    np.testing.assert_allclose(
                        posed.model.body_pos[palm], [0.012, -0.034, 0.056]
                    )
                    np.testing.assert_allclose(
                        posed.model.body_quat[palm],
                        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
                    )
                    child_mask = np.arange(posed.model.nbody) != palm
                    np.testing.assert_array_equal(
                        posed.model.body_pos[child_mask],
                        baseline.model.body_pos[child_mask],
                    )
                    np.testing.assert_array_equal(
                        posed.model.body_quat[child_mask],
                        baseline.model.body_quat[child_mask],
                    )
                    np.testing.assert_array_equal(
                        posed.model.jnt_pos, baseline.model.jnt_pos
                    )
                    np.testing.assert_array_equal(
                        posed.model.jnt_axis, baseline.model.jnt_axis
                    )
                    self.assertEqual(posed.action_space.shape, baseline.action_space.shape)
                    self.assertEqual(
                        posed.observation_space.shape, baseline.observation_space.shape
                    )
                np.testing.assert_allclose(
                    environments[1].model.body_pos[
                        mujoco.mj_name2id(
                            environments[1].model, mujoco.mjtObj.mjOBJ_BODY, "palm"
                        )
                    ],
                    environments[3].model.body_pos[
                        mujoco.mj_name2id(
                            environments[3].model, mujoco.mjtObj.mjOBJ_BODY, "palm"
                        )
                    ],
                )
            finally:
                for env in environments:
                    env.close()

    def test_reset_observation_and_action_dimensions_remain_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_pose(Path(tmp))
            env = RodRotationEnv(
                hand_model="allegro",
                physics_mode="tip_connect",
                hand_pose_config=str(path),
                reset_joint_noise=0.0,
                grasp_ramp_steps=1,
                grasp_hold_steps=0,
            )
            try:
                observation, _ = env.reset(seed=3)
                self.assertEqual(observation.shape, (48,))
                self.assertEqual(env.action_space.shape, (12,))
                self.assertTrue(np.isfinite(observation).all())
            finally:
                env.close()

    def test_invalid_json_and_quaternion_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            malformed = directory / "malformed.json"
            malformed.write_text("{not json")
            with self.assertRaisesRegex(ValueError, "invalid hand pose JSON"):
                RodRotationEnv(hand_pose_config=str(malformed))

            invalid_quaternion = self._write_pose(
                directory, quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]
            )
            with self.assertRaisesRegex(ValueError, "must be normalized"):
                RodRotationEnv(hand_pose_config=str(invalid_quaternion))

    def test_incompatible_model_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_pose(
                Path(tmp), compatible_model_variants=["allegro_three_finger_rod"]
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                RodRotationEnv(
                    physics_mode="revolute", hand_pose_config=str(path)
                )

    def test_save_refuses_overwrite_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._write_pose(directory)
            content = json.loads(source.read_text())
            with self.assertRaises(FileExistsError):
                write_hand_pose(source, content)
            write_hand_pose(source, content, overwrite=True)
            self.assertEqual(json.loads(source.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
