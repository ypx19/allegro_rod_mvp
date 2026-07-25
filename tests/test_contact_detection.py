import unittest

import numpy as np

from allegro_rod_mvp import RodRotationEnv


class ContactDetectionTest(unittest.TestCase):
    def test_reset_detects_two_known_contacts(self):
        env = RodRotationEnv(
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
