import time
from allegro_rod_mvp import RodRotationEnv

env = RodRotationEnv(render_mode="human", curriculum_stage=0, episode_seconds=20)
obs, _ = env.reset(seed=0)
try:
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.25)
        time.sleep(0.01)
        if terminated or truncated:
            obs, _ = env.reset()
finally:
    env.close()
