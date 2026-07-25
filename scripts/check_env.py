from stable_baselines3.common.env_checker import check_env
from allegro_rod_mvp import RodRotationEnv

if __name__ == "__main__":
    env = RodRotationEnv(curriculum_stage=0)
    check_env(env, warn=True)
    obs, info = env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    print("Environment check passed.")
    print("Observation shape:", obs.shape)
    print("Last metrics:", info)
    env.close()
