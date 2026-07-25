import argparse
import time
from stable_baselines3 import PPO
from allegro_rod_mvp import RodRotationEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()
    env = RodRotationEnv(render_mode="human", curriculum_stage=args.stage, episode_seconds=20)
    model = PPO.load(args.model, device="cpu")
    obs, _ = env.reset(seed=2)
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if env.step_count % 25 == 0:
                print({k: info[k] for k in ["axis_rotation_deg", "tip_error_m", "contact_count"]})
            time.sleep(0.01)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
