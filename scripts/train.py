import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from allegro_rod_mvp import RodRotationEnv


def make_env(
    stage: int,
    tip_connect_solref: float | None,
    tip_connect_enabled: bool | None,
    axis_stabilizer_scale: float | None,
    axis_stabilizer_scale_range: tuple[float, float] | None,
    axis_tilt_penalty_weight: float,
    axis_tilt_recovery_scale: float,
    rotation_reward_scale: float,
    contact_reward_mode: str,
    three_contact_reward: float,
    contact_window_steps: int,
    contact_window_threshold: float,
):
    return Monitor(
        RodRotationEnv(
            curriculum_stage=stage,
            tip_connect_solref=tip_connect_solref,
            tip_connect_enabled=tip_connect_enabled,
            axis_stabilizer_scale=axis_stabilizer_scale,
            axis_stabilizer_scale_range=axis_stabilizer_scale_range,
            axis_tilt_penalty_weight=axis_tilt_penalty_weight,
            axis_tilt_recovery_scale=axis_tilt_recovery_scale,
            rotation_reward_scale=rotation_reward_scale,
            contact_reward_mode=contact_reward_mode,
            three_contact_reward=three_contact_reward,
            contact_window_steps=contact_window_steps,
            contact_window_threshold=contact_window_threshold,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--tip-connect-solref", type=float, default=None)
    parser.add_argument("--tip-connect", dest="tip_connect_enabled", action="store_true")
    parser.add_argument("--no-tip-connect", dest="tip_connect_enabled", action="store_false")
    parser.set_defaults(tip_connect_enabled=None)
    parser.add_argument("--axis-stabilizer-scale", type=float, default=None)
    parser.add_argument("--axis-stabilizer-min", type=float, default=None)
    parser.add_argument("--axis-stabilizer-max", type=float, default=None)
    parser.add_argument("--axis-tilt-penalty-weight", type=float, default=1.0)
    parser.add_argument("--axis-tilt-recovery-scale", type=float, default=0.0)
    parser.add_argument("--rotation-reward-scale", type=float, default=16.0)
    parser.add_argument(
        "--contact-reward-mode",
        choices=["linear", "discrete"],
        default="linear",
    )
    parser.add_argument("--three-contact-reward", type=float, default=10.0)
    parser.add_argument("--contact-window-steps", type=int, default=0)
    parser.add_argument("--contact-window-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    args = parser.parse_args()
    if (args.axis_stabilizer_min is None) != (args.axis_stabilizer_max is None):
        parser.error("--axis-stabilizer-min and --axis-stabilizer-max must be provided together")
    stabilizer_range = (
        None
        if args.axis_stabilizer_min is None
        else (args.axis_stabilizer_min, args.axis_stabilizer_max)
    )

    out = Path(args.out) if args.out else Path("checkpoints") / f"stage{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv(
        [
            lambda: make_env(
                args.stage,
                args.tip_connect_solref,
                args.tip_connect_enabled,
                args.axis_stabilizer_scale,
                stabilizer_range,
                args.axis_tilt_penalty_weight,
                args.axis_tilt_recovery_scale,
                args.rotation_reward_scale,
                args.contact_reward_mode,
                args.three_contact_reward,
                args.contact_window_steps,
                args.contact_window_threshold,
            )
        ]
    )
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        model.verbose = 1
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = lambda _: args.learning_rate
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = args.learning_rate
        if args.ent_coef is not None:
            model.ent_coef = args.ent_coef
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4 if args.learning_rate is None else args.learning_rate,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01 if args.ent_coef is None else args.ent_coef,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            verbose=1,
            seed=args.seed,
            device="cpu",
            tensorboard_log=str(out / "tb"),
        )
    callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(out),
        name_prefix="ppo_rod",
    )
    # Avoid rich progress bars when stdout is redirected to a log file.
    use_bar = sys.stdout.isatty()
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        progress_bar=use_bar,
        reset_num_timesteps=not bool(args.resume),
    )
    model.save(out / "final_model")
    env.close()
