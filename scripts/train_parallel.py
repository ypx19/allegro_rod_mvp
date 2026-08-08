#!/usr/bin/env python3
"""Parallel MuJoCo PPO trainer (SubprocVecEnv + CUDA + VecNormalize).

EXP-infra entrypoint for the DexScrew-style track. Stage 0 reward/physics are
unchanged; this script only changes training plumbing.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from allegro_rod_mvp import RodRotationEnv

ROOT = Path(__file__).resolve().parents[1]


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
    three_contact_required: bool = False,
    physics_mode: str = "tip_connect",
    reward_style: str = "stage",
    privileged_obs: bool = False,
    dexscrew_rotate_scale: float = 2.5,
    dexscrew_prox_scale: float = 2.0,
    dexscrew_pose_scale: float = 0.1,
    dexscrew_energy_scale: float = 0.05,
    dexscrew_excess_scale: float = 0.1,
    dexscrew_tilt_scale: float | None = None,
    dexscrew_tip_penalty_scale: float = 0.5,
    omega_success_threshold: float = 0.5,
    omega_success_hold_seconds: float = 10.0,
    adaptive_reward_mass: bool = False,
    mass_target_rot: float = 0.45,
    mass_target_tilt: float = 0.45,
    mass_ema_tau_steps: float = 2000.0,
    mass_kappa: float = 0.08,
    rod_mass_scale: float = 1.0,
    rod_friction_cap: float = 4.0,
    tilt_terminate_rad: float = 0.7,
    tip_anchor: str = "top",
    dexscrew_tip_sigma: float = 0.025,
    rank: int = 0,
    seed: int = 0,
) -> gym.Env:
    """Top-level picklable env factory for SubprocVecEnv."""
    env = Monitor(
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
            three_contact_required=three_contact_required,
            physics_mode=physics_mode,
            reward_style=reward_style,
            privileged_obs=privileged_obs,
            dexscrew_rotate_scale=dexscrew_rotate_scale,
            dexscrew_prox_scale=dexscrew_prox_scale,
            dexscrew_pose_scale=dexscrew_pose_scale,
            dexscrew_energy_scale=dexscrew_energy_scale,
            dexscrew_excess_scale=dexscrew_excess_scale,
            dexscrew_tilt_scale=dexscrew_tilt_scale,
            dexscrew_tip_penalty_scale=dexscrew_tip_penalty_scale,
            omega_success_threshold=omega_success_threshold,
            omega_success_hold_seconds=omega_success_hold_seconds,
            adaptive_reward_mass=adaptive_reward_mass,
            mass_target_rot=mass_target_rot,
            mass_target_tilt=mass_target_tilt,
            mass_ema_tau_steps=mass_ema_tau_steps,
            mass_kappa=mass_kappa,
            rod_mass_scale=rod_mass_scale,
            rod_friction_cap=rod_friction_cap,
            tilt_terminate_rad=tilt_terminate_rad,
            tip_anchor=tip_anchor,
            dexscrew_tip_sigma=dexscrew_tip_sigma,
        )
    )
    env.reset(seed=seed + rank)
    return env


def parse_net_arch(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("net-arch must be a comma-separated list of ints")
    return [int(p) for p in parts]


def git_metadata() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def framework_versions() -> dict[str, str]:
    import gymnasium
    import mujoco
    import stable_baselines3
    import torch

    return {
        "gymnasium": gymnasium.__version__,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }


class MetricsCsvCallback(BaseCallback):
    """Append SB3 logger scalars to metrics.csv each rollout."""

    def __init__(self, csv_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._header_written = False
        self._start_wall = None

    def _on_training_start(self) -> None:
        self._start_wall = datetime.now(timezone.utc).timestamp()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        row: dict[str, Any] = {
            "step": int(self.num_timesteps),
            "wall_time": float(datetime.now(timezone.utc).timestamp() - (self._start_wall or 0.0)),
        }
        # Monitor episode buffer (available before SB3 dumps train/*).
        if len(self.model.ep_info_buffer) > 0:
            rews = [ep["r"] for ep in self.model.ep_info_buffer if "r" in ep]
            lens = [ep["l"] for ep in self.model.ep_info_buffer if "l" in ep]
            if rews:
                row["episode_return"] = float(np.mean(rews))
            if lens:
                row["episode_length"] = float(np.mean(lens))
        if self.logger is not None:
            for key, val in sorted(self.logger.name_to_value.items()):
                row[key.replace("/", "_")] = float(val)
        if len(row) <= 2:
            return
        write_header = not self._header_written or not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
                self._header_written = True
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")


class VecNormalizeCheckpointCallback(BaseCallback):
    """Save VecNormalize stats whenever the model is checkpointed (and periodically)."""

    def __init__(self, save_path: Path, save_freq: int, name_prefix: str = "ppo_rod", verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = max(1, int(save_freq))
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            self.save_path.mkdir(parents=True, exist_ok=True)
            step = int(self.num_timesteps)
            env.save(str(self.save_path / f"{self.name_prefix}_{step}_steps_vecnormalize.pkl"))
            env.save(str(self.save_path / "vecnormalize.pkl"))
        return True


class AdaptiveMassCallback(BaseCallback):
    """Log per-step adaptive mass fractions / live scales to TensorBoard each rollout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._mass_rot: list[float] = []
        self._mass_tilt: list[float] = []
        self._rot_scale: list[float] = []
        self._tilt_scale: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if not info:
                continue
            if "mass_rot" in info:
                self._mass_rot.append(float(info["mass_rot"]))
            if "mass_tilt" in info:
                self._mass_tilt.append(float(info["mass_tilt"]))
            if "dexscrew_rotate_scale_live" in info:
                self._rot_scale.append(float(info["dexscrew_rotate_scale_live"]))
            if "dexscrew_tilt_scale_live" in info:
                self._tilt_scale.append(float(info["dexscrew_tilt_scale_live"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is None:
            self._mass_rot.clear()
            self._mass_tilt.clear()
            self._rot_scale.clear()
            self._tilt_scale.clear()
            return
        if self._mass_rot:
            self.logger.record("mass/rot", float(np.mean(self._mass_rot)))
        if self._mass_tilt:
            self.logger.record("mass/tilt", float(np.mean(self._mass_tilt)))
        if self._rot_scale:
            self.logger.record("mass/rotate_scale", float(np.mean(self._rot_scale)))
        if self._tilt_scale:
            self.logger.record("mass/tilt_scale", float(np.mean(self._tilt_scale)))
        self._mass_rot.clear()
        self._mass_tilt.clear()
        self._rot_scale.clear()
        self._tilt_scale.clear()


def build_vec_env(
    args: argparse.Namespace,
    stabilizer_range: tuple[float, float] | None,
    vecnormalize_path: str | None = None,
):
    env_fns = [
        partial(
            make_env,
            stage=args.stage,
            tip_connect_solref=args.tip_connect_solref,
            tip_connect_enabled=args.tip_connect_enabled,
            axis_stabilizer_scale=args.axis_stabilizer_scale,
            axis_stabilizer_scale_range=stabilizer_range,
            axis_tilt_penalty_weight=args.axis_tilt_penalty_weight,
            axis_tilt_recovery_scale=args.axis_tilt_recovery_scale,
            rotation_reward_scale=args.rotation_reward_scale,
            contact_reward_mode=args.contact_reward_mode,
            three_contact_reward=args.three_contact_reward,
            contact_window_steps=args.contact_window_steps,
            contact_window_threshold=args.contact_window_threshold,
            three_contact_required=args.three_contact_required,
            physics_mode=args.physics,
            reward_style=args.reward_style,
            privileged_obs=args.privileged_obs,
            dexscrew_rotate_scale=args.dexscrew_rotate_scale,
            dexscrew_prox_scale=args.dexscrew_prox_scale,
            dexscrew_pose_scale=args.dexscrew_pose_scale,
            dexscrew_energy_scale=args.dexscrew_energy_scale,
            dexscrew_excess_scale=args.dexscrew_excess_scale,
            dexscrew_tilt_scale=args.dexscrew_tilt_scale,
            dexscrew_tip_penalty_scale=args.dexscrew_tip_penalty_scale,
            omega_success_threshold=args.omega_success_threshold,
            omega_success_hold_seconds=args.omega_success_hold_seconds,
            adaptive_reward_mass=args.adaptive_reward_mass,
            mass_target_rot=args.mass_target_rot,
            mass_target_tilt=args.mass_target_tilt,
            mass_ema_tau_steps=args.mass_ema_tau_steps,
            mass_kappa=args.mass_kappa,
            rod_mass_scale=args.rod_mass_scale,
            rod_friction_cap=args.rod_friction_cap,
            tilt_terminate_rad=args.tilt_terminate_rad,
            tip_anchor=args.tip_anchor,
            dexscrew_tip_sigma=args.dexscrew_tip_sigma,
            rank=rank,
            seed=args.seed,
        )
        for rank in range(args.num_envs)
    ]
    vec = SubprocVecEnv(env_fns)
    if args.vec_normalize:
        if vecnormalize_path:
            vec = VecNormalize.load(vecnormalize_path, vec)
            vec.training = True
            vec.norm_reward = True
        else:
            vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec


def smoke_load_checkpoint(
    model_path: Path,
    vecnorm_path: Path | None,
    args: argparse.Namespace,
    stabilizer_range: tuple[float, float] | None,
) -> None:
    """Load final model (+ VecNormalize) and take a few steps on a fresh env."""
    env = build_vec_env(
        args,
        stabilizer_range,
        vecnormalize_path=str(vecnorm_path) if vecnorm_path and vecnorm_path.exists() else None,
    )
    if isinstance(env, VecNormalize):
        env.training = False
        env.norm_reward = False
    model = PPO.load(str(model_path), env=env, device=args.device)
    obs = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _infos = env.step(action)
        if not np.isfinite(rewards).all():
            raise RuntimeError(f"Non-finite rewards during load smoke: {rewards}")
    env.close()
    print(f"Load smoke OK: {model_path}", flush=True)


def write_run_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    net_arch: list[int],
    command: list[str],
    notes: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    config = {
        "run_id": args.run_id,
        "stage": args.stage,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "num_envs": args.num_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate if args.learning_rate is not None else 3e-4,
        "ent_coef": args.ent_coef if args.ent_coef is not None else 0.01,
        "net_arch": net_arch,
        "vec_normalize": args.vec_normalize,
        "reward_style": args.reward_style,
        "physics": args.physics,
        "tip_connect_solref": args.tip_connect_solref,
        "tip_connect_enabled": args.tip_connect_enabled,
        "axis_stabilizer_scale": args.axis_stabilizer_scale,
        "axis_tilt_penalty_weight": args.axis_tilt_penalty_weight,
        "axis_tilt_recovery_scale": args.axis_tilt_recovery_scale,
        "rotation_reward_scale": args.rotation_reward_scale,
        "contact_reward_mode": args.contact_reward_mode,
        "three_contact_reward": args.three_contact_reward,
        "contact_window_steps": args.contact_window_steps,
        "contact_window_threshold": args.contact_window_threshold,
        "three_contact_required": args.three_contact_required,
        "omega_success_threshold": args.omega_success_threshold,
        "omega_success_hold_seconds": args.omega_success_hold_seconds,
        "adaptive_reward_mass": args.adaptive_reward_mass,
        "mass_target_rot": args.mass_target_rot,
        "mass_target_tilt": args.mass_target_tilt,
        "mass_ema_tau_steps": args.mass_ema_tau_steps,
        "mass_kappa": args.mass_kappa,
        "rod_mass_scale": args.rod_mass_scale,
        "rod_friction_cap": args.rod_friction_cap,
        "tilt_terminate_rad": args.tilt_terminate_rad,
        "tip_anchor": args.tip_anchor,
        "dexscrew_tip_penalty_scale": args.dexscrew_tip_penalty_scale,
        "dexscrew_tip_sigma": args.dexscrew_tip_sigma,
        "dexscrew_tilt_scale": (
            args.dexscrew_tilt_scale
            if args.dexscrew_tilt_scale is not None
            else (
                1.0
                if args.reward_style == "dexscrew" and args.physics == "tip_connect"
                else 0.0
            )
        ),
        "checkpoint_freq": args.checkpoint_freq,
        "change_from_baseline": (
            "Training stack only: SubprocVecEnv + CUDA + net_arch [512,256,128] + VecNormalize; "
            "Stage 0 reward/physics unchanged."
        ),
        "success_criteria": (
            "DexScrew: sustain axial ω > omega_success_threshold for "
            "omega_success_hold_seconds (plus tip/tilt/drop gates). "
            "Stage reward_style: unwrapped_angle > π (legacy)."
        ),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    meta = {
        "run_id": args.run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **git_metadata(),
        "command": " ".join(command),
        "seed": args.seed,
        "device": args.device,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "framework_versions": framework_versions(),
        "baseline_run": "scripts/train.py DummyVecEnv CPU Stage 0",
        "notes": notes,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def write_summary(run_dir: Path, args: argparse.Namespace, ok: bool, detail: str) -> None:
    status = "passed" if ok else "failed"
    text = f"""# EXP-infra: Parallel MuJoCo + CUDA PPO

## Question
Does SubprocVecEnv + CUDA + net_arch [512,256,128] + VecNormalize train Stage 0 without NaNs and produce a loadable checkpoint?

## Change from Baseline
Only the training stack changed relative to `scripts/train.py` (DummyVecEnv, CPU, [256,256]). Stage 0 reward and physics are unchanged.

## Result
{status}. {detail}

## Key Settings
- num_envs: {args.num_envs}
- device: {args.device}
- net_arch: {args.net_arch}
- vec_normalize: {args.vec_normalize}
- steps: {args.steps}
- seed: {args.seed}

## Artifacts
- Config: `config.yaml`
- Metadata: `metadata.json`
- Metrics: `metrics.csv`
- Checkpoints: `checkpoints/`
- VecNormalize: `checkpoints/vecnormalize.pkl`
- TensorBoard: `tb/`

## Conclusion
{"Adopt parallel training plumbing for subsequent DexScrew-style EXPs." if ok else "Investigate infra failure before Arm A."}

## Recommended Next Step
{"EXP-A0: revolute MJCF + DexScrew-style ω reward core, n_envs=8, 2e5 smoke." if ok else "Fix infra failure, then rerun EXP-infra."}
"""
    (run_dir / "summary.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel PPO trainer for DexScrew-style track")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Run directory (default runs/<run-id>)")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--net-arch", type=parse_net_arch, default="512,256,128")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vec-normalize", dest="vec_normalize", action="store_true")
    parser.add_argument("--no-vec-normalize", dest="vec_normalize", action="store_false")
    parser.set_defaults(vec_normalize=True)
    parser.add_argument("--reward-style", choices=["stage", "dexscrew"], default="stage")
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--privileged-obs", action="store_true")
    parser.add_argument("--dexscrew-rotate-scale", type=float, default=2.5)
    parser.add_argument("--dexscrew-prox-scale", type=float, default=2.0)
    parser.add_argument("--dexscrew-pose-scale", type=float, default=0.1)
    parser.add_argument("--dexscrew-energy-scale", type=float, default=0.05)
    parser.add_argument("--dexscrew-excess-scale", type=float, default=0.1)
    parser.add_argument(
        "--dexscrew-tilt-scale",
        type=float,
        default=None,
        help="Tilt punishment scale for DexScrew. Default: 1.0 on tip_connect, 0.0 on revolute.",
    )
    parser.add_argument("--dexscrew-tip-penalty-scale", type=float, default=0.5)
    parser.add_argument(
        "--omega-success-threshold",
        type=float,
        default=0.5,
        help="DexScrew success: require axial ω above this (rad/s).",
    )
    parser.add_argument(
        "--omega-success-hold-seconds",
        type=float,
        default=10.0,
        help="DexScrew success: consecutive seconds with ω above threshold.",
    )
    parser.add_argument(
        "--adaptive-reward-mass",
        action="store_true",
        help="EMA-adapt rotate/tilt scales toward mass targets (tip-connect+dexscrew only).",
    )
    parser.add_argument("--mass-target-rot", type=float, default=0.45)
    parser.add_argument("--mass-target-tilt", type=float, default=0.45)
    parser.add_argument("--mass-ema-tau-steps", type=float, default=2000.0)
    parser.add_argument("--mass-kappa", type=float, default=0.08)
    parser.add_argument(
        "--rod-mass-scale",
        type=float,
        default=1.0,
        help="Scale rod mass and inertia. Friction uses min(scale, --rod-friction-cap).",
    )
    parser.add_argument(
        "--rod-friction-cap",
        type=float,
        default=4.0,
        help="Cap on friction scale relative to baseline (default 4).",
    )
    parser.add_argument(
        "--tilt-terminate-rad",
        type=float,
        default=0.7,
        help="Tip-connect hard tilt termination threshold (rad).",
    )
    parser.add_argument(
        "--tip-anchor",
        choices=["top", "bottom"],
        default="top",
        help="Rod tip / equality location: top hang (stable) or bottom support (inverted).",
    )
    parser.add_argument("--dexscrew-tip-sigma", type=float, default=0.025)
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
    parser.add_argument("--contact-reward-mode", choices=["linear", "discrete"], default="linear")
    parser.add_argument("--three-contact-reward", type=float, default=10.0)
    parser.add_argument("--contact-window-steps", type=int, default=0)
    parser.add_argument("--contact-window-threshold", type=float, default=0.0)
    parser.add_argument(
        "--three-contact-required",
        action="store_true",
        help="Require 3-finger contact for rotation credit; hard-gate on a 3-contact window.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument(
        "--vecnormalize-path",
        type=str,
        default=None,
        help="Optional VecNormalize stats to load when resuming (pkl).",
    )
    parser.add_argument(
        "--continue-timesteps",
        action="store_true",
        help="Keep SB3 num_timesteps from the resumed model (default: reset so --steps is additional).",
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--skip-load-smoke", action="store_true")
    args = parser.parse_args(argv)

    if (args.axis_stabilizer_min is None) != (args.axis_stabilizer_max is None):
        parser.error("--axis-stabilizer-min and --axis-stabilizer-max must be provided together")
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            parser.error("CUDA requested but torch.cuda.is_available() is False")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]

    if args.run_id is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.run_id = f"{stamp}-exp-infra-subproc{args.num_envs}-cuda-seed{args.seed}"
    run_dir = Path(args.out) if args.out else ROOT / "runs" / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    notes = args.notes or (
        "Parallel PPO run via scripts/train_parallel.py (SubprocVecEnv + CUDA + VecNormalize)."
    )
    write_run_artifacts(
        run_dir,
        args,
        net_arch=list(args.net_arch),
        command=command,
        notes=notes,
    )

    stabilizer_range = (
        None
        if args.axis_stabilizer_min is None
        else (args.axis_stabilizer_min, args.axis_stabilizer_max)
    )

    env = build_vec_env(args, stabilizer_range, vecnormalize_path=args.vecnormalize_path)
    net_arch = {"pi": list(args.net_arch), "vf": list(args.net_arch)}
    lr = 3e-4 if args.learning_rate is None else args.learning_rate
    ent = 0.01 if args.ent_coef is None else args.ent_coef

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.verbose = 1
        # Keep tensorboard under this run even when resuming.
        model.tensorboard_log = str(run_dir / "tb")
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
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            policy_kwargs={"net_arch": net_arch},
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
        )

    # CheckpointCallback save_freq counts env.step() calls (= timesteps / n_envs).
    ckpt_every = max(1, args.checkpoint_freq // args.num_envs)
    callbacks = [
        CheckpointCallback(
            save_freq=ckpt_every,
            save_path=str(ckpt_dir),
            name_prefix="ppo_rod",
        ),
        VecNormalizeCheckpointCallback(
            save_path=ckpt_dir,
            save_freq=ckpt_every,
            name_prefix="ppo_rod",
        ),
        MetricsCsvCallback(run_dir / "metrics.csv"),
        AdaptiveMassCallback(),
    ]

    use_bar = sys.stdout.isatty()
    log_path = run_dir / "logs" / "train.log"
    ok = False
    detail = ""
    reset_ts = not args.continue_timesteps
    try:
        print(
            f"Starting run: stage={args.stage} n_envs={args.num_envs} device={args.device} "
            f"net_arch={args.net_arch} steps={args.steps} ent_coef={ent} "
            f"reset_timesteps={reset_ts} run={args.run_id}",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(f"command: {' '.join(command)}\n")
            log_f.flush()
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=use_bar,
            reset_num_timesteps=reset_ts,
            tb_log_name="PPO",
        )
        final_path = ckpt_dir / "final_model"
        model.save(str(final_path))
        vecnorm_path = None
        if args.vec_normalize and isinstance(env, VecNormalize):
            vecnorm_path = ckpt_dir / "vecnormalize.pkl"
            env.save(str(vecnorm_path))

        # Finite-reward sanity from latest metrics row if present.
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) >= 2:
                detail = f"Wrote {len(lines) - 1} metrics rows; final checkpoint at {final_path}.zip"
            else:
                detail = f"Checkpoint saved at {final_path}.zip"
        else:
            detail = f"Checkpoint saved at {final_path}.zip"

        if not args.skip_load_smoke:
            zip_path = Path(str(final_path) + ".zip")
            smoke_load_checkpoint(zip_path, vecnorm_path, args, stabilizer_range)
            detail += " Load smoke passed."
        ok = True
    except Exception as exc:  # noqa: BLE001 — surface infra failures in summary
        detail = f"{type(exc).__name__}: {exc}"
        print(f"EXP-infra FAILED: {detail}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"ERROR: {detail}\n")
        ok = False
        write_summary(run_dir, args, ok=False, detail=detail)
        env.close()
        raise
    env.close()
    write_summary(run_dir, args, ok=ok, detail=detail)

    print(f"EXP-infra complete: ok={ok} run_dir={run_dir}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
