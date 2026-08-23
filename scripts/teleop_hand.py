#!/usr/bin/env python3
"""Keyboard teleoperation of the Allegro hand in MuJoCo.

Controls:
  1-9       Select actuator 0..8
  , / .     Select previous / next actuator (covers all 12 Allegro joints)
  i / k     Increment / decrement selected joint target by step_size
  I / K     Increment / decrement by 5x step_size
  [ / ]     Decrease / increase step_size
  0         Zero action (hold current position)
  r         Reset episode
  p         Pause / unpause simulation
  q / ESC   Quit

  Space     Print current joint targets
  v         Save PNG snapshot (with --snapshots on headless/offscreen)

HUD prints axis_tilt_deg, axis_rotation_deg, contact info every 0.5s.
On episode end, prints summary with peak rotation, tilt stats, termination reason.
Logs per-step metrics to a CSV file for later analysis.

On SSH servers without DISPLAY, use --headless (terminal HUD only) or ssh -X for --viewer.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _bootstrap_python() -> None:
    """Use project .venv when base/system Python lacks dependencies."""
    try:
        import gymnasium  # noqa: F401
    except ModuleNotFoundError:
        venv_py = _ROOT / ".venv" / "bin" / "python"
        in_project_venv = Path(sys.prefix).resolve() == (_ROOT / ".venv").resolve()
        if venv_py.is_file() and not in_project_venv:
            os.execv(str(venv_py), [str(venv_py), *sys.argv])
        sys.stderr.write(
            "Missing dependencies (gymnasium/mujoco). Install with:\n"
            "  pip install -e .\n"
            "Or activate the project venv:\n"
            "  source .venv/bin/activate\n"
        )
        raise SystemExit(1) from None


_bootstrap_python()

import numpy as np

from allegro_rod_mvp import RodRotationEnv

FINGER_NAMES = ["finger0", "finger1", "finger2"]


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _print_display_help() -> None:
    sys.stderr.write(
        "No graphical display available (DISPLAY / WAYLAND_DISPLAY unset).\n\n"
        "Options:\n"
        "  1) SSH with X11 forwarding:  ssh -X user@host\n"
        "     then:  python scripts/teleop_hand.py --viewer\n"
        "  2) Headless teleop (terminal HUD only):\n"
        "     python scripts/teleop_hand.py --headless\n"
        "  3) Virtual framebuffer (if xvfb installed):\n"
        "     xvfb-run -a python scripts/teleop_hand.py --viewer\n"
    )


def _resolve_render_mode(args: argparse.Namespace) -> str | None:
    if args.headless:
        return "rgb_array" if args.snapshots else None
    if args.viewer:
        if not _has_display():
            _print_display_help()
            raise SystemExit(1)
        return "human"
    if _has_display():
        return "human"
    print(
        "WARNING: No DISPLAY detected — running headless (terminal HUD only).\n"
        "  Use --viewer when X11 is available, or --headless to silence this.\n"
        "  Press 'v' during teleop to save PNG snapshots when --snapshots is set.\n"
    )
    return "rgb_array" if args.snapshots else None


def _setup_terminal():
    """Put terminal in raw/cbreak mode for single-keypress reading."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old_settings


def _restore_terminal(fd, old_settings):
    import termios

    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key(fd) -> str | None:
    """Non-blocking single character read. Returns None if nothing available."""
    import select

    if select.select([fd], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def _print_hud(
    step: int,
    info: dict,
    selected_joint: int,
    ctrl_target: np.ndarray,
    step_size: float,
    paused: bool,
    joints_per_finger: int,
):
    tilt = info.get("axis_tilt_deg", 0.0)
    rot = info.get("axis_rotation_deg", 0.0)
    cc = info.get("contact_count", 0)
    fc = info.get("finger_contacts", [0, 0, 0])
    tip_err = info.get("tip_error_m", 0.0)
    omega = info.get("axial_omega", 0.0)
    pause_str = " [PAUSED]" if paused else ""
    finger_idx = selected_joint // joints_per_finger
    local_idx = selected_joint % joints_per_finger
    joint_label = f"{FINGER_NAMES[finger_idx]}.j{local_idx}"
    tilt_marker = "OK" if info.get("axis_tilt_rad", 1.0) < 0.25 else "!!"
    print(
        f"\r\033[K"
        f"step={step:5d} | "
        f"tilt={tilt:5.1f}°[{tilt_marker}] rot={rot:7.1f}° ω={omega:+5.2f} | "
        f"tip={tip_err*100:4.1f}cm | "
        f"contact={cc} [{fc[0]},{fc[1]},{fc[2]}] | "
        f"sel={selected_joint}({joint_label}) tgt={ctrl_target[selected_joint]:+.3f} Δ={step_size:.3f}"
        f"{pause_str}",
        end="",
        flush=True,
    )


def _print_episode_summary(metrics: list[dict], episode_num: int):
    if not metrics:
        return
    rots = [m["axis_rotation_deg"] for m in metrics]
    tilts = [m["axis_tilt_rad"] for m in metrics]
    peak_rot = max(rots)
    min_rot = min(rots)
    tilt_ok_frac = sum(1 for t in tilts if t < 0.25) / len(tilts)
    max_tilt = max(tilts)
    mean_tilt = np.mean(tilts)
    final = metrics[-1]
    term = final.get("termination_reason", "none")
    contacts = [m["contact_count"] for m in metrics]
    mean_contacts = np.mean(contacts)

    achievable = peak_rot >= 180.0 and tilt_ok_frac > 0.0
    print(f"\n{'='*72}")
    print(f"Episode {episode_num} Summary ({len(metrics)} steps)")
    print(f"  Peak rotation:     {peak_rot:+.1f}° (min {min_rot:+.1f}°)")
    print(f"  Final rotation:    {rots[-1]:+.1f}°")
    print(f"  Tilt OK fraction:  {tilt_ok_frac:.1%} (< 0.25 rad)")
    print(f"  Max tilt:          {np.degrees(max_tilt):.1f}° ({max_tilt:.3f} rad)")
    print(f"  Mean tilt:         {np.degrees(mean_tilt):.1f}° ({mean_tilt:.3f} rad)")
    print(f"  Mean contacts:     {mean_contacts:.2f}")
    print(f"  Tip error (final): {final.get('tip_error_m', 0)*100:.2f} cm")
    print(f"  Termination:       {term}")
    print(f"  Balance+Rotate achievable (rot>=180° & any tilt<0.25): {'YES' if achievable else 'NO'}")
    print(f"{'='*72}\n")


def run_teleop(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_mode = _resolve_render_mode(args)
    if render_mode == "rgb_array" and not _has_display():
        os.environ.setdefault("MUJOCO_GL", "egl")

    env = RodRotationEnv(
        render_mode=render_mode,
        curriculum_stage=args.stage,
        episode_seconds=args.episode_seconds,
        axis_stabilizer_scale=args.axis_stabilizer_scale,
        tip_connect_solref=args.tip_connect_solref,
        rod_mass_scale=args.rod_mass_scale,
        rod_friction_cap=args.rod_friction_cap,
        tilt_terminate_rad=args.tilt_terminate_rad,
        tip_anchor=args.tip_anchor,
        physics_mode=args.physics,
        hand_model=args.hand_model,
        hand_pose_config=args.hand_pose_config,
    )

    obs, _ = env.reset(seed=args.seed)
    nu = env.nu
    joints_per_finger = nu // 3
    ctrl_target = env.data.ctrl[:nu].copy()
    snapshot_idx = 0

    selected_joint = 0
    step_size = 0.05
    paused = False
    episode_num = 0
    episode_metrics: list[dict] = []

    csv_path = out_dir / f"teleop_ep{episode_num:03d}.csv"
    csv_fields = [
        "step", "axis_rotation_deg", "axis_tilt_rad", "axis_tilt_deg",
        "tip_error_m", "contact_count", "finger0", "finger1", "finger2",
        "axial_omega", "termination_reason", "is_success",
    ]
    for i in range(nu):
        csv_fields.append(f"ctrl_{i}")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    def _save_snapshot(step: int, info: dict) -> None:
        nonlocal snapshot_idx
        if render_mode != "rgb_array":
            print("\n  Snapshots require --snapshots (rgb_array rendering).")
            return
        frame = env.render()
        if frame is None:
            print("\n  Snapshot failed: no frame rendered.")
            return
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore

        tilt = info.get("axis_tilt_deg", 0.0)
        rot = info.get("axis_rotation_deg", 0.0)
        path = out_dir / f"snapshot_ep{episode_num:03d}_step{step:05d}_rot{rot:.0f}_tilt{tilt:.0f}.png"
        imageio.imwrite(path, frame)
        snapshot_idx += 1
        print(f"\n  Saved snapshot: {path}")

    def _new_csv():
        nonlocal csv_file, csv_writer, csv_path
        csv_file.close()
        csv_path = out_dir / f"teleop_ep{episode_num:03d}.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

    fd, old_settings = _setup_terminal()
    mode_label = render_mode or "none"
    print("Teleop started. Press 'q' to quit, 'r' to reset, digits or ,/. to select, i/k to move.")
    if render_mode == "human":
        print("Viewer: ON (MuJoCo window)")
    else:
        print(f"Viewer: OFF (render_mode={mode_label}; terminal HUD only)")
        if args.snapshots:
            print("Press 'v' to save PNG snapshots to the log directory.")
    print(f"Stabilizer scale: {args.axis_stabilizer_scale}, physics: {args.physics}, tip: {args.tip_anchor}")
    print(f"Logging to: {out_dir}/")
    print()

    last_hud_time = 0.0
    step_count = 0
    info: dict = {}

    try:
        while True:
            key = _read_key(fd)
            if key is not None:
                if key in ("q", "\x1b"):
                    break
                elif key == "r":
                    _print_episode_summary(episode_metrics, episode_num)
                    episode_num += 1
                    episode_metrics = []
                    obs, _ = env.reset(seed=args.seed + episode_num)
                    ctrl_target = env.data.ctrl[:nu].copy()
                    step_count = 0
                    _new_csv()
                    print(f"\n--- Episode {episode_num} reset ---")
                elif key == "p":
                    paused = not paused
                elif key in "123456789":
                    selected_joint = min(int(key) - 1, nu - 1)
                elif key == ",":
                    selected_joint = (selected_joint - 1) % nu
                elif key == ".":
                    selected_joint = (selected_joint + 1) % nu
                elif key == "i":
                    ctrl_target[selected_joint] += step_size
                elif key == "k":
                    ctrl_target[selected_joint] -= step_size
                elif key == "I":
                    ctrl_target[selected_joint] += step_size * 5
                elif key == "K":
                    ctrl_target[selected_joint] -= step_size * 5
                elif key == "[":
                    step_size = max(0.005, step_size * 0.5)
                elif key == "]":
                    step_size = min(0.5, step_size * 2.0)
                elif key == "0":
                    pass  # zero action handled below
                elif key == " ":
                    print(f"\n  ctrl_target = {np.array2string(ctrl_target, precision=3)}")
                elif key == "v":
                    if info:
                        _save_snapshot(step_count, info)

                lo = env.model.actuator_ctrlrange[:, 0]
                hi = env.model.actuator_ctrlrange[:, 1]
                ctrl_target = np.clip(ctrl_target, lo, hi)

            if paused:
                time.sleep(0.02)
                if info:
                    _print_hud(
                        step_count,
                        info,
                        selected_joint,
                        ctrl_target,
                        step_size,
                        paused,
                        joints_per_finger,
                    )
                continue

            action = (ctrl_target - env.data.ctrl[:nu]) / env.ctrl_scale
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            fc = info.get("finger_contacts", [0, 0, 0])
            row = {
                "step": step_count,
                "axis_rotation_deg": info.get("axis_rotation_deg", 0.0),
                "axis_tilt_rad": info.get("axis_tilt_rad", 0.0),
                "axis_tilt_deg": info.get("axis_tilt_deg", 0.0),
                "tip_error_m": info.get("tip_error_m", 0.0),
                "contact_count": info.get("contact_count", 0),
                "finger0": fc[0],
                "finger1": fc[1],
                "finger2": fc[2],
                "axial_omega": info.get("axial_omega", 0.0),
                "termination_reason": info.get("termination_reason", "none"),
                "is_success": info.get("is_success", False),
            }
            for i in range(nu):
                row[f"ctrl_{i}"] = float(ctrl_target[i])
            csv_writer.writerow(row)
            episode_metrics.append(info)

            now = time.time()
            if now - last_hud_time > 0.1:
                _print_hud(
                    step_count,
                    info,
                    selected_joint,
                    ctrl_target,
                    step_size,
                    paused,
                    joints_per_finger,
                )
                last_hud_time = now

            if terminated or truncated:
                _print_episode_summary(episode_metrics, episode_num)
                episode_num += 1
                episode_metrics = []
                obs, _ = env.reset(seed=args.seed + episode_num)
                ctrl_target = env.data.ctrl[:nu].copy()
                step_count = 0
                _new_csv()
                print(f"--- Episode {episode_num} auto-reset ---")

            time.sleep(1.0 / env.policy_hz)

    except KeyboardInterrupt:
        pass
    finally:
        _print_episode_summary(episode_metrics, episode_num)
        csv_file.close()
        _restore_terminal(fd, old_settings)
        env.close()
        print(f"\nTeleop ended. Logs saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation of the Allegro hand")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--episode-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis-stabilizer-scale", type=float, default=0.0,
                        help="0 for pure kinematic test (no external help)")
    parser.add_argument("--tip-connect-solref", type=float, default=None)
    parser.add_argument("--rod-mass-scale", type=float, default=1.0)
    parser.add_argument("--rod-friction-cap", type=float, default=4.0)
    parser.add_argument("--tilt-terminate-rad", type=float, default=10.0,
                        help="Set high to not auto-terminate on tilt during teleop")
    parser.add_argument("--tip-anchor", choices=["top", "bottom"], default="top")
    parser.add_argument("--physics", choices=["tip_connect", "revolute"], default="tip_connect")
    parser.add_argument("--hand-model", choices=["allegro", "surrogate"], default="allegro")
    parser.add_argument("--hand-pose-config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/teleop")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No MuJoCo window; use terminal HUD only (for SSH servers without DISPLAY)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Require MuJoCo interactive viewer (needs DISPLAY / X11 forwarding)",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Enable offscreen rendering; press 'v' during teleop to save PNG frames",
    )
    args = parser.parse_args()
    if args.headless and args.viewer:
        parser.error("Use only one of --headless or --viewer")
    run_teleop(args)


if __name__ == "__main__":
    main()
