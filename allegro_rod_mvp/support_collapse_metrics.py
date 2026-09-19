"""Episode-level diagnostics for the T00 2-finger → 1-finger support collapse."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TILT_EXCURSION_RAD = 0.25
TILT_RECOVERY_RAD = 0.20
DEFAULT_RECONTACT_WINDOW = 10
DEFAULT_COLLAPSE_LOOKBACK = 10
DEFAULT_POST_LOSS_HORIZON = 5


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for flag in mask.tolist():
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _mean_or_nan(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def episode_support_collapse_metrics(
    n_contact: Sequence[int] | np.ndarray,
    omega_perp: Sequence[float] | np.ndarray,
    tilt_rad: Sequence[float] | np.ndarray,
    *,
    termination_reason: str = "none",
    recontact_window: int = DEFAULT_RECONTACT_WINDOW,
    collapse_lookback: int = DEFAULT_COLLAPSE_LOOKBACK,
    post_loss_horizon: int = DEFAULT_POST_LOSS_HORIZON,
) -> dict[str, Any]:
    """Compute support-collapse diagnostics for one episode.

    Support-loss event: n_contact(t-1) >= 2 and n_contact(t) <= 1.
    Successful re-contact: return to n_contact >= 2 within `recontact_window`.
    """
    contacts = _as_int_array(n_contact)
    omega = _as_float_array(omega_perp)
    tilt = _as_float_array(tilt_rad)
    n_steps = int(contacts.size)
    if omega.size != n_steps or tilt.size != n_steps:
        raise ValueError("n_contact, omega_perp, and tilt_rad must have equal length")

    window = max(int(recontact_window), 0)
    lookback = max(int(collapse_lookback), 0)
    horizon = max(int(post_loss_horizon), 0)

    zero_mask = contacts == 0
    one_mask = contacts == 1
    twoplus_mask = contacts >= 2
    time_0 = int(np.sum(zero_mask))
    time_1 = int(np.sum(one_mask))
    time_2plus = int(np.sum(twoplus_mask))

    one_durations = _run_lengths(one_mask)

    loss_idx: list[int] = []
    recover_idx: list[int] = []
    recovered_flags: list[bool] = []
    steps_to_recontact: list[int] = []
    omega_at_loss: list[float] = []
    omega_max_after: list[float] = []
    delta_omega_after: list[float] = []
    delay_to_omega_spike: list[float] = []

    for t in range(1, n_steps):
        if contacts[t - 1] >= 2 and contacts[t] <= 1:
            loss_idx.append(t)
            end = min(n_steps, t + window + 1)
            recovered_at = -1
            for k in range(t + 1, end):
                if contacts[k] >= 2:
                    recovered_at = k
                    break
            recovered = recovered_at >= 0
            recovered_flags.append(recovered)
            if recovered:
                delay = recovered_at - t
                steps_to_recontact.append(int(delay))
            prev_omega = float(omega[t - 1])
            omega_at_loss.append(float(omega[t]))
            after_end = min(n_steps, t + horizon + 1)
            after = omega[t:after_end]
            max_after = float(np.max(after)) if after.size else float(omega[t])
            omega_max_after.append(max_after)
            delta_omega_after.append(max_after - prev_omega)
            spike_delay = float("nan")
            if after.size:
                rel = int(np.argmax(after))
                if max_after > prev_omega:
                    spike_delay = float(rel)
            delay_to_omega_spike.append(spike_delay)
        elif contacts[t - 1] <= 1 and contacts[t] >= 2:
            recover_idx.append(t)

    n_loss = len(loss_idx)
    n_recontact_events = int(
        np.sum(
            (contacts[1:] >= 2) & (contacts[:-1] <= 1)
        )
    )
    n_recovered = int(np.sum(recovered_flags)) if recovered_flags else 0
    recontact_success_rate = (
        float(n_recovered / n_loss) if n_loss else float("nan")
    )

    # Tilt excursions: upward cross of 0.25 rad, recovery if later < 0.20
    # before episode end. 0.25 is a diagnostic boundary, not termination.
    n_up_cross = 0
    time_above_025 = int(np.sum(tilt >= TILT_EXCURSION_RAD))
    excursion_active = False
    recovered_excursions = 0
    n_excursions = 0
    recover_delays: list[int] = []
    recovered_peak_tilts: list[float] = []
    current_start = -1
    current_peak = 0.0
    prev_tilt = float(tilt[0]) if n_steps else 0.0
    if n_steps and prev_tilt >= TILT_EXCURSION_RAD:
        excursion_active = True
        n_excursions = 1
        current_start = 0
        current_peak = prev_tilt
    for t in range(1, n_steps):
        curr = float(tilt[t])
        if prev_tilt < TILT_EXCURSION_RAD <= curr:
            n_up_cross += 1
            if not excursion_active:
                excursion_active = True
                n_excursions += 1
                current_start = t
                current_peak = curr
        if excursion_active:
            current_peak = max(current_peak, curr)
            if curr < TILT_RECOVERY_RAD:
                recovered_excursions += 1
                recover_delays.append(int(t - current_start))
                recovered_peak_tilts.append(float(current_peak))
                excursion_active = False
        prev_tilt = curr

    reason = str(termination_reason)
    preceded_by_loss = False
    delay_loss_to_term = float("nan")
    if reason == "axis_tilt" and n_steps > 0 and loss_idx:
        last_t = n_steps - 1
        window_start = max(0, last_t - lookback)
        recent = [idx for idx in loss_idx if window_start <= idx <= last_t]
        if recent:
            preceded_by_loss = True
            delay_loss_to_term = float(last_t - recent[-1])

    return {
        "episode_length": n_steps,
        "n_support_loss_events": n_loss,
        "n_recontact_events": n_recontact_events,
        "n_2plus_to_1_transitions": n_loss,
        "n_1_to_2plus_transitions": n_recontact_events,
        "time_steps_0_contact": time_0,
        "time_steps_1_contact": time_1,
        "time_steps_2plus_contact": time_2plus,
        "frac_steps_0_contact": float(time_0 / n_steps) if n_steps else 0.0,
        "frac_steps_1_contact": float(time_1 / n_steps) if n_steps else 0.0,
        "frac_steps_2plus_contact": float(time_2plus / n_steps) if n_steps else 0.0,
        "one_contact_duration_mean": (
            float(np.mean(one_durations)) if one_durations else 0.0
        ),
        "one_contact_duration_max": (
            int(np.max(one_durations)) if one_durations else 0
        ),
        "had_support_loss": bool(n_loss > 0),
        "n_support_loss_recovered": n_recovered,
        "recontact_success_rate": recontact_success_rate,
        "steps_to_recontact_mean": _mean_or_nan(steps_to_recontact),
        "omega_perp_mean": float(np.mean(omega)) if n_steps else 0.0,
        "omega_perp_max": float(np.max(omega)) if n_steps else 0.0,
        "omega_perp_at_support_loss_mean": _mean_or_nan(omega_at_loss),
        "omega_perp_max_after_support_loss_mean": _mean_or_nan(omega_max_after),
        "delta_omega_perp_after_support_loss_mean": _mean_or_nan(delta_omega_after),
        "delay_support_loss_to_omega_spike_mean": _mean_or_nan(delay_to_omega_spike),
        "tilt_max": float(np.max(tilt)) if n_steps else 0.0,
        "tilt_final": float(tilt[-1]) if n_steps else 0.0,
        "time_steps_tilt_above_0_25": time_above_025,
        "n_tilt_upward_cross_0_25": n_up_cross,
        "n_tilt_excursions": n_excursions,
        "n_tilt_excursions_recovered": recovered_excursions,
        "tilt_recovery_rate": (
            float(recovered_excursions / n_excursions) if n_excursions else float("nan")
        ),
        "tilt_recovery_steps_mean": _mean_or_nan(recover_delays),
        "tilt_recovered_peak_mean": _mean_or_nan(recovered_peak_tilts),
        "termination_reason": reason,
        "axis_tilt_preceded_by_support_loss": bool(preceded_by_loss),
        "delay_support_loss_to_axis_tilt": delay_loss_to_term,
        "support_loss_steps": loss_idx,
        "recontact_steps": recover_idx,
        "tilt_cross_0_25_steps": [
            int(t)
            for t in range(1, n_steps)
            if float(tilt[t - 1]) < TILT_EXCURSION_RAD <= float(tilt[t])
        ],
    }


def aggregate_support_collapse_metrics(
    episodes: Iterable[Mapping[str, Any]],
    *,
    recontact_window: int = DEFAULT_RECONTACT_WINDOW,
    collapse_lookback: int = DEFAULT_COLLAPSE_LOOKBACK,
) -> dict[str, Any]:
    rows = list(episodes)
    if not rows:
        return {
            "episodes": 0,
            "episode_length_mean": 0.0,
        }

    def _col(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    def _finite_mean(name: str) -> float:
        return _mean_or_nan(_col(name))

    n = len(rows)
    axis_tilt_rows = [row for row in rows if row.get("termination_reason") == "axis_tilt"]
    n_axis_tilt = len(axis_tilt_rows)
    preceded = [
        row for row in axis_tilt_rows if row.get("axis_tilt_preceded_by_support_loss")
    ]
    total_loss = int(np.sum(_col("n_support_loss_events")))
    total_recovered = int(np.sum(_col("n_support_loss_recovered")))
    return {
        "episodes": n,
        "episode_length_mean": float(np.mean(_col("episode_length"))),
        "episode_length_std": float(np.std(_col("episode_length"))),
        "n_2plus_to_1_mean": float(np.mean(_col("n_2plus_to_1_transitions"))),
        "n_1_to_2plus_mean": float(np.mean(_col("n_1_to_2plus_transitions"))),
        "time_steps_0_contact_mean": float(np.mean(_col("time_steps_0_contact"))),
        "time_steps_1_contact_mean": float(np.mean(_col("time_steps_1_contact"))),
        "time_steps_2plus_contact_mean": float(np.mean(_col("time_steps_2plus_contact"))),
        "frac_steps_0_contact_mean": float(np.mean(_col("frac_steps_0_contact"))),
        "frac_steps_1_contact_mean": float(np.mean(_col("frac_steps_1_contact"))),
        "frac_steps_2plus_contact_mean": float(np.mean(_col("frac_steps_2plus_contact"))),
        "one_contact_duration_mean": float(np.mean(_col("one_contact_duration_mean"))),
        "one_contact_duration_max_mean": float(np.mean(_col("one_contact_duration_max"))),
        "one_contact_duration_max_max": float(np.max(_col("one_contact_duration_max"))),
        "fraction_episodes_with_support_loss": float(
            np.mean(_col("had_support_loss"))
        ),
        "recontact_success_rate": (
            float(total_recovered / total_loss) if total_loss else float("nan")
        ),
        "recontact_success_rate_episode_mean": _finite_mean("recontact_success_rate"),
        "steps_to_recontact_mean": _finite_mean("steps_to_recontact_mean"),
        "omega_perp_mean": float(np.mean(_col("omega_perp_mean"))),
        "omega_perp_max_mean": float(np.mean(_col("omega_perp_max"))),
        "omega_perp_max_max": float(np.max(_col("omega_perp_max"))),
        "omega_perp_at_support_loss_mean": _finite_mean(
            "omega_perp_at_support_loss_mean"
        ),
        "omega_perp_max_after_support_loss_mean": _finite_mean(
            "omega_perp_max_after_support_loss_mean"
        ),
        "delta_omega_perp_after_support_loss_mean": _finite_mean(
            "delta_omega_perp_after_support_loss_mean"
        ),
        "delay_support_loss_to_omega_spike_mean": _finite_mean(
            "delay_support_loss_to_omega_spike_mean"
        ),
        "tilt_max_mean": float(np.mean(_col("tilt_max"))),
        "tilt_final_mean": float(np.mean(_col("tilt_final"))),
        "time_steps_tilt_above_0_25_mean": float(
            np.mean(_col("time_steps_tilt_above_0_25"))
        ),
        "n_tilt_upward_cross_0_25_mean": float(
            np.mean(_col("n_tilt_upward_cross_0_25"))
        ),
        "n_tilt_excursions_mean": float(np.mean(_col("n_tilt_excursions"))),
        "n_tilt_excursions_recovered_mean": float(
            np.mean(_col("n_tilt_excursions_recovered"))
        ),
        "tilt_recovery_rate": _finite_mean("tilt_recovery_rate"),
        "tilt_recovery_steps_mean": _finite_mean("tilt_recovery_steps_mean"),
        "tilt_recovered_peak_mean": _finite_mean("tilt_recovered_peak_mean"),
        "n_axis_tilt_terminations": n_axis_tilt,
        "fraction_axis_tilt_deaths_preceded_by_support_loss": (
            float(len(preceded) / n_axis_tilt) if n_axis_tilt else float("nan")
        ),
        "delay_support_loss_to_axis_tilt_mean": _mean_or_nan(
            [
                row["delay_support_loss_to_axis_tilt"]
                for row in preceded
            ]
        ),
        "recontact_window_steps": int(recontact_window),
        "collapse_lookback_steps": int(collapse_lookback),
    }


def plot_support_collapse_trace(
    trace: Mapping[str, Sequence[float] | np.ndarray],
    out_path: str,
    *,
    title: str = "",
    termination_step: int | None = None,
) -> None:
    """Save a 6-panel diagnostic figure aligned by timestep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = _as_int_array(trace["timestep"])
    n_contact = _as_float_array(trace["n_contact"])
    omega_perp = _as_float_array(trace["omega_perp_norm"])
    tilt = _as_float_array(trace["tilt"])
    axial = _as_float_array(trace["axial_omega"])
    angle = _as_float_array(trace["net_angle"])
    forces = np.asarray(trace["finger_force"], dtype=np.float64)
    if forces.ndim == 1:
        forces = forces.reshape(-1, 1)

    loss_steps = [int(s) for s in trace.get("support_loss_steps", [])]
    recontact_steps = [int(s) for s in trace.get("recontact_steps", [])]
    tilt_cross_steps = [int(s) for s in trace.get("tilt_cross_0_25_steps", [])]

    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    axes[0].step(steps, n_contact, where="post", color="tab:blue")
    axes[0].set_ylabel("n_contact")
    axes[0].set_ylim(-0.2, 3.2)
    for i, color in enumerate(("tab:red", "tab:orange", "tab:green")):
        if forces.shape[1] > i:
            axes[1].plot(steps, forces[:, i], color=color, label=f"f{i}")
    axes[1].set_ylabel("contact force (N)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(steps, omega_perp, color="tab:purple")
    axes[2].set_ylabel(r"$\|\omega_\perp\|$ (rad/s)")
    axes[3].plot(steps, np.degrees(tilt), color="tab:brown")
    axes[3].axhline(np.degrees(TILT_EXCURSION_RAD), color="k", ls="--", lw=0.8)
    axes[3].set_ylabel("tilt (deg)")
    axes[4].plot(steps, axial, color="tab:cyan")
    axes[4].set_ylabel(r"$\omega_{axial}$ (rad/s)")
    axes[5].plot(steps, np.degrees(angle), color="tab:gray")
    axes[5].set_ylabel("net angle (deg)")
    axes[5].set_xlabel("timestep")

    for ax in axes:
        for s in loss_steps:
            ax.axvline(s, color="tab:red", ls="--", lw=0.9, alpha=0.8)
        for s in recontact_steps:
            ax.axvline(s, color="tab:green", ls=":", lw=0.9, alpha=0.8)
        for s in tilt_cross_steps:
            ax.axvline(s, color="tab:orange", ls="-.", lw=0.8, alpha=0.7)
        if termination_step is not None:
            ax.axvline(int(termination_step), color="k", ls="-", lw=1.0, alpha=0.7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
