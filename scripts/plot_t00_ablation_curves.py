#!/usr/bin/env python3
"""Plot the T00 2×2 ablation 6-panel training curves from metrics.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXTRA_COLS = [
    "approx_kl",
    "clip_fraction",
    "clip_range",
    "entropy_loss",
    "explained_variance",
    "learning_rate",
    "loss",
    "n_updates",
    "policy_gradient_loss",
    "std",
    "value_loss",
]

CONDITIONS = {
    "A": {"color": "#4c8dde", "label": "A 48-D hist1 baseline"},
    "B": {"color": "#e8a54b", "label": "B 192-D hist4 baseline"},
    "C": {"color": "#5cb8a0", "label": "C 48-D hist1 support-aware"},
    "D": {"color": "#c45c48", "label": "D 192-D hist4 support-aware"},
}

PANELS = [
    ("episode_return", "episode return"),
    ("episode_length", "episode length"),
    ("approx_kl", "approx KL"),
    ("explained_variance", "explained variance"),
    ("clip_fraction", "clip fraction"),
    ("std", "action std"),
]


def _float(value: str) -> float:
    return float("nan") if value in ("", None) else float(value)


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        n_named = min(4, len(header))
        names = list(header[:n_named]) + EXTRA_COLS
        for raw in reader:
            if not raw:
                continue
            rec: dict[str, float] = {}
            for i, name in enumerate(names):
                rec[name] = _float(raw[i]) if i < len(raw) else float("nan")
            rows.append(rec)
    if not rows:
        raise FileNotFoundError(f"no data rows in {path}")
    rows.sort(key=lambda r: r["step"])
    dedup: dict[float, dict[str, float]] = {r["step"]: r for r in rows}
    ordered = [dedup[k] for k in sorted(dedup)]
    keys = ["step", "episode_return", "episode_length", *EXTRA_COLS]
    return {key: np.array([r.get(key, float("nan")) for r in ordered], dtype=float) for key in keys}


def merge_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks = [load_metrics(p) for p in paths if p.exists()]
    if not chunks:
        raise FileNotFoundError(f"no metrics.csv among {paths}")
    keys = chunks[0].keys()
    merged = {key: np.concatenate([c[key] for c in chunks]) for key in keys}
    order = np.argsort(merged["step"], kind="stable")
    merged = {key: merged[key][order] for key in keys}
    _, uniq = np.unique(merged["step"], return_index=True)
    uniq = np.sort(uniq)
    return {key: merged[key][uniq] for key in keys}


def slice_to(frame: dict[str, np.ndarray], max_step: float) -> dict[str, np.ndarray]:
    mask = frame["step"] <= max_step
    return {key: value[mask] for key, value in frame.items()}


def nearest_row(frame: dict[str, np.ndarray], target: float) -> dict[str, float]:
    idx = int(np.argmin(np.abs(frame["step"] - target)))
    return {key: float(value[idx]) for key, value in frame.items()}


def last_row(frame: dict[str, np.ndarray]) -> dict[str, float]:
    return {key: float(value[-1]) for key, value in frame.items()}


def plot_curves(series: dict[str, dict[str, np.ndarray]], title: str, out: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), constrained_layout=True)
    for ax, (col, ylabel) in zip(axes.ravel(), PANELS):
        for name, frame in series.items():
            spec = CONDITIONS[name]
            ax.plot(
                frame["step"],
                frame[col],
                color=spec["color"],
                marker="o",
                markersize=3.2,
                linewidth=1.4,
                label=spec["label"],
            )
        ax.set_xlabel("env steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument("--A", nargs="+", required=True)
    parser.add_argument("--B", nargs="+", required=True)
    parser.add_argument("--C", nargs="+", required=True)
    parser.add_argument("--D", nargs="+", required=True)
    args = parser.parse_args()
    series = {
        name: merge_metrics([Path(p) for p in getattr(args, name)])
        for name in ("A", "B", "C", "D")
    }
    if args.max_step is not None:
        series = {name: slice_to(frame, args.max_step) for name, frame in series.items()}
    plot_curves(series, args.title, Path(args.out))
    print(f"wrote {args.out}")
    for name, frame in series.items():
        stats = last_row(frame)
        print(
            f"{name} step={stats['step']:.0f} return={stats['episode_return']:.3f} "
            f"length={stats['episode_length']:.2f} kl={stats['approx_kl']:.5f} "
            f"ev={stats['explained_variance']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
