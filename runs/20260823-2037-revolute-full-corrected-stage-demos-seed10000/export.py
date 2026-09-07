#!/usr/bin/env python3
"""Export all ten accepted corrected-lineage revolute stage demos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE_EXPORTER = (
    ROOT
    / "runs/20260823-2016-revolute-stage-demos-seed10000/export.py"
)
EXPECTED = [
    (400.0, 4.0, "revolute_s400.mp4"),
    (200.0, 2.0, "revolute_s200.mp4"),
    (100.0, 1.0, "revolute_s100.mp4"),
    (50.0, 0.5, "revolute_s50.mp4"),
    (25.0, 0.25, "revolute_s25.mp4"),
    (12.5, 0.125, "revolute_s12p5.mp4"),
    (6.25, 0.0625, "revolute_s6p25.mp4"),
    (3.125, 0.03125, "revolute_s3p125.mp4"),
    (1.5625, 0.015625, "revolute_s1p5625.mp4"),
    (1.0, 0.01, "revolute_s1.mp4"),
]


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "accepted_revolute_stage_exporter", SOURCE_EXPORTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE_EXPORTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(module) -> None:
    state = json.loads(module.STATE_PATH.read_text())
    completed = state.get("completed", [])
    if len(completed) != len(EXPECTED):
        raise RuntimeError(
            f"Expected exactly {len(EXPECTED)} completed stages, found {len(completed)}"
        )
    missing: list[str] = []
    for record, (mass, friction, _) in zip(completed, EXPECTED, strict=True):
        actual = (float(record["mass_scale"]), float(record["friction_scale"]))
        if actual != (mass, friction) or not record.get("accepted", False):
            raise RuntimeError(
                f"Expected accepted {(mass, friction)}, found {actual}, "
                f"accepted={record.get('accepted')}"
            )
        checkpoint = Path(record["checkpoint"])
        vecnormalize = checkpoint.parent / "vecnormalize.pkl"
        for path in (checkpoint, vecnormalize):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing accepted checkpoint/config artifacts; no substitutions made: "
            + ", ".join(missing)
        )


def finalize_metadata() -> None:
    metadata_path = OUT / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance_note"] = (
        "All ten records are the accepted revolute stages in corrected curriculum "
        "state 20260823-2015-proportional-physics-C-seed0. s=400 is the state's "
        "accepted C parent checkpoint; s=200 through s=1 are the exact sequential "
        "corrected-lineage final_model.zip checkpoints. No tip-connect or diagnostic "
        "checkpoint was substituted."
    )
    metadata["source_exporter"] = str(SOURCE_EXPORTER.resolve())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    index_path = OUT / "INDEX.md"
    index = index_path.read_text()
    marker = f"- Curriculum state snapshot: `{metadata['source_state']}`\n"
    provenance = (
        "- Checkpoint provenance: exact ten accepted corrected-lineage revolute "
        "records; the s=400 record points to accepted C and s=200 through s=1 point "
        "to their sequential corrected-curriculum checkpoints.\n"
    )
    if marker not in index:
        raise RuntimeError("Could not add provenance to INDEX.md")
    index_path.write_text(index.replace(marker, marker + provenance, 1))


def main() -> int:
    module = load_exporter()
    preflight(module)
    module.OUT = OUT
    module.EXPECTED = EXPECTED
    result = module.main()
    finalize_metadata()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
