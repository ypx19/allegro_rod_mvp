#!/usr/bin/env bash
# Curriculum: Arm A revolute → Arm B tip-connect+tilt (shared obs transfer).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

STAMP="$(date +%Y%m%d-%H%M)"
A_RUN="${STAMP}-exp-a1-revolute-sharedObs-omegaHold10-subproc64-1e6-seed0"
B_RUN="${STAMP}-exp-b1-from-a1-tipconnect-tilt-subproc64-1e6-seed0"

echo "[curriculum] Arm A revolute → ${A_RUN}"
"${PY}" scripts/train_parallel.py \
  --run-id "${A_RUN}" \
  --physics revolute \
  --reward-style dexscrew \
  --no-tip-connect \
  --axis-stabilizer-scale 0 \
  --dexscrew-tilt-scale 0 \
  --omega-success-threshold 0.5 \
  --omega-success-hold-seconds 10 \
  --num-envs 64 \
  --n-steps 256 \
  --batch-size 256 \
  --steps 1000000 \
  --ent-coef 0.0 \
  --checkpoint-freq 200000 \
  --device cuda \
  --seed 0 \
  --notes "EXP-A1: revolute + DexScrew ω reward; shared obs (42); success=ω>0.5 for 10s; episode 20s. Parent for tip-connect transfer."

A_CKPT="${ROOT}/runs/${A_RUN}/checkpoints/final_model.zip"
if [[ ! -f "${A_CKPT}" ]]; then
  echo "[curriculum] missing Arm A ckpt: ${A_CKPT}" >&2
  exit 1
fi

echo "[curriculum] Arm B tip-connect+tilt from ${A_CKPT} → ${B_RUN}"
# Fresh VecNormalize (do not load Arm A stats); policy weights transfer via --resume.
"${PY}" scripts/train_parallel.py \
  --run-id "${B_RUN}" \
  --resume "${A_CKPT}" \
  --physics tip_connect \
  --reward-style dexscrew \
  --tip-connect \
  --tip-connect-solref 0.008 \
  --axis-stabilizer-scale 0 \
  --dexscrew-tilt-scale 1.0 \
  --omega-success-threshold 0.5 \
  --omega-success-hold-seconds 10 \
  --num-envs 64 \
  --n-steps 256 \
  --batch-size 256 \
  --steps 1000000 \
  --ent-coef 0.0 \
  --checkpoint-freq 200000 \
  --device cuda \
  --seed 0 \
  --notes "EXP-B1: tip-connect+tilt fine-tune from A1 revolute ckpt; shared obs; auto tilt punishment 1.0; fresh VecNormalize."

echo "[curriculum] done A=${A_RUN} B=${B_RUN}"
