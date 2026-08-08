#!/usr/bin/env bash
# EXP-B2: A1 revolute ckpt → tip-connect + adaptive 45/45 rot–tilt mass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

A1_CKPT="${ROOT}/runs/20260802-1439-exp-a1-revolute-sharedObs-omegaHold10-subproc64-1e6-seed0/checkpoints/final_model.zip"
STAMP="$(date +%Y%m%d-%H%M)"
B2_RUN="${STAMP}-exp-b2-from-a1-adaptiveMass45-subproc64-1e6-seed0"

if [[ ! -f "${A1_CKPT}" ]]; then
  echo "missing A1 ckpt: ${A1_CKPT}" >&2
  exit 1
fi

echo "[B2] tip-connect adaptive mass from ${A1_CKPT} → ${B2_RUN}"
"${PY}" scripts/train_parallel.py \
  --run-id "${B2_RUN}" \
  --resume "${A1_CKPT}" \
  --physics tip_connect \
  --reward-style dexscrew \
  --tip-connect \
  --tip-connect-solref 0.008 \
  --axis-stabilizer-scale 0 \
  --dexscrew-tilt-scale 1.0 \
  --adaptive-reward-mass \
  --mass-target-rot 0.45 \
  --mass-target-tilt 0.45 \
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
  --notes "EXP-B2: tip-connect+tilt from A1; online EMA mass targets rot=tilt=0.45; fresh VecNormalize; compare to B1."

echo "[B2] done run=${B2_RUN}"
echo "${B2_RUN}" > "${ROOT}/runs/logs/latest_b2_run_id.txt"
