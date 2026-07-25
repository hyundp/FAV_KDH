#!/bin/bash
#
# Peak memory benchmark: GPU peak (JAX allocator high-water mark) and host RAM peak
# (process RSS high-water mark) for every method in the comparison table.
#
# Usage:
#   conda activate aligen
#   nohup bash bash_scripts/qam_dsrl_mem.sh > nohup_mem.out 2>&1 &

set -uo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=bash_scripts/env_paths.sh
source bash_scripts/env_paths.sh

# Fail loudly rather than measuring CPU memory by accident.
python - <<'PY' || exit 1
import sys

import jax

if jax.default_backend() != 'gpu':
    sys.exit(f'JAX backend is {jax.default_backend()!r}, not GPU. See bash_scripts/env_paths.sh.')
PY

# Report the JAX allocator high-water mark, not the size of the preallocated pool.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep host RSS comparable across machines with different core counts.
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=8

ENV="${ENV_NAME:-scene-play-singletask-task1-v0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-10000}"
RUN_GROUP="${RUN_GROUP:-mem}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
# Policy evaluation is not part of the training footprint: it holds 50 episodes of
# trajectories plus a MuJoCo eval environment, adding roughly 0.27 GB of host RSS to every
# method alike, and it dominates wall-clock. 0 disables it; set 100000 to measure with it.
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
# The validation forward pass runs every LOG_INTERVAL steps and compiles its own graph.
LOG_INTERVAL="${LOG_INTERVAL:-5000}"

# agents/aligen.py is FAV. RLPD is run with the same 512x4 networks as every other method;
# the defaults in agents/rlpd.py are (256, 256), which makes it incomparable to the rest.
AGENTS=(
  "agents/qam.py"
  "agents/dsrl.py"
  "agents/fql.py"
  "agents/ifql.py"
  "agents/iql.py"
  "agents/rebrac.py"
  "agents/rlpd.py"
  "agents/aligen.py"
)

# One run at a time on a single GPU, so every method is measured under identical conditions.
for AGENT in "${AGENTS[@]}"; do
  EXTRA_ARGS=()
  if [[ "${AGENT}" == "agents/rlpd.py" ]]; then
    EXTRA_ARGS+=(--agent.actor_hidden_dims="(512,512,512,512)")
    EXTRA_ARGS+=(--agent.value_hidden_dims="(512,512,512,512)")
  fi

  echo "GPU ${GPU_ID}: ${AGENT} | ${ENV} | ${OFFLINE_STEPS} steps | seed ${SEED} | eval_interval ${EVAL_INTERVAL}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
    --agent="${AGENT}" \
    --env_name="${ENV}" \
    --offline_steps="${OFFLINE_STEPS}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --log_interval="${LOG_INTERVAL}" \
    --run_group="${RUN_GROUP}" \
    --seed="${SEED}" \
    --nouse_wandb \
    --mem_logging \
    "${EXTRA_ARGS[@]}"
done

echo
echo "=== Summary ==="
python bash_scripts/collect_mem.py --run_group="${RUN_GROUP}"
