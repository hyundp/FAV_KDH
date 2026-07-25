#!/bin/bash
#
# One-shot setup for the peak-memory benchmark on a fresh machine.
#
# Usage:
#   bash bash_scripts/setup_env.sh [env_name]
#
# Creates the conda environment, installs the pinned dependencies, installs MuJoCo 2.1.0
# (mujoco_py loads even on the OGBench code path, so it has to be present), downloads the
# scene-play dataset, and verifies that JAX sees the GPU.

set -euo pipefail

ENV_NAME="${1:-aligen}"
PYTHON_VERSION="3.12"
MUJOCO_DIR="$HOME/.mujoco"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "=== 1/5 conda environment: ${ENV_NAME} (python ${PYTHON_VERSION}) ==="
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists, reusing it."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

echo "=== 2/5 MuJoCo 2.1.0 (for mujoco_py) ==="
if [ -d "${MUJOCO_DIR}/mujoco210" ]; then
  echo "${MUJOCO_DIR}/mujoco210 already present."
else
  mkdir -p "${MUJOCO_DIR}"
  curl -L -o /tmp/mujoco210.tar.gz \
    https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C "${MUJOCO_DIR}"
  rm -f /tmp/mujoco210.tar.gz
fi

echo "=== 3/5 python dependencies ==="
# d4rl is not on PyPI; pip freeze cannot round-trip it, so install it from source first.
pip install --upgrade pip
pip install "cython<3"
pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"
pip install -r requirements.lock.txt

echo "=== 4/5 scene-play dataset ==="
python - <<'PY'
import ogbench

ogbench.download_datasets(['scene-play-v0'])
print('dataset ready')
PY

echo "=== 5/5 verifying JAX sees the GPU ==="
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/env_paths.sh"
python - <<'PY'
import sys

import jax

backend = jax.default_backend()
print('jax backend:', backend)
print('devices:', jax.devices())
if backend != 'gpu':
    sys.exit('JAX is not using the GPU. See bash_scripts/env_paths.sh.')
PY

echo
echo "Setup complete. Run the benchmark with:"
echo "  conda activate ${ENV_NAME}"
echo "  bash bash_scripts/qam_dsrl_mem.sh"
