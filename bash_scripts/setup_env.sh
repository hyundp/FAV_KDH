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

echo "=== 1/7 conda environment: ${ENV_NAME} (python ${PYTHON_VERSION}) ==="
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

echo "=== 2/7 MuJoCo 2.1.0 (for mujoco_py) ==="
if [ -d "${MUJOCO_DIR}/mujoco210" ]; then
  echo "${MUJOCO_DIR}/mujoco210 already present."
else
  mkdir -p "${MUJOCO_DIR}"
  curl -L -o /tmp/mujoco210.tar.gz \
    https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C "${MUJOCO_DIR}"
  rm -f /tmp/mujoco210.tar.gz
fi

echo "=== 3/7 build prerequisites for mujoco_py ==="
# mujoco_py compiles a Cython extension against OSMesa headers. Missing system packages
# surface as a confusing "GL/osmesa.h: No such file or directory" much later, so check here.
APT_REQUIRED="libosmesa6-dev libglew-dev patchelf build-essential"
# Names vary across Ubuntu releases (libgl1-mesa-glx was renamed to libgl1 in 24.04), so
# these are attempted individually and a failure is not fatal.
APT_OPTIONAL="libgl1-mesa-glx libgl1 libglfw3"

missing=0
command -v gcc >/dev/null 2>&1 || missing=1
command -v patchelf >/dev/null 2>&1 || missing=1
[ -f /usr/include/GL/osmesa.h ] || missing=1

if [ "$missing" -eq 1 ]; then
  if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
    echo "Installing: ${APT_REQUIRED}"
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq ${APT_REQUIRED}
    for pkg in ${APT_OPTIONAL}; do
      apt-get install -y -qq "$pkg" >/dev/null 2>&1 || true
    done
  else
    echo "Missing build prerequisites for mujoco_py. Install them and re-run:" >&2
    echo "  sudo apt-get install -y ${APT_REQUIRED}" >&2
    exit 1
  fi
fi

if [ ! -f /usr/include/GL/osmesa.h ]; then
  echo "GL/osmesa.h still missing after install; mujoco_py cannot build." >&2
  exit 1
fi

echo "=== 4/7 python dependencies ==="
pip install --upgrade pip
# d4rl needs Cython < 3 present before its own build runs.
pip install "cython<3"
pip install -r requirements.lock.txt

echo "=== 5/7 building mujoco_py ==="
# mujoco_py compiles a Cython extension on first import. Trigger it now so a build failure
# surfaces during setup instead of halfway through the benchmark.
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/env_paths.sh"
python -c "import mujoco_py; print('mujoco_py ok')"

echo "=== 6/7 scene-play dataset ==="
python - <<'PY'
import ogbench

ogbench.download_datasets(['scene-play-v0'])
print('dataset ready')
PY

echo "=== 7/7 verifying JAX sees the GPU ==="
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
