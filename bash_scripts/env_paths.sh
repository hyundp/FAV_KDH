# Shared loader-path setup. Source this, do not execute it.
#
# The CUDA libraries ship inside the pip packages but are not on the loader path, and
# without them the JAX CUDA plugin fails to initialise and silently falls back to CPU --
# it only prints a warning, so a benchmark run would quietly measure CPU memory instead.
# mujoco_py additionally refuses to load once LD_LIBRARY_PATH is set unless its own two
# directories are on it as well.

_ENV_ROOT="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
if [ -n "$_ENV_ROOT" ]; then
  for _lib in cusparse cuda_runtime cublas cusolver cufft nvjitlink cudnn; do
    for _d in "$_ENV_ROOT"/lib/python*/site-packages/nvidia/$_lib/lib; do
      [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
    done
  done
fi
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
