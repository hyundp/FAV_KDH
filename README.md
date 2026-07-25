# FAV peak-memory benchmark

Measures two numbers for each offline RL method in the comparison table:

- **GPU peak** — the JAX allocator high-water mark (`peak_bytes_in_use`), i.e. bytes actually
  live on the device, not the size of the preallocated pool.
- **Host RAM peak** — the process resident-set high-water mark (`ru_maxrss`).

## Setup

Needs `conda`, an NVIDIA driver, and the system packages mujoco_py compiles against. As root
the setup script installs those itself; otherwise install them first:

```bash
sudo apt-get install -y libosmesa6-dev libglew-dev patchelf build-essential
```

Then:

```bash
git clone https://github.com/hyundp/FAV_KDH.git
cd FAV_KDH
bash bash_scripts/setup_env.sh          # optional arg: environment name, default "aligen"
```

This creates the conda environment, installs the pinned dependencies from
`requirements.lock.txt`, installs MuJoCo 2.1.0 under `~/.mujoco`, downloads the
`scene-play-v0` dataset, and checks that JAX actually sees the GPU.

## Run

```bash
conda activate aligen
bash bash_scripts/qam_dsrl_mem.sh
```

Each method runs alone on one GPU, sequentially, and prints a summary table at the end.
Per-run detail lands in `exp/mem/<run>/mem_summary.json` (raw bytes plus MiB/GB/GiB).
Re-print the table at any time with:

```bash
python bash_scripts/collect_mem.py --run_group=mem
```

Overridable via environment variables: `ENV_NAME`, `OFFLINE_STEPS`, `RUN_GROUP`, `SEED`,
`GPU_ID`, `EVAL_INTERVAL`, `LOG_INTERVAL`.

Policy evaluation is off by default (`EVAL_INTERVAL=0`). It is not part of the training
memory footprint — it holds 50 episodes of trajectories plus a MuJoCo eval environment,
which adds about 0.27 GB of host RSS to every method alike — and it dominates wall-clock,
taking longer than the 10000 training steps it accompanies. To measure with evaluation
included:

```bash
EVAL_INTERVAL=100000 bash bash_scripts/qam_dsrl_mem.sh
```

## Reference results

`scene-play-singletask-task1-v0`, 10000 offline steps, seed 0, one method at a time on a
single GPU, measured on an **RTX A6000**. The default column has evaluation off; the right
column is the same runs with `EVAL_INTERVAL=100000`.

| Method | GPU peak | Host RAM peak | GPU peak (with eval) | Host RAM peak (with eval) |
|---|---|---|---|---|
| QAM | 254.0 MiB | 2.91 GB | 266.9 MiB | 3.04 GB |
| DSRL | 180.3 MiB | 2.98 GB | 180.5 MiB | 3.12 GB |
| FQL | 138.8 MiB | 2.65 GB | 138.8 MiB | 2.79 GB |
| IFQL | 132.3 MiB | 2.50 GB | 132.3 MiB | 2.63 GB |
| IQL | 140.2 MiB | 2.54 GB | 140.2 MiB | 2.61 GB |
| ReBRAC | 134.2 MiB | 2.43 GB | 134.2 MiB | 2.59 GB |
| RLPD | 113.4 MiB | 2.57 GB | 113.4 MiB | 2.66 GB |
| FAV | 254.0 MiB | 2.62 GB | 254.0 MiB | 2.74 GB |

GPU peak is reproducible to well under 1% across repeated runs; host RAM peak to about 1%.

Note that QAM and FAV are indistinguishable at 254.0 MiB with evaluation off. QAM's higher
figure with evaluation on comes from the inference path — sampling through a 10-step flow —
not from training.

## Notes

**Two settings matter for the numbers to mean anything.** `XLA_PYTHON_CLIENT_PREALLOCATE=false`
makes the reported GPU figure the allocator high-water mark instead of the preallocated pool
(which would just be 75% of whatever card is installed, so a 24 GB and a 48 GB card would
differ by 2x for no real reason). `MALLOC_ARENA_MAX=2` pins the glibc arena count, which
otherwise scales with core count and inflates RSS on many-core hosts. Both are set by
`bash_scripts/qam_dsrl_mem.sh`.

**JAX falls back to CPU silently.** The CUDA libraries ship inside the pip packages but are
not on the loader path, so the CUDA plugin fails to initialise and prints only a warning.
`bash_scripts/env_paths.sh` fixes the path and the benchmark aborts if the backend is not GPU.

**RLPD is run with 512x4 networks.** The defaults in `agents/rlpd.py` are `(256, 256)`, which
is smaller than every other method in the table. The benchmark overrides them on the command
line so the comparison is like-for-like; the file's defaults are left untouched.

**Do not shorten `OFFLINE_STEPS`.** GPU peak climbs with step count before it saturates,
because JAX dispatches asynchronously: the Python loop runs ahead of the device and in-flight
buffers accumulate until backpressure caps the pipeline depth. For FAV, with eval and
validation disabled so only step count varies:

| steps | 200 | 500 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|
| GPU peak | 227.7 | 231.7 | 238.0 | 253.1 | 254.0 MiB |

Light methods (FQL, IFQL, IQL, ReBRAC, RLPD) read the same at 200 steps as at 10000 — their
per-iteration buffers are too small for pipeline depth to matter. Heavy ones (QAM, FAV) do
not. A table that mixes the two is comparing saturated against unsaturated numbers. The
default 10000 steps is past saturation for every method here.

**GPU peak is sensitive to device synchronisation, not just to the model.** Disabling the
validation pass (`LOG_INTERVAL=999999`) *raises* FQL's GPU peak from 138.8 to 154.9 MiB: that
pass pulls values back to the host every `LOG_INTERVAL` steps, which drains the dispatch
pipeline. Remove the sync and the pipeline runs deeper. Treat this metric as a property of
the configuration, not an intrinsic property of the method.

**Host RAM peak has a floor around 2.5 GB and barely moves.** Only two knobs affect it:
evaluation (about 0.14 GB) and the validation pass (about 0.17 GB). `buffer_size` does not
matter at all, despite the replay buffer over-allocating to 2M rows for a 1.001M-transition
dataset — the untouched pages never become resident, so they never enter RSS. Step count
does not matter either. Roughly 0.9 GB of the floor is imports plus JAX GPU context
initialisation, and the rest is the dataset.

**GPU peak plateaus near 254 MiB.** Several unrelated configurations land on exactly 254.0
MiB — FAV at `gen_multiplier` 8, 7 and 6, and both FAV and QAM at 10000 steps with eval and
validation off. Differences between methods in that range are not resolvable by this metric.
