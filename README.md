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

Overridable via environment variables: `ENV_NAME`, `OFFLINE_STEPS`, `RUN_GROUP`, `SEED`, `GPU_ID`.

## Reference results

`scene-play-singletask-task1-v0`, 10000 offline steps, seed 0, one method at a time on a
single GPU, measured on an **RTX A6000**:

| Method | GPU peak | Host RAM peak |
|---|---|---|
| QAM | 266.9 MiB | 3.04 GB |
| DSRL | 180.5 MiB | 3.12 GB |
| FQL | 138.8 MiB | 2.79 GB |
| IFQL | 132.3 MiB | 2.63 GB |
| IQL | 140.2 MiB | 2.61 GB |
| ReBRAC | 134.2 MiB | 2.59 GB |
| RLPD | 113.4 MiB | 2.70 GB |
| FAV | 254.0 MiB | 2.74 GB |

GPU peak is reproducible to well under 1% across repeated runs; host RAM peak to about 1%.

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

**GPU peak plateaus near 254 MiB.** Several unrelated configurations land on exactly 254.0
MiB — FAV at `gen_multiplier` 8, 7 and 6, and both FAV and QAM at 10000 steps with eval and
validation off. Differences between methods in that range are not resolvable by this metric.
