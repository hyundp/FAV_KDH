import resource

import jax


def get_gpu_peak_bytes():
    """Return the peak bytes ever live in the JAX device allocator, summed over local devices.

    This is the allocator high-water mark, not the size of the preallocated pool, so it is
    unaffected by how much memory the GPU physically has. Returns None on backends that do
    not expose allocator stats (e.g. CPU).
    """
    total = 0
    for device in jax.local_devices():
        try:
            stats = device.memory_stats()
        except (AttributeError, NotImplementedError):
            return None
        if not stats or 'peak_bytes_in_use' not in stats:
            return None
        total += stats['peak_bytes_in_use']
    return total


def get_host_peak_bytes():
    """Return the peak resident set size of this process. ru_maxrss is in KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def get_memory_metrics(prefix='memory'):
    """Return peak memory metrics in the units used for reporting."""
    metrics = {f'{prefix}/host_ram_peak(GB)': get_host_peak_bytes() / 1e9}
    gpu_peak_bytes = get_gpu_peak_bytes()
    if gpu_peak_bytes is not None:
        metrics[f'{prefix}/gpu_peak(MiB)'] = gpu_peak_bytes / 2**20
    return metrics


def get_memory_summary():
    """Return peak memory as raw bytes plus every unit convention, for the run summary file."""
    host_peak_bytes = get_host_peak_bytes()
    summary = {
        'host_ram_peak_bytes': host_peak_bytes,
        'host_ram_peak_GB': host_peak_bytes / 1e9,
        'host_ram_peak_GiB': host_peak_bytes / 2**30,
    }
    gpu_peak_bytes = get_gpu_peak_bytes()
    if gpu_peak_bytes is not None:
        summary['gpu_peak_bytes'] = gpu_peak_bytes
        summary['gpu_peak_MiB'] = gpu_peak_bytes / 2**20
        summary['gpu_peak_MB'] = gpu_peak_bytes / 1e6
    return summary
