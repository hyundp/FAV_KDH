"""Collect peak memory numbers from measurement runs into one table.

Repeated runs of the same agent are aggregated so that run-to-run spread is visible
alongside the mean.
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='exp/')
    parser.add_argument('--run_group', default='mem')
    args = parser.parse_args()

    pattern = os.path.join(args.save_dir, args.run_group, '*', 'mem_summary.json')
    by_agent = defaultdict(list)
    for summary_path in sorted(glob.glob(pattern)):
        with open(summary_path) as f:
            summary = json.load(f)
        by_agent[summary.get('agent_name', '?')].append(summary)

    if not by_agent:
        print(f'No mem_summary.json found under {pattern}')
        return

    def stats(summaries, key):
        values = [s[key] for s in summaries if s.get(key) is not None]
        if not values:
            return None, None
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / mean * 100 if mean else 0.0
        return mean, spread

    def fmt(value, spec):
        return format(value, spec) if value is not None else '-'

    header = f'{"Method":<10}{"n":>3}{"GPU peak":>12}{"spread":>9}{"Host RAM peak":>16}{"spread":>9}'
    print(header)
    print('-' * len(header))
    for agent in sorted(by_agent):
        summaries = by_agent[agent]
        gpu_mean, gpu_spread = stats(summaries, 'gpu_peak_MiB')
        host_mean, host_spread = stats(summaries, 'host_ram_peak_GB')
        print(
            f'{agent:<10}'
            f'{len(summaries):>3}'
            f'{fmt(gpu_mean, ".0f") + " MiB":>12}'
            f'{fmt(gpu_spread, ".1f") + "%":>9}'
            f'{fmt(host_mean, ".2f") + " GB":>16}'
            f'{fmt(host_spread, ".1f") + "%":>9}'
        )


if __name__ == '__main__':
    main()
