#!/usr/bin/env python3
"""
CARLA Determinism Experiment Runner (CLI entrypoint).

Core execution logic lives in experiments_helper.py to keep this file concise.
"""

import argparse
import csv
import os
import sys
import time
from datetime import timedelta

try:
    import carla
except ImportError:
    carla = None

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not found. Run: pip install pyyaml")

from experiments_helper import (
    CSV_COLUMNS,
    CarlaKilledDueToVRAM,
    _build_csv_row,
    deduplicate_summary_csv,
    generate_runs,
    load_best_status_by_run_id,
    run_single_experiment,
)


def normalize_map_name(map_name: str) -> str:
    """Normalize map string to short CARLA name (e.g., Town03)."""
    if not map_name:
        return ""
    name = map_name.split('/')[-1]
    if name.endswith('.umap'):
        name = name[:-5]
    return name


def main():
    runner_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(runner_dir, 'experiment_config.yaml')
    default_replay_script = os.path.abspath(os.path.join(runner_dir, '..', 'carla_replay.py'))

    parser = argparse.ArgumentParser(description='CARLA Determinism Experiment Runner')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Experiment config file')
    parser.add_argument('--port', type=int, default=None,
                        help='CARLA server port (overrides config)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print all runs without executing')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directory to store experiment results (default: experiments/)')
    parser.add_argument('--carla-replay-script', type=str, default=default_replay_script,
                        help='Path to carla_replay.py (default: ../carla_replay.py)')
    parser.add_argument('--failed-only', action='store_true',
                        help='Only run entries currently marked failed/timeout/no_results/vram_kill in summary.csv')
    parser.add_argument('--vram-kill-threshold', type=float, default=80.0,
                        help='Kill CARLA if VRAM usage exceeds this percent (0=disabled, default: 80)')
    parser.add_argument('--gpu-index', type=int, default=None,
                        help='GPU index to check for VRAM (default: check all, use max)')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: Config file not found: {config_path}")

    replay_script = os.path.abspath(args.carla_replay_script)
    if not os.path.exists(replay_script):
        sys.exit(f"ERROR: carla_replay.py not found: {replay_script}")

    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    port = args.port if args.port else exp_config.get('port', 2000)
    fixed = exp_config['fixed']

    runs = generate_runs(exp_config)
    runs.sort(key=lambda r: r['map'])

    print(f"\n{'='*70}")
    print("CARLA DETERMINISM EXPERIMENT SWEEP")
    print(f"{'='*70}")
    print(f"Total runs:       {len(runs)}")
    print("Decision vars:    settling_ticks x max_substeps x spawn_z_offset")
    print("Scenario params:  maps x spawn_configs x num_npcs x seeds")
    print(f"Port:             {port}")
    vram_str = f"{args.vram_kill_threshold}%" if args.vram_kill_threshold > 0 else "disabled"
    print(f"VRAM kill:        {vram_str}")
    print(f"Output:           {args.experiments_dir}/")
    print(f"Replay script:    {replay_script}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("DRY RUN -- listing all combinations:\n")
        current_map = None
        for run in runs:
            if run['map'] != current_map:
                current_map = run['map']
                print(f"\n--- {current_map} ---")
            print(f"  {run['run_id']}: settle={run['settling_ticks']} "
                  f"substeps={run['max_substeps']} z={run['spawn_z_offset']} "
                  f"npcs={run['num_npcs']} seed={run['seed']} "
                  f"ego={run['ego_spawn_index']} npc_start={run['npc_spawn_start']}")
        print(f"\nTotal: {len(runs)} runs")
        return

    experiments_dir = os.path.abspath(args.experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)

    with open(os.path.join(experiments_dir, 'experiment_config.yaml'), 'w') as f:
        yaml.dump(exp_config, f)

    csv_file = os.path.join(experiments_dir, 'summary.csv')
    status_by_run_id = load_best_status_by_run_id(csv_file)

    if args.failed_only:
        rerun_statuses = {'failed', 'timeout', 'no_results', 'vram_kill'}
        original_total = len(runs)
        runs = [r for r in runs if status_by_run_id.get(r['run_id']) in rerun_statuses]
        print(f"Filtering (--failed-only): {len(runs)}/{original_total} runs selected")
        print(f"Target statuses: {sorted(rerun_statuses)}")
        if not runs:
            print("No failed/timeout/no_results runs found in summary.csv. Exiting.")
            return

    run_maps = sorted({normalize_map_name(r['map']) for r in runs})
    print(f"Run maps requested (normalized): {run_maps}")

    completed = 0
    skipped = 0
    failed = 0
    total = len(runs)
    start_time = time.time()
    current_map = None
    work_dir = os.path.dirname(replay_script)
    csv_fh = None
    writer = None

    try:
        if not args.dry_run:
            if carla is None:
                sys.exit("ERROR: Python CARLA package not found; cannot run map preflight.")
            try:
                client = carla.Client('localhost', port)
                client.set_timeout(60.0)
                available_maps_raw = sorted(client.get_available_maps())
                available_maps_norm = sorted({normalize_map_name(m) for m in available_maps_raw})

                print("\nAvailable maps reported by CARLA:")
                for m in available_maps_raw:
                    print(f"  - {m} (normalized: {normalize_map_name(m)})")

                missing = sorted(m for m in run_maps if m not in set(available_maps_norm))
                if missing:
                    sys.exit(f"ERROR: Requested maps not available on server: {missing}")
            except Exception as e:
                sys.exit(f"ERROR: CARLA map preflight failed: {e}")

        csv_exists = os.path.exists(csv_file)
        csv_fh = open(csv_file, 'a', newline='')
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
        if not csv_exists:
            writer.writeheader()

        for i, run in enumerate(runs):
            if run['map'] != current_map:
                current_map = run['map']
                print(f"\n{'='*50}")
                print(f"MAP: {current_map}")
                print(f"{'='*50}")

            elapsed = time.time() - start_time
            if completed + skipped > 0:
                avg_time = elapsed / (completed + skipped)
                remaining = avg_time * (total - i)
                eta = timedelta(seconds=int(remaining))
            else:
                eta = "estimating..."
            print(f"\n[{i+1}/{total}] ETA: {eta}")

            try:
                row = run_single_experiment(
                    run=run,
                    fixed=fixed,
                    port=port,
                    experiments_dir=experiments_dir,
                    replay_script=replay_script,
                    work_dir=work_dir,
                    vram_kill_threshold=args.vram_kill_threshold,
                    gpu_index=args.gpu_index,
                )
            except CarlaKilledDueToVRAM as e:
                row = _build_csv_row(run, None, duration=0.0, status='vram_kill')
                writer.writerow(row)
                csv_fh.flush()
                print(f"\n{'='*70}")
                print("CARLA KILLED (VRAM threshold exceeded)")
                print(f"{'='*70}")
                print(f"VRAM usage: {e.usage:.1f}% (threshold: {e.threshold}%)")
                print("Restart CARLA, then re-run with --failed-only to resume.")
                print(f"{'='*70}\n")
                failed += 1
                break

            writer.writerow(row)
            csv_fh.flush()

            if row['status'].startswith('completed'):
                completed += 1
            elif row['status'] == 'completed_cached':
                skipped += 1
            else:
                failed += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved. Re-run to resume.\n")
    finally:
        if csv_fh is not None:
            csv_fh.close()
        if os.path.exists(csv_file):
            before_rows, after_rows = deduplicate_summary_csv(csv_file)
            removed = before_rows - after_rows
            print(f"Deduplicated summary.csv: {before_rows} -> {after_rows} rows (removed {removed})")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("EXPERIMENT SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Completed:  {completed}")
    print(f"Skipped:    {skipped} (already had results)")
    print(f"Failed:     {failed}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Results:    {csv_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

