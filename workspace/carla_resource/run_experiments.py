#!/usr/bin/env python3
"""
CARLA Determinism Experiment Runner

Generates all parameter combinations from experiment_config.yaml,
runs carla_replay.py for each one, and aggregates results into a CSV.

Supports resumability: skips runs whose results.json already exists.
Sorts runs by map to minimize expensive map reloads.

Usage:
    python run_experiments.py --config experiment_config.yaml --port 2000
    python run_experiments.py --config experiment_config.yaml --port 2000 --dry-run
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not found. Run: pip install pyyaml")

# =============================================================================
# COMBINATION GENERATION
# =============================================================================

def generate_runs(exp_config: dict) -> list:
    """
    Generate all parameter combinations from experiment config.

    Returns a list of dicts, each representing one run with all parameters
    needed to build a carla_replay.py config.
    """
    dv = exp_config['decision_variables']
    sp = exp_config['scenario_parameters']

    # Decision variable grid
    dv_keys = sorted(dv.keys())
    dv_values = [dv[k] for k in dv_keys]
    dv_combos = list(itertools.product(*dv_values))

    runs = []
    run_id = 0

    for dv_combo in dv_combos:
        dv_dict = dict(zip(dv_keys, dv_combo))

        for map_name, map_cfg in sp['maps'].items():
            for spawn_cfg in map_cfg['spawn_configs']:
                for num_npcs in sp['num_npcs']:
                    for seed in sp['seeds']:
                        run = {
                            'run_id': f"run_{run_id:04d}",
                            # Decision variables
                            'settling_ticks': dv_dict['settling_ticks'],
                            'max_substeps': dv_dict['max_substeps'],
                            'spawn_z_offset': dv_dict['spawn_z_offset'],
                            # Scenario parameters
                            'map': map_name,
                            'ego_spawn_index': spawn_cfg['ego_spawn_index'],
                            'npc_spawn_start': spawn_cfg['npc_spawn_start'],
                            'num_npcs': num_npcs,
                            'seed': seed,
                        }
                        runs.append(run)
                        run_id += 1

    return runs


def build_replay_config(run: dict, fixed: dict) -> dict:
    """Build a complete carla_replay.py config dict from a run and fixed params."""
    config = {}
    config.update(fixed)
    config['map'] = run['map']
    config['seed'] = run['seed']
    config['num_npcs'] = run['num_npcs']
    config['ego_spawn_index'] = run['ego_spawn_index']
    config['npc_spawn_start'] = run['npc_spawn_start']
    config['settling_ticks'] = run['settling_ticks']
    config['max_substeps'] = run['max_substeps']
    config['spawn_z_offset'] = run['spawn_z_offset']
    return config

# =============================================================================
# EXECUTION
# =============================================================================

CSV_COLUMNS = [
    'run_id', 'map', 'ego_spawn_index', 'npc_spawn_start', 'num_npcs',
    'seed', 'settling_ticks', 'max_substeps', 'spawn_z_offset',
    'ego_max_error_cm', 'ego_mean_error_cm', 'num_frames',
    'duration_seconds', 'status',
]


def run_single_experiment(run: dict, fixed: dict, port: int,
                          experiments_dir: str, script_dir: str) -> dict:
    """
    Execute a single experiment run via subprocess.

    Returns a dict with result columns for the CSV.
    """
    run_id = run['run_id']
    output_dir = os.path.join(experiments_dir, run_id)
    results_file = os.path.join(output_dir, 'results.json')

    # Check if already completed (resumability)
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"  [{run_id}] SKIPPED (already completed)")
            return _build_csv_row(run, results, duration=0.0, status='completed_cached')
        except (json.JSONDecodeError, KeyError):
            pass  # Re-run if results file is corrupt

    # Build config and write to run directory
    os.makedirs(output_dir, exist_ok=True)
    config = build_replay_config(run, fixed)
    config_file = os.path.join(output_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Run carla_replay.py via subprocess
    replay_script = os.path.join(script_dir, 'carla_replay.py')
    cmd = [
        sys.executable, replay_script,
        '--config', config_file,
        '--port', str(port),
        '--output-dir', output_dir,
    ]

    print(f"  [{run_id}] Running: {run['map']} | "
          f"settle={run['settling_ticks']} substeps={run['max_substeps']} "
          f"z={run['spawn_z_offset']} npcs={run['num_npcs']} seed={run['seed']} "
          f"ego={run['ego_spawn_index']} npc_start={run['npc_spawn_start']}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per run
            cwd=script_dir,
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"  [{run_id}] FAILED (exit code {result.returncode})")
            # Save stderr for debugging
            stderr_file = os.path.join(output_dir, 'stderr.log')
            with open(stderr_file, 'w') as f:
                f.write(result.stderr)
            stdout_file = os.path.join(output_dir, 'stdout.log')
            with open(stdout_file, 'w') as f:
                f.write(result.stdout)
            return _build_csv_row(run, None, duration=duration, status='failed')

        # Save stdout for reference
        stdout_file = os.path.join(output_dir, 'stdout.log')
        with open(stdout_file, 'w') as f:
            f.write(result.stdout)

        # Read results
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"  [{run_id}] DONE in {duration:.1f}s | "
                  f"max={results['ego_max_error_cm']:.2f}cm "
                  f"mean={results['ego_mean_error_cm']:.2f}cm")
            return _build_csv_row(run, results, duration=duration, status='completed')
        else:
            print(f"  [{run_id}] FAILED (no results.json produced)")
            return _build_csv_row(run, None, duration=duration, status='no_results')

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"  [{run_id}] TIMEOUT after {duration:.1f}s")
        return _build_csv_row(run, None, duration=duration, status='timeout')
    except Exception as e:
        duration = time.time() - start_time
        print(f"  [{run_id}] ERROR: {e}")
        return _build_csv_row(run, None, duration=duration, status=f'error:{e}')


def _build_csv_row(run: dict, results: dict, duration: float, status: str) -> dict:
    """Build a CSV row dict from run params and results."""
    row = {
        'run_id': run['run_id'],
        'map': run['map'],
        'ego_spawn_index': run['ego_spawn_index'],
        'npc_spawn_start': run['npc_spawn_start'],
        'num_npcs': run['num_npcs'],
        'seed': run['seed'],
        'settling_ticks': run['settling_ticks'],
        'max_substeps': run['max_substeps'],
        'spawn_z_offset': run['spawn_z_offset'],
        'ego_max_error_cm': '',
        'ego_mean_error_cm': '',
        'num_frames': '',
        'duration_seconds': f"{duration:.1f}",
        'status': status,
    }
    if results:
        row['ego_max_error_cm'] = f"{results['ego_max_error_cm']:.4f}"
        row['ego_mean_error_cm'] = f"{results['ego_mean_error_cm']:.4f}"
        row['num_frames'] = results.get('num_frames', '')
    return row

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CARLA Determinism Experiment Runner')
    parser.add_argument('--config', type=str, default='experiment_config.yaml',
                        help='Experiment config file (default: experiment_config.yaml)')
    parser.add_argument('--port', type=int, default=None,
                        help='CARLA server port (overrides config)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print all runs without executing')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Directory to store experiment results (default: experiments/)')
    args = parser.parse_args()

    # Load experiment config
    if not os.path.exists(args.config):
        sys.exit(f"ERROR: Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)

    port = args.port if args.port else exp_config.get('port', 2000)
    fixed = exp_config['fixed']
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate all runs
    runs = generate_runs(exp_config)

    # Sort by map to minimize map reloads
    runs.sort(key=lambda r: r['map'])

    print(f"\n{'='*70}")
    print(f"CARLA DETERMINISM EXPERIMENT SWEEP")
    print(f"{'='*70}")
    print(f"Total runs:       {len(runs)}")
    print(f"Decision vars:    settling_ticks x max_substeps x spawn_z_offset")
    print(f"Scenario params:  maps x spawn_configs x num_npcs x seeds")
    print(f"Port:             {port}")
    print(f"Output:           {args.experiments_dir}/")
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

    # Setup experiments directory
    experiments_dir = args.experiments_dir
    os.makedirs(experiments_dir, exist_ok=True)

    # Save experiment config for reproducibility
    exp_config_copy = os.path.join(experiments_dir, 'experiment_config.yaml')
    with open(exp_config_copy, 'w') as f:
        yaml.dump(exp_config, f)

    # CSV summary file
    csv_file = os.path.join(experiments_dir, 'summary.csv')
    csv_exists = os.path.exists(csv_file)

    # Load existing results to track what's been done
    existing_run_ids = set()
    if csv_exists:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status', '').startswith('completed'):
                    existing_run_ids.add(row['run_id'])

    # Open CSV for appending
    csv_fh = open(csv_file, 'a', newline='')
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    if not csv_exists:
        writer.writeheader()

    # Run experiments
    completed = 0
    skipped = 0
    failed = 0
    total = len(runs)
    start_time = time.time()
    current_map = None

    try:
        for i, run in enumerate(runs):
            # Print map change header
            if run['map'] != current_map:
                current_map = run['map']
                print(f"\n{'='*50}")
                print(f"MAP: {current_map}")
                print(f"{'='*50}")

            # Progress
            elapsed = time.time() - start_time
            if completed + skipped > 0:
                avg_time = elapsed / (completed + skipped)
                remaining = avg_time * (total - i)
                eta = timedelta(seconds=int(remaining))
            else:
                eta = "estimating..."
            print(f"\n[{i+1}/{total}] ETA: {eta}")

            # Run experiment
            row = run_single_experiment(run, fixed, port, experiments_dir, script_dir)

            # Write to CSV immediately (crash-safe)
            writer.writerow(row)
            csv_fh.flush()

            if row['status'].startswith('completed'):
                completed += 1
            elif row['status'] == 'completed_cached':
                skipped += 1
            else:
                failed += 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Progress saved. Re-run to resume.\n")
    finally:
        csv_fh.close()

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Completed:  {completed}")
    print(f"Skipped:    {skipped} (already had results)")
    print(f"Failed:     {failed}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Results:    {csv_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

