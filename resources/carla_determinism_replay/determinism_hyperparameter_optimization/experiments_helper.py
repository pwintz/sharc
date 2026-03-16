#!/usr/bin/env python3
"""Helper utilities for CARLA determinism sweep experiments."""

import csv
import itertools
import json
import os
import subprocess
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not found. Run: pip install pyyaml")

CSV_COLUMNS = [
    'run_id', 'map', 'ego_spawn_index', 'npc_spawn_start', 'num_npcs',
    'seed', 'settling_ticks', 'max_substeps', 'spawn_z_offset',
    'ego_max_error_cm', 'ego_mean_error_cm', 'num_frames',
    'ego_had_collision', 'ego_collision_count', 'ego_collision_max_impulse',
    'ego_first_collision_frame',
    'duration_seconds', 'status',
]

STATUS_PRIORITY = {
    'completed': 5,
    'completed_cached': 4,
    'timeout': 3,
    'no_results': 2,
    'failed': 1,
    'vram_kill': 0,  # CARLA killed due to VRAM threshold; re-run with --failed-only
}


def generate_runs(exp_config: dict) -> list:
    """Generate all parameter combinations from experiment config."""
    dv = exp_config['decision_variables']
    sp = exp_config['scenario_parameters']

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
                        runs.append({
                            'run_id': f"run_{run_id:04d}",
                            'settling_ticks': dv_dict['settling_ticks'],
                            'max_substeps': dv_dict['max_substeps'],
                            'spawn_z_offset': dv_dict['spawn_z_offset'],
                            'map': map_name,
                            'ego_spawn_index': spawn_cfg['ego_spawn_index'],
                            'npc_spawn_start': spawn_cfg['npc_spawn_start'],
                            'num_npcs': num_npcs,
                            'seed': seed,
                        })
                        run_id += 1
    return runs


def get_vram_usage_percent(gpu_index: int | None = None) -> float | None:
    """
    Get VRAM usage as percentage (0-100). If gpu_index is None, returns max across all GPUs.
    Returns None if nvidia-smi unavailable.
    """
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total',
            '--format=csv,noheader,nounits',
        ]
        if gpu_index is not None:
            cmd.append(f'--id={gpu_index}')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split('\n')
        max_pct = 0.0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            used_mb = float(parts[0].split()[0])
            total_mb = float(parts[1].split()[0])
            if total_mb > 0:
                max_pct = max(max_pct, 100.0 * used_mb / total_mb)
        return max_pct if max_pct > 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def kill_carla() -> bool:
    """Kill any running CarlaUE4 process. Returns True if a process was killed."""
    try:
        result = subprocess.run(
            ['pkill', '-9', 'CarlaUE4'],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_and_kill_carla_if_vram_high(threshold: float, gpu_index: int | None = None) -> bool:
    """
    If VRAM usage exceeds threshold (0-100), kill CARLA and return True.
    Returns False if no action taken. Threshold 0 disables the check.
    gpu_index: specific GPU to check, or None for max across all GPUs.
    """
    if threshold <= 0:
        return False
    usage = get_vram_usage_percent(gpu_index)
    if usage is None:
        return False
    if usage >= threshold:
        print(f"\n  [VRAM] Usage {usage:.1f}% >= {threshold}% threshold. Killing CARLA...")
        if kill_carla():
            print(f"  [VRAM] CARLA killed. Restart CARLA and re-run with --failed-only to resume.\n")
            return True
    return False


def build_replay_config(run: dict, fixed: dict) -> dict:
    """Build a complete carla_replay.py config dict from run + fixed params."""
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


class CarlaKilledDueToVRAM(Exception):
    """Raised when CARLA was killed because VRAM exceeded threshold."""

    def __init__(self, usage: float, threshold: float):
        self.usage = usage
        self.threshold = threshold
        super().__init__(f"VRAM {usage:.1f}% >= {threshold}%. CARLA killed. Restart CARLA and re-run with --failed-only to resume.")


def run_single_experiment(run: dict, fixed: dict, port: int,
                          experiments_dir: str, replay_script: str,
                          work_dir: str, *,
                          vram_kill_threshold: float = 0,
                          gpu_index: int | None = None) -> dict:
    """Execute one experiment via subprocess and return one CSV row."""
    run_id = run['run_id']
    output_dir = os.path.join(experiments_dir, run_id)
    results_file = os.path.join(output_dir, 'results.json')

    # Resumability: trust existing valid results.json
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"  [{run_id}] SKIPPED (already completed)")
            return _build_csv_row(run, results, duration=0.0, status='completed_cached')
        except (json.JSONDecodeError, KeyError):
            pass

    # VRAM guard: kill CARLA if usage exceeds threshold before starting a new run
    if vram_kill_threshold > 0:
        usage = get_vram_usage_percent(gpu_index)
        if usage is not None and usage >= vram_kill_threshold:
            if check_and_kill_carla_if_vram_high(vram_kill_threshold, gpu_index):
                raise CarlaKilledDueToVRAM(usage, vram_kill_threshold)

    os.makedirs(output_dir, exist_ok=True)
    config = build_replay_config(run, fixed)
    config_file = os.path.join(output_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

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
    print(f"  [{run_id}] Passing map to carla_replay.py via config: {config['map']}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=work_dir,
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"  [{run_id}] FAILED (exit code {result.returncode})")
            with open(os.path.join(output_dir, 'stderr.log'), 'w') as f:
                f.write(result.stderr)
            with open(os.path.join(output_dir, 'stdout.log'), 'w') as f:
                f.write(result.stdout)
            return _build_csv_row(run, None, duration=duration, status='failed')

        with open(os.path.join(output_dir, 'stdout.log'), 'w') as f:
            f.write(result.stdout)

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            collision_count = results.get('ego_collision_count', 0)
            print(f"  [{run_id}] DONE in {duration:.1f}s | "
                  f"max={results['ego_max_error_cm']:.2f}cm "
                  f"mean={results['ego_mean_error_cm']:.2f}cm "
                  f"collisions={collision_count}")
            return _build_csv_row(run, results, duration=duration, status='completed')

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
        'ego_had_collision': '',
        'ego_collision_count': '',
        'ego_collision_max_impulse': '',
        'ego_first_collision_frame': '',
        'duration_seconds': f"{duration:.1f}",
        'status': status,
    }
    if results:
        row['ego_max_error_cm'] = f"{results['ego_max_error_cm']:.4f}"
        row['ego_mean_error_cm'] = f"{results['ego_mean_error_cm']:.4f}"
        row['num_frames'] = results.get('num_frames', '')
        row['ego_had_collision'] = results.get('ego_had_collision', '')
        row['ego_collision_count'] = results.get('ego_collision_count', '')
        max_impulse = results.get('ego_collision_max_impulse', '')
        row['ego_collision_max_impulse'] = (
            f"{max_impulse:.4f}" if isinstance(max_impulse, (int, float)) else max_impulse
        )
        row['ego_first_collision_frame'] = results.get('ego_first_collision_frame', '')
    return row


def _status_rank(status: str) -> int:
    if status.startswith('error:'):
        return 0
    return STATUS_PRIORITY.get(status, 0)


def _run_id_sort_key(run_id: str):
    try:
        return int(run_id.split('_')[-1])
    except (ValueError, AttributeError, IndexError):
        return run_id


def deduplicate_summary_csv(csv_file: str) -> tuple[int, int]:
    """Deduplicate summary.csv by run_id and keep best/latest status."""
    if not os.path.exists(csv_file):
        return 0, 0

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames or CSV_COLUMNS

    if not all_rows:
        return 0, 0

    best_by_run_id = {}
    best_meta = {}
    for idx, row in enumerate(all_rows):
        run_id = row.get('run_id')
        if not run_id:
            continue
        rank = _status_rank(row.get('status', ''))
        prev = best_meta.get(run_id)
        if prev is None or rank > prev[0] or (rank == prev[0] and idx > prev[1]):
            best_meta[run_id] = (rank, idx)
            best_by_run_id[run_id] = row

    dedup_rows = [best_by_run_id[rid] for rid in sorted(best_by_run_id.keys(), key=_run_id_sort_key)]

    temp_file = f"{csv_file}.tmp"
    with open(temp_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dedup_rows)
    os.replace(temp_file, csv_file)

    return len(all_rows), len(dedup_rows)


def load_best_status_by_run_id(csv_file: str) -> dict:
    """Load best/latest status per run_id from summary.csv."""
    if not os.path.exists(csv_file):
        return {}

    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    best_status_by_run_id = {}
    best_meta = {}
    for idx, row in enumerate(all_rows):
        run_id = row.get('run_id')
        if not run_id:
            continue
        status = row.get('status', '')
        rank = _status_rank(status)
        prev = best_meta.get(run_id)
        if prev is None or rank > prev[0] or (rank == prev[0] and idx > prev[1]):
            best_meta[run_id] = (rank, idx)
            best_status_by_run_id[run_id] = status

    return best_status_by_run_id
