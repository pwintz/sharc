#!/usr/bin/env python3
"""
Compute and report experiment metrics from SHARC experiment data.

Reads experiment_data.json and carla_extra.jsonl from an experiment
directory and produces metrics.json with:
  - collision_count / collision_detected
  - avg_mpc_cost (per step)
  - path_tracking_rmse (ego vs lane-centre waypoints)
  - speed_tracking_rmse (ego speed vs target)
  - min_obstacle_distance (closest approach to any NPC)
  - feasibility_rate (fraction of MPC solves that are feasible)
  - avg_computation_time (mean solver delay)
  - max_deceleration (strongest braking, m/s²)
  - total_distance (m)
  - final_speed (km/h)

Usage:
  python3 compute_metrics.py <experiment_dir>

The experiment_dir should contain experiment_data.json (or
experiment_data_incremental.json) and optionally carla_extra.jsonl.
"""

import argparse
import json
import math
import os
import sys

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_experiment_data(exp_dir):
    """Load experiment_data.json (or incremental fallback)."""
    for name in ("experiment_data.json", "experiment_data_incremental.json"):
        path = os.path.join(exp_dir, name)
        if os.path.isfile(path):
            with open(path) as fh:
                data = json.load(fh)
            # Normalise key name
            if "pending_computation" in data and "pending_computations" not in data:
                data["pending_computations"] = data.pop("pending_computation")
            return data
    return None


def _load_carla_extra(exp_dir):
    """Load all carla_extra.jsonl records (handles batch sub-dirs too)."""
    import re as _re

    def _batch_key(name):
        m = _re.search(r'batch(\d+)', name)
        return int(m.group(1)) if m else float('inf')

    files = []
    top = os.path.join(exp_dir, "carla_extra.jsonl")
    if os.path.isfile(top):
        files.append(top)
    try:
        for entry in sorted(os.listdir(exp_dir), key=_batch_key):
            sub = os.path.join(exp_dir, entry, "carla_extra.jsonl")
            if os.path.isfile(sub):
                files.append(sub)
    except OSError:
        pass

    records = []
    for fpath in files:
        with open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Core metrics computation ─────────────────────────────────────────────────

def compute_metrics(exp_dir):
    """Compute all metrics and return as a dict."""
    data = _load_experiment_data(exp_dir)
    if data is None:
        print(f"ERROR: No experiment data found in {exp_dir}", file=sys.stderr)
        sys.exit(1)

    extra_records = _load_carla_extra(exp_dir)

    # Extract per-step data (odd indices = end-of-step)
    x_arr = np.asarray(data.get("x", []), dtype=float)
    u_arr = np.asarray(data.get("u", []), dtype=float)
    t_arr = np.asarray(data.get("t", []), dtype=float)
    w_arr = np.asarray(data.get("w", []), dtype=float) if data.get("w") else np.array([])
    pc_arr = data.get("pending_computations", [])

    # Per-step = every other entry starting at index 1
    t_steps = t_arr[1::2]
    x_steps = x_arr[1::2] if x_arr.ndim == 2 else np.asarray(x_arr[1::2], dtype=float)
    u_steps = u_arr[1::2] if u_arr.ndim == 2 else np.asarray(u_arr[1::2], dtype=float)
    w_steps = w_arr[1::2] if w_arr.size > 0 else np.array([])
    pc_steps = pc_arr[1::2] if pc_arr else []

    if x_steps.ndim == 1:
        x_steps = x_steps.reshape(-1, 1) if x_steps.size else x_steps
    if u_steps.ndim == 1:
        u_steps = u_steps.reshape(-1, 1) if u_steps.size else u_steps
    if w_steps.ndim == 1 and w_steps.size:
        w_steps = w_steps.reshape(-1, 1)

    n_steps = len(t_steps)
    cfg = data.get("config", {})
    sp = cfg.get("system_parameters", {})
    mpc_opts = sp.get("mpc_options", {})
    n_wp = mpc_opts.get("n_waypoints", 40)
    target_speed = sp.get("target_speed", None)

    metrics = {}

    # ── 1. Collision metrics ─────────────────────────────────────────────
    collision_events = []
    for rec in extra_records:
        col = rec.get("collision")
        if col:
            collision_events.append(col)

    metrics["collision_detected"] = len(collision_events) > 0
    metrics["collision_count"] = len(collision_events)

    # ── 2. MPC cost per step ─────────────────────────────────────────────
    costs = []
    feasible_count = 0
    total_solves = 0
    delays = []
    seen_k = set()

    for pc in pc_steps:
        if not isinstance(pc, dict):
            continue
        meta = pc.get("metadata", {})
        if not meta:
            continue
        k = meta.get("k")
        if k is not None and k in seen_k:
            continue  # skip duplicate (same computation repeated across 2 entries)
        if k is not None:
            seen_k.add(k)

        cost = meta.get("cost")
        if cost is not None:
            costs.append(cost)
        if "is_feasible" in meta:
            total_solves += 1
            if meta["is_feasible"]:
                feasible_count += 1
        delay = pc.get("delay")
        if delay is not None:
            delays.append(delay)

    metrics["avg_mpc_cost"] = float(np.mean(costs)) if costs else None
    metrics["min_mpc_cost"] = float(np.min(costs)) if costs else None
    metrics["max_mpc_cost"] = float(np.max(costs)) if costs else None
    metrics["std_mpc_cost"] = float(np.std(costs)) if costs else None

    # ── 3. Feasibility rate ──────────────────────────────────────────────
    metrics["feasibility_rate"] = feasible_count / total_solves if total_solves > 0 else None
    metrics["total_solves"] = total_solves

    # ── 4. Computation delay stats ───────────────────────────────────────
    metrics["avg_computation_time"] = float(np.mean(delays)) if delays else None
    metrics["max_computation_time"] = float(np.max(delays)) if delays else None

    # ── 5. Path tracking RMSE ────────────────────────────────────────────
    # Compare ego position (x[0], x[1]) to the nearest waypoint at each step
    path_errors = []
    if w_steps.size > 0 and n_wp > 0 and x_steps.shape[1] >= 2:
        for i in range(min(n_steps, len(w_steps))):
            ego_x, ego_y = x_steps[i, 0], x_steps[i, 1]
            w = w_steps[i]
            # Waypoints are pairs: w[0]=x1, w[1]=y1, w[2]=x2, w[3]=y2, ...
            wp_xs = w[0:2*n_wp:2]
            wp_ys = w[1:2*n_wp:2]
            # Closest waypoint distance = cross-track error proxy
            dists = np.sqrt((wp_xs - ego_x)**2 + (wp_ys - ego_y)**2)
            path_errors.append(float(np.min(dists)))

    metrics["path_tracking_rmse"] = float(np.sqrt(np.mean(np.array(path_errors)**2))) if path_errors else None
    metrics["path_tracking_max_error"] = float(np.max(path_errors)) if path_errors else None

    # ── 6. Speed tracking RMSE ───────────────────────────────────────────
    if target_speed is not None and x_steps.shape[1] >= 4:
        speed = x_steps[:, 3]  # km/h
        speed_errors = speed - target_speed
        metrics["speed_tracking_rmse"] = float(np.sqrt(np.mean(speed_errors**2)))
    else:
        metrics["speed_tracking_rmse"] = None

    # ── 7. Min obstacle distance ─────────────────────────────────────────
    min_obs_dist = float('inf')
    for rec in extra_records:
        ego_x = rec.get("ego_x", 0)
        ego_y = rec.get("ego_y", 0)
        for npc in rec.get("npcs", []):
            dx = npc["x"] - ego_x
            dy = npc["y"] - ego_y
            dist = math.sqrt(dx*dx + dy*dy)
            min_obs_dist = min(min_obs_dist, dist)

    metrics["min_obstacle_distance"] = float(min_obs_dist) if min_obs_dist < float('inf') else None

    # ── 8. Max deceleration ──────────────────────────────────────────────
    if u_steps.size > 0:
        accel = u_steps[:, 0]  # acceleration (m/s²) for MPC-style
        metrics["max_deceleration"] = float(np.min(accel))  # most negative = strongest braking
    else:
        metrics["max_deceleration"] = None

    # ── 9. Total distance ────────────────────────────────────────────────
    if x_steps.shape[1] >= 2 and n_steps > 1:
        dx = np.diff(x_steps[:, 0])
        dy = np.diff(x_steps[:, 1])
        metrics["total_distance"] = float(np.sum(np.sqrt(dx**2 + dy**2)))
    else:
        metrics["total_distance"] = None

    # ── 10. Final speed ──────────────────────────────────────────────────
    if x_steps.shape[1] >= 4 and n_steps > 0:
        metrics["final_speed"] = float(x_steps[-1, 3])
    else:
        metrics["final_speed"] = None

    # ── 11. Number of steps ──────────────────────────────────────────────
    metrics["n_steps"] = n_steps

    # Round floats for clean output
    for k, v in metrics.items():
        if isinstance(v, float) and v is not None:
            metrics[k] = round(v, 6)

    return metrics


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_metrics(metrics):
    """Print metrics in a clean, readable format."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT METRICS")
    print("=" * 60)

    col = "\033[92m✓ No collision\033[0m" if not metrics["collision_detected"] else \
          f"\033[91m✗ {metrics['collision_count']} COLLISION(S)\033[0m"
    print(f"  Collision:              {col}")
    print(f"  Steps:                  {metrics['n_steps']}")
    print()

    print("  MPC Performance:")
    if metrics["avg_mpc_cost"] is not None:
        print(f"    Avg cost / step:      {metrics['avg_mpc_cost']:.2f}")
        print(f"    Min / Max cost:       {metrics['min_mpc_cost']:.2f} / {metrics['max_mpc_cost']:.2f}")
        print(f"    Std cost:             {metrics['std_mpc_cost']:.2f}")
    if metrics["feasibility_rate"] is not None:
        pct = metrics["feasibility_rate"] * 100
        print(f"    Feasibility rate:     {pct:.1f}% ({metrics['total_solves']} solves)")
    if metrics["avg_computation_time"] is not None:
        print(f"    Avg compute time:     {metrics['avg_computation_time']*1000:.1f} ms")
        print(f"    Max compute time:     {metrics['max_computation_time']*1000:.1f} ms")
    print()

    print("  Tracking:")
    if metrics["path_tracking_rmse"] is not None:
        print(f"    Path RMSE:            {metrics['path_tracking_rmse']:.4f} m")
        print(f"    Path max error:       {metrics['path_tracking_max_error']:.4f} m")
    if metrics["speed_tracking_rmse"] is not None:
        print(f"    Speed RMSE:           {metrics['speed_tracking_rmse']:.2f} km/h")
    print()

    print("  Safety & Dynamics:")
    if metrics["min_obstacle_distance"] is not None:
        print(f"    Min obstacle dist:    {metrics['min_obstacle_distance']:.2f} m")
    if metrics["max_deceleration"] is not None:
        print(f"    Max deceleration:     {metrics['max_deceleration']:.2f} m/s²")
    if metrics["total_distance"] is not None:
        print(f"    Total distance:       {metrics['total_distance']:.2f} m")
    if metrics["final_speed"] is not None:
        print(f"    Final speed:          {metrics['final_speed']:.2f} km/h")

    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute SHARC experiment metrics")
    parser.add_argument("experiment_dir", help="Path to the experiment directory")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only write JSON, no pretty print")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(exp_dir):
        print(f"ERROR: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(exp_dir)

    # Save metrics.json
    out_path = os.path.join(exp_dir, "metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved to {out_path}")

    if not args.quiet:
        print_metrics(metrics)

    return metrics


if __name__ == "__main__":
    main()
