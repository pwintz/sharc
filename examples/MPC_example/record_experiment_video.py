"""
Record a video from a completed SHARC experiment by replaying the trajectory.

Reads the ego trajectory from experiment_data.json and NPC trajectories from
carla_extra.jsonl (across batch subdirectories), then replays them in CARLA
using teleportation (set_transform) with the same vehicle blueprint, spawn
logic, and camera setup as the live experiment (carla_mpc_dynamics.py).

Usage:
  python3 record_experiment_video.py /path/to/experiment_list_dir
  python3 record_experiment_video.py /path/to/experiment_list_dir --highres
  python3 record_experiment_video.py /path/to/experiment_list_dir --fps 30

Default: 640x360 (low-res, fast).  --highres: 1280x720.
Frames are auto-deleted after the video is created.

Requires CARLA running with GPU rendering (xvfb-run, no -nullrhi).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import carla
import numpy as np
import random


# ── Data loading ──────────────────────────────────────────────────────


def load_experiment_data(experiment_list_dir):
    """Load experiment data from a SHARC experiment_list directory."""
    exp_data_files = []
    for entry in os.listdir(experiment_list_dir):
        subdir = os.path.join(experiment_list_dir, entry)
        if os.path.isdir(subdir):
            edata = os.path.join(subdir, "experiment_data.json")
            if os.path.exists(edata):
                exp_data_files.append((entry, edata))

    if not exp_data_files:
        edata = os.path.join(experiment_list_dir, "experiment_data.json")
        if os.path.exists(edata):
            exp_data_files.append(
                (os.path.basename(experiment_list_dir), edata))

    if not exp_data_files:
        raise FileNotFoundError(
            f"No experiment_data.json found in {experiment_list_dir}")

    label, edata_path = exp_data_files[0]
    print(f"Loading experiment: {label}")
    with open(edata_path, 'r') as f:
        data = json.load(f)

    return label, data, os.path.dirname(edata_path)


def _batch_sort_key(name):
    """Numeric sort key for batch directories (batch0, batch1, ..., batch15)."""
    m = re.search(r'batch(\d+)', name)
    return int(m.group(1)) if m else -1


def load_npc_data(experiment_dir):
    """Load NPC trajectory data from carla_extra.jsonl in batch subdirs.

    Batch directories are sorted numerically (not lexicographically)
    to avoid the batch0→batch10→batch1 temporal ordering bug.
    """
    records = []
    # Collect all directories containing carla_extra.jsonl
    extra_files = []
    for dirpath, _dirnames, filenames in os.walk(experiment_dir):
        if "carla_extra.jsonl" in filenames:
            extra_files.append(dirpath)
    # Sort by batch number (numeric)
    extra_files.sort(key=lambda p: _batch_sort_key(os.path.basename(p)))
    for dirpath in extra_files:
        fpath = os.path.join(dirpath, "carla_extra.jsonl")
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    # Records should now be in correct temporal order; sort as safety net
    records.sort(key=lambda r: r.get("t", 0))
    return records


def extract_trajectory(data):
    """Extract ego trajectory from experiment data.

    The x/t arrays have 2 entries per timestep (pre-step, post-step).
    We take the post-step state (odd indices) and skip the (0,0,0,0)
    placeholder at index 0.
    """
    x_data = data.get("x", [])
    t_data = data.get("t", [])
    config = data.get("config", {})

    # Deduplicate: take one state per unique time value.
    # Pattern: t = [0.0, 0.1, 0.1, 0.2, 0.2, ...]
    # Take the LAST x entry for each unique t (post-step state).
    seen_t = {}
    for i, x in enumerate(x_data):
        if isinstance(x, list) and len(x) >= 3:
            ti = t_data[i] if i < len(t_data) else i * 0.1
            px = x[0][0] if isinstance(x[0], list) else x[0]
            py = x[1][0] if isinstance(x[1], list) else x[1]
            psi = x[2][0] if isinstance(x[2], list) else x[2]
            v = 0.0
            if len(x) >= 4:
                v = x[3][0] if isinstance(x[3], list) else x[3]
            seen_t[ti] = {"t": ti, "px": px, "py": py, "psi": psi, "v": v}

    trajectory = [seen_t[k] for k in sorted(seen_t.keys())]

    # Skip the (0,0,0,0) placeholder at t=0
    if len(trajectory) > 1 and trajectory[0]["px"] == 0 and trajectory[0]["py"] == 0:
        trajectory = trajectory[1:]

    return trajectory, config


def extract_mpc_overlay_data(data):
    """Extract per-step waypoints and MPC predicted trajectory for overlay.

    Returns dict:  step_index -> {
        "waypoints": [(wx, wy), ...],
        "traj_x": [...],
        "traj_y": [...],
        "obstacles": [(ox, oy, r), ...],
    }
    """
    config = data.get("config", {})
    mpc_opts = config.get("system_parameters", {}).get("mpc_options", {})
    n_wp = mpc_opts.get("n_waypoints", 40)
    n_obs = mpc_opts.get("n_obstacles", 0)
    sample_time = config.get("system_parameters", {}).get("sample_time", 0.1)

    w_data = data.get("w", [])
    t_data = data.get("t", [])
    pcs = data.get("pending_computations", [])

    overlay = {}  # step_index -> dict

    # Build step_index -> w array  (take post-step entries: odd indices)
    for i in range(len(w_data)):
        ti = t_data[i] if i < len(t_data) else 0
        step_idx = round(ti / sample_time) if sample_time > 0 else i
        w = w_data[i]
        if not isinstance(w, list) or len(w) < 2:
            continue

        entry = overlay.setdefault(step_idx, {})

        # Waypoints: first 2*n_wp entries
        wps = []
        for j in range(n_wp):
            idx_x = 2 * j
            idx_y = 2 * j + 1
            if idx_y < len(w):
                wx = w[idx_x][0] if isinstance(w[idx_x], list) else w[idx_x]
                wy = w[idx_y][0] if isinstance(w[idx_y], list) else w[idx_y]
                wps.append((float(wx), float(wy)))
        entry["waypoints"] = wps

        # Obstacles: after waypoints, 5 values each (x, y, vx, vy, r)
        obs = []
        offset = 2 * n_wp
        for j in range(n_obs):
            base = offset + 5 * j
            if base + 4 < len(w):
                ox = w[base][0] if isinstance(w[base], list) else w[base]
                oy = w[base + 1][0] if isinstance(w[base + 1], list) else w[base + 1]
                r = w[base + 4][0] if isinstance(w[base + 4], list) else w[base + 4]
                ox, oy, r = float(ox), float(oy), float(r)
                if ox < 1e5:  # skip sentinel values
                    obs.append((ox, oy, r))
        entry["obstacles"] = obs

    # Build step_index -> MPC trajectory (from pending_computations metadata)
    for i, pc in enumerate(pcs):
        if not isinstance(pc, dict):
            continue
        md = pc.get("metadata", {})
        if not isinstance(md, dict):
            continue
        tx = md.get("traj_x")
        ty = md.get("traj_y")
        if tx and ty:
            # Map to step index: pending_computations has 2 entries per step
            # (pre-step, post-step); take the post-step entry
            ti = t_data[i] if i < len(t_data) else 0
            step_idx = round(ti / sample_time) if sample_time > 0 else i // 2
            entry = overlay.setdefault(step_idx, {})
            entry["traj_x"] = [float(v) for v in tx]
            entry["traj_y"] = [float(v) for v in ty]

    return overlay


def draw_mpc_overlay(debug, overlay_data, step_idx, z, life_time):
    """Draw waypoints (blue), MPC trajectory (red), obstacles (orange)
    as 3D CARLA debug primitives on the world."""
    entry = overlay_data.get(step_idx)
    if entry is None:
        return

    # Waypoints — blue dots and connecting line
    wps = entry.get("waypoints", [])
    for wx, wy in wps:
        debug.draw_point(
            carla.Location(x=wx, y=wy, z=z),
            size=0.08,
            color=carla.Color(0, 128, 255),  # blue
            life_time=life_time,
        )
    # Connect waypoints with a blue line
    for i in range(len(wps) - 1):
        debug.draw_line(
            carla.Location(x=wps[i][0], y=wps[i][1], z=z),
            carla.Location(x=wps[i+1][0], y=wps[i+1][1], z=z),
            thickness=0.03,
            color=carla.Color(0, 128, 255),  # blue
            life_time=life_time,
        )

    # MPC predicted trajectory — red dots and connecting line
    tx = entry.get("traj_x", [])
    ty = entry.get("traj_y", [])
    if tx and ty:
        n = min(len(tx), len(ty))
        for i in range(n):
            debug.draw_point(
                carla.Location(x=tx[i], y=ty[i], z=z),
                size=0.12,
                color=carla.Color(255, 0, 0),  # red
                life_time=life_time,
            )
        for i in range(n - 1):
            debug.draw_line(
                carla.Location(x=tx[i], y=ty[i], z=z),
                carla.Location(x=tx[i+1], y=ty[i+1], z=z),
                thickness=0.05,
                color=carla.Color(255, 0, 0),  # red
                life_time=life_time,
            )

    # Obstacles — orange dots with radius ring
    for ox, oy, r in entry.get("obstacles", []):
        debug.draw_point(
            carla.Location(x=ox, y=oy, z=z),
            size=0.2,
            color=carla.Color(255, 165, 0),  # orange
            life_time=life_time,
        )
        for j in range(12):
            theta = j * math.pi / 6
            debug.draw_point(
                carla.Location(x=ox + r * math.cos(theta),
                               y=oy + r * math.sin(theta), z=z),
                size=0.05,
                color=carla.Color(255, 165, 0),
                life_time=life_time,
            )


# ── NPC yaw estimation ────────────────────────────────────────────────


def estimate_npc_yaws(npc_records, sample_time):
    """Compute NPC yaw angles from consecutive positions.

    carla_extra.jsonl stores (x, y) but not yaw.  We estimate yaw from
    a look-ahead window (not just the next point) for stability.  For
    near-stationary NPCs the last known heading is held to prevent
    rapid oscillation.

    NPCs are identified by positional index (sorted by ID within each
    record) rather than raw CARLA actor IDs, because IDs change across
    batch resets.
    """
    MIN_DIST_FOR_YAW = 0.3  # metres — must move this far to update heading

    # Collect per-NPC position traces ordered by time, keyed by index.
    # Within each record, NPCs are sorted by their CARLA actor ID so the
    # 0th NPC is always the same physical vehicle.
    traces = {}  # npc_index -> [(t, x, y), ...]
    for rec in npc_records:
        t = rec.get("t", 0)
        npcs_sorted = sorted(rec.get("npcs", []), key=lambda n: n["id"])
        for idx, npc in enumerate(npcs_sorted):
            traces.setdefault(idx, []).append((t, npc["x"], npc["y"]))

    yaw_lookup = {}  # (npc_index, t) -> yaw_degrees
    for npc_idx, pts in traces.items():
        pts.sort()
        last_yaw = 0.0
        for i in range(len(pts)):
            # Look ahead up to 5 steps to find a point with enough displacement
            yaw = None
            for j in range(i + 1, min(i + 6, len(pts))):
                dx = pts[j][1] - pts[i][1]
                dy = pts[j][2] - pts[i][2]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist >= MIN_DIST_FOR_YAW:
                    # CARLA yaw convention: atan2(y, x) in degrees
                    yaw = math.degrees(math.atan2(dy, dx))
                    break
            if yaw is not None:
                last_yaw = yaw
            yaw_lookup[(npc_idx, pts[i][0])] = last_yaw

    return yaw_lookup


# ── Video recorder ────────────────────────────────────────────────────


class VideoRecorder:
    """Attach a camera to a vehicle, capture frames, stitch into MP4."""

    def __init__(self, world, vehicle, output_dir, prefix="experiment",
                 width=640, height=360, fmt="jpg"):
        self.output_dir = output_dir
        self.prefix = prefix
        self.width = width
        self.height = height
        self.fmt = fmt
        self._frame_count = 0
        self._frames_dir = os.path.join(output_dir, f"{prefix}_frames")
        os.makedirs(self._frames_dir, exist_ok=True)

        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        # Match the experiment camera in CarlaMPCDynamics.CameraManager
        bp.set_attribute('fov', '90')

        spawn_tf = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15))
        self.camera = world.spawn_actor(bp, spawn_tf, attach_to=vehicle)
        self.camera.listen(self._on_image)

    def _on_image(self, image):
        fname = os.path.join(
            self._frames_dir,
            f"{self.prefix}_{self._frame_count:06d}.{self.fmt}")
        image.save_to_disk(fname)
        self._frame_count += 1

    def destroy(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

    def make_video(self, fps=20, delete_frames=True):
        """Stitch frames into MP4 with ffmpeg, optionally delete frames."""
        pattern = os.path.join(
            self._frames_dir, f"{self.prefix}_%06d.{self.fmt}")
        outfile = os.path.join(self.output_dir, f"{self.prefix}_video.mp4")

        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-crf', '23', '-preset', 'fast',
            outfile
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(outfile):
            size_mb = os.path.getsize(outfile) / 1024 / 1024
            print(f"  Video saved: {outfile} ({size_mb:.1f} MB)")
            if delete_frames:
                shutil.rmtree(self._frames_dir, ignore_errors=True)
                print(f"  Frames deleted.")
            return outfile
        else:
            print(f"  Video creation FAILED")
            if result.stderr:
                print(f"  ffmpeg: {result.stderr[-500:]}")
            return None

    @property
    def frame_count(self):
        return self._frame_count


# ── Main replay logic ─────────────────────────────────────────────────


def record_video(experiment_list_dir, fps=20, width=640, height=360,
                 keep_frames=False):
    """Replay experiment in CARLA and record video.

    Matches the experiment dynamics exactly:
      - Same vehicle blueprint (vehicle.tesla.model3)
      - Same spawn point (seed % len(spawn_points))
      - Same NPC spawn logic (nearest spawn points, sorted by distance)
      - Same camera setup (fov=90, x=-6, z=3, pitch=-15)
      - Traffic lights all green + frozen (same as experiment)
    """
    label, data, experiment_dir = load_experiment_data(experiment_list_dir)
    trajectory, config = extract_trajectory(data)
    npc_records = load_npc_data(experiment_dir)

    if not trajectory:
        print("ERROR: No trajectory data found in experiment")
        return None

    sample_time = config.get("system_parameters", {}).get("sample_time", 0.1)
    carla_cfg = config.get("carla", {})
    seed = carla_cfg.get("seed", 0)
    npc_cfg = carla_cfg.get("npcs", {})
    n_npc_vehicles = npc_cfg.get("n_vehicles", 0)

    print(f"  Trajectory: {len(trajectory)} steps, sample_time={sample_time}s")
    print(f"  NPC records: {len(npc_records)} (across batch subdirs)")
    print(f"  Config: seed={seed}, n_vehicles={n_npc_vehicles}")
    print(f"  Resolution: {width}x{height}")

    # ── Connect to CARLA ──────────────────────────────────────────────
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    world = client.get_world()

    # Clean up stale actors
    existing = world.get_actors()
    stale_ids = [
        a.id for a in existing
        if a.type_id.startswith(
            ('vehicle.', 'sensor.', 'walker.', 'controller.'))
    ]
    if stale_ids:
        client.apply_batch_sync(
            [carla.command.DestroyActor(aid) for aid in stale_ids], True)
    time.sleep(0.5)

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = sample_time
    world.apply_settings(settings)

    # Freeze traffic lights (same as experiment)
    for tl in world.get_actors().filter('traffic.traffic_light'):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)
    time.sleep(0.3)

    # ── Spawn ego: same logic as CarlaMPCDynamics ─────────────────────
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    random.seed(seed)
    np.random.seed(seed)
    ego_spawn_idx = seed % len(spawn_points)

    ego = None
    actual_ego_idx = None
    for offset in range(len(spawn_points)):
        idx = (ego_spawn_idx + offset) % len(spawn_points)
        ego = world.try_spawn_actor(vehicle_bp, spawn_points[idx])
        if ego is not None:
            actual_ego_idx = idx
            break
    if ego is None:
        print("ERROR: Failed to spawn ego vehicle")
        return None

    print(f"  Ego spawn index: {actual_ego_idx}"
          f" ({spawn_points[actual_ego_idx].location})")

    ego.set_simulate_physics(False)

    # ── Spawn NPCs using initial positions from carla_extra.jsonl ────────
    # The recorded NPC data uses CARLA actor IDs that change each batch
    # reset.  We ignore IDs and map NPCs by positional index (sorted by
    # ID within each record).  This ensures continuity across batches.
    npc_bps = sorted(bp_lib.filter('vehicle.*'), key=lambda bp: bp.id)
    carla_map = world.get_map()

    # Determine how many NPCs and their initial positions from first record
    n_npcs_in_data = 0
    initial_npc_positions = []
    if npc_records:
        first_npcs = sorted(npc_records[0].get("npcs", []),
                            key=lambda n: n["id"])
        n_npcs_in_data = len(first_npcs)
        for npc in first_npcs:
            initial_npc_positions.append((npc["x"], npc["y"]))

    n_to_spawn = max(n_npc_vehicles, n_npcs_in_data)
    npc_actors = []
    for i in range(n_to_spawn):
        bp = npc_bps[i % len(npc_bps)]
        if bp.has_attribute('color'):
            colors = bp.get_attribute('color').recommended_values
            bp.set_attribute('color', colors[i % len(colors)])
        # Spawn at the recorded initial position if available
        if i < len(initial_npc_positions):
            nx, ny = initial_npc_positions[i]
            npc_wp = carla_map.get_waypoint(
                carla.Location(x=nx, y=ny, z=0), project_to_road=True)
            npc_z = npc_wp.transform.location.z if npc_wp else 0.3
            spawn_tf = carla.Transform(
                carla.Location(x=nx, y=ny, z=npc_z + 0.3),
                npc_wp.transform.rotation if npc_wp else carla.Rotation())
        else:
            # Fallback: nearest spawn points
            ego_loc = spawn_points[actual_ego_idx].location
            available_sp = [(j, sp) for j, sp in enumerate(spawn_points)
                           if j != actual_ego_idx]
            available_sp.sort(
                key=lambda pair: (pair[1].location.x - ego_loc.x) ** 2
                               + (pair[1].location.y - ego_loc.y) ** 2)
            spawn_tf = available_sp[i % len(available_sp)][1]
        npc = world.try_spawn_actor(bp, spawn_tf)
        if npc is not None:
            npc.set_simulate_physics(False)
            npc_actors.append(npc)

    print(f"  Spawned {len(npc_actors)}/{n_to_spawn} NPC vehicles")

    world.tick()
    world.tick()

    # ── Build NPC position+yaw lookup ─────────────────────────────────
    # Map each NPC record to step index; estimate yaw from consecutive pos.
    # NPCs are keyed by positional index (sorted by ID within each record),
    # NOT by raw CARLA actor IDs which change across batch resets.
    yaw_lookup = estimate_npc_yaws(npc_records, sample_time)

    npc_step_data = {}  # step_idx -> [(x, y, yaw), ...]  (by positional index)
    for rec in npc_records:
        t = rec.get("t", 0)
        step_idx = round(t / sample_time)
        npcs_sorted = sorted(rec.get("npcs", []), key=lambda n: n["id"])
        positions = []
        for idx, npc in enumerate(npcs_sorted):
            nx = npc["x"]
            ny = npc["y"]
            yaw = yaw_lookup.get((idx, t), 0.0)
            positions.append((nx, ny, yaw))
        npc_step_data[step_idx] = positions

    # ── Extract MPC overlay data (waypoints + predicted trajectory) ───
    overlay_data = extract_mpc_overlay_data(data)
    if overlay_data:
        print(f"  MPC overlay: {len(overlay_data)} steps with data")

    # Cache the CARLA map for road-surface z queries
    carla_map = world.get_map()

    # ── Attach camera and record ──────────────────────────────────────
    fmt = "jpg"  # JPEG is ~5x faster than PNG for frame capture
    recorder = VideoRecorder(world, ego, experiment_dir,
                             prefix=label, width=width, height=height,
                             fmt=fmt)
    debug = world.debug
    n_frames = len(trajectory)
    print(f"  Recording {n_frames} frames ({fmt.upper()})...")

    for step_idx, wp in enumerate(trajectory):
        # Query road surface z at ego position for correct ground placement
        ego_wp = carla_map.get_waypoint(
            carla.Location(x=wp["px"], y=wp["py"], z=0),
            project_to_road=True)
        ego_z = ego_wp.transform.location.z if ego_wp else 0.3

        # Teleport ego to recorded position at road surface height
        tf = carla.Transform(
            carla.Location(x=wp["px"], y=wp["py"], z=ego_z + 0.05),
            carla.Rotation(yaw=math.degrees(wp["psi"]))
        )
        ego.set_transform(tf)

        # Teleport NPCs to recorded positions at road surface height
        if step_idx in npc_step_data:
            for actor_idx, pos in enumerate(npc_step_data[step_idx]):
                if pos is not None and actor_idx < len(npc_actors):
                    nx, ny, nyaw = pos
                    # Query road z for this NPC position
                    npc_wp = carla_map.get_waypoint(
                        carla.Location(x=nx, y=ny, z=0),
                        project_to_road=True)
                    npc_z = npc_wp.transform.location.z if npc_wp else ego_z
                    npc_tf = carla.Transform(
                        carla.Location(x=nx, y=ny, z=npc_z + 0.05),
                        carla.Rotation(yaw=nyaw)
                    )
                    npc_actors[actor_idx].set_transform(npc_tf)

        # Draw MPC overlay (waypoints=blue, trajectory=red, obstacles=orange)
        overlay_z = ego_z + 0.5  # slightly above road for visibility
        draw_mpc_overlay(debug, overlay_data, step_idx, overlay_z,
                         life_time=sample_time + 0.05)

        world.tick()

        if step_idx % 100 == 0:
            print(f"    Step {step_idx}/{n_frames}")

    # Wait for final frame callback
    time.sleep(1.0)
    print(f"  Frames captured: {recorder.frame_count}")

    video_path = recorder.make_video(fps=fps, delete_frames=not keep_frames)

    # ── Cleanup ───────────────────────────────────────────────────────
    recorder.destroy()
    for npc in npc_actors:
        npc.destroy()
    ego.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    time.sleep(0.3)

    return video_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Record video from a SHARC experiment")
    parser.add_argument(
        'experiment_dir',
        help="Path to experiment_list dir (or experiment subdir)")
    parser.add_argument(
        '--fps', type=int, default=20,
        help="Video frame rate (default: 20)")
    parser.add_argument(
        '--highres', action='store_true',
        help="Record at 1280x720 (default: 640x360)")
    parser.add_argument(
        '--width', type=int, default=None,
        help="Custom width (overrides --highres)")
    parser.add_argument(
        '--height', type=int, default=None,
        help="Custom height (overrides --highres)")
    parser.add_argument(
        '--keep-frames', action='store_true',
        help="Keep individual frame images after creating video")
    args = parser.parse_args()

    # Resolution: custom > highres > default (640x360)
    if args.width and args.height:
        w, h = args.width, args.height
    elif args.highres:
        w, h = 1280, 720
    else:
        w, h = 640, 360

    print("=" * 60)
    print("SHARC Experiment Video Recorder")
    print("=" * 60)
    video = record_video(args.experiment_dir, args.fps, w, h,
                         keep_frames=args.keep_frames)
    if video:
        print(f"\nVideo recording complete: {video}")
    else:
        print("\nVideo recording failed!", file=sys.stderr)
        sys.exit(1)
