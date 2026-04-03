"""
Record a video from a completed SHARC experiment by replaying the CARLA
native recording.

Uses CARLA's replay_file() API for pixel-perfect actor positioning (no
teleportation artefacts), with MPC overlay (waypoints, trajectory,
obstacles) drawn via debug primitives.

Phase 1 (during experiment):  carla_mpc_dynamics.py records the simulation
                              using client.start_recorder().
Phase 2 (this script):        replays the recording and captures video
                              frames with a camera sensor, adding MPC
                              overlay visualisation.

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


def load_recording_meta(experiment_dir):
    """Load CARLA native recording metadata.

    Returns dict with keys: ego_actor_id, recording_file, time_step.
    The metadata JSON is written by carla_mpc_dynamics.py at experiment
    start.
    """
    # experiment_dir is typically .../experiment_list/<config_name>/
    # The meta file is in the experiment_list directory (one level up).
    search_dirs = [
        experiment_dir,
        os.path.dirname(experiment_dir),
        os.path.dirname(os.path.dirname(experiment_dir)),
    ]
    for d in search_dirs:
        meta_path = os.path.join(d, 'carla_recording_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"  Recording meta: {meta_path}")
            return meta
    raise FileNotFoundError(
        "carla_recording_meta.json not found. "
        "Re-run the experiment to generate a CARLA native recording.")


def count_experiment_steps(data):
    """Count the number of unique post-step states in the experiment.

    The x/t arrays have 2 entries per timestep (pre-step, post-step).
    We count unique post-step times, skipping the (0,0,0,0) placeholder.
    """
    x_data = data.get("x", [])
    t_data = data.get("t", [])

    seen_t = set()
    for i, x in enumerate(x_data):
        if isinstance(x, list) and len(x) >= 3:
            ti = t_data[i] if i < len(t_data) else i * 0.1
            px = x[0][0] if isinstance(x[0], list) else x[0]
            py = x[1][0] if isinstance(x[1], list) else x[1]
            # Skip (0,0,...) placeholder at t=0
            if ti == 0 and px == 0 and py == 0:
                continue
            seen_t.add(ti)

    return len(seen_t)


# ── MPC overlay ───────────────────────────────────────────────────────


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
    for i in range(len(wps) - 1):
        debug.draw_line(
            carla.Location(x=wps[i][0], y=wps[i][1], z=z),
            carla.Location(x=wps[i+1][0], y=wps[i+1][1], z=z),
            thickness=0.03,
            color=carla.Color(0, 128, 255),
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
                color=carla.Color(255, 0, 0),
                life_time=life_time,
            )
        for i in range(n - 1):
            debug.draw_line(
                carla.Location(x=tx[i], y=ty[i], z=z),
                carla.Location(x=tx[i+1], y=ty[i+1], z=z),
                thickness=0.05,
                color=carla.Color(255, 0, 0),
                life_time=life_time,
            )

    # Obstacles — orange dots with radius ring
    for ox, oy, r in entry.get("obstacles", []):
        debug.draw_point(
            carla.Location(x=ox, y=oy, z=z),
            size=0.2,
            color=carla.Color(255, 165, 0),
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


def _find_ego_vehicle(world, ego_actor_id, data):
    """Find the ego vehicle in a CARLA replay using multiple strategies.

    Strategy 1: exact actor-ID match (CARLA replay preserves IDs).
    Strategy 2: closest vehicle to experiment's initial position.
    Strategy 3: first vehicle found.
    """
    # Strategy 1 — exact ID
    ego = world.get_actor(ego_actor_id)
    if ego is not None and ego.type_id.startswith('vehicle.'):
        return ego

    vehicles = list(world.get_actors().filter('vehicle.*'))
    if not vehicles:
        return None

    ids_str = ', '.join(f"{v.id}:{v.type_id}" for v in vehicles)
    print(f"  WARNING: Ego ID {ego_actor_id} not found. "
          f"Vehicles present: [{ids_str}]")

    # Strategy 2 — match initial position from experiment data
    x_data = data.get("x", [])
    if x_data and len(x_data) > 0:
        x0 = x_data[0]
        if isinstance(x0, list) and len(x0) >= 2:
            sx = x0[0][0] if isinstance(x0[0], list) else x0[0]
            sy = x0[1][0] if isinstance(x0[1], list) else x0[1]
            best, best_d = None, float('inf')
            for v in vehicles:
                loc = v.get_location()
                d = (loc.x - sx) ** 2 + (loc.y - sy) ** 2
                if d < best_d:
                    best, best_d = v, d
            if best is not None:
                print(f"  Using closest vehicle to start pos "
                      f"({sx:.1f},{sy:.1f}): {best.id} "
                      f"(dist={best_d**.5:.1f}m)")
                return best

    # Strategy 3 — first vehicle
    print(f"  Using first vehicle: {vehicles[0].id}")
    return vehicles[0]


def record_video(experiment_list_dir, fps=20, width=640, height=360,
                 keep_frames=False):
    """Replay CARLA native recording and capture video with MPC overlay.

    Uses client.replay_file() for pixel-perfect actor positioning,
    then draws MPC overlay (waypoints, trajectory, obstacles) per-step.
    """
    label, data, experiment_dir = load_experiment_data(experiment_list_dir)
    n_steps = count_experiment_steps(data)

    if n_steps == 0:
        print("ERROR: No trajectory data found in experiment")
        return None

    config = data.get("config", {})
    sample_time = config.get("system_parameters", {}).get("sample_time", 0.1)

    # Load CARLA native recording metadata
    meta = load_recording_meta(experiment_dir)
    ego_actor_id = meta["ego_actor_id"]
    recording_file = meta["recording_file"]
    rec_time_step = meta.get("time_step", sample_time)

    if not os.path.exists(recording_file):
        print(f"ERROR: Recording file not found: {recording_file}")
        return None

    print(f"  Steps: {n_steps}, sample_time={sample_time}s")
    print(f"  Recording: {recording_file}")
    print(f"  Ego actor ID (from recording): {ego_actor_id}")
    print(f"  Resolution: {width}x{height}")

    # ── Connect to CARLA ──────────────────────────────────────────────
    port = int(os.getenv('_EXP_PORT', 2000))
    client = carla.Client('localhost', port)
    client.set_timeout(60.0)
    world = client.get_world()
    print(f"  Connected to CARLA on port {port}")

    # ── Thorough world cleanup ────────────────────────────────────────
    # 1. Stop any ongoing replay from a previous run
    try:
        client.stop_replayer(keep_actors=False)
    except Exception:
        pass
    time.sleep(0.5)

    # 2. Switch to synchronous mode FIRST — prevents CARLA from
    #    auto-advancing the replay before we're ready to capture.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = rec_time_step
    world.apply_settings(settings)

    # 3. Destroy stale actors (vehicles, sensors, walkers, controllers)
    existing = world.get_actors()
    stale_ids = [
        a.id for a in existing
        if a.type_id.startswith(
            ('vehicle.', 'sensor.', 'walker.', 'controller.'))
    ]
    if stale_ids:
        client.apply_batch_sync(
            [carla.command.DestroyActor(aid) for aid in stale_ids], True)
        print(f"  Cleaned {len(stale_ids)} stale actors")

    # 4. Tick to flush the destructions
    world.tick()
    time.sleep(0.5)

    # Verify the world is clean
    remaining = [
        a for a in world.get_actors()
        if a.type_id.startswith(
            ('vehicle.', 'sensor.', 'walker.', 'controller.'))
    ]
    if remaining:
        print(f"  WARNING: {len(remaining)} actors still present after cleanup")
        for a in remaining:
            print(f"    - {a.id}: {a.type_id}")

    # ── Start CARLA native replay (world is already in sync mode) ─────
    # replay_file() recreates all actors from the recording with their
    # original IDs and positions.  Because we're in sync mode, the replay
    # will NOT advance until we call world.tick().
    print(f"  Starting CARLA replay...")
    replay_result = client.replay_file(
        recording_file, 0.0, 0.0, ego_actor_id, False)
    print(f"  Replay started: {replay_result[:200] if replay_result else '(ok)'}")

    # Let replay initialise actors.  Each tick in sync mode advances the
    # recording by fixed_delta_seconds.  We need a minimum number of ticks
    # for CARLA to create actors from the recording header, but every init
    # tick consumes a recording frame.  We track the count to offset the
    # MPC overlay accordingly.
    INIT_TICKS = 2
    for _ in range(INIT_TICKS):
        world.tick()
    time.sleep(0.5)

    # ── Find the ego vehicle from the replay ──────────────────────────
    ego = _find_ego_vehicle(world, ego_actor_id, data)

    if ego is None:
        print("ERROR: Could not find ego vehicle in replay")
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        client.stop_replayer(keep_actors=False)
        return None

    print(f"  Found ego: {ego.type_id} (id={ego.id}) at "
          f"({ego.get_location().x:.1f}, {ego.get_location().y:.1f})")

    # Log all actors for debugging
    all_actors = world.get_actors()
    actor_summary = [
        f"{a.id}:{a.type_id}"
        for a in all_actors
        if a.type_id.startswith(('vehicle.', 'walker.'))
    ]
    print(f"  Replay actors: [{', '.join(actor_summary)}]")

    # ── Extract MPC overlay data ──────────────────────────────────────
    overlay_data = extract_mpc_overlay_data(data)
    if overlay_data:
        print(f"  MPC overlay: {len(overlay_data)} steps with data")

    # ── Attach camera and record ──────────────────────────────────────
    fmt = "jpg"
    recorder = VideoRecorder(world, ego, experiment_dir,
                             prefix=label, width=width, height=height,
                             fmt=fmt)
    debug = world.debug

    # Determine z for overlay from ego position
    ego_loc = ego.get_location()
    overlay_z = ego_loc.z + 0.5

    # The INIT_TICKS consumed recording frames before the camera was
    # attached.  After INIT_TICKS, the replay is at recording frame
    # INIT_TICKS.  Each tick in the main loop advances by one frame.
    # The camera captures AFTER the tick (actors at the new position),
    # so the frame captured by tick N in the main loop shows actors at
    # recording frame (INIT_TICKS + N + 1).
    #
    # During the original experiment, recording frame F corresponds to
    # the state after experiment step (F - 1), because frame 0 is the
    # initial state (before any evolve_state tick) and frame 1 is after
    # step 0.  The MPC overlay drawn during step k used the plan
    # computed at step k, stored in overlay_data[k].
    #
    # Therefore: overlay_step = (INIT_TICKS + frame_idx + 1) - 1
    #                         = INIT_TICKS + frame_idx
    #
    # We cap the number of frames to avoid running past the recording end.
    frames_to_record = min(n_steps, n_steps - INIT_TICKS + 2)

    print(f"  Recording {frames_to_record} frames ({fmt.upper()}, "
          f"overlay offset={INIT_TICKS})...")

    for frame_idx in range(frames_to_record):
        # Update overlay z from current ego position (road may slope)
        ego_loc = ego.get_location()
        overlay_z = ego_loc.z + 0.5

        # Overlay step accounts for the INIT_TICKS consumed before camera
        overlay_step = frame_idx + INIT_TICKS
        draw_mpc_overlay(debug, overlay_data, overlay_step, overlay_z,
                         life_time=rec_time_step + 0.05)

        # Advance replay by one frame; camera captures the rendered scene
        world.tick()

        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{frames_to_record} "
                  f"(overlay step {overlay_step})")

    # Wait for final frame callback
    time.sleep(1.0)
    print(f"  Frames captured: {recorder.frame_count}")

    video_path = recorder.make_video(fps=fps, delete_frames=not keep_frames)

    # ── Cleanup ───────────────────────────────────────────────────────
    recorder.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    client.stop_replayer(keep_actors=False)
    time.sleep(0.5)

    return video_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Record video from a SHARC experiment (CARLA native replay)")
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
    print("SHARC Experiment Video Recorder (CARLA Native Replay)")
    print("=" * 60)
    video = record_video(args.experiment_dir, args.fps, w, h,
                         keep_frames=args.keep_frames)
    if video:
        print(f"\nVideo recording complete: {video}")
    else:
        print("\nVideo recording failed!", file=sys.stderr)
        sys.exit(1)