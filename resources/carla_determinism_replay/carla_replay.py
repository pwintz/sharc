#!/usr/bin/env python3
"""
CARLA Deterministic Recording and Replay System

Records CARLA scenarios with full determinism and replays them with sub-20cm accuracy.
Uses PID lane following for realistic, collision-free driving.

Usage:
    python carla_replay.py --config config.yaml [--port PORT] [--output-dir DIR]
    
    Example:
    python carla_replay.py --config config.yaml --port 2010
    python carla_replay.py --config config.yaml --port 2010 --output-dir experiments/run_0001
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import carla
except ImportError:
    sys.exit("ERROR: CARLA Python API not found. Run: conda activate carla")

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not found. Run: pip install pyyaml")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# =============================================================================
# PID LANE FOLLOWING CONTROLLER
# =============================================================================

class PIDLaneFollowController: # TODO: might be better to use this controller for NPCs as well as ego
    """Deterministic PID controller that follows the road."""
    
    def __init__(self, vehicle: carla.Vehicle, world: carla.World, config: dict):
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed = config['pid_target_speed'] / 3.6  # km/h to m/s
        self.kp = config['pid_kp']
        self.kd = config['pid_kd']
        self.lookahead = config['pid_lookahead']
        self.previous_error = 0.0
        
    def get_control(self) -> carla.VehicleControl:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get target waypoint
        current_wp = self.map.get_waypoint(transform.location)
        target_wp = current_wp.next(self.lookahead)[0]
        
        # Compute steering (PID)
        target_vec = target_wp.transform.location - transform.location
        norm = math.sqrt(target_vec.x**2 + target_vec.y**2)
        if norm > 0.01:
            target_vec.x /= norm
            target_vec.y /= norm
        
        forward = transform.get_forward_vector()
        cross = forward.x * target_vec.y - forward.y * target_vec.x
        lateral_error = math.asin(max(-1.0, min(1.0, cross)))
        
        derivative = lateral_error - self.previous_error
        steer = self.kp * lateral_error + self.kd * derivative
        steer = max(-1.0, min(1.0, steer))
        self.previous_error = lateral_error
        
        # Speed control
        speed_error = self.target_speed - speed
        throttle = min(0.7, max(0.0, 0.5 * speed_error)) if speed_error > 0 else 0.0
        brake = min(0.5, max(0.0, -0.5 * speed_error)) if speed_error <= 0 else 0.0
        
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

# =============================================================================
# UTILITIES
# =============================================================================

def setup_world(client: carla.Client, config: dict) -> carla.World:
    """Setup deterministic CARLA world."""
    world = client.load_world(config['map'])
    
    # Set world to sync mode with deterministic settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = config['fixed_delta']
    settings.substepping = True
    settings.max_substeps = config['max_substeps'] # TODO: try playing with this parameter to see performance impact
    settings.max_substep_delta_time = config['fixed_delta'] / config['max_substeps']
    settings.deterministic_ragdolls = True  # Ensure deterministic physics
    world.apply_settings(settings)
    
    # Set deterministic weather
    weather = carla.WeatherParameters(
        cloudiness=20.0, 
        precipitation=0.0, 
        sun_altitude_angle=30.0,
        sun_azimuth_angle=20.0,
        fog_density=0.0,
        fog_distance=0.0,
        wetness=30.0
    )
    world.set_weather(weather)
    
    # Tick to apply settings
    world.tick()
    return world

def serialize_transform(tf: carla.Transform) -> Dict:
    return {
        'location': {'x': tf.location.x, 'y': tf.location.y, 'z': tf.location.z},
        'rotation': {'pitch': tf.rotation.pitch, 'yaw': tf.rotation.yaw, 'roll': tf.rotation.roll}
    }

def deserialize_transform(data: Dict) -> carla.Transform:
    return carla.Transform(
        carla.Location(data['location']['x'], data['location']['y'], data['location']['z']),
        carla.Rotation(data['rotation']['pitch'], data['rotation']['yaw'], data['rotation']['roll'])
    )

def serialize_vector3d(vec: carla.Vector3D) -> Dict:
    return {'x': vec.x, 'y': vec.y, 'z': vec.z}

def deserialize_vector3d(data: Dict) -> carla.Vector3D:
    return carla.Vector3D(data['x'], data['y'], data['z'])

# =============================================================================
# RECORDING
# =============================================================================

def record_scenario(client: carla.Client, config: dict, output_dir: str) -> Dict:
    """Record a deterministic scenario."""
    print(f"\n{'='*60}")
    print(f"RECORDING: {config['duration']}s, {config['num_npcs']} NPCs, {config['map']}")
    print(f"Seed: {config['seed']}")
    print(f"{'='*60}")
    
    # Set Python random seed for any randomness in our code
    import random
    random.seed(config['seed'])
    
    world = setup_world(client, config)
    
    # Destroy any existing actors for clean slate
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    for v in vehicles:
        v.destroy()
    world.tick()
    
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(config['seed'])
    tm.global_percentage_speed_difference(0.0)
    tm.set_hybrid_physics_mode(False)  # Disable for full determinism
    
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    
    # Spawn ego
    spawn_z_offset = config.get('spawn_z_offset', 0.5)
    ego_spawn = spawns[config['ego_spawn_index']]
    ego_spawn.location.z += spawn_z_offset
    ego = world.try_spawn_actor(bp_lib.filter('model3')[0], ego_spawn)
    if not ego:
        raise RuntimeError("Failed to spawn ego vehicle")
    
    # Spawn NPCs with deterministic vehicle selection
    npcs = []
    npc_indices = []
    
    # Get deterministic list of vehicle blueprints (sorted for consistency)
    all_vehicles = sorted([bp.id for bp in bp_lib.filter('vehicle.*')])
    # Use only a subset of common vehicles for consistency
    vehicle_pool = [
        'vehicle.audi.a2',
        'vehicle.audi.tt',
        'vehicle.dodge.charger_2020',
        'vehicle.tesla.model3',
        'vehicle.toyota.prius',
        'vehicle.volkswagen.t2'
    ]
    
    for i in range(config['num_npcs']):
        idx = config['npc_spawn_start'] + i * 10
        if idx >= len(spawns):
            break
        npc_spawn = spawns[idx]
        npc_spawn.location.z += spawn_z_offset
        
        # Use deterministic vehicle selection
        vehicle_id = vehicle_pool[i % len(vehicle_pool)]
        try:
            npc_bp = bp_lib.find(vehicle_id)
        except:
            # Fallback to model3 if specific vehicle not found
            npc_bp = bp_lib.find('vehicle.tesla.model3')
        
        npc = world.try_spawn_actor(npc_bp, npc_spawn)
        if npc:
            tm.auto_lane_change(npc, False)
            tm.vehicle_percentage_speed_difference(npc, 0.0)
            tm.distance_to_leading_vehicle(npc, 2.0)
            npc.set_autopilot(True, 8000)
            npcs.append(npc)
            npc_indices.append({'index': idx, 'blueprint': vehicle_id})
    
    settling_ticks = config.get('settling_ticks', 20)
    print(f"Settling physics for {settling_ticks} ticks...")
    for _ in range(settling_ticks):
        world.tick()
    
    # Setup spectator to follow ego
    if config.get('follow_camera', True):
        spectator = world.get_spectator()
        print("Spectator camera following ego during recording")
    
    # Record
    log = {
        'config': config,
        'ego_spawn_transform': serialize_transform(ego_spawn),
        'npc_spawn_indices': npc_indices,
        'frames': []
    }
    
    controller = PIDLaneFollowController(ego, world, config)
    num_frames = int(config['duration'] / config['fixed_delta'])
    
    for i in range(num_frames):
        # Update spectator camera
        if config.get('follow_camera', True):
            ego_tf = ego.get_transform()
            cam_tf = carla.Transform(
                carla.Location(
                    x=ego_tf.location.x - 8 * math.cos(math.radians(ego_tf.rotation.yaw)),
                    y=ego_tf.location.y - 8 * math.sin(math.radians(ego_tf.rotation.yaw)),
                    z=ego_tf.location.z + 3
                ),
                carla.Rotation(pitch=-15, yaw=ego_tf.rotation.yaw, roll=0)
            )
            spectator.set_transform(cam_tf)
        
        # Apply control BEFORE tick
        ctrl = controller.get_control()
        ego.apply_control(ctrl)
        world.tick()
        
        # Record state AFTER tick (result of control)
        current_transform = ego.get_transform()
        #npc_states = [serialize_transform(n.get_transform()) for n in npcs]
        #edited 26/01/26 to get NPC controls
        npc_states = []
        for n in npcs:
            npc_tf = n.get_transform()
            npc_vel = n.get_velocity()
            npc_control = n.get_control()  # this will still work with Traffic Manager autopilot
            npc_states.append({
                'transform': serialize_transform(npc_tf),
                'velocity': serialize_vector3d(npc_vel),
                'control': {
                    'throttle': npc_control.throttle,
                    'steer': npc_control.steer,
                    'brake': npc_control.brake,
                    'hand_brake': npc_control.hand_brake,
                    'reverse': npc_control.reverse,
                    'gear': npc_control.gear
                }
            })
        
        log['frames'].append({
            'time': i * config['fixed_delta'],
            'ego': {
                'transform': serialize_transform(current_transform),
                'control': {'throttle': ctrl.throttle, 'steer': ctrl.steer, 'brake': ctrl.brake}
            },
            'npcs': npc_states
        })
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_frames} frames")
    
    log_file = os.path.join(output_dir, 'scenario.json')
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Saved: {log_file}")
    
    ego.destroy()
    for n in npcs:
        n.destroy()
    
    return log

# =============================================================================
# REPLAY
# =============================================================================

def replay_scenario(client: carla.Client, log: Dict, output_dir: str):
    """Replay a recorded scenario."""
    print(f"\n{'='*60}")
    print(f"REPLAYING: {len(log['frames'])} frames")
    print(f"{'='*60}")
    
    config = log['config']
    world = setup_world(client, config)
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    
    # Spawn ego
    ego_start = deserialize_transform(log['ego_spawn_transform'])
    ego = world.try_spawn_actor(bp_lib.filter('model3')[0], ego_start)
    if not ego:
        ego_start.location.z += 0.2
        ego = world.try_spawn_actor(bp_lib.filter('model3')[0], ego_start)
    if not ego:
        raise RuntimeError("Failed to spawn ego")

    # Optional ego collision tracking during replay
    track_ego_collisions = config.get('track_ego_collisions', True)
    save_collision_events = config.get('save_collision_events', False)
    ego_collision_events = []
    ego_collision_sensor = None
    if track_ego_collisions:
        collision_bp = bp_lib.find('sensor.other.collision')
        ego_collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)

        def on_ego_collision(event):
            impulse = event.normal_impulse
            impulse_mag = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            other = event.other_actor
            ego_collision_events.append({
                'frame': event.frame,
                'other_actor_id': int(other.id) if other else -1,
                'other_actor_type': other.type_id if other else 'unknown',
                'impulse': impulse_mag,
            })

        ego_collision_sensor.listen(on_ego_collision)
    
    # Spawn NPCs - use EXACT same vehicle types from recording
    npc_replay_mode = config.get('npc_replay_mode', 'physics')
    npcs = []
    print(f"[REPLAY] Spawning NPCs with exact blueprints from recording:")
    for npc_info in log['npc_spawn_indices']:
        idx = npc_info['index'] if isinstance(npc_info, dict) else npc_info
        blueprint_id = npc_info['blueprint'] if isinstance(npc_info, dict) else 'vehicle.tesla.model3'
        
        try:
            npc_bp = bp_lib.find(blueprint_id)
            print(f"  Spawn {blueprint_id} at index {idx}")
        except:
            npc_bp = bp_lib.find('vehicle.tesla.model3')
            print(f"  Fallback to model3 at index {idx}")
        
        npc = world.try_spawn_actor(npc_bp, spawns[idx])
        if npc:
            # if npc_replay_mode is 'physics', enable physics for NPCs so we can replay using recorded control inputs
            # if npc_replay_mode is 'teleport', disable physics for NPCs so we can teleport them to the recorded transform
            npc.set_simulate_physics(npc_replay_mode != 'teleport')
            npcs.append(npc)
        else:
            print(f"  WARNING: Failed to spawn NPC at index {idx}")
    
    print(f"[REPLAY] Successfully spawned {len(npcs)} NPCs (replay mode: {npc_replay_mode})")
    
    # Settle - match recording settling
    settling_ticks = config.get('settling_ticks', 20)
    print(f"Settling physics for {settling_ticks} ticks...")
    for _ in range(settling_ticks):
        world.tick()
    
    # Setup spectator
    if config.get('follow_camera', True):
        spectator = world.get_spectator()
        print("Spectator camera following ego during replay")
    
    # Replay
    recorded_traj = []
    replayed_traj = []
    
    # ------- NPC CODE YASH ---------------
    # NPC evaluation: which NPC indices (in the npcs list) to track for error metrics
    eval_npc_indices = config.get('eval_npc_indices', [])
    eval_npc_indices = [i for i in eval_npc_indices if 0 <= i < len(npcs)]
    
    # Per-NPC recorded and replayed trajectories (x, y)
    npc_recorded_traj = {i: [] for i in eval_npc_indices}
    npc_replayed_traj = {i: [] for i in eval_npc_indices}
    # ------- NPC CODE YASH --------------- 
        
    for i, frame in enumerate(log['frames']):
        # Update camera
        if config.get('follow_camera', True):
            ego_tf = ego.get_transform()
            cam_tf = carla.Transform(
                carla.Location(
                    x=ego_tf.location.x - 8 * math.cos(math.radians(ego_tf.rotation.yaw)),
                    y=ego_tf.location.y - 8 * math.sin(math.radians(ego_tf.rotation.yaw)),
                    z=ego_tf.location.z + 3
                ),
                carla.Rotation(pitch=-15, yaw=ego_tf.rotation.yaw, roll=0)
            )
            spectator.set_transform(cam_tf)
        
        # Apply ego control
        ctrl = frame['ego']['control']
        ego.apply_control(carla.VehicleControl(throttle=ctrl['throttle'], steer=ctrl['steer'], brake=ctrl['brake']))
        
        # Apply NPC replay (physics or teleport)
        npc_frame_states = frame['npcs']
        for j, npc in enumerate(npcs):
            if j >= len(npc_frame_states):
                break
            npc_state = npc_frame_states[j]

            if npc_replay_mode == 'teleport':
                # Teleport NPC to recorded transform directly
                target_tf = deserialize_transform(npc_state['transform'])
                npc.set_transform(target_tf)
            else:
                # Physics-based: replay recorded control inputs
                npc_ctrl = npc_state['control']
                npc.apply_control(carla.VehicleControl(
                    throttle=npc_ctrl['throttle'],
                    steer=npc_ctrl['steer'],
                    brake=npc_ctrl['brake'],
                    hand_brake=npc_ctrl['hand_brake'],
                    reverse=npc_ctrl['reverse'],
                    gear=npc_ctrl['gear'],
                ))
        
        world.tick()
        
        # Compare ego state AFTER tick (matching recording)
        rec_tf = deserialize_transform(frame['ego']['transform'])
        rep_tf = ego.get_transform()
        recorded_traj.append((rec_tf.location.x, rec_tf.location.y))
        replayed_traj.append((rep_tf.location.x, rep_tf.location.y))
        
        # ------- NPC CODE YASH ---------------
        # Compare NPC states AFTER tick for selected indices
        for j in eval_npc_indices:
            if j >= len(frame['npcs']) or j >= len(npcs):
                continue
            npc_state = frame['npcs'][j]
            rec_npc_tf = deserialize_transform(npc_state['transform'])
            rep_npc_tf = npcs[j].get_transform()
            npc_recorded_traj[j].append((rec_npc_tf.location.x, rec_npc_tf.location.y))
            npc_replayed_traj[j].append((rep_npc_tf.location.x, rep_npc_tf.location.y))
        # ------- NPC CODE YASH ---------------
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(log['frames'])} frames")
    
    if ego_collision_sensor is not None:
        ego_collision_sensor.stop()
        ego_collision_sensor.destroy()
    ego.destroy()
    for n in npcs:
        n.destroy()
    
    # Analyze ego
    errors = [math.sqrt((r[0]-p[0])**2 + (r[1]-p[1])**2) * 100 for r, p in zip(recorded_traj, replayed_traj)]
    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    
    # ------- NPC CODE YASH ---------------
    # Analyze NPCs
    npc_errors = {}
    for j in eval_npc_indices:
        rec_traj = npc_recorded_traj[j]
        rep_traj = npc_replayed_traj[j]
        if not rec_traj or not rep_traj:
            continue
        npc_errs = [
            math.sqrt((r[0] - p[0])**2 + (r[1] - p[1])**2) * 100
            for r, p in zip(rec_traj, rep_traj)
        ]
        if npc_errs:
            npc_errors[j] = {
                'max': max(npc_errs),
                'mean': sum(npc_errs) / len(npc_errs),
            }
    # ------- NPC CODE YASH ---------------

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Max Error (ego):  {max_err:.2f} cm")
    print(f"Mean Error (ego): {mean_err:.2f} cm")
    print(f"Ego collisions:   {len(ego_collision_events)}")
    print(f"Status: {'✅ Good' if max_err < 20 else '⚠ Check config'}")
    
    # ------- NPC CODE YASH ---------------
    if npc_errors:
        print("\nNPC trajectory errors (cm):")
        for j, stats in npc_errors.items():
            print(f"  NPC {j}: Max {stats['max']:.2f} cm, Mean {stats['mean']:.2f} cm")
    else:
        print("\nNPC trajectory errors: (no eval_npc_indices configured or no data)")
    # ------- NPC CODE YASH ---------------
    
    print(f"{'='*60}\n")
    
    # Write machine-readable results for experiment runner
    ego_collision_count = len(ego_collision_events)
    ego_had_collision = ego_collision_count > 0
    ego_collision_max_impulse = max((e['impulse'] for e in ego_collision_events), default=0.0)
    ego_first_collision_frame = ego_collision_events[0]['frame'] if ego_collision_events else None

    results = {
        'ego_max_error_cm': max_err,
        'ego_mean_error_cm': mean_err,
        'num_frames': len(errors),
        'ego_had_collision': ego_had_collision,
        'ego_collision_count': ego_collision_count,
        'ego_collision_max_impulse': ego_collision_max_impulse,
        'ego_first_collision_frame': ego_first_collision_frame,
    }
    if npc_errors:
        results['npc_errors'] = {str(k): v for k, v in npc_errors.items()}
    if save_collision_events and ego_collision_events:
        results['ego_collision_events'] = ego_collision_events
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")
    
    # Plot
    if HAS_MATPLOTLIB:
        plot_results(recorded_traj, replayed_traj, errors, output_dir, config)
        plot_npc_results(npc_recorded_traj, npc_replayed_traj, npc_errors, output_dir, config)
    
    return max_err, mean_err

def plot_results(recorded, replayed, errors, output_dir, config=None):
    """Generate ego comparison plot."""
    fixed_delta = config['fixed_delta'] if config else 0.05
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    rx, ry = zip(*recorded)
    px, py = zip(*replayed)
    
    axes[0].plot(rx, ry, 'b-', label='Recorded', linewidth=2)
    axes[0].plot(px, py, 'r--', label='Replayed', linewidth=2)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Ego Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    times = [i * fixed_delta for i in range(len(errors))]
    axes[1].plot(times, errors, 'k-', linewidth=1.5)
    axes[1].fill_between(times, 0, errors, alpha=0.3, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error (cm)')
    axes[1].set_title(f'Ego Position Error (Max: {max(errors):.2f} cm)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_file}")


def plot_npc_results(npc_recorded_traj, npc_replayed_traj, npc_errors, output_dir, config=None):
    """Generate per-NPC comparison plots (trajectory + error over time)."""
    if not npc_errors:
        return
    
    fixed_delta = config['fixed_delta'] if config else 0.05
    
    for j, stats in npc_errors.items():
        rec_traj = npc_recorded_traj[j]
        rep_traj = npc_replayed_traj[j]
        
        if not rec_traj or not rep_traj:
            continue
        
        # Compute per-frame errors for this NPC
        errs = [
            math.sqrt((r[0] - p[0])**2 + (r[1] - p[1])**2) * 100
            for r, p in zip(rec_traj, rep_traj)
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        rx, ry = zip(*rec_traj)
        px, py = zip(*rep_traj)
        
        axes[0].plot(rx, ry, 'b-', label='Recorded', linewidth=2)
        axes[0].plot(px, py, 'r--', label='Replayed', linewidth=2)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title(f'NPC {j} Trajectory')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        
        times = [i * fixed_delta for i in range(len(errs))]
        axes[1].plot(times, errs, 'k-', linewidth=1.5)
        axes[1].fill_between(times, 0, errs, alpha=0.3, color='orange')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Error (cm)')
        axes[1].set_title(f'NPC {j} Position Error (Max: {stats["max"]:.2f} cm, Mean: {stats["mean"]:.2f} cm)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'npc_{j}_comparison.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_file}")

# =============================================================================
# MAIN
# =============================================================================

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='CARLA Deterministic Replay')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port (default: 2000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-timestamped under results/)')
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        sys.exit(f"ERROR: Config file not found: {args.config}")
    
    config = load_config(args.config)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('results', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Connect to CARLA
    client = carla.Client('localhost', args.port)
    client.set_timeout(60.0)
    
    # Record
    log = record_scenario(client, config, output_dir)
    
    # Replay
    replay_scenario(client, log, output_dir)
    
    print(f"\nComplete! Results in: {output_dir}\n")

if __name__ == '__main__':
    main()
