"""
CarlaMPCDynamics — CARLA dynamics bridge for the MPC controller.

Differences from the PID-oriented CarlaDynamics:
  * State  x = (px, py, psi, v)  ∈ ℝ⁴    (no waypoints in state)
  * Input  u = (a, delta)        ∈ ℝ²    (acceleration + steering)
  * Exogenous input w  = dense waypoint array  ∈ ℝ^{2·N_w}
  * Converts acceleration → CARLA throttle/brake internally.

Batch replay support:
  Every CARLA ``world.tick()`` is logged together with the
  ``VehicleControl`` that was applied immediately before it.  When SHARC
  detects a missed-computation deadline and rolls back to an earlier
  time-step, ``prepare_for_batch()`` destroys all actors, respawns them
  with identical parameters, and fast-forwards the CARLA world by
  re-applying the logged controls — yielding a world state that matches
  the original run to within CARLA's intrinsic physics tolerance
  (typically < 0.1 m for batch-sized fast-forwards of ~8–64 steps).
"""

import json
import math
import os
import time as _time
import numpy as np
from sharc.dynamics_base import Dynamics
import carla
import random
import pygame
import sys


def _has_display():
    """Return True if a graphical display is available."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def pygame_init(w=1280, h=720):
    if not _has_display():
        print("[CarlaMPCDynamics] No display detected — running headless (no pygame window).")
        return None
    pygame.init()
    display = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Camera View — MPC")
    return display


class CameraManager:
    """Third-person camera attached to ego vehicle."""
    def __init__(self, world, vehicle, width=1280, height=720):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.surface = None

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', '90')

        spawn_point = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15)
        )
        self.camera = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        self.camera.listen(lambda data: self._on_image(data))

    def _on_image(self, image):
        if not _has_display():
            return
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 4))
        img = img[:, :, :3][:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def destroy(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None


class CarlaMPCDynamics(Dynamics):
    """
    CARLA dynamics for the bicycle-model MPC controller.

    Config keys (under ``carla``):
      seed           – deterministic seed
      n_waypoints    – number of dense waypoints to provide (N_w)
      waypoint_spacing – distance [m] between consecutive waypoints
      npcs.n_vehicles, npcs.n_walkers – NPC counts
    """

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _wrap_angle(angle_rad):
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return angle_rad

    def _choose_reference_successor(self, wp, candidates):
        """Choose a deterministic successor when CARLA offers multiple branches."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        current_tf = wp.transform
        current_yaw = math.radians(current_tf.rotation.yaw)
        current_loc = current_tf.location

        def candidate_key(candidate):
            tf = candidate.transform
            yaw = math.radians(tf.rotation.yaw)
            heading_error = abs(self._wrap_angle(yaw - current_yaw))
            same_lane_penalty = 0 if candidate.lane_id == wp.lane_id else 1
            same_road_penalty = 0 if candidate.road_id == wp.road_id else 1
            lateral_offset = abs(
                -(tf.location.x - current_loc.x) * math.sin(current_yaw) +
                (tf.location.y - current_loc.y) * math.cos(current_yaw)
            )
            return (
                same_lane_penalty,
                same_road_penalty,
                heading_error,
                lateral_offset,
                tf.location.x,
                tf.location.y,
            )

        return min(candidates, key=candidate_key)

    def _build_reference_route(self):
        """Precompute a stable forward route from the ego spawn lane."""
        start_wp = self._carla_map.get_waypoint(
            self.vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if start_wp is None:
            raise RuntimeError("Failed to build reference route: ego vehicle is not on a driving lane.")

        route_len = max(self.n_wp * 25, 1000)
        route = [(start_wp.transform.location.x, start_wp.transform.location.y)]
        wp = start_wp

        for _ in range(route_len - 1):
            nxt = self._choose_reference_successor(wp, wp.next(self.wp_spacing))
            if nxt is None:
                route.append(route[-1])
                continue
            wp = nxt
            route.append((wp.transform.location.x, wp.transform.location.y))

        self._reference_route = route
        self._reference_route_idx = 0

    def _closest_reference_route_index(self, location):
        if not getattr(self, "_reference_route", None):
            return 0

        best_idx = 0
        best_dist_sq = float("inf")
        for i, (wx, wy) in enumerate(self._reference_route):
            dx = location.x - wx
            dy = location.y - wy
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = i

        self._reference_route_idx = best_idx
        return best_idx

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def setup_system(self):
        self.time_step = float(self.config["system_parameters"]["sample_time"])

        carla_cfg     = self.config.get("carla", {})
        self.seed     = carla_cfg.get("seed", 0)
        mpc_opts      = self.config["system_parameters"].get("mpc_options", {})
        self.n_wp     = mpc_opts.get("n_waypoints", 10)
        self.wp_spacing = mpc_opts.get("waypoint_spacing", 2.0)

        # Obstacle-aware MPC options (0 disables obstacle packing)
        self.n_obs            = mpc_opts.get("n_obstacles", 0)
        self.detection_radius = mpc_opts.get("detection_radius", 30.0)

        random.seed(self.seed)
        np.random.seed(self.seed)

        # ---- Connect to CARLA ---------------------------------------- #
        print("[CarlaMPCDynamics] Creating CARLA session …")
        carla_port = int(os.environ.get('_EXP_PORT', 2000))
        self.client = carla.Client("localhost", carla_port)
        self.client.set_timeout(60.0)   # generous timeout; nullrhi can be slow to settle
        self.world = self.client.get_world()

        # Reset to async mode first (in case previous run left sync on)
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        # Destroy leftover actors from a previous session
        existing = self.world.get_actors()
        stale_ids = [
            a.id for a in existing
            if a.type_id.startswith(('vehicle.', 'sensor.', 'walker.', 'controller.'))
        ]
        if stale_ids:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(aid) for aid in stale_ids], True
            )
            print(f"[CarlaMPCDynamics] Removed {len(stale_ids)} leftover actor(s).")
        _time.sleep(0.5)  # let CARLA settle

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.time_step
        # Guard: ensure substep settings are valid (invalid values permanently
        # corrupt CARLA physics, requiring a server restart).
        if settings.max_substeps < 1 or settings.max_substeps > 16:
            print(f"[CarlaMPCDynamics] WARNING: invalid max_substeps={settings.max_substeps}, resetting to 10")
            settings.max_substeps = 10
        if settings.max_substep_delta_time <= 0 or settings.max_substep_delta_time > 0.05:
            print(f"[CarlaMPCDynamics] WARNING: invalid max_substep_delta_time={settings.max_substep_delta_time}, resetting to 0.01")
            settings.max_substep_delta_time = 0.01
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager(8100)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed)
        self.world.tick()

        # ---- Spawn ego ------------------------------------------------ #
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_points = self.world.get_map().get_spawn_points()
        ego_spawn_idx = self.seed % len(spawn_points)

        # Try preferred spawn point first, then others if it fails
        self.vehicle = None
        actual_ego_idx = None
        for offset in range(len(spawn_points)):
            idx = (ego_spawn_idx + offset) % len(spawn_points)
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[idx])
            if self.vehicle is not None:
                actual_ego_idx = idx
                print(f"[CarlaMPCDynamics] Spawned ego at spawn-point index {idx}")
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")

        # Dimensions (already set by Dynamics.__init__)
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

        # Cache the CARLA map for waypoint queries (needed by NPC road spawning)
        self._carla_map = self.world.get_map()

        # ---- NPC spawning -------------------------------------------- #
        npc_cfg = carla_cfg.get("npcs", {})
        n_vehicles = npc_cfg.get("n_vehicles", 0)
        n_walkers  = npc_cfg.get("n_walkers", 0)
        self.npc_vehicles = self._spawn_npc_vehicles(bp_lib, spawn_points, actual_ego_idx, n_vehicles)
        self.npc_walkers, self.npc_walker_controllers = self._spawn_npc_walkers(bp_lib, n_walkers)

        # ---- Force all traffic lights to stay green ------------------- #
        for tl in self.world.get_actors().filter('traffic.traffic_light'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # ---- Collision sensor ---------------------------------------- #
        self._collision_events = []
        collision_bp = bp_lib.find('sensor.other.collision')
        self._collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle)
        self._collision_sensor.listen(self._on_collision)

        # ---- Sidecar file state -------------------------------------- #
        self._sim_dir   = None
        self._extra_fh  = None

        # ---- Pygame + camera (skip in headless mode) ----------------------- #
        self.display = pygame_init()
        if _has_display():
            self.camera_manager = CameraManager(self.world, self.vehicle)
        else:
            self.camera_manager = None
            print("[CarlaMPCDynamics] Headless mode — camera window disabled.")

        # ---- Batch replay state -------------------------------------- #
        # Every world.tick() is logged with the VehicleControl that was
        # applied beforehand. prepare_for_batch() uses this to reset and
        # fast-forward the world when a batch rolls back.
        self._tick_log = []          # list of carla.VehicleControl per tick
        self._npc_tick_log = []      # list of [carla.VehicleControl, ...] per tick (one per NPC)
        self._walker_tick_log = []   # list of [carla.WalkerControl, ...] per tick
        self._tick_count = 0         # total CARLA ticks driven so far
        # Map time-step index → tick index at the START of that step.
        # Used to find how far to fast-forward after a rollback.
        self._step_to_tick = {}
        self._current_step = 0       # current time-step index

        # Remember spawn parameters for deterministic respawn.
        self._ego_spawn_idx = actual_ego_idx
        self._n_npc_vehicles = npc_cfg.get("n_vehicles", 0)
        self._n_npc_walkers = npc_cfg.get("n_walkers", 0)
        self._road_npc_count = 0  # set by _spawn_npc_vehicles
        # Optimal settle ticks for determinism.  Validated via a comprehensive
        # parameter sweep (determinism_sweep.py / determinism_sweep_v2.py):
        #   - settle_ticks=1 is best (non-monotonic: 2-5 are *worse*).
        #   - Warmup reset + tick-by-tick replay achieves 0.000 m deviation.
        #   - Independent sessions (no replay) show 0.5–5 m at 1024 steps.
        self._n_settle_ticks = 1

        # ---- Warm-up reset ------------------------------------------- #
        # A single destroy-respawn cycle before the first real batch
        # eliminates the "first-reset divergence" observed in benchmarks.
        # Validated: safety-critical scenarios achieve 0.000 m ego deviation
        # over 512 steps with warmup + replay.
        self._do_warmup_reset()
        self._build_reference_route()

    def _do_warmup_reset(self):
        """Perform one dummy reset cycle to stabilise CARLA's internal state."""
        print("[CarlaMPCDynamics] Warm-up reset …")
        self._reset_world()
        print("[CarlaMPCDynamics] Warm-up reset complete.")

    # ------------------------------------------------------------------ #
    #  Initial state                                                      #
    # ------------------------------------------------------------------ #

    def get_initial_state(self):
        """Return the ego vehicle's current CARLA state as a column vector.

        Called by plant_runner to override config ``x0`` with the actual
        simulator spawn position so the MPC planner starts from the
        correct location.
        """
        transform = self.vehicle.get_transform()
        velocity  = self.vehicle.get_velocity()
        yaw_rad   = math.radians(transform.rotation.yaw)
        v_x_local = velocity.x * math.cos(yaw_rad) + velocity.y * math.sin(yaw_rad)

        if self.n == 4:
            return np.array([
                [transform.location.x],
                [transform.location.y],
                [yaw_rad],
                [v_x_local],
            ], dtype=float)
        else:
            v_y_local = -velocity.x * math.sin(yaw_rad) + velocity.y * math.cos(yaw_rad)
            ang_vel = self.vehicle.get_angular_velocity()
            r_rad_s = math.radians(ang_vel.z)
            return np.array([
                [transform.location.x],
                [transform.location.y],
                [yaw_rad],
                [v_x_local],
                [v_y_local],
                [r_rad_s],
            ], dtype=float)

    # ------------------------------------------------------------------ #
    #  Sidecar / collision helpers                                        #
    # ------------------------------------------------------------------ #

    def set_sim_dir(self, sim_dir: str):
        """Called by plant_runner at the start of each batch."""
        self._sim_dir = sim_dir if sim_dir.endswith('/') else sim_dir + '/'
        # Open a fresh sidecar file for this batch (one file per batch dir).
        if self._extra_fh is not None:
            try:
                self._extra_fh.close()
            except Exception:
                pass
        try:
            self._extra_fh = open(
                os.path.join(self._sim_dir, 'carla_extra.jsonl'), 'w', buffering=1)
        except OSError:
            self._extra_fh = None

        # ---- Start CARLA native recording on the first batch --------- #
        # The recorder captures all actor positions/controls in a binary
        # file that can later be replayed via client.replay_file() for
        # high-fidelity video generation (no teleportation artefacts).
        if not getattr(self, '_recorder_started', False):
            # Derive experiment root (the experiment_list directory that
            # contains this batch/sim sub-directory):
            #   _sim_dir = .../experiment_list/<config_name>/
            #   experiment_root = .../experiment_list/
            experiment_root = os.path.dirname(
                self._sim_dir.rstrip('/'))
            self._recording_file = os.path.join(
                experiment_root, 'carla_recording.log')
            try:
                self.client.start_recorder(self._recording_file, True)
                self._recorder_started = True
                # Save metadata for the video replay script.
                meta = {
                    'ego_actor_id': self.vehicle.id,
                    'recording_file': self._recording_file,
                    'time_step': self.time_step,
                }
                meta_path = os.path.join(
                    experiment_root, 'carla_recording_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f"[CarlaMPCDynamics] Started CARLA recorder → "
                      f"{self._recording_file}")
            except Exception as e:
                print(f"[CarlaMPCDynamics] WARNING: Failed to start "
                      f"CARLA recorder: {e}")
                self._recorder_started = False

    # ------------------------------------------------------------------ #
    #  Batch replay (reset + fast-forward)                                #
    # ------------------------------------------------------------------ #

    def prepare_for_batch(self, first_time_index: int, sim_config: dict):
        """Reset and fast-forward CARLA when a batch rolls back.

        Called by ``plant_runner`` before each batch begins.  If
        *first_time_index* is behind the dynamics' current position (a
        rollback after a missed-computation deadline), the method:

        1. destroys all actors and respawns them deterministically,
        2. replays the logged ``VehicleControl`` sequence from tick 0
           up to the tick that corresponds to *first_time_index*.

        If *first_time_index* matches the current position (no rollback,
        normal continuation), the method is a no-op.
        """
        target_tick = self._step_to_tick.get(first_time_index)

        if first_time_index == 0 and self._tick_count == 0:
            # Very first batch — nothing to replay.
            self._current_step = 0
            return

        if target_tick is not None and target_tick == self._tick_count:
            # Normal continuation — dynamics is already at the right state.
            self._current_step = first_time_index
            return

        if target_tick is None:
            # first_time_index was never reached yet.  This shouldn't happen
            # in normal operation but guard against it.
            print(f"[CarlaMPCDynamics] WARNING: step {first_time_index} not in "
                  f"tick map (keys: {sorted(self._step_to_tick.keys())}). "
                  f"Skipping replay.")
            self._current_step = first_time_index
            return

        # ── Rollback detected ─────────────────────────────────────────
        print(f"[CarlaMPCDynamics] Rollback detected: dynamics at tick "
              f"{self._tick_count} but batch starts at step "
              f"{first_time_index} (tick {target_tick}).  "
              f"Resetting and fast-forwarding …")

        self._reset_world()
        self._fast_forward(target_tick)

        # Trim the tick log and step map to discard the invalidated future.
        self._tick_log = self._tick_log[:target_tick]
        self._npc_tick_log = self._npc_tick_log[:target_tick]
        self._walker_tick_log = self._walker_tick_log[:target_tick]
        self._tick_count = target_tick
        invalidated = [k for k in self._step_to_tick if k > first_time_index]
        for k in invalidated:
            del self._step_to_tick[k]
        self._current_step = first_time_index

        print(f"[CarlaMPCDynamics] Fast-forward complete.  "
              f"Dynamics now at tick {self._tick_count}, "
              f"step {self._current_step}.")

    def _reset_world(self):
        """Destroy all actors and respawn ego + NPCs deterministically."""

        # ---- Stop collision sensor stream ----------------------------- #
        if getattr(self, '_collision_sensor', None) is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None

        # ---- Stop walker controllers ---------------------------------- #
        for ctrl in getattr(self, 'npc_walker_controllers', []):
            try:
                if ctrl is not None:
                    ctrl.stop()
            except Exception:
                pass

        # ---- Destroy camera ------------------------------------------- #
        if getattr(self, 'camera_manager', None) is not None:
            self.camera_manager.destroy()
            self.camera_manager = None

        # ---- Disable autopilot BEFORE destroy (avoids TM warnings) ---- #
        tm_port = self.traffic_manager.get_port()
        for npc in getattr(self, 'npc_vehicles', []):
            try:
                if npc.is_alive:
                    npc.set_autopilot(False, tm_port)
            except Exception:
                pass

        # ---- Batch destroy all managed actors ------------------------- #
        destroy_ids = []
        for ctrl in getattr(self, 'npc_walker_controllers', []):
            if ctrl is not None:
                destroy_ids.append(ctrl.id)
        for walker in getattr(self, 'npc_walkers', []):
            destroy_ids.append(walker.id)
        for npc in getattr(self, 'npc_vehicles', []):
            destroy_ids.append(npc.id)
        if getattr(self, 'vehicle', None) is not None:
            destroy_ids.append(self.vehicle.id)
        if destroy_ids:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(aid) for aid in destroy_ids], True)
        _time.sleep(0.5)

        # ---- Reset TM seed ------------------------------------------- #
        self.traffic_manager.set_synchronous_mode(False)
        self.traffic_manager = self.client.get_trafficmanager(8100)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed)

        # ---- Respawn ego --------------------------------------------- #
        random.seed(self.seed)
        np.random.seed(self.seed)

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.try_spawn_actor(
            vehicle_bp, spawn_points[self._ego_spawn_idx])
        if self.vehicle is None:
            raise RuntimeError(
                f"Failed to respawn ego at index {self._ego_spawn_idx}")

        # ---- Respawn NPCs -------------------------------------------- #
        self.npc_vehicles = self._spawn_npc_vehicles(
            bp_lib, spawn_points, self._ego_spawn_idx, self._n_npc_vehicles)
        self._npcs_stop_commanded = False
        self.npc_walkers, self.npc_walker_controllers = self._spawn_npc_walkers(
            bp_lib, self._n_npc_walkers)

        # ---- Freeze traffic lights ----------------------------------- #
        for tl in self.world.get_actors().filter('traffic.traffic_light'):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        # ---- Re-attach collision sensor ------------------------------ #
        self._collision_events = []
        collision_bp = bp_lib.find('sensor.other.collision')
        self._collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle)
        self._collision_sensor.listen(self._on_collision)

        # ---- Re-attach camera (if display available) ----------------- #
        if _has_display():
            self.camera_manager = CameraManager(self.world, self.vehicle)

        # ---- Settle ticks -------------------------------------------- #
        for _ in range(self._n_settle_ticks):
            self.world.tick()

    def _fast_forward(self, target_tick):
        """Replay logged ego + NPC controls from tick 0 to *target_tick*."""
        n = min(target_tick, len(self._tick_log))
        if n == 0:
            return
        print(f"[CarlaMPCDynamics] Fast-forwarding {n} ticks …")

        # Disable TM autopilot and walker AI so we can apply logged controls
        tm_port = self.traffic_manager.get_port()
        for npc in self.npc_vehicles:
            try:
                if npc.is_alive:
                    npc.set_autopilot(False, tm_port)
            except Exception:
                pass
        for ctrl in self.npc_walker_controllers:
            try:
                if ctrl is not None:
                    ctrl.stop()
            except Exception:
                pass

        for i in range(n):
            self.vehicle.apply_control(self._tick_log[i])
            # Apply logged NPC vehicle controls
            if i < len(self._npc_tick_log):
                for j, npc in enumerate(self.npc_vehicles):
                    if j < len(self._npc_tick_log[i]) and self._npc_tick_log[i][j] is not None:
                        try:
                            if npc.is_alive:
                                npc.apply_control(self._npc_tick_log[i][j])
                        except Exception:
                            pass
            # Apply logged walker controls
            if i < len(self._walker_tick_log):
                for j, walker in enumerate(self.npc_walkers):
                    if j < len(self._walker_tick_log[i]) and self._walker_tick_log[i][j] is not None:
                        try:
                            if walker.is_alive:
                                walker.apply_control(self._walker_tick_log[i][j])
                        except Exception:
                            pass
            self.world.tick()

        # Re-enable TM autopilot and walker AI for ongoing simulation
        for npc in self.npc_vehicles:
            try:
                if npc.is_alive:
                    npc.set_autopilot(True, tm_port)
            except Exception:
                pass
        self._configure_road_npcs_tm()
        for ctrl in self.npc_walker_controllers:
            try:
                if ctrl is not None:
                    ctrl.start()
                    dest = self.world.get_random_location_from_navigation()
                    if dest is not None:
                        ctrl.go_to_location(dest)
                    ctrl.set_max_speed(1.4)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Tick logging helpers                                               #
    # ------------------------------------------------------------------ #

    def _log_tick(self, control: 'carla.VehicleControl'):
        """Record ego + NPC controls for a single CARLA tick."""
        self._tick_log.append(carla.VehicleControl(
            throttle=control.throttle,
            steer=control.steer,
            brake=control.brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse,
            manual_gear_shift=control.manual_gear_shift,
            gear=control.gear,
        ))
        # Log NPC vehicle controls (read what TM actually applied)
        npc_ctrls = []
        for npc in getattr(self, 'npc_vehicles', []):
            try:
                if npc.is_alive:
                    c = npc.get_control()
                    npc_ctrls.append(carla.VehicleControl(
                        throttle=c.throttle, steer=c.steer, brake=c.brake,
                        hand_brake=c.hand_brake, reverse=c.reverse,
                        manual_gear_shift=c.manual_gear_shift, gear=c.gear))
                else:
                    npc_ctrls.append(None)
            except Exception:
                npc_ctrls.append(None)
        self._npc_tick_log.append(npc_ctrls)
        # Log walker controls
        walker_ctrls = []
        for walker in getattr(self, 'npc_walkers', []):
            try:
                if walker.is_alive:
                    wc = walker.get_control()
                    walker_ctrls.append(carla.WalkerControl(
                        direction=wc.direction, speed=wc.speed, jump=wc.jump))
                else:
                    walker_ctrls.append(None)
            except Exception:
                walker_ctrls.append(None)
        self._walker_tick_log.append(walker_ctrls)
        self._tick_count += 1

    def _register_step_start(self, time_step_index: int):
        """Map a time-step index to the current tick count."""
        if time_step_index not in self._step_to_tick:
            self._step_to_tick[time_step_index] = self._tick_count

    def _on_collision(self, event):
        """CARLA collision sensor callback."""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._collision_events.append({
            'other': event.other_actor.type_id,
            'intensity': round(intensity, 2),
        })

    def _write_extra(self, tf: float, x: np.ndarray):
        """Append one JSON record (NPC positions + collisions) to the sidecar."""
        if self._extra_fh is None:
            return
        # Use the managed NPC list (sorted by actor ID for stable ordering)
        # instead of scanning world.get_actors(), which may include stale
        # actors or miss NPCs during batch transitions.
        npc_data = []
        for npc in sorted(getattr(self, 'npc_vehicles', []),
                          key=lambda a: a.id):
            try:
                if not npc.is_alive:
                    continue
                loc = npc.get_transform().location
                npc_data.append({'id': npc.id,
                                 'x': round(loc.x, 2),
                                 'y': round(loc.y, 2),
                                 'type': 'vehicle'})
            except Exception:
                pass
        for walker in sorted(getattr(self, 'npc_walkers', []),
                             key=lambda a: a.id):
            try:
                if not walker.is_alive:
                    continue
                loc = walker.get_transform().location
                npc_data.append({'id': walker.id,
                                 'x': round(loc.x, 2),
                                 'y': round(loc.y, 2),
                                 'type': 'walker'})
            except Exception:
                pass
        col_events = self._collision_events[:]
        self._collision_events.clear()
        record = {
            't':         round(tf, 3),
            'ego_x':     round(float(x[0, 0]), 2),
            'ego_y':     round(float(x[1, 0]), 2),
            'npcs':      npc_data,
            'collision': col_events[0] if col_events else None,
        }
        try:
            self._extra_fh.write(json.dumps(record) + '\n')
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    #  Exogenous input: dense waypoints                                   #
    # ------------------------------------------------------------------ #

    def get_exogenous_input(self, t):
        """Return waypoints (+ optional obstacle data) as w column vector.

        Layout:
          w[0 .. 2*N_w - 1]                     = waypoints  (wx1, wy1, …)
          w[2*N_w + 5*i + 0..4]  (if n_obs > 0) = obstacle i (x, y, vx, vy, radius)
        Sentinel for empty obstacle slot: x = 1e6, y = 1e6, vx = vy = radius = 0.

        Waypoints that fall inside the safety exclusion zone of any in-lane
        obstacle are replaced by the last safe waypoint, so the MPC has a
        collision-free reference trajectory even when an obstacle blocks the road.
        """
        transform = self.vehicle.get_transform()
        route_idx = self._closest_reference_route_index(transform.location)
        waypoints = []
        for offset in range(1, self.n_wp + 1):
            idx = min(route_idx + offset, len(self._reference_route) - 1)
            waypoints.append(self._reference_route[idx])

        # ---- Filter blocked waypoints --------------------------------- #
        # The generic obstacle-avoidance controller benefits from a truncated
        # path once the lane ahead is blocked. The follow-MPC variant needs
        # the unmodified centerline so it can still project the lead vehicle
        # onto the route and regulate following distance instead of treating
        # the lead as "off path".
        controller_type = (
            self.config.get("system_parameters", {}).get("controller_type", "")
        )
        should_truncate_waypoints = (
            self.n_obs > 0 and controller_type != "CarlaNPCFollowMPCController"
        )

        # Fetch in-lane obstacles (same filtering as in _get_nearby_obstacles).
        # Any waypoint whose distance to an obstacle centroid is less than
        # ego_radius + obs_radius + safe_margin is replaced by the last
        # safe waypoint, keeping the reference path out of blocked zones.
        if should_truncate_waypoints:
            mpc_opts   = self.config["system_parameters"]["mpc_options"]
            weights    = mpc_opts.get("cost_weights", {})
            ego_r      = weights.get("ego_radius",  2.5)
            safe_margin= weights.get("safe_margin",  1.5)
            obs_data   = self._get_nearby_obstacles()   # already lateral-filtered

            if obs_data:
                last_safe = None
                truncated = False
                for i, (wx, wy) in enumerate(waypoints):
                    if truncated:
                        # All waypoints after the first blocked one are
                        # replaced, so MPC sees no path beyond the obstacle.
                        if last_safe is not None:
                            waypoints[i] = last_safe
                        continue
                    blocked = False
                    for ox, oy, _ovx, _ovy, obs_r in obs_data:
                        r_safe = ego_r + obs_r + safe_margin
                        if math.sqrt((wx - ox)**2 + (wy - oy)**2) < r_safe:
                            blocked = True
                            break
                    if blocked:
                        truncated = True
                        if last_safe is not None:
                            waypoints[i] = last_safe
                    else:
                        last_safe = (wx, wy)

        dim = 2 * self.n_wp + 5 * self.n_obs
        w = np.zeros((dim, 1), dtype=float)

        # Pack waypoints
        for i, (wx, wy) in enumerate(waypoints):
            w[2 * i]     = wx
            w[2 * i + 1] = wy

        # Pack obstacles (if configured)
        if self.n_obs > 0:
            obs_data = self._get_nearby_obstacles()
            offset = 2 * self.n_wp
            for i in range(self.n_obs):
                if i < len(obs_data):
                    ox, oy, ovx, ovy, r = obs_data[i]
                    w[offset + 5 * i + 0] = ox
                    w[offset + 5 * i + 1] = oy
                    w[offset + 5 * i + 2] = ovx
                    w[offset + 5 * i + 3] = ovy
                    w[offset + 5 * i + 4] = r
                else:
                    # Sentinel: no obstacle in this slot
                    w[offset + 5 * i + 0] = 1e6
                    w[offset + 5 * i + 1] = 1e6

        return w

    # ------------------------------------------------------------------ #
    #  Obstacle detection                                                 #
    # ------------------------------------------------------------------ #

    def _get_nearby_obstacles(self):
        """Return up to N_obs nearby actors along the ego reference route.

        Vehicles do not need to be moving to count as obstacles. Actors are
        filtered by distance to the future route corridor instead of only the
        ego's instantaneous heading, which makes stopped lead vehicles much
        more stable to detect on curves and during yaw transients.

        Returns list of (x, y, vx, vy, bounding_radius) tuples sorted by
        route progress ahead of the ego.
        """
        ego_tf  = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_id  = self.vehicle.id
        route_idx = self._closest_reference_route_index(ego_loc)
        route_window_end = min(route_idx + max(self.n_wp * 2, 40),
                               len(self._reference_route) - 1)
        route_window = self._reference_route[route_idx:route_window_end + 1]

        if len(route_window) < 2:
            return []

        corridor_half_width = 3.5

        candidates = []
        for actor in self.world.get_actors():
            if actor.id == ego_id:
                continue
            if not (actor.type_id.startswith('vehicle.') or
                    actor.type_id.startswith('walker.')):
                continue

            loc  = actor.get_transform().location
            dx   = loc.x - ego_loc.x
            dy   = loc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.detection_radius:
                continue

            best_idx = None
            best_lat = float("inf")
            for i in range(len(route_window) - 1):
                ax, ay = route_window[i]
                bx, by = route_window[i + 1]
                abx = bx - ax
                aby = by - ay
                ab2 = abx * abx + aby * aby
                if ab2 < 1e-9:
                    continue
                apx = loc.x - ax
                apy = loc.y - ay
                tau = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
                proj_x = ax + tau * abx
                proj_y = ay + tau * aby
                lat = math.hypot(loc.x - proj_x, loc.y - proj_y)
                if lat < best_lat:
                    best_lat = lat
                    best_idx = i

            if best_idx is None or best_lat > corridor_half_width:
                continue

            route_progress = route_idx + best_idx
            if route_progress < route_idx:
                continue

            vel  = actor.get_velocity()
            ext  = actor.bounding_box.extent
            # Top-down bounding circle radius
            radius = math.sqrt(ext.x ** 2 + ext.y ** 2)

            candidates.append((route_progress, best_lat, dist, loc.x, loc.y, vel.x, vel.y, radius))

        candidates.sort(key=lambda c: (c[0], c[1], c[2]))

        return [(ox, oy, ovx, ovy, r)
                for (_, _, _, ox, oy, ovx, ovy, r) in candidates[:self.n_obs]]

    # ------------------------------------------------------------------ #
    #  Visualization                                                       #
    # ------------------------------------------------------------------ #

    def _draw_trajectory(self, x0, metadata, w):
        """Draw waypoints (blue), MPC predicted trajectory (red), and
        detected obstacles (orange) as 3D CARLA world debug primitives."""
        if self.world is None:
            return
        debug = self.world.debug
        ego_loc = self.vehicle.get_location()
        z = ego_loc.z + 0.5  # slightly above ground to avoid z-fighting
        life_time = max(0.1, self.time_step + 0.05)

        # 1. Waypoints — blue dots (first 2*n_wp entries of w)
        for i in range(self.n_wp):
            try:
                wx = float(w[2 * i])
                wy = float(w[2 * i + 1])
            except (IndexError, TypeError):
                break
            debug.draw_point(
                carla.Location(x=wx, y=wy, z=z),
                size=0.1,
                color=carla.Color(0, 128, 255),
                life_time=life_time,
            )

        # 2. MPC predicted trajectory — red dots from metadata
        if metadata and isinstance(metadata, dict):
            tx = metadata.get("traj_x")
            ty = metadata.get("traj_y")
            if tx is not None and ty is not None:
                for i in range(len(tx)):
                    debug.draw_point(
                        carla.Location(x=float(tx[i]), y=float(ty[i]), z=z),
                        size=0.12,
                        color=carla.Color(255, 0, 0),
                        life_time=life_time,
                    )

        # 3. Detected obstacles — orange center + radius ring
        if self.n_obs > 0:
            offset = 2 * self.n_wp
            for i in range(self.n_obs):
                try:
                    ox = float(w[offset + 5 * i + 0])
                    oy = float(w[offset + 5 * i + 1])
                except (IndexError, TypeError):
                    break
                if ox > 1e5:  # sentinel — empty slot
                    continue
                r = float(w[offset + 5 * i + 4])
                # Center dot
                debug.draw_point(
                    carla.Location(x=ox, y=oy, z=z),
                    size=0.2,
                    color=carla.Color(255, 165, 0),
                    life_time=life_time,
                )
                # Radius ring (12 points)
                for j in range(12):
                    theta = j * math.pi / 6
                    debug.draw_point(
                        carla.Location(x=ox + r * math.cos(theta),
                                       y=oy + r * math.sin(theta), z=z),
                        size=0.05,
                        color=carla.Color(255, 165, 0),
                        life_time=life_time,
                    )

    # ------------------------------------------------------------------ #
    #  State evolution                                                    #
    # ------------------------------------------------------------------ #

    def _update_npc_speed(self, t):
        """Command road NPCs to stop per config schedule."""
        npc_cfg = self.config.get("carla", {}).get("npcs", {})
        stop_after = npc_cfg.get("road_stop_after_s")
        if stop_after is None or t < stop_after:
            return
        if getattr(self, '_npcs_stop_commanded', False):
            return
        self._npcs_stop_commanded = True
        for npc in getattr(self, 'npc_vehicles', []):
            try:
                if npc.is_alive:
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        npc, 100)
            except Exception:
                pass
        print(f"[CarlaMPCDynamics] t={t:.2f}s: Commanded NPCs to stop "
              f"(road_stop_after_s={stop_after})")

    def evolve_state(self, t0, x0, u, w, tf, metadata=None):
        """Apply control u = (a, delta) to CARLA and return x = (px, py, psi, v)."""
        # Infer the time-step index from t0 and register the tick mapping.
        step_index = round(t0 / self.time_step)
        self._current_step = step_index
        self._register_step_start(step_index)

        # Update NPC behavior (e.g., scheduled stop)
        self._update_npc_speed(t0)

        accel  = float(u[0])  # longitudinal acceleration [m/s^2]
        delta  = float(u[1])  # steering angle [rad]  (-1..+1)

        # Convert MPC acceleration → CARLA throttle / brake.
        # Normalize by the MPC input limits so the full solver range maps
        # linearly onto CARLA's [0, 1] throttle/brake.
        mpc_limits = self.config["system_parameters"]["mpc_options"]["input_limits"]
        max_accel_limit = mpc_limits["max_accel"]
        min_accel_limit = abs(mpc_limits["min_accel"])

        if accel >= 0:
            throttle = min(accel / max_accel_limit, 1.0) if max_accel_limit > 0 else 0.0
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = min(-accel / min_accel_limit, 1.0) if min_accel_limit > 0 else 0.0

        # Map steering angle → CARLA steer in [-1, 1]
        # CARLA expects steer in [-1, 1]; max physical angle ≈ 0.7 rad
        steer = max(-1.0, min(1.0, delta))

        control = carla.VehicleControl(
            throttle=throttle, steer=steer, brake=brake)

        steps = max(1, math.floor((tf - t0) / self.time_step))

        for step in range(steps):
            self.vehicle.apply_control(control)

            # Render camera view (skip when headless)
            if self.display is not None and self.camera_manager is not None:
                if self.camera_manager.surface is not None:
                    self.display.blit(self.camera_manager.surface, (0, 0))
                self._draw_trajectory(x0, metadata, w.flatten())
                pygame.display.flip()
            else:
                self._draw_trajectory(x0, metadata, w.flatten())

            # Logging
            vel = self.vehicle.get_velocity()
            speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            log_msg = (
                f"[MPC] t={t0:.2f}→{tf:.2f} sub {step+1}/{steps} | "
                f"a={accel:.3f} δ={delta:.3f} → thr={throttle:.3f} brk={brake:.3f} str={steer:.3f} | "
                f"v={speed_ms:.1f} m/s\n"
            )
            sys.__stdout__.write(log_msg)
            sys.__stdout__.flush()

            # Log the tick BEFORE world.tick() so replay reproduces exactly.
            self._log_tick(control)
            # Apply direct walker control each tick (fixed-route walkers).
            for wkr, wkr_ctrl in getattr(self, '_direct_walker_controls', []):
                try:
                    if wkr.is_alive:
                        wkr.apply_control(wkr_ctrl)
                except Exception:
                    pass
            self.world.tick()

        # Register the tick position after this evolve_state completes.
        # If tf lands on a full step boundary, this records the start tick
        # for the NEXT time step — needed when a subsequent batch rolls
        # back to that step.
        next_step = round(tf / self.time_step)
        self._register_step_start(next_step)

        # Read state
        transform = self.vehicle.get_transform()
        velocity  = self.vehicle.get_velocity()
        yaw_rad   = math.radians(transform.rotation.yaw)

        # Project global velocity into body frame
        v_x_local = velocity.x * math.cos(yaw_rad) + velocity.y * math.sin(yaw_rad)
        v_y_local = -velocity.x * math.sin(yaw_rad) + velocity.y * math.cos(yaw_rad)

        # Angular velocity: CARLA returns deg/s, convert to rad/s
        ang_vel = self.vehicle.get_angular_velocity()
        r_rad_s = math.radians(ang_vel.z)

        if self.n == 4:
            x = np.array([
                [transform.location.x],
                [transform.location.y],
                [yaw_rad],
                [v_x_local],
            ], dtype=float)
        else:
            x = np.array([
                [transform.location.x],
                [transform.location.y],
                [yaw_rad],
                [v_x_local],
                [v_y_local],
                [r_rad_s],
            ], dtype=float)

        self._write_extra(tf, x)
        return tf, x

    # ------------------------------------------------------------------ #
    #  Teardown                                                           #
    # ------------------------------------------------------------------ #

    def teardown(self):
        print("[CarlaMPCDynamics] Tearing down …")
        # Stop CARLA native recorder (before destroying actors so the final
        # frames are properly captured).
        if getattr(self, '_recorder_started', False):
            try:
                self.client.stop_recorder()
                print(f"[CarlaMPCDynamics] Stopped CARLA recorder → "
                      f"{getattr(self, '_recording_file', '?')}")
            except Exception as e:
                print(f"[CarlaMPCDynamics] WARNING: stop_recorder: {e}")
            self._recorder_started = False
        # Close sidecar file
        if getattr(self, '_extra_fh', None) is not None:
            try:
                self._extra_fh.close()
            except Exception:
                pass
            self._extra_fh = None
        # Stop and destroy collision sensor first (stops the sensor stream)
        if getattr(self, '_collision_sensor', None) is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
                print("[CarlaMPCDynamics] Collision sensor stopped and destroyed.")
            except Exception as e:
                print(f"[CarlaMPCDynamics] Warning: collision sensor cleanup: {e}")
            self._collision_sensor = None
        try:
            for ctrl in getattr(self, 'npc_walker_controllers', []):
                try:
                    if ctrl is not None:
                        ctrl.stop()
                except Exception: pass

            if hasattr(self, 'camera_manager') and self.camera_manager is not None:
                self.camera_manager.destroy()

            destroy_ids = []
            for ctrl in getattr(self, 'npc_walker_controllers', []):
                if ctrl is not None:
                    destroy_ids.append(ctrl.id)
            for walker in getattr(self, 'npc_walkers', []):
                destroy_ids.append(walker.id)
            for npc in getattr(self, 'npc_vehicles', []):
                destroy_ids.append(npc.id)
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                destroy_ids.append(self.vehicle.id)

            if destroy_ids and hasattr(self, 'client'):
                self.client.apply_batch_sync(
                    [carla.command.DestroyActor(aid) for aid in destroy_ids], True
                )
                print(f"[CarlaMPCDynamics] Destroyed {len(destroy_ids)} actors.")

            if hasattr(self, 'world') and self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            if hasattr(self, 'traffic_manager'):
                self.traffic_manager.set_synchronous_mode(False)

            if _has_display():
                pygame.quit()

            # Give CARLA time to flush sensor streams and settle before the
            # next session connects (avoids "Invalid session: no stream" spam)
            _time.sleep(2.0)
            print("[CarlaMPCDynamics] Teardown complete.")
        except Exception as e:
            print(f"[CarlaMPCDynamics] cleanup error: {e}")

    # ------------------------------------------------------------------ #
    #  NPC helpers (identical to CarlaDynamics)                           #
    # ------------------------------------------------------------------ #

    def _spawn_npc_vehicles(self, bp_lib, spawn_points, ego_idx, count):
        if count == 0:
            self._road_npc_count = 0
            return []
        vehicle_bps = sorted(bp_lib.filter('vehicle.*'), key=lambda bp: bp.id)

        npc_cfg = self.config.get("carla", {}).get("npcs", {})
        road_count = min(npc_cfg.get("road_vehicles", 0), count)
        road_spacing = npc_cfg.get("road_spacing", 25.0)
        road_speed_pct = npc_cfg.get("road_speed_pct", 60)

        vehicles = []
        tm_port = self.traffic_manager.get_port()
        spawned_road = 0

        # Use the known spawn-point location instead of querying the vehicle
        # actor.  In CARLA synchronous mode, get_location() returns the
        # default position (near origin) until the next world.tick().
        ego_spawn_loc = spawn_points[ego_idx].location

        # ---- Phase 1: Spawn NPCs on the ego's road ahead ------------ #
        if road_count > 0:
            ego_wp = self._carla_map.get_waypoint(
                ego_spawn_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            print(f"[CarlaMPCDynamics] Ego spawn loc: "
                  f"({ego_spawn_loc.x:.1f}, {ego_spawn_loc.y:.1f}), "
                  f"ego waypoint: ({ego_wp.transform.location.x:.1f}, "
                  f"{ego_wp.transform.location.y:.1f}), "
                  f"road_id={ego_wp.road_id}, lane_id={ego_wp.lane_id}")
            ego_road_id = ego_wp.road_id
            ego_lane_id = ego_wp.lane_id
            current_wp = ego_wp
            for i in range(road_count):
                nxt = current_wp.next(road_spacing)
                if not nxt:
                    print(f"[CarlaMPCDynamics] WARNING: next({road_spacing}) "
                          f"returned empty at step {i}")
                    break
                # At junctions, prefer the branch that stays on the same
                # road or at least the same lane direction.
                best = nxt[0]
                if len(nxt) > 1:
                    for w in nxt:
                        if w.road_id == ego_road_id and w.lane_id == ego_lane_id:
                            best = w
                            break
                    # Fallback: pick the closest one geometrically
                    else:
                        best = min(nxt, key=lambda w: (
                            (w.transform.location.x - current_wp.transform.location.x) ** 2 +
                            (w.transform.location.y - current_wp.transform.location.y) ** 2))
                current_wp = best
                npc_loc = current_wp.transform.location
                dx = npc_loc.x - ego_spawn_loc.x
                dy = npc_loc.y - ego_spawn_loc.y
                geom_dist = math.sqrt(dx * dx + dy * dy)
                print(f"[CarlaMPCDynamics] NPC road wp {i}: "
                      f"({npc_loc.x:.1f}, {npc_loc.y:.1f}), "
                      f"road_id={current_wp.road_id}, "
                      f"lane_id={current_wp.lane_id}, "
                      f"geom_dist={geom_dist:.1f}m")

                bp = vehicle_bps[i % len(vehicle_bps)]
                if bp.has_attribute('color'):
                    colors = bp.get_attribute('color').recommended_values
                    bp.set_attribute('color', colors[i % len(colors)])
                spawn_tf = current_wp.transform
                spawn_tf.location.z += 0.3
                npc = self.world.try_spawn_actor(bp, spawn_tf)
                if npc is not None:
                    npc.set_autopilot(True, tm_port)
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        npc, road_speed_pct)
                    self.traffic_manager.auto_lane_change(npc, False)
                    self.traffic_manager.distance_to_leading_vehicle(npc, 5.0)
                    vehicles.append(npc)
                    spawned_road += 1
                else:
                    print(f"[CarlaMPCDynamics] WARNING: try_spawn_actor "
                          f"failed for road NPC {i}")
            if spawned_road > 0:
                print(f"[CarlaMPCDynamics] Road NPCs: {spawned_road}/{road_count} "
                      f"(spacing={road_spacing}m, speed_pct={road_speed_pct})")

        # ---- Phase 2: Spawn remaining NPCs at nearest spawn points --- #
        remaining = count - len(vehicles)
        if remaining > 0:
            ego_loc = ego_spawn_loc
            available = [sp for i, sp in enumerate(spawn_points) if i != ego_idx]
            available.sort(
                key=lambda sp: (sp.location.x - ego_loc.x) ** 2
                             + (sp.location.y - ego_loc.y) ** 2)
            for i in range(min(remaining, len(available))):
                bp_idx = (len(vehicles) + i) % len(vehicle_bps)
                bp = vehicle_bps[bp_idx]
                if bp.has_attribute('color'):
                    colors = bp.get_attribute('color').recommended_values
                    bp.set_attribute('color',
                                     colors[(len(vehicles) + i) % len(colors)])
                npc = self.world.try_spawn_actor(bp, available[i])
                if npc is not None:
                    npc.set_autopilot(True, tm_port)
                    vehicles.append(npc)

        self._road_npc_count = spawned_road
        print(f"[CarlaMPCDynamics] Spawned {len(vehicles)}/{count} NPC vehicles "
              f"({spawned_road} on road).")
        return vehicles

    def _configure_road_npcs_tm(self):
        """Re-apply TM settings for road NPCs after autopilot re-enable."""
        if not hasattr(self, '_road_npc_count') or self._road_npc_count == 0:
            return
        npc_cfg = self.config.get("carla", {}).get("npcs", {})
        road_speed_pct = npc_cfg.get("road_speed_pct", 60)
        for i in range(min(self._road_npc_count, len(self.npc_vehicles))):
            npc = self.npc_vehicles[i]
            try:
                if npc.is_alive:
                    self.traffic_manager.vehicle_percentage_speed_difference(
                        npc, road_speed_pct)
                    self.traffic_manager.auto_lane_change(npc, False)
                    self.traffic_manager.distance_to_leading_vehicle(npc, 5.0)
            except Exception:
                pass

    def _spawn_npc_walkers(self, bp_lib, count):
        npc_cfg = self.config.get("carla", {}).get("npcs", {})
        walker_routes = npc_cfg.get("walker_routes", [])

        if count == 0 and not walker_routes:
            return [], []

        walker_bps = sorted(bp_lib.filter('walker.pedestrian.*'), key=lambda bp: bp.id)
        controller_bp = bp_lib.find('controller.ai.walker')

        walkers, controllers = [], []
        self._direct_walker_controls = []   # (walker, WalkerControl) for fixed-route

        # --- Fixed-route walkers: direct control (no AI) --- #
        for i, route in enumerate(walker_routes):
            bp = walker_bps[i % len(walker_bps)]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            sx, sy, sz = route["spawn"]
            spawn_tf = carla.Transform(carla.Location(x=sx, y=sy, z=sz))
            walker = self.world.try_spawn_actor(bp, spawn_tf)
            if walker is None:
                # Fallback: spawn at a random navmesh location, then teleport.
                print(f"[CarlaMPCDynamics] Direct spawn failed at ({sx},{sy},{sz}), "
                      f"trying spawn-then-teleport...")
                fallback_loc = self.world.get_random_location_from_navigation()
                if fallback_loc is not None:
                    walker = self.world.try_spawn_actor(
                        bp, carla.Transform(fallback_loc))
                if walker is not None:
                    walker.set_transform(spawn_tf)
                    self.world.tick()
                else:
                    print(f"[CarlaMPCDynamics] WARNING: Could not spawn "
                          f"walker {i} (all attempts failed)")
                    continue
            # Compute walk direction from spawn to destination
            dx, dy, dz = route["dest"]
            dir_x, dir_y, dir_z = dx - sx, dy - sy, dz - sz
            length = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            if length > 1e-6:
                dir_x /= length; dir_y /= length; dir_z /= length
            speed = route.get("speed", 1.4)
            ctrl_cmd = carla.WalkerControl(
                direction=carla.Vector3D(x=dir_x, y=dir_y, z=dir_z),
                speed=speed, jump=False)
            self._direct_walker_controls.append((walker, ctrl_cmd))
            walkers.append(walker)
            controllers.append(None)  # placeholder — no AI controller
            loc = walker.get_transform().location
            print(f"[CarlaMPCDynamics] Fixed-route walker {i} spawned at"
                  f" ({loc.x:.1f},{loc.y:.1f},{loc.z:.1f}), target dir"
                  f" ({dir_x:.2f},{dir_y:.2f}), speed={speed}")

        # --- Random walkers (original behaviour) --- #
        remaining = count - len(walker_routes)
        if remaining > 0:
            spawn_locations = []
            for _ in range(remaining * 3):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_locations.append(loc)
                if len(spawn_locations) >= remaining:
                    break
            offset = len(walkers)
            for j, loc in enumerate(spawn_locations[:remaining]):
                bp = walker_bps[(offset + j) % len(walker_bps)]
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'false')
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
                if walker is None:
                    continue
                ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                walkers.append(walker)
                controllers.append(ctrl)

        self.world.tick()

        # --- Start AI controllers for random walkers only --- #
        route_count = len(walker_routes)
        for i, ctrl in enumerate(controllers):
            if ctrl is None:
                continue  # fixed-route walker — uses direct control
            ctrl.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4)

        # Apply first direct control tick
        for walker, ctrl_cmd in self._direct_walker_controls:
            try:
                walker.apply_control(ctrl_cmd)
            except Exception:
                pass

        print(f"[CarlaMPCDynamics] Spawned {len(walkers)}/{count + len(walker_routes)} NPC walkers"
              f" ({len(walker_routes)} fixed-route direct-control).")
        return walkers, controllers
