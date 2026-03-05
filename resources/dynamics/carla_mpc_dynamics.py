"""
CarlaMPCDynamics — CARLA dynamics bridge for the MPC controller.

Differences from the PID-oriented CarlaDynamics:
  * State  x = (px, py, psi, v)  ∈ ℝ⁴    (no waypoints in state)
  * Input  u = (a, delta)        ∈ ℝ²    (acceleration + steering)
  * Exogenous input w  = dense waypoint array  ∈ ℝ^{2·N_w}
  * Converts acceleration → CARLA throttle/brake internally.
"""

import math
import numpy as np
from sharc.dynamics_base import Dynamics
import carla
import random
import pygame
import sys


def pygame_init(w=1280, h=720):
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

        random.seed(self.seed)
        np.random.seed(self.seed)

        # ---- Connect to CARLA ---------------------------------------- #
        print("[CarlaMPCDynamics] Creating CARLA session …")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
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
        import time; time.sleep(0.5)  # let CARLA settle

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.time_step
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager(8000)
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
        for offset in range(len(spawn_points)):
            idx = (ego_spawn_idx + offset) % len(spawn_points)
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_points[idx])
            if self.vehicle is not None:
                print(f"[CarlaMPCDynamics] Spawned ego at spawn-point index {idx}")
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")

        # Dimensions (already set by Dynamics.__init__)
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

        # ---- NPC spawning -------------------------------------------- #
        npc_cfg = carla_cfg.get("npcs", {})
        n_vehicles = npc_cfg.get("n_vehicles", 0)
        n_walkers  = npc_cfg.get("n_walkers", 0)
        self.npc_vehicles = self._spawn_npc_vehicles(bp_lib, spawn_points, ego_spawn_idx, n_vehicles)
        self.npc_walkers, self.npc_walker_controllers = self._spawn_npc_walkers(bp_lib, n_walkers)

        # ---- Pygame + camera ----------------------------------------- #
        self.display = pygame_init()
        self.camera_manager = CameraManager(self.world, self.vehicle)

        # Cache the CARLA map for waypoint queries
        self._carla_map = self.world.get_map()

    # ------------------------------------------------------------------ #
    #  Exogenous input: dense waypoints                                   #
    # ------------------------------------------------------------------ #

    def get_exogenous_input(self, t):
        """Return N_w dense waypoints as w ∈ ℝ^{2·N_w} column vector."""
        transform = self.vehicle.get_transform()
        wp = self._carla_map.get_waypoint(transform.location)
        waypoints = []
        for _ in range(self.n_wp):
            nxt = wp.next(self.wp_spacing)
            if not nxt:
                # Road ends — repeat last known waypoint
                waypoints.append((wp.transform.location.x, wp.transform.location.y))
                continue
            wp = nxt[0]
            waypoints.append((wp.transform.location.x, wp.transform.location.y))

        # Pack [wx1, wy1, wx2, wy2, …]
        w = np.zeros((2 * self.n_wp, 1), dtype=float)
        for i, (wx, wy) in enumerate(waypoints):
            w[2 * i]     = wx
            w[2 * i + 1] = wy
        return w

    # ------------------------------------------------------------------ #
    #  State evolution                                                    #
    # ------------------------------------------------------------------ #

    def _draw_trajectory(self, x, metadata, waypoints):
        """Draw waypoints (blue) and MPC trajectory (red) in the 3D CARLA world."""
        if self.world is None: return
        
        debug = self.world.debug
        ego_loc = self.vehicle.get_location()
        z = ego_loc.z + 0.5  # Slightly above ground to avoid z-fighting
        
        life_time = max(0.1, self.time_step + 0.05) # ensure it lasts until next frame
        
        # 1. Draw waypoints (blue dots)
        if waypoints is not None:
            # waypoints is [wx1, wy1, wx2, wy2, ...]
            for i in range(len(waypoints) // 2):
                wx = float(waypoints[2*i])
                wy = float(waypoints[2*i+1])
                debug.draw_point(carla.Location(x=wx, y=wy, z=z), size=0.1, color=carla.Color(0, 1, 255), life_time=life_time)
        
        # 2. Draw MPC planned trajectory (red dots)
        if metadata and "traj_x" in metadata and "traj_y" in metadata:
            tx = metadata["traj_x"]
            ty = metadata["traj_y"]
            for i in range(len(tx)):
                px = float(tx[i])
                py = float(ty[i])
                debug.draw_point(carla.Location(x=px, y=py, z=z), size=0.1, color=carla.Color(255, 0, 0), life_time=life_time)

    def evolve_state(self, t0, x0, u, w, tf, metadata=None):
        """Apply control u = (a, delta) to CARLA and return x = (px, py, psi, v)."""
        accel  = float(u[0])  # longitudinal acceleration [m/s^2]
        delta  = float(u[1])  # steering angle [rad]  (-1..+1)

        # Convert acceleration → throttle / brake
        if accel >= 0:
            throttle = min(accel, 1.0)   # normalise by max_accel
            brake    = 0.0
        else:
            throttle = 0.0
            brake    = min(-accel, 1.0)  # normalise by |min_accel|

        # Map steering angle → CARLA steer in [-1, 1]
        steer = max(-1.0, min(1.0, delta))

        steps = max(1, math.floor((tf - t0) / self.time_step))

        for step in range(steps):
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            ))

            # Render camera view
            if self.camera_manager.surface is not None:
                self.display.blit(self.camera_manager.surface, (0, 0))
            
            # Draw waypoints and trajectory overlay
            self._draw_trajectory(x0, metadata, w)
            
            pygame.display.flip()

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

            self.world.tick()

        # Read state
        transform = self.vehicle.get_transform()
        velocity  = self.vehicle.get_velocity()
        yaw_rad   = math.radians(transform.rotation.yaw)
        
        # Transform global velocity to local (body) frame
        v_x_global = velocity.x
        v_y_global = velocity.y
        v_x_local = v_x_global * math.cos(yaw_rad) + v_y_global * math.sin(yaw_rad)
        v_y_local = -v_x_global * math.sin(yaw_rad) + v_y_global * math.cos(yaw_rad)
        
        # Read angular velocity (CARLA returns deg/s)
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

        return tf, x

    # ------------------------------------------------------------------ #
    #  Teardown                                                           #
    # ------------------------------------------------------------------ #

    def teardown(self):
        print("[CarlaMPCDynamics] Tearing down …")
        try:
            for ctrl in getattr(self, 'npc_walker_controllers', []):
                try: ctrl.stop()
                except Exception: pass

            if hasattr(self, 'world') and self.world is not None:
                self.world.tick()

            if hasattr(self, 'camera_manager') and self.camera_manager is not None:
                self.camera_manager.destroy()

            destroy_ids = []
            for ctrl in getattr(self, 'npc_walker_controllers', []):
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

            pygame.quit()
        except Exception as e:
            print(f"[CarlaMPCDynamics] cleanup error: {e}")

    # ------------------------------------------------------------------ #
    #  NPC helpers (identical to CarlaDynamics)                           #
    # ------------------------------------------------------------------ #

    def _spawn_npc_vehicles(self, bp_lib, spawn_points, ego_idx, count):
        if count == 0:
            return []
        vehicle_bps = sorted(bp_lib.filter('vehicle.*'), key=lambda bp: bp.id)
        available = [sp for i, sp in enumerate(spawn_points) if i != ego_idx]
        count = min(count, len(available))
        vehicles = []
        for i in range(count):
            bp = vehicle_bps[i % len(vehicle_bps)]
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                bp.set_attribute('color', colors[i % len(colors)])
            npc = self.world.try_spawn_actor(bp, available[i])
            if npc is not None:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                vehicles.append(npc)
        print(f"[CarlaMPCDynamics] Spawned {len(vehicles)}/{count} NPC vehicles.")
        return vehicles

    def _spawn_npc_walkers(self, bp_lib, count):
        if count == 0:
            return [], []
        walker_bps = sorted(bp_lib.filter('walker.pedestrian.*'), key=lambda bp: bp.id)
        controller_bp = bp_lib.find('controller.ai.walker')
        spawn_locations = []
        for _ in range(count * 3):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_locations.append(loc)
            if len(spawn_locations) >= count:
                break

        walkers, controllers = [], []
        for i, loc in enumerate(spawn_locations[:count]):
            bp = walker_bps[i % len(walker_bps)]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue
            ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            walkers.append(walker)
            controllers.append(ctrl)

        self.world.tick()
        for ctrl in controllers:
            ctrl.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4)

        print(f"[CarlaMPCDynamics] Spawned {len(walkers)}/{count} NPC walkers.")
        return walkers, controllers
