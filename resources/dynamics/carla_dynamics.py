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
    pygame.display.set_caption("CARLA Camera View")
    return display


def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6


class CameraManager:
    def __init__(self, world, vehicle, width=1280, height=720):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        self.surface = None

        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', '90')

        # Attach camera behind the car with slight height
        spawn_point = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15)
        )

        self.camera = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        self.camera.listen(lambda data: self._on_image(data))

    def _on_image(self, image):
        # Convert raw BGRA -> RGB array for pygame
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 4))
        img = img[:, :, :3][:, :, ::-1]   # BGR -> RGB

        self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def destroy(self):
        """Stop listening and destroy the camera sensor."""
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None


class CarlaDynamics(Dynamics):

    def __init__(self, config):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def setup_system(self):
        self.time_step = float(self.config["system_parameters"]["sample_time"])

        # ---- Deterministic seeding ----------------------------------- #
        carla_cfg = self.config.get("carla", {})
        self.seed = carla_cfg.get("seed", 0)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # ---- Connect to CARLA ---------------------------------------- #
        print("[CarlaDynamics] Creating new CARLA session...")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # ---- Destroy any leftover actors from a previous session ---------- #
        # This handles crashes/SIGKILL where teardown() never ran.
        existing = self.world.get_actors()
        stale_ids = [
            a.id for a in existing
            if a.type_id.startswith(('vehicle.', 'sensor.', 'walker.', 'controller.'))
        ]
        if stale_ids:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(aid) for aid in stale_ids], True
            )
            print(f"[CarlaDynamics] Removed {len(stale_ids)} leftover actor(s) from previous session.")
            self.world.tick()

        # Synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.time_step
        self.world.apply_settings(settings)

        # Traffic manager — deterministic seed
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed)

        self.world.tick()

        s2 = self.world.get_settings()
        print(f"sync: {s2.synchronous_mode}  dt: {s2.fixed_delta_seconds}  seed: {self.seed}")

        # ---- Spawn ego vehicle at deterministic spawn point ---------- #
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        spawn_points = self.world.get_map().get_spawn_points()
        ego_spawn_idx = self.seed % len(spawn_points)
        spawn = spawn_points[ego_spawn_idx]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        print(f"Spawned ego: {self.vehicle} at spawn index {ego_spawn_idx}")

        # State dimensions (also set by base class, kept for clarity)
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

        # ---- NPC spawning -------------------------------------------- #
        npc_cfg = carla_cfg.get("npcs", {})
        n_vehicles = npc_cfg.get("n_vehicles", 0)
        n_walkers = npc_cfg.get("n_walkers", 0)

        self.npc_vehicles = self._spawn_npc_vehicles(
            bp_lib, spawn_points, ego_spawn_idx, n_vehicles
        )
        self.npc_walkers, self.npc_walker_controllers = self._spawn_npc_walkers(
            bp_lib, n_walkers
        )

        # ---- Pygame display & camera --------------------------------- #
        self.display = pygame_init()
        self.camera_manager = CameraManager(self.world, self.vehicle)

    def teardown(self):
        """Destroy all CARLA actors (NPCs, ego, camera) and quit pygame.

        Uses ``client.apply_batch_sync`` for bulk actor destruction, which
        is the pattern recommended by CARLA's own traffic-generation scripts
        and avoids C++-level crashes from operating on stale actor handles.
        """
        print("[CarlaDynamics] Tearing down CARLA session...")
        try:
            # 1. Stop walker AI controllers (must happen before any destroy)
            for ctrl in getattr(self, 'npc_walker_controllers', []):
                try:
                    ctrl.stop()
                except Exception:
                    pass

            # Let the server process the stop commands
            if hasattr(self, 'world') and self.world is not None:
                self.world.tick()

            # 2. Destroy camera sensor (stop listener first)
            if hasattr(self, 'camera_manager') and self.camera_manager is not None:
                self.camera_manager.destroy()

            # 3. Batch-destroy all spawned actors in one server call
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
                    [carla.command.DestroyActor(aid) for aid in destroy_ids],
                    True,
                )
                print(f"[CarlaDynamics] Destroyed {len(destroy_ids)} actors.")

            # 4. Restore simulator to asynchronous mode
            if hasattr(self, 'world') and self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            if hasattr(self, 'traffic_manager'):
                self.traffic_manager.set_synchronous_mode(False)

            pygame.quit()
        except Exception as e:
            print(f"[CarlaDynamics] cleanup error: {e}")

    # ------------------------------------------------------------------ #
    #  NPC helpers                                                        #
    # ------------------------------------------------------------------ #

    def _spawn_npc_vehicles(self, bp_lib, spawn_points, ego_idx, count):
        """Spawn *count* NPC vehicles at deterministic spawn points with autopilot.

        Spawn-point ordering and blueprint selection are fully determined by
        the seed set in ``setup_system``.
        """
        if count == 0:
            return []

        # Sorted blueprints for deterministic ordering across runs
        vehicle_bps = sorted(bp_lib.filter('vehicle.*'), key=lambda bp: bp.id)
        available = [sp for i, sp in enumerate(spawn_points) if i != ego_idx]

        if count > len(available):
            print(f"[CarlaDynamics] Warning: requested {count} NPC vehicles "
                  f"but only {len(available)} spawn points available.")
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

        print(f"[CarlaDynamics] Spawned {len(vehicles)}/{count} NPC vehicles.")
        return vehicles

    def _spawn_npc_walkers(self, bp_lib, count):
        """Spawn *count* NPC walkers with AI controllers.

        Each walker is placed at a navigable location and given a random
        destination.  The traffic-manager seed set in ``setup_system``
        governs autopilot randomness; Python ``random`` (also seeded)
        governs selection of walker blueprints.
        """
        if count == 0:
            return [], []

        walker_bps = sorted(
            bp_lib.filter('walker.pedestrian.*'), key=lambda bp: bp.id
        )
        controller_bp = bp_lib.find('controller.ai.walker')

        # Collect navigable spawn locations (over-sample for robustness)
        spawn_locations = []
        for _ in range(count * 3):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_locations.append(loc)
            if len(spawn_locations) >= count:
                break

        walkers = []
        controllers = []
        for i, loc in enumerate(spawn_locations[:count]):
            bp = walker_bps[i % len(walker_bps)]
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue

            ctrl = self.world.spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
            walkers.append(walker)
            controllers.append(ctrl)

        # Tick so all actors are registered before starting controllers
        self.world.tick()

        for ctrl in controllers:
            ctrl.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.4)  # ~5 km/h normal walking speed

        print(f"[CarlaDynamics] Spawned {len(walkers)}/{count} NPC walkers.")
        return walkers, controllers



    def evolve_state(self, t0: float, x0: np.ndarray, u: np.ndarray, w: np.ndarray, tf: float):
        # make required global in evolve_state
        # apply control and tick world
        # 1. initialize carla from x0, 2. run the step function from tf to t0 of steps 3. return x some representation of carla states 
        # state of the car: for simulation: x,y,velocity might need other states

        # decompose input u
        throttle = float(u[0])
        steer = float(u[2])
        brake = float(u[1])

        # calculate the amount of steps to evolve
        steps = math.floor((tf-t0)/self.time_step)

        #print("DEBUG apply_control target:", type(self.vehicle), self.vehicle, flush=True)
        #print("DEBUG self dict keys:", [k for k in self.__dict__.keys() if "veh" in k.lower()], flush=True)
        #assert self.vehicle is not None, "self.vehicle is None at apply_control"

        for step in range(steps):
            # apply control

            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            ))

            # --- NEW: Render camera frame ----------------------------------------
            if self.camera_manager.surface is not None:
                self.display.blit(self.camera_manager.surface, (0, 0))
            pygame.display.flip()

            # Logging Information (Bypassing stdout redirection to log file)
            log_msg = (
                "-" * 40 + "\n" +
                f"Time Step: t={t0:.3f}s -> tf={tf:.3f}s (substep {step+1}/{steps})\n" +
                f"State Vector (x, y, yaw, speed, wp_x, wp_y):\n" +
                f"  {x0.flatten()}\n" +
                f"Applied Control:\n" +
                f"  Throttle: {throttle:.4f}\n" +
                f"  Steer:    {steer:.4f}\n" +
                f"  Brake:    {brake:.4f}\n" +
                f"Current Speed: {get_speed(self.vehicle):.2f} km/h\n" +
                "-" * 40 + "\n"
            )
            sys.__stdout__.write(log_msg)
            sys.__stdout__.flush()

            # tick the world
            self.world.tick()

        # read state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        yaw_deg = transform.rotation.yaw
        yaw = math.radians(yaw_deg)

        # x = np.array([
        #     [transform.location.x],
        #     [transform.location.y],
        #     [yaw],
        #     [math.sqrt(velocity.x**2 + velocity.y**2)],
        # ], dtype=float)

        # Get next waypoint from CARLA map
        map = self.world.get_map()
        waypoint = map.get_waypoint(transform.location)
        next_wp = waypoint.next(2.0)[0]  # 2 meters ahead
        wp_x = next_wp.transform.location.x
        wp_y = next_wp.transform.location.y

        # w = np.array([
        #     [wp_x],
        #     [wp_y],
        # ], dtype=float)

        x = np.array([
            [transform.location.x],
            [transform.location.y],
            [yaw],
            [get_speed(self.vehicle)],
            [wp_x],
            [wp_y],
        ], dtype=float)

        return tf,x






        


