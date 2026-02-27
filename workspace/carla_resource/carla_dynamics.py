import math 
import numpy as np
from sharc.dynamics_base import Dynamics
import carla
import random
import pygame

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
        # Convert raw BGRA → RGB array for pygame
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((self.height, self.width, 4))
        img = img[:, :, :3][:, :, ::-1]   # BGR → RGB

        self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

class CarlaDynamics(Dynamics):

    def __init__(self, config):
        super().__init__(config)

    def setup_system(self):
        # initialize carla once
        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        # set tick to sample time
        self.time_step = float(self.config["system_parameters"]["sample_time"])
        #print("requested dt:", self.time_step)

        
        #print("Setting Updates")
        settings = self.world.get_settings()
        #print("sync:", settings.synchronous_mode, "dt:", settings.fixed_delta_seconds)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.time_step
        self.world.apply_settings(settings)
        self.world.tick()

        # Optional: verify it actually took effect
        s2 = self.world.get_settings()
        print("sync:", s2.synchronous_mode, "dt:", s2.fixed_delta_seconds)

        # Spawn vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")

        spawn_points = self.world.get_map().get_spawn_points()
        spawn = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn)

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")

        print("Spawned:", self.vehicle)

        # State Parameter
        self.n = self.config["system_parameters"]["state_dimension"]
        self.m = self.config["system_parameters"]["input_dimension"]
        self.p = self.config["system_parameters"]["output_dimension"]

        # camera
        self.display = pygame_init()
        self.camera_manager = CameraManager(self.world, self.vehicle)



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

            print("Step: ", step)
            print("Speed: %.2f km/h" % get_speed(self.vehicle))
            print("throttle", throttle)
            print("steer", steer)
            print("brake", brake)

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






        


