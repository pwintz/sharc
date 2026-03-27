"""
Warmup CARLA after start — spawns a vehicle with camera, ticks a few times,
then cleans up. This ensures GPU shaders/caches are loaded so the first real
experiment run is deterministic.

Usage: python3 warmup_carla.py          # blocks until warmup is done
       python3 warmup_carla.py --wait   # also waits for CARLA to be ready
"""
import carla
import sys
import time
import os

PORT = 2010

def wait_for_carla(timeout=120):
    """Block until CARLA accepts connections."""
    for i in range(timeout):
        try:
            port = int(os.getenv('_EXP_PORT', PORT))
            c = carla.Client('localhost', port)
            c.set_timeout(5.0)
            w = c.get_world()
            return c, w
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"CARLA not ready after {timeout}s")


def warmup(client, world):
    """Spawn vehicle + camera, tick, destroy."""
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()

    # Enable sync
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    time.sleep(0.5)

    # Spawn vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    world.tick()

    # Attach camera to force GPU shader compilation
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1280')
    cam_bp.set_attribute('image_size_y', '720')
    cam_bp.set_attribute('fov', '110')
    cam_tf = carla.Transform(carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-20))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    camera.listen(lambda img: None)

    # Drive a few steps
    for _ in range(20):
        vehicle.apply_control(carla.VehicleControl(throttle=0.5))
        world.tick()

    # Cleanup
    camera.stop()
    camera.destroy()
    vehicle.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    time.sleep(0.5)


if __name__ == '__main__':
    do_wait = '--wait' in sys.argv
    if do_wait:
        print("Waiting for CARLA to be ready...")
        client, world = wait_for_carla()
    else:
        port = int(os.getenv('_EXP_PORT', PORT))
        client = carla.Client('localhost', port)
        client.set_timeout(30.0)
        world = client.get_world()

    print(f"Connected: {world.get_map().name}")
    print("Running GPU warmup...")
    warmup(client, world)
    print("Warmup complete. CARLA is ready for deterministic experiments.")
