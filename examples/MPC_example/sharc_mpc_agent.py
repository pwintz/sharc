#!/usr/bin/env python3
"""
SHARC MPC Agent for CARLA ScenarioRunner.

This agent uses our compiled C++ MPC controller (CarlaConstraintMPCController)
to control the ego vehicle within CARLA scenario_runner scenarios.

Usage with scenario_runner:
  python scenario_runner.py --scenario FollowLeadingVehicle_1 \
      --agent /path/to/sharc_mpc_agent.py \
      --agentConfig /path/to/sharc_agent_config.json \
      --port 2010 --sync --reloadWorld

  python scenario_runner.py --route srunner/data/routes_devtest.xml \
      --route-id 0 \
      --agent /path/to/sharc_mpc_agent.py \
      --agentConfig /path/to/sharc_agent_config.json \
      --port 2010 --sync
"""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import carla
import numpy as np

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# ---------------------------------------------------------------------------
# Default MPC configuration (can be overridden via --agentConfig JSON file)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "prediction_horizon": 10,
    "control_horizon": 10,
    "n_waypoints": 40,
    "waypoint_spacing": 0.5,
    "n_obstacles": 5,
    "detection_radius": 50.0,
    "target_speed": 15,
    "sample_time": 0.05,
    "controller_type": "CarlaConstraintMPCController",
    "cost_weights": {
        "q_path": 2.0,
        "q_speed": 1.0,
        "r_accel": 1.0,
        "r_steer": 10.0,
        "r_jerk_v": 0.0,
        "r_jerk_yaw": 0.0,
        "gamma": 0.90,
        "q_obs": 0.0,
        "sigma_obs": 5.0,
        "q_prog": 0.0,
        "ego_radius": 2.5,
        "safe_margin": 2.0,
        "lane_half_width": 2.0,
    },
    "input_limits": {
        "max_accel": 1.0,
        "min_accel": -5.0,
        "max_steer": 0.5,
        "min_steer": -0.5,
    },
    "vehicle": {
        "wheelbase": 2.87,
        "mass": 1845.0,
        "Iz": 2500.0,
        "l_f": 1.35,
        "l_r": 1.52,
        "C_f": 80000.0,
        "C_r": 80000.0,
    },
}


def _find_sharc_root():
    """Walk up from this file to find the SHARC repository root."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "resources", "sharc")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Cannot locate SHARC root (resources/sharc not found)")


class SharcMpcAgent(AutonomousAgent):
    """Autonomous agent that uses the SHARC C++ MPC controller."""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self, path_to_conf_file):
        self._sharc_root = _find_sharc_root()
        self._example_dir = os.path.join(self._sharc_root, "examples", "MPC_example")

        # Load config
        self._cfg = dict(DEFAULT_CONFIG)
        if path_to_conf_file and os.path.isfile(path_to_conf_file):
            with open(path_to_conf_file) as f:
                user_cfg = json.load(f)
            self._deep_update(self._cfg, user_cfg)
            print(f"[SharcMpcAgent] Loaded config from {path_to_conf_file}")

        # Derived dimensions
        self._n_wp = self._cfg["n_waypoints"]
        self._wp_spacing = self._cfg["waypoint_spacing"]
        self._n_obs = self._cfg["n_obstacles"]
        self._det_radius = self._cfg["detection_radius"]
        self._target_speed = self._cfg["target_speed"]
        self._sample_time = self._cfg["sample_time"]

        state_dim = 4   # px, py, psi, v
        input_dim = 2   # accel, steer
        w_dim = 2 * self._n_wp + 5 * self._n_obs
        output_dim = 2

        # Step counter and logging
        self._step = 0
        self._log_data = []
        self._output_dir = os.environ.get(
            "SHARC_AGENT_OUTPUT_DIR",
            os.path.join(self._example_dir, "scenario_runner_results",
                         time.strftime("%Y%m%d_%H%M%S")),
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # Create simulation directory for pipe communication
        self._sim_dir = os.path.join(self._output_dir, "sim")
        os.makedirs(self._sim_dir, exist_ok=True)

        # Write config.json for the C++ controller
        controller_config = self._build_controller_config(
            state_dim, input_dim, w_dim, output_dim
        )
        config_path = os.path.join(self._sim_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(controller_config, f, indent=2)

        # Compile the MPC binary
        self._executable = self._compile_mpc(
            state_dim, input_dim, w_dim, output_dim
        )

        # Create named pipes
        self._pipe_names = {
            "k_py_to_c++":        os.path.join(self._sim_dir, "k_py_to_c++"),
            "t_py_to_c++":        os.path.join(self._sim_dir, "t_py_to_c++"),
            "x_py_to_c++":        os.path.join(self._sim_dir, "x_py_to_c++"),
            "w_py_to_c++":        os.path.join(self._sim_dir, "w_py_to_c++"),
            "u_c++_to_py":        os.path.join(self._sim_dir, "u_c++_to_py"),
            "metadata_c++_to_py": os.path.join(self._sim_dir, "metadata_c++_to_py"),
            "t_delay_py_to_c++":  os.path.join(self._sim_dir, "t_delay_py_to_c++"),
        }
        for name, path in self._pipe_names.items():
            if os.path.exists(path):
                os.unlink(path)
            os.mkfifo(path)

        # Write simulator status
        status_path = os.path.join(self._sim_dir, "status_py_to_c++")
        with open(status_path, "w") as f:
            f.write("RUNNING")

        # Start C++ controller as background process
        self._controller_log = open(
            os.path.join(self._output_dir, "controller.log"), "w"
        )
        self._controller_proc = subprocess.Popen(
            [self._executable],
            cwd=self._sim_dir,
            stdout=self._controller_log,
            stderr=self._controller_log,
        )
        print(f"[SharcMpcAgent] Started controller PID={self._controller_proc.pid}")

        # Open pipes (must happen after C++ process starts, since pipe open blocks
        # until both sides are ready). Open in a specific order matching C++.
        # Writers (Python → C++)
        self._k_pipe = open(self._pipe_names["k_py_to_c++"], "w", buffering=1)
        self._t_pipe = open(self._pipe_names["t_py_to_c++"], "w", buffering=1)
        self._x_pipe = open(self._pipe_names["x_py_to_c++"], "w", buffering=1)
        self._w_pipe = open(self._pipe_names["w_py_to_c++"], "w", buffering=1)
        self._delay_pipe = open(self._pipe_names["t_delay_py_to_c++"], "w", buffering=1)

        # Readers (C++ → Python)
        self._u_pipe = open(self._pipe_names["u_c++_to_py"], "r", buffering=1)
        self._metadata_pipe = open(self._pipe_names["metadata_c++_to_py"], "r", buffering=1)

        self._hero = None
        self._carla_map = None
        print("[SharcMpcAgent] Setup complete. Pipes open, controller running.")

    def sensors(self):
        """No external sensors needed — we read state from CarlaDataProvider."""
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 300,
                "height": 200,
                "fov": 100,
                "id": "Center",
            }
        ]

    def run_step(self, input_data, timestamp):
        """Compute MPC control for current step."""
        t_wall_start = time.monotonic()

        # Lazy-init hero actor and map
        if self._hero is None:
            self._hero = self._find_hero()
            if self._hero is None:
                return carla.VehicleControl()
            self._carla_map = CarlaDataProvider.get_map()

        # 1) Extract ego state: [px, py, psi, v]
        transform = self._hero.get_transform()
        velocity = self._hero.get_velocity()
        px = transform.location.x
        py = transform.location.y
        psi = math.radians(transform.rotation.yaw)
        v = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        x_state = np.array([[px], [py], [psi], [v]], dtype=np.float64)

        # 2) Build exogenous input w = [waypoints | obstacles]
        w = self._build_exogenous_input(transform)

        # 3) Send to C++ MPC via pipes
        k = self._step
        t = timestamp if isinstance(timestamp, float) else float(timestamp)

        self._write_int(self._k_pipe, k)
        self._write_float(self._t_pipe, t)
        self._write_vector(self._x_pipe, x_state)
        self._write_vector(self._w_pipe, w)

        # 4) Read control u and metadata from C++
        u = self._read_vector(self._u_pipe)
        metadata = self._read_json(self._metadata_pipe)

        # 5) Compute wall-clock delay and send it
        t_wall_end = time.monotonic()
        computation_time = t_wall_end - t_wall_start
        self._write_float(self._delay_pipe, computation_time)

        # 6) Convert u = [accel, steer] to VehicleControl
        accel = float(u[0])
        steer = float(u[1])
        control = self._accel_steer_to_vehicle_control(accel, steer)

        # 7) Log step data
        self._log_step(k, t, x_state, u, w, metadata, computation_time)
        self._step += 1

        return control

    def destroy(self):
        """Clean up controller subprocess and pipes."""
        print("[SharcMpcAgent] Destroying agent...")

        # Signal controller to stop
        try:
            status_path = os.path.join(self._sim_dir, "status_py_to_c++")
            with open(status_path, "w") as f:
                f.write("FINISHED")
        except Exception:
            pass

        # Close pipes
        for pipe in [
            self._k_pipe, self._t_pipe, self._x_pipe,
            self._w_pipe, self._delay_pipe,
            self._u_pipe, self._metadata_pipe,
        ]:
            try:
                pipe.close()
            except Exception:
                pass

        # Terminate controller
        if hasattr(self, "_controller_proc") and self._controller_proc.poll() is None:
            self._controller_proc.terminate()
            try:
                self._controller_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._controller_proc.kill()

        if hasattr(self, "_controller_log"):
            self._controller_log.close()

        # Write collected log data
        self._write_results()

        # Clean up pipe files
        for path in self._pipe_names.values():
            try:
                os.unlink(path)
            except Exception:
                pass

        print(f"[SharcMpcAgent] Results saved to {self._output_dir}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _find_hero(self):
        for actor in CarlaDataProvider.get_world().get_actors():
            if "role_name" in actor.attributes and actor.attributes["role_name"] == "hero":
                return actor
        return None

    def _build_exogenous_input(self, ego_transform):
        """Build the w vector: [waypoints | obstacles]."""
        dim = 2 * self._n_wp + 5 * self._n_obs
        w = np.zeros((dim, 1), dtype=np.float64)

        # --- Waypoints from global plan ---
        waypoints = self._get_waypoints_from_plan(ego_transform)
        for i, (wx, wy) in enumerate(waypoints):
            w[2 * i] = wx
            w[2 * i + 1] = wy

        # --- Obstacles ---
        if self._n_obs > 0:
            obs_data = self._get_nearby_obstacles(ego_transform)

            # Filter blocked waypoints (replace with last safe)
            ego_r = self._cfg["cost_weights"].get("ego_radius", 2.5)
            safe_margin = self._cfg["cost_weights"].get("safe_margin", 2.0)
            if obs_data:
                last_safe = None
                truncated = False
                for i in range(len(waypoints)):
                    wx, wy = w[2 * i, 0], w[2 * i + 1, 0]
                    if truncated:
                        if last_safe is not None:
                            w[2 * i] = last_safe[0]
                            w[2 * i + 1] = last_safe[1]
                        continue
                    blocked = False
                    for ox, oy, _ovx, _ovy, obs_r in obs_data:
                        r_safe = ego_r + obs_r + safe_margin
                        if math.sqrt((wx - ox) ** 2 + (wy - oy) ** 2) < r_safe:
                            blocked = True
                            break
                    if blocked:
                        truncated = True
                        if last_safe is not None:
                            w[2 * i] = last_safe[0]
                            w[2 * i + 1] = last_safe[1]
                    else:
                        last_safe = (wx, wy)

            # Pack obstacles
            offset = 2 * self._n_wp
            for i in range(self._n_obs):
                if i < len(obs_data):
                    ox, oy, ovx, ovy, r = obs_data[i]
                    w[offset + 5 * i + 0] = ox
                    w[offset + 5 * i + 1] = oy
                    w[offset + 5 * i + 2] = ovx
                    w[offset + 5 * i + 3] = ovy
                    w[offset + 5 * i + 4] = r
                else:
                    w[offset + 5 * i + 0] = 1e6
                    w[offset + 5 * i + 1] = 1e6

        return w

    def _get_waypoints_from_plan(self, ego_transform):
        """Get waypoints from the global plan or from CARLA map."""
        waypoints = []

        if self._global_plan_world_coord:
            # Use scenario_runner global plan
            ego_loc = ego_transform.location
            ego_x, ego_y = ego_loc.x, ego_loc.y

            # Find closest waypoint in global plan
            min_dist = float("inf")
            closest_idx = 0
            for i, (tf, _) in enumerate(self._global_plan_world_coord):
                dx = tf.location.x - ego_x
                dy = tf.location.y - ego_y
                dist = dx * dx + dy * dy
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Collect waypoints ahead from global plan
            plan_len = len(self._global_plan_world_coord)
            collected = 0
            idx = closest_idx
            last_wp = None

            while collected < self._n_wp and idx < plan_len:
                tf, _ = self._global_plan_world_coord[idx]
                wp_x, wp_y = tf.location.x, tf.location.y

                # If global plan waypoints are too far apart, interpolate
                # using CARLA map waypoints
                if last_wp is not None:
                    dx = wp_x - last_wp[0]
                    dy = wp_y - last_wp[1]
                    gap = math.sqrt(dx * dx + dy * dy)
                    if gap > self._wp_spacing * 2:
                        # Use map waypoints to fill the gap
                        map_wp = self._carla_map.get_waypoint(
                            carla.Location(x=last_wp[0], y=last_wp[1], z=0),
                            project_to_road=True,
                        )
                        if map_wp:
                            n_fill = int(gap / self._wp_spacing)
                            for _ in range(min(n_fill, self._n_wp - collected)):
                                nxt = map_wp.next(self._wp_spacing)
                                if not nxt:
                                    break
                                map_wp = nxt[0]
                                waypoints.append(
                                    (map_wp.transform.location.x,
                                     map_wp.transform.location.y)
                                )
                                collected += 1
                            if collected >= self._n_wp:
                                break

                waypoints.append((wp_x, wp_y))
                last_wp = (wp_x, wp_y)
                collected += 1
                idx += 1

            # If not enough waypoints from plan, extend using map
            if collected < self._n_wp and waypoints:
                last_x, last_y = waypoints[-1]
                map_wp = self._carla_map.get_waypoint(
                    carla.Location(x=last_x, y=last_y, z=0),
                    project_to_road=True,
                )
                while collected < self._n_wp:
                    if map_wp:
                        nxt = map_wp.next(self._wp_spacing)
                        if nxt:
                            map_wp = nxt[0]
                            waypoints.append(
                                (map_wp.transform.location.x,
                                 map_wp.transform.location.y)
                            )
                        else:
                            waypoints.append(waypoints[-1])
                    else:
                        waypoints.append(waypoints[-1] if waypoints else (0, 0))
                    collected += 1
        else:
            # No global plan — use CARLA map waypoints ahead
            wp = self._carla_map.get_waypoint(ego_transform.location)
            for _ in range(self._n_wp):
                nxt = wp.next(self._wp_spacing)
                if not nxt:
                    waypoints.append(
                        (wp.transform.location.x, wp.transform.location.y)
                    )
                    continue
                wp = nxt[0]
                waypoints.append(
                    (wp.transform.location.x, wp.transform.location.y)
                )

        return waypoints[: self._n_wp]

    def _get_nearby_obstacles(self, ego_transform):
        """Detect nearby dynamic actors (vehicles + walkers)."""
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        ego_id = self._hero.id

        fwd_x = math.cos(ego_yaw)
        fwd_y = math.sin(ego_yaw)
        LATERAL_FILTER = 2.5

        candidates = []
        for actor in CarlaDataProvider.get_world().get_actors():
            if actor.id == ego_id:
                continue
            if not (
                actor.type_id.startswith("vehicle.")
                or actor.type_id.startswith("walker.")
            ):
                continue

            loc = actor.get_transform().location
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self._det_radius:
                continue

            lat_offset = abs(-dx * fwd_y + dy * fwd_x)
            if lat_offset > LATERAL_FILTER:
                continue

            vel = actor.get_velocity()
            ext = actor.bounding_box.extent
            radius = math.sqrt(ext.x ** 2 + ext.y ** 2)
            candidates.append((dist, loc.x, loc.y, vel.x, vel.y, radius))

        candidates.sort(key=lambda c: c[0])
        return [
            (ox, oy, ovx, ovy, r)
            for (_, ox, oy, ovx, ovy, r) in candidates[: self._n_obs]
        ]

    def _accel_steer_to_vehicle_control(self, accel, steer):
        """Convert MPC output (accel, steer_angle) to VehicleControl."""
        control = carla.VehicleControl()

        max_accel = self._cfg["input_limits"]["max_accel"]
        min_accel = self._cfg["input_limits"]["min_accel"]

        if accel >= 0:
            control.throttle = min(accel / max(max_accel, 0.01), 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel / max(abs(min_accel), 0.01), 1.0)

        control.steer = max(-1.0, min(1.0, steer))
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    # ------------------------------------------------------------------ #
    # Pipe I/O
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_int(pipe, val):
        pipe.write(f"{val:d}\n")
        pipe.flush()

    @staticmethod
    def _write_float(pipe, val):
        pipe.write(f"{val:.8g}\n")
        pipe.flush()

    @staticmethod
    def _write_vector(pipe, vec):
        flat = vec.flatten()
        csv_str = ", ".join(f"{v:.8g}" for v in flat)
        pipe.write(csv_str + "\n")
        pipe.flush()

    @staticmethod
    def _read_vector(pipe):
        line = ""
        while not line.endswith("\n"):
            line += pipe.readline()
        chars = " []\n"
        parts = line.split(",")
        return np.array(
            [[float(p.strip(chars))] for p in parts], dtype=np.float64
        )

    @staticmethod
    def _read_json(pipe):
        line = ""
        while not line.endswith("\n"):
            line += pipe.readline()
        return json.loads(line)

    # ------------------------------------------------------------------ #
    # Build / compile
    # ------------------------------------------------------------------ #
    def _compile_mpc(self, state_dim, input_dim, w_dim, output_dim):
        """Compile the MPC controller binary if needed."""
        build_dir = os.path.join(self._example_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        ph = self._cfg["prediction_horizon"]
        ch = self._cfg["control_horizon"]
        n_obs = self._cfg["n_obstacles"]
        n_ineq = ph * (n_obs + 1) if n_obs > 0 else 0

        executable_name = "main_controller_MPC_v1"
        executable_path = os.path.join(build_dir, executable_name)

        # Only rebuild if binary doesn't exist
        if os.path.isfile(executable_path):
            print(f"[SharcMpcAgent] Using existing binary: {executable_path}")
            return executable_path

        print(f"[SharcMpcAgent] Compiling MPC controller...")
        cmake_args = [
            "cmake",
            "-S", self._example_dir,
            "-B", build_dir,
            f"-DPREDICTION_HORIZON={ph}",
            f"-DCONTROL_HORIZON={ch}",
            f"-DTNX={state_dim}",
            f"-DTNU={input_dim}",
            f"-DTNDU={w_dim}",
            f"-DTNY={output_dim}",
            f"-DTNIEQ={n_ineq}",
            "-DUSE_DYNAMORIO=OFF",
        ]
        subprocess.run(cmake_args, cwd=build_dir, check=True)
        subprocess.run(
            ["cmake", "--build", build_dir, "-j", str(os.cpu_count())],
            cwd=build_dir,
            check=True,
        )

        if not os.path.isfile(executable_path):
            raise RuntimeError(f"MPC binary not found after build: {executable_path}")

        print(f"[SharcMpcAgent] Compiled: {executable_path}")
        return executable_path

    def _build_controller_config(self, state_dim, input_dim, w_dim, output_dim):
        """Build the config.json that the C++ controller reads."""
        cfg = self._cfg
        w_names = []
        for i in range(1, self._n_wp + 1):
            w_names.extend([f"wp_x_{i}", f"wp_y_{i}"])
        for i in range(1, self._n_obs + 1):
            w_names.extend([
                f"obs{i}_x", f"obs{i}_y",
                f"obs{i}_vx", f"obs{i}_vy", f"obs{i}_r",
            ])

        return {
            "label": "sharc_mpc_agent",
            "n_time_steps": 999999,
            "only_update_control_at_sample_times": True,
            "delay_multiplier": 1,
            "fake_delays": {"enable": False, "time_steps": [], "sample_time_multipliers": []},
            "x0": [0, 0, 0, 0],
            "system_parameters": {
                "state_dimension": state_dim,
                "input_dimension": input_dim,
                "exogenous_input_dimension": w_dim,
                "output_dimension": output_dim,
                "sample_time": cfg["sample_time"],
                "controller_type": cfg["controller_type"],
                "x_names": ["px", "py", "psi", "v"],
                "u_names": ["acceleration", "steering_angle"],
                "w_names": w_names,
                "y_names": ["pos_x", "pos_y"],
                "target_speed": cfg["target_speed"],
                "mpc_options": {
                    "prediction_horizon": cfg["prediction_horizon"],
                    "control_horizon": cfg["control_horizon"],
                    "wheelbase": cfg["vehicle"]["wheelbase"],
                    "mass": cfg["vehicle"]["mass"],
                    "Iz": cfg["vehicle"]["Iz"],
                    "l_f": cfg["vehicle"]["l_f"],
                    "l_r": cfg["vehicle"]["l_r"],
                    "C_f": cfg["vehicle"]["C_f"],
                    "C_r": cfg["vehicle"]["C_r"],
                    "n_waypoints": self._n_wp,
                    "waypoint_spacing": self._wp_spacing,
                    "n_obstacles": self._n_obs,
                    "detection_radius": self._det_radius,
                    "cost_weights": cfg["cost_weights"],
                    "input_limits": cfg["input_limits"],
                },
            },
            "carla": {"seed": 0, "npcs": {"n_vehicles": 0, "n_walkers": 0}},
            "Simulation Options": {
                "in-the-loop_delay_provider": "onestep",
                "parallel_scarab_simulation": False,
                "max_batches": 1,
            },
            "PARAMS_patch_values": {"chip_cycle_time": 500000},
            "simulation_label": "sharc_mpc_agent",
        }

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_step(self, k, t, x, u, w, metadata, comp_time):
        entry = {
            "k": k,
            "t": t,
            "x": x.flatten().tolist(),
            "u": u.flatten().tolist(),
            "computation_time_s": comp_time,
            "is_feasible": metadata.get("is_feasible", True),
            "cost": metadata.get("cost", 0.0),
        }
        self._log_data.append(entry)

        if k % 50 == 0:
            speed_kmh = float(x[3]) * 3.6
            print(
                f"  [Step {k:4d}] t={t:.2f}s  "
                f"pos=({float(x[0]):.1f}, {float(x[1]):.1f})  "
                f"v={speed_kmh:.1f} km/h  "
                f"comp={comp_time*1000:.1f}ms  "
                f"feasible={metadata.get('is_feasible', '?')}"
            )

    def _write_results(self):
        """Write collected log data to JSON."""
        results_path = os.path.join(self._output_dir, "agent_results.json")
        results = {
            "total_steps": len(self._log_data),
            "steps": self._log_data,
        }
        if self._log_data:
            comp_times = [s["computation_time_s"] for s in self._log_data]
            results["summary"] = {
                "mean_computation_ms": 1000 * sum(comp_times) / len(comp_times),
                "max_computation_ms": 1000 * max(comp_times),
                "min_computation_ms": 1000 * min(comp_times),
                "total_infeasible": sum(
                    1 for s in self._log_data if not s["is_feasible"]
                ),
                "deadline_misses": sum(
                    1
                    for s in self._log_data
                    if s["computation_time_s"] > self._sample_time
                ),
            }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SharcMpcAgent] Results: {results_path}")
        if results.get("summary"):
            s = results["summary"]
            print(
                f"  Mean comp: {s['mean_computation_ms']:.1f}ms  "
                f"Max: {s['max_computation_ms']:.1f}ms  "
                f"Infeasible: {s['total_infeasible']}  "
                f"Deadline misses: {s['deadline_misses']}"
            )

    @staticmethod
    def _deep_update(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                SharcMpcAgent._deep_update(base[k], v)
            else:
                base[k] = v
