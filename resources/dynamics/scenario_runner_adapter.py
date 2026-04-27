"""
Thin Scenario Runner adapter used by CarlaMPCDynamics.

This module keeps integration lightweight:
- It can parse Leaderboard-style route/scenario assets.
- It owns optional scenario-managed actors for obstacle extraction.
- It stays feature-flagged and does nothing when disabled.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import carla


Waypoint2D = Tuple[float, float]
ObstacleTuple = Tuple[float, float, float, float, float]


class ScenarioRunnerAdapter:
    """Minimal runtime bridge between SHARC dynamics and scenario_runner assets."""

    def __init__(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager,
        cfg: Dict,
        seed: int,
    ) -> None:
        self.world = world
        self.traffic_manager = traffic_manager
        self.cfg = cfg or {}
        self.seed = int(self.cfg.get("seed", seed))
        self.enabled = bool(self.cfg.get("enabled", False))
        self._managed_actors: List[carla.Actor] = []
        self._route_waypoints: List[Waypoint2D] = []
        self._scenario_event: Optional[Dict] = None
        self._setup_done = False

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    @property
    def use_scenario_obstacles_only(self) -> bool:
        return bool(self.cfg.get("use_scenario_obstacles_only", False))

    def setup(
        self,
        ego_vehicle: carla.Vehicle,
        carla_map: carla.Map,
        bp_lib: carla.BlueprintLibrary,
    ) -> None:
        if not self.enabled:
            return

        random.seed(self.seed)
        self._route_waypoints = self._load_route_waypoints()
        self._scenario_event = self._load_scenario_event()
        self._spawn_oncoming_flow(ego_vehicle, carla_map, bp_lib)
        self._setup_done = True

    def get_route_waypoints(self) -> List[Waypoint2D]:
        return list(self._route_waypoints)

    def on_world_reset(
        self,
        ego_vehicle: carla.Vehicle,
        carla_map: carla.Map,
        bp_lib: carla.BlueprintLibrary,
    ) -> None:
        if not self.enabled:
            return
        self._destroy_managed_actors()
        self.setup(ego_vehicle, carla_map, bp_lib)

    def tick(self, sim_time_s: float) -> None:
        # Hook for future scenario-specific time logic.
        _ = sim_time_s

    def get_obstacles(
        self,
        detection_radius: float,
        ego_loc: carla.Location,
    ) -> List[ObstacleTuple]:
        if not self.enabled:
            return []

        obstacles: List[ObstacleTuple] = []
        for actor in self._managed_actors:
            try:
                if not actor.is_alive:
                    continue
                tf = actor.get_transform()
                loc = tf.location
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                if math.hypot(dx, dy) > detection_radius:
                    continue

                vel = actor.get_velocity()
                radius = self._get_actor_radius(actor)
                obstacles.append((loc.x, loc.y, vel.x, vel.y, radius))
            except RuntimeError:
                continue

        return obstacles

    def teardown(self) -> None:
        self._destroy_managed_actors()
        self._setup_done = False

    def _load_route_waypoints(self) -> List[Waypoint2D]:
        inline_points = self.cfg.get("route_waypoints", [])
        if inline_points:
            return self._normalize_waypoints(inline_points)

        route_file = self.cfg.get("route_file")
        if not route_file:
            return []

        route_id = self.cfg.get("route_id")
        try:
            return self._parse_route_file(route_file, route_id)
        except Exception as exc:
            print(f"[ScenarioRunnerAdapter] WARNING: failed to parse route file '{route_file}': {exc}")
            return []

    def _load_scenario_event(self) -> Optional[Dict]:
        scenario_cfg_path = self.cfg.get("scenario_config_path")
        if not scenario_cfg_path:
            return None

        scenario_id = str(self.cfg.get("scenario_id", "Scenario2"))
        scenario_index = int(self.cfg.get("scenario_index", 0))

        try:
            with open(scenario_cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[ScenarioRunnerAdapter] WARNING: cannot read scenario config '{scenario_cfg_path}': {exc}")
            return None

        town = self.cfg.get("town")
        if not town:
            town = self.world.get_map().name.split("/")[-1]

        available = data.get("available_scenarios", [])
        if not available:
            return None

        town_table = available[0].get(town, [])
        for entry in town_table:
            if entry.get("scenario_type") != scenario_id:
                continue
            events = entry.get("available_event_configurations", [])
            if not events:
                return None
            idx = max(0, min(scenario_index, len(events) - 1))
            return events[idx]
        return None

    def _spawn_oncoming_flow(
        self,
        ego_vehicle: carla.Vehicle,
        carla_map: carla.Map,
        bp_lib: carla.BlueprintLibrary,
    ) -> None:
        count = int(self.cfg.get("oncoming_actor_count", 1))
        if count <= 0:
            return

        base_wp = self._get_base_waypoint(ego_vehicle, carla_map)
        if base_wp is None:
            print("[ScenarioRunnerAdapter] WARNING: no base waypoint for Scenario2 flow")
            return

        vehicle_bps = sorted(bp_lib.filter("vehicle.*"), key=lambda bp: bp.id)
        if not vehicle_bps:
            return

        tm_port = self.traffic_manager.get_port()
        source_dist = float(self.cfg.get("oncoming_source_dist", 30.0))
        speed_mps = float(self.cfg.get("oncoming_speed_mps", 10.0))

        spawned = 0
        for i in range(count):
            wp_candidates = base_wp.previous(max(2.0, source_dist + 10.0 * i))
            if not wp_candidates:
                continue
            spawn_wp = wp_candidates[0]
            spawn_tf = spawn_wp.transform
            spawn_tf.location.z += 0.3

            bp = vehicle_bps[(self.seed + i) % len(vehicle_bps)]
            actor = self.world.try_spawn_actor(bp, spawn_tf)
            if actor is None:
                continue

            actor.set_autopilot(True, tm_port)
            try:
                speed_limit = max(actor.get_speed_limit(), 1.0)
                pct = 100.0 * (1.0 - speed_mps / speed_limit)
                pct = max(-50.0, min(95.0, pct))
            except RuntimeError:
                pct = 30.0
            self.traffic_manager.vehicle_percentage_speed_difference(actor, pct)
            self.traffic_manager.auto_lane_change(actor, False)
            self.traffic_manager.distance_to_leading_vehicle(actor, 5.0)

            self._managed_actors.append(actor)
            spawned += 1

        print(f"[ScenarioRunnerAdapter] Scenario actors spawned: {spawned}/{count}")

    def _get_base_waypoint(
        self,
        ego_vehicle: carla.Vehicle,
        carla_map: carla.Map,
    ) -> Optional[carla.Waypoint]:
        if self._scenario_event:
            transform = self._scenario_event.get("transform", {})
            try:
                loc = carla.Location(
                    x=float(transform["x"]),
                    y=float(transform["y"]),
                    z=float(transform.get("z", 0.0)),
                )
                wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp is not None:
                    return wp
            except Exception:
                pass

        ego_loc = ego_vehicle.get_transform().location
        return carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _destroy_managed_actors(self) -> None:
        if not self._managed_actors:
            return

        tm_port = self.traffic_manager.get_port()
        destroy_ids = []
        for actor in self._managed_actors:
            try:
                if actor.is_alive and actor.type_id.startswith("vehicle."):
                    actor.set_autopilot(False, tm_port)
            except RuntimeError:
                pass
            try:
                destroy_ids.append(actor.id)
            except RuntimeError:
                pass

        self._managed_actors = []
        if destroy_ids:
            for actor_id in destroy_ids:
                try:
                    actor = self.world.get_actor(actor_id)
                    if actor is not None:
                        actor.destroy()
                except RuntimeError:
                    pass

    @staticmethod
    def _normalize_waypoints(points: Sequence) -> List[Waypoint2D]:
        out: List[Waypoint2D] = []
        for point in points:
            if isinstance(point, dict):
                out.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                out.append((float(point[0]), float(point[1])))
        return out

    @staticmethod
    def _parse_route_file(route_file: str, route_id: Optional[str]) -> List[Waypoint2D]:
        tree = ET.parse(route_file)
        root = tree.getroot()

        selected_route = None
        for route in root.iter("route"):
            if route_id is None or str(route.attrib.get("id")) == str(route_id):
                selected_route = route
                break

        if selected_route is None:
            return []

        waypoints_tag = selected_route.find("waypoints")
        if waypoints_tag is None:
            return []

        route: List[Waypoint2D] = []
        for position in waypoints_tag.iter("position"):
            route.append((float(position.attrib["x"]), float(position.attrib["y"])))
        return route

    @staticmethod
    def _get_actor_radius(actor: carla.Actor) -> float:
        bb = actor.bounding_box.extent
        return math.sqrt(bb.x * bb.x + bb.y * bb.y)
