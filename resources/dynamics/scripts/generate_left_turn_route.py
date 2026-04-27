#!/usr/bin/env python3
"""Generate a dense left-turn route XML from the active CARLA map."""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import carla


def yaw_delta(a_deg, b_deg):
    return ((b_deg - a_deg + 540.0) % 360.0) - 180.0


def pick_straight(current_wp, next_wps):
    if not next_wps:
        return None
    cur_yaw = current_wp.transform.rotation.yaw
    return min(next_wps, key=lambda w: abs(yaw_delta(cur_yaw, w.transform.rotation.yaw)))


def pick_left(current_wp, next_wps, min_left_deg):
    cur_yaw = current_wp.transform.rotation.yaw
    cur_rad = math.radians(cur_yaw)
    cur_vec = (math.cos(cur_rad), math.sin(cur_rad))
    best_wp, best_angle = None, None
    for wp in next_wps:
        d_yaw = yaw_delta(cur_yaw, wp.transform.rotation.yaw)
        wp_rad = math.radians(wp.transform.rotation.yaw)
        wp_vec = (math.cos(wp_rad), math.sin(wp_rad))
        cross = cur_vec[0] * wp_vec[1] - cur_vec[1] * wp_vec[0]
        if cross > 0 and d_yaw >= min_left_deg:
            if best_wp is None or d_yaw > best_angle:
                best_wp, best_angle = wp, d_yaw
    return best_wp, best_angle


def trace_left_turn(start_wp, step_m, max_points, min_left_deg, min_after_turn):
    route = [start_wp]
    current = start_wp
    turned_left = False
    left_angle = 0.0
    after_turn = 0
    for _ in range(max_points - 1):
        nxt = current.next(step_m)
        if not nxt:
            break
        if (not turned_left) and current.is_junction:
            left_wp, ang = pick_left(current, nxt, min_left_deg)
            chosen = left_wp if left_wp is not None else pick_straight(current, nxt)
            if left_wp is not None:
                turned_left = True
                left_angle = ang
        else:
            chosen = pick_straight(current, nxt)
        if chosen is None:
            break
        route.append(chosen)
        current = chosen
        if turned_left:
            after_turn += 1
            if after_turn >= min_after_turn and not current.is_junction:
                break
    if not turned_left:
        return None, None
    return route, left_angle


def find_best_route(map_obj, spawn_points, preferred_idx, step_m, max_points, min_left_deg, min_after_turn):
    indices = list(range(len(spawn_points)))
    if preferred_idx is not None and 0 <= preferred_idx < len(indices):
        indices.remove(preferred_idx)
        indices.insert(0, preferred_idx)

    best = None  # (left_angle, n_points, spawn_idx, route_wps)
    for idx in indices:
        wp = map_obj.get_waypoint(
            spawn_points[idx].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            continue
        route_wps, left_angle = trace_left_turn(wp, step_m, max_points, min_left_deg, min_after_turn)
        if route_wps is None:
            continue
        cand = (left_angle, len(route_wps), idx, route_wps)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best


def write_xml(output_xml, route_id, town_name, route_wps):
    out_dir = os.path.dirname(output_xml)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id=str(route_id), town=town_name)
    waypoints = ET.SubElement(route, "waypoints")
    for wp in route_wps:
        p = wp.transform.location
        ET.SubElement(waypoints, "position", x=f"{p.x:.3f}", y=f"{p.y:.3f}", z=f"{p.z:.3f}")
    ET.SubElement(route, "scenarios")
    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)


def main():
    p = argparse.ArgumentParser(description="Generate left-turn route XML for SHARC Scenario 02.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--output-xml", required=True)
    p.add_argument("--route-id", default="scenario02_left_turn")
    p.add_argument("--preferred-spawn-index", type=int, default=None)
    p.add_argument("--step-m", type=float, default=2.0)
    p.add_argument("--max-points", type=int, default=220)
    p.add_argument("--min-left-deg", type=float, default=20.0)
    p.add_argument("--min-points-after-turn", type=int, default=50)
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on current map.")

    best = find_best_route(
        map_obj,
        spawn_points,
        args.preferred_spawn_index,
        args.step_m,
        args.max_points,
        args.min_left_deg,
        args.min_points_after_turn,
    )
    if best is None:
        raise RuntimeError("Could not find a suitable left-turn route.")

    left_angle, n_pts, spawn_idx, route_wps = best
    write_xml(args.output_xml, args.route_id, map_obj.name.split("/")[-1], route_wps)
    first = route_wps[0].transform.location
    last = route_wps[-1].transform.location
    print(f"Town: {map_obj.name.split('/')[-1]}")
    print(f"Spawn index: {spawn_idx}")
    print(f"Left turn angle: {left_angle:.1f} deg")
    print(f"Route points: {n_pts}")
    print(f"Start: ({first.x:.2f}, {first.y:.2f}, {first.z:.2f})")
    print(f"End:   ({last.x:.2f}, {last.y:.2f}, {last.z:.2f})")
    print(f"Wrote: {args.output_xml}")


if __name__ == "__main__":
    main()
