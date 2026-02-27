#!/usr/bin/env python3
"""
Enumerate and visualize CARLA spawn points for a given map.

Connects to a running CARLA server, loads the specified map, retrieves all
spawn points, prints their indices and coordinates, and saves a 2D bird's-eye
scatter plot with index labels.

Usage:
    python enumerate_spawns.py --map Town01 --port 2000
    python enumerate_spawns.py --map Town02 --port 2000 --no-plot
"""

import argparse
import sys

try:
    import carla
except ImportError:
    sys.exit("ERROR: CARLA Python API not found. Run: conda activate carla")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def enumerate_spawns(client: carla.Client, map_name: str, save_plot: bool = True):
    """Load map, print spawn points, and optionally plot them."""
    print(f"Loading map: {map_name} ...")
    world = client.load_world(map_name)

    # Brief tick to let the world settle
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.tick()

    spawn_points = world.get_map().get_spawn_points()
    print(f"\nFound {len(spawn_points)} spawn points on {map_name}:\n")
    print(f"{'Index':>6}  {'X':>10}  {'Y':>10}  {'Z':>8}  {'Yaw':>8}")
    print("-" * 52)

    xs, ys, indices = [], [], []
    for i, sp in enumerate(spawn_points):
        loc = sp.location
        rot = sp.rotation
        print(f"{i:>6}  {loc.x:>10.2f}  {loc.y:>10.2f}  {loc.z:>8.2f}  {rot.yaw:>8.1f}")
        xs.append(loc.x)
        ys.append(loc.y)
        indices.append(i)

    # Reset to async mode so the server doesn't hang
    settings.synchronous_mode = False
    world.apply_settings(settings)

    if save_plot and HAS_MATPLOTLIB:
        _plot_spawns(xs, ys, indices, map_name)
    elif save_plot and not HAS_MATPLOTLIB:
        print("\nWARNING: matplotlib not available, skipping plot.")

    return spawn_points


def _plot_spawns(xs, ys, indices, map_name):
    """Save a 2D scatter plot of spawn points labeled by index."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(xs, ys, c='royalblue', s=40, zorder=5)

    for i, x, y in zip(indices, xs, ys):
        ax.annotate(str(i), (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color='darkred')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{map_name} Spawn Points ({len(indices)} total)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    out_file = f'spawns_{map_name}.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"\nPlot saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Enumerate and visualize CARLA spawn points')
    parser.add_argument('--map', type=str, default='Town01',
                        help='CARLA map name (default: Town01)')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port (default: 2000)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating the plot')
    args = parser.parse_args()

    client = carla.Client('localhost', args.port)
    client.set_timeout(20.0)

    enumerate_spawns(client, args.map, save_plot=not args.no_plot)


if __name__ == '__main__':
    main()

