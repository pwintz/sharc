import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation  # <-- needed for FuncAnimation
from math import nan
from typing import Union, Dict, List


def main():
    filename = input("Enter experiment folder name (the date-time stamp): ").strip()
    
    car_example_dir = os.path.abspath('.') + '/examples/CarExample/experiments'
    experiment_dir = os.path.join(car_example_dir, filename)
    assertFileExists(experiment_dir)

    # Paths
    incremental_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
    final_path = os.path.join(experiment_dir, "experiment_list_data.json")
    out_dir = os.path.join(experiment_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    # Read experiment data
    experiment_results = {}
    if os.path.exists(incremental_path):
        experiment_results = readJson(incremental_path)
    if not experiment_results and os.path.exists(final_path):
        print("⚠️ Incremental results empty, using final results file.")
        experiment_results = readJson(final_path)
    if not experiment_results:
        raise RuntimeError(
            f"No experiment data found in:\n  {incremental_path}\n  or\n  {final_path}"
        )

    # (key, value) → (label, value)
    results = [
        (val["experiment config"]["label"], val)
        for val in experiment_results.values()
    ]
    if not results:
        raise RuntimeError("Experiment results list is empty — make sure your experiment ran to completion.")

    # Make animations for all experiments
    sim_experiment_list(results, out_dir)


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------
def assertFileExists(path: str, help_msg=None):
    if not os.path.exists(path):
        err_msg = f"Expected {path} to exist but it does not. "
        if os.path.abspath(path) != path:
            err_msg += f"The absolute path is {os.path.abspath(path)}"
        if help_msg:
            err_msg += "\n" + help_msg
        raise IOError(err_msg)


def readJson(filename: str) -> Union[Dict, List]:
    assertFileExists(filename)
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except json.decoder.JSONDecodeError as err:
        raise ValueError(f"Error parsing JSON: {filename}") from err


# --------------------------------------------------------------------------
# Plotting / animation
# --------------------------------------------------------------------------
def sim_experiment_list(results, out_dir: str):
    """
    results: list of (label, result_dict)
    For each result, create and save a GIF animation of x vs y.
    """
    for label, result in results:
        result_data = result["experiment data"]
        gif_path = os.path.join(out_dir, f"{label}_car_sim.gif")
        print(f"📈 Creating animation for experiment '{label}'...")
        sim_experiment_result(result_data, gif_path)
        print(f"✅ Saved figure to: {gif_path}")


def sim_experiment_result(result_data: Dict, save_path: str):
    """
    Create an animation of the car trajectory (x vs y) and save as GIF.
    """
    t = np.array(result_data["t"])
    x = np.array(result_data["x"])  # (N, 4): [x, y, v, psi]
    x_pos, y_pos, v, psi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Car trajectory")
    ax.grid(True)

    # Initialize the line object
    line, = ax.plot([], [], lw=2)

    # Fix axis limits so they don't rescale every frame
    margin = 0.1
    xmin, xmax = x_pos.min(), x_pos.max()
    ymin, ymax = y_pos.min(), y_pos.max()
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
    ax.set_ylim(ymin - margin * dy, ymax + margin * dy)

    # Define animation functions that close over x_pos, y_pos, line
    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        # Show trajectory up to frame i
        line.set_data(x_pos[:i], y_pos[:i])
        return line,

    frames = len(t)

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=25,  # ms between frames
        blit=True
    )

    # Save as GIF (requires pillow installed: pip install pillow)
    ani.save(save_path, writer="pillow", fps=30)
    plt.close(fig)  # close figure so it doesn't pop up when running as script


if __name__ == "__main__":
    main()
