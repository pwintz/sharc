import argparse
import json
import os
from itertools import cycle
from math import nan
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for a Rocket example experiment."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default="latest",
        help=(
            "Experiment folder name, absolute path, or path to an experiment directory. "
            "Defaults to 'latest'."
        ),
    )
    args = parser.parse_args()

    experiment_dir = resolve_experiment_dir(args.experiment)

    incremental_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
    final_path = os.path.join(experiment_dir, "experiment_list_data.json")
    out_dir = os.path.join(experiment_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    experiment_results = {}
    if os.path.exists(incremental_path):
        experiment_results = readJson(incremental_path)
    if not experiment_results and os.path.exists(final_path):
        print("Incremental results missing/empty, using final results file.")
        experiment_results = readJson(final_path)
    if not experiment_results:
        raise RuntimeError(
            "No experiment results found in:\n"
            f"  {incremental_path}\n"
            f"  {final_path}"
        )

    results = [(val["experiment config"]["label"], val) for val in experiment_results.values()]
    if not results:
        raise RuntimeError("Experiment results are empty.")

    plot_module = plot_experiment_list(results)
    image_save_path = os.path.join(out_dir, "rocket_plots.png")
    plot_module.savefig(image_save_path)
    print(f"Saved file to {image_save_path}")


def resolve_experiment_dir(experiment_arg: str) -> str:
    example_dir = os.path.dirname(os.path.abspath(__file__))
    experiments_dir = os.path.join(example_dir, "experiments")

    candidate_paths = []
    if os.path.isabs(experiment_arg):
        candidate_paths.append(experiment_arg)
    else:
        candidate_paths.append(os.path.abspath(experiment_arg))
        candidate_paths.append(os.path.join(example_dir, experiment_arg))
        candidate_paths.append(os.path.join(experiments_dir, experiment_arg))

    for path in candidate_paths:
        if os.path.isdir(path):
            return path

    raise IOError(
        "Could not find experiment directory from argument "
        f"'{experiment_arg}'. Tried:\n  " + "\n  ".join(candidate_paths)
    )


def assertFileExists(path:str, help_msg=None):
  if not os.path.exists(path):
    err_msg = f'Expected {path} to exist but it does not. '
    if os.path.abspath(path) != path:
      err_msg += f'The absolute path is {os.path.abspath(path)}'
    if help_msg:
      err_msg += '\n' + help_msg
    raise IOError(err_msg)


def readJson(filename: str) -> Union[Dict,List]:
  assertFileExists(filename)
  try:
    with open(filename, 'r') as json_file:
      return json.load(json_file)
  except json.decoder.JSONDecodeError as err:
    raise ValueError(f'An error occurred while parsing {filename}.') from err


def plot_experiment_list(experiment_list):
    colors = cycle(plt.cm.tab10.colors)
    n_axs = 4
    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 13), sharex=True)

    velocity_ax, height_ax, delay_ax, control_ax = axs
    plts_for_legend = []

    for label, result in experiment_list:
        result_data = result["experiment data"]
        color = next(colors)
        plot_experiment_result(result_data, velocity_ax, height_ax, delay_ax, control_ax, color)

        # Invisible line for legend
        line = velocity_ax.plot(nan, nan, c=color)[0]
        plts_for_legend.append((line, label))

    # Example parameters
    result_data = experiment_list[0][1]["experiment data"]
    sample_time = experiment_list[0][1]["experiment config"]["system_parameters"]["sample_time"]
    t = np.array(result_data["t"])
    xlim = (0, t[-1])

    # Axis formatting
    setup_axes(velocity_ax, height_ax, delay_ax, control_ax, xlim, sample_time)
    fig.legend(*zip(*plts_for_legend), loc='lower center', bbox_to_anchor=(0.55, 0.12),
               ncol=4, columnspacing=1.0, handlelength=1.5, handletextpad=0.5,
               bbox_transform=fig.transFigure, borderaxespad=0.2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)
    return plt


def setup_axes(velocity_ax, height_ax, delay_ax, control_ax, xlim, sample_time):
    # Velocity (can go negative for descent)
    velocity_ax.set(
        ylabel='Velocity $v$ [m/s]', 
        title='Velocity (Upward + / Downward -)', 
        xlim=xlim
    )
    velocity_ax.grid(True)

    # Height (always positive)
    height_ax.set(
        ylabel='Height $h$ [m]', 
        title='Height', 
        xlim=xlim,
        ylim=(0, None)
    )
    height_ax.grid(True)

    # Delay
    delay_ax.set(
        ylabel='Delay $\\tau$ [s]', 
        title='Computation / Actuation Delays', 
        xlim=xlim
    )
    delay_ax.axhline(
        y=sample_time, color='black', linestyle='--', linewidth=2,
        label=f"Sample Time ({sample_time}s)"
    )
    delay_ax.grid(True)
    delay_ax.legend()

    # Control input (thrust positive upward)
    control_ax.set(
        xlabel='Time [s]', 
        ylabel='Thrust $u$ [N]', 
        title='Control Input (Thrust Upward +)', 
        xlim=xlim
    )
    control_ax.grid(True)


def plot_experiment_result(result_data, velocity_ax, height_ax, delay_ax, control_ax, color):
    t = np.array(result_data["t"])
    u = np.array(result_data["u"])           # (N, 1)
    x = np.array(result_data["x"])           # (N, 2): [height, velocity]
    h = x[:, 0]
    v = x[:, 1]

    # Plot states
    velocity_ax.plot(t, v, c=color)
    height_ax.plot(t, h, c=color)

    # Plot delays
    pc_t, pc_delay = [], []
    for pc in result_data.get("pending_computations", []):
        if pc:
            t_start = float(pc["t_start"])
            delay = float(pc["delay"])
            pc_t.extend([t_start, t_start + delay, np.nan])
            pc_delay.extend([delay, delay, np.nan])
    delay_ax.plot(pc_t, pc_delay, color=color)

    # Plot control input (u is thrust, positive upward)
    control_ax.plot(t, u[:, 0], color=color, linestyle='-')

    # Plot batch starting points (if present)
    if result_data.get("batches"):
        for batch in result_data["batches"]:
            start_time = batch["valid_simulation_data"]["t"][0]
            idx = np.where(t == start_time)[0][0]
            velocity_ax.plot(t[idx], v[idx], 'kx', markersize=4)
            height_ax.plot(t[idx], h[idx], 'kx', markersize=4)

if __name__ == "__main__":
    main()
