import numpy as np
import copy
import os
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from math import nan
from typing import Union
from typing import List, Set, Dict, Tuple




def main():
    
    filename = input("What is the file name(Its the date and time)")
    
    rocket_example_dir = os.path.abspath('.') + '/examples/Rocket_example/experiments'
    experiment_dir = os.path.join(rocket_example_dir, filename)
    assertFileExists(experiment_dir)

    #paths
    incremental_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
    final_path = os.path.join(experiment_dir, "experiment_list_data.json")
    out_dir = os.path.join(experiment_dir, "images")
    os.makedirs(out_dir, exist_ok = True)
    
    #read data
    experiment_results = readJson(incremental_path)
    results = [(val["experiment config"]["label"], val) for val in experiment_results.values()]

    #plot and save
    plt = plot_experiment_list(results)
    image_save_path = os.path.join(out_dir + '/plots.png')
    plt.savefig(image_save_path)
    print("Saved file to ", image_save_path)


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
      json_data = json.load(json_file)
      return json_data
  except json.decoder.JSONDecodeError as err:
    raise ValueError(f'An error occured while parsing {filename}.') from err
def plot_experiment_list(experiment_list):
    colors = iter(plt.cm.tab10.colors)
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

    print("x[0]:", result_data["x"][0])


    # Plot states
    velocity_ax.plot(t, v, c=color)
    height_ax.plot(t, h, c=color)

    # Plot delays
    pc_t, pc_delay = [], []
    for pc in result_data["pending_computations"]:
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