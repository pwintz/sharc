import os
import json
import numpy as np
import matplotlib.pyplot as plt
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
    experiment_results = readJson(incremental_path)
    results = [(val["experiment config"]["label"], val) for val in experiment_results.values()]

    # Plot results
    plt = plot_experiment_list(results)
    image_save_path = os.path.join(out_dir, 'car_plots.png')
    plt.savefig(image_save_path)
    print(f"✅ Saved figure to: {image_save_path}")


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
# Plotting
# --------------------------------------------------------------------------
def plot_experiment_list(experiment_list):
    colors = iter(plt.cm.tab10.colors)
    n_axs = 5
    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 15), sharex=True)
    pos_ax, vel_ax, psi_ax, delay_ax, control_ax = axs
    plts_for_legend = []

    for label, result in experiment_list:
        result_data = result["experiment data"]
        color = next(colors)
        plot_experiment_result(result_data, pos_ax, vel_ax, psi_ax, delay_ax, control_ax, color)
        # Invisible handle for legend
        line = vel_ax.plot(nan, nan, c=color)[0]
        plts_for_legend.append((line, label))

    # Axis formatting
    sample_time = experiment_list[0][1]["experiment config"]["system_parameters"]["sample_time"]
    t = np.array(experiment_list[0][1]["experiment data"]["t"])
    xlim = (0, t[-1])
    setup_axes(pos_ax, vel_ax, psi_ax, delay_ax, control_ax, xlim, sample_time)

    fig.legend(*zip(*plts_for_legend), loc='lower center', bbox_to_anchor=(0.5, 0.08),
               ncol=4, handlelength=1.5, handletextpad=0.5, borderaxespad=0.2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    return plt


def setup_axes(pos_ax, vel_ax, psi_ax, delay_ax, control_ax, xlim, sample_time):
    pos_ax.set(title='Position (X, Y)', ylabel='Position [m]', xlim=xlim)
    pos_ax.grid(True)

    vel_ax.set(title='Velocity', ylabel='Speed v [m/s]', xlim=xlim)
    vel_ax.grid(True)

    psi_ax.set(title='Heading Angle', ylabel='ψ [rad]', xlim=xlim)
    psi_ax.grid(True)

    delay_ax.set(title='Computation Delays', ylabel='Delay [s]', xlim=xlim)
    delay_ax.axhline(y=sample_time, color='black', linestyle='--', linewidth=2, label=f'Sample Time = {sample_time}s')
    delay_ax.grid(True)
    delay_ax.legend()

    control_ax.set(title='Control Inputs', xlabel='Time [s]', ylabel='[a, β]', xlim=xlim)
    control_ax.grid(True)


def plot_experiment_result(result_data, pos_ax, vel_ax, psi_ax, delay_ax, control_ax, color):
    t = np.array(result_data["t"])
    u = np.array(result_data["u"])           # (N, 2): [a, beta]
    x = np.array(result_data["x"])           # (N, 4): [x, y, v, psi]
    x_pos, y_pos, v, psi = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    # Debug print
    print(f"Loaded trajectory: {len(t)} steps")

    # Position over time (optional — could plot x vs y on separate fig)
    pos_ax.plot(t, x_pos, color=color, label='x')
    pos_ax.plot(t, y_pos, color=color, linestyle='--', label='y')

    vel_ax.plot(t, v, color=color)
    psi_ax.plot(t, psi, color=color)

    # Delays
    pc_t, pc_delay = [], []
    for pc in result_data.get("pending_computations", []):
        if pc:
            t_start = float(pc["t_start"])
            delay = float(pc["delay"])
            pc_t.extend([t_start, t_start + delay, np.nan])
            pc_delay.extend([delay, delay, np.nan])
    delay_ax.plot(pc_t, pc_delay, color=color)

    # Control inputs
    control_ax.plot(t, u[:, 0], color=color, label='a (accel)')
    control_ax.plot(t, u[:, 1], color=color, linestyle='--', label='β (slip)')
    control_ax.legend()


if __name__ == "__main__":
    main()
