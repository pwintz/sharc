#! /bin/env python3

import unittest
import sharc
from sharc import Simulation, BatchInit, Batch
import sharc.scarabizor as scarabizor
from sharc.data_types import *
import numpy as np
import copy
import os
import sharc.make_plots
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sharc.utils import readJson
from math import nan

def main():
  # eamples_dir = os.getenv("EXAMPLES_DIR")
  acc_example_dir = os.path.abspath('.')
  experiment_dir = os.path.join(acc_example_dir, "latest")
  assertFileExists(experiment_dir)
  experiment_incremental_results_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
  experiment_final_results_path = os.path.join(experiment_dir, "experiment_list_data.json")
  out_dir = os.path.join(experiment_dir, "images")
  os.makedirs(out_dir, exist_ok = True)
  # try:
  experiment_results = readJson(experiment_incremental_results_path)
  
  # sharc.make_plots.run(experiment_results, out_dir)

  results = []
  for key, val in experiment_results.items():
    results.append((val["experiment config"]["label"], val))
  plt = plot_experiment_list(results)

  image_save_path = os.path.join(experiment_dir + '/plots.png')
  plt.savefig(image_save_path)
  print("Saved file to ", image_save_path)

def plot_experiment_list(experiment_list):
    colors = iter(plt.cm.tab10.colors)
    
    n_axs = 4
    fig, axs = plt.subplots(n_axs, 1, figsize=(10, 13), sharex=True)
    plts_for_bottom_legend = []
    velocity_ax = axs[0]
    headway_ax  = axs[1]
    delay_ax    = axs[2]
    control_ax  = axs[3]
    for experiment in experiment_list:
      label = experiment[0]
      result = experiment[1]
      result_data = result["experiment data"]
      color = next(colors)
      try:
        plot_experiment_result(result_data, velocity_ax=velocity_ax, headway_ax=headway_ax, delay_ax=delay_ax, control_ax=control_ax, color=color)
      except Exception as err:
        raise Exception(f'Failed to plot {result["label"]}') from err
    
      # Plot invisible point to use in creating the legend.
      line = velocity_ax.plot(nan, nan, c=color)[0]
      plts_for_bottom_legend.append((line, label))
    
    sample_time = result["experiment config"]["system_parameters"]["sample_time"]
    d_min = result["experiment config"]["system_parameters"]["d_min"]
    w = np.array(result_data["w"])
    u = np.array(result_data["w"])
    v_front = w[:,0]
    t = np.array(result_data["t"])

    # Plot velocity of front vehicle.
    velocity_ax.plot(t, v_front, label="Front Velocity", color='black')
    # Plot sample time
    delay_ax.axhline(y=sample_time, color='black', linestyle='--', linewidth=2, label = f"Sample Time ({sample_time} s)")
    # Plot minimum headway.
    headway_ax.axhline(y=d_min, color='black', linestyle='--', linewidth=2, label = "Minimum Headway $d_{min}$")
    
    control_ax.plot(nan, nan, label="Acceleration Force", color="black", linestyle='-')
    control_ax.plot(nan, nan, label="Braking Force", color="black", linestyle=':')

    # Add legend entry for batch starts.
    batch_start_markers = velocity_ax.plot(nan, nan, 'kx', markersize=4)[0]
    plts_for_bottom_legend.append((batch_start_markers, "Batch start"))

    xlim = (0, result_data["t"][-1])

    # Velocity Axis
    velocity_ax.set(
          ylabel='Velocity $v$ [m/s]', 
          # yscale='log', 
          title='Velocity', 
          xlim=xlim, 
          ylim=(0, None)
        )
    velocity_ax.grid(True)
    velocity_ax.legend()

    # Headway Axis
    headway_ax.set(
          ylabel='Headway $h$ [m]', 
          # yscale='log', 
          title='Headway', 
          xlim=xlim,
          ylim=(0, None)
        )
    headway_ax.grid(True)
    headway_ax.legend()
    
    # Delays Axis
    delay_ax.set(
        ylabel='Delay $\\tau$ [s]', 
        title='Delays', 
        xlim=xlim, 
        # yscale='log', 
      #  ylim=(0,1.6)
    )
    delay_ax.grid(True)
    delay_ax.legend()
    
    # Control Axis
    control_ax.set(
        xlabel='Time [s]', 
        ylabel='Control Input $u$ [N]', 
        title='Control', 
        xlim=xlim, 
        # ylim = [-147, 147]
      )
    control_ax.grid(True)
    control_ax.legend()
    
    fig.legend(*zip(*plts_for_bottom_legend), 
              loc='lower center',
              bbox_to_anchor=(0.55, 0.12),
              ncol=4,
              columnspacing=1.0,
              handlelength=1.5,
              handletextpad=0.5,
              bbox_transform=fig.transFigure,
              borderaxespad=0.2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.23)
    # plt.show()
    return plt


def plot_experiment_result(result_data, velocity_ax, headway_ax, delay_ax, control_ax, color):
    t = np.array(result_data["t"])
    u = np.array(result_data["u"])
    x = np.array(result_data["x"])
    w = np.array(result_data["w"])
    h = x[:, 1]
    v = x[:, 2]
    # norm = np.sqrt(np.sum(np.array(x)[:, :2]**2, axis=1))

    velocity_ax.plot(t, v, label=None, c=color)
    headway_ax.plot(t, h, label=None, c=color)
    # plot_time_series(velocity_ax, v, t, name=None, color=color)
    # plot_time_series(headway_ax, h, t, name=None, color=color)
    
    pc_t, pc_delay = [], []
    for pc in result_data["pending_computations"]:
        if pc:
            t_start = float(pc["t_start"])
            delay = float(pc["delay"])
            pc_t.extend([t_start, t_start + delay, np.nan])
            pc_delay.extend([delay, delay, np.nan])
    delay_ax.plot(pc_t, pc_delay, color=color)
    
    control_ax.plot(t, u[:,0], color=color, linestyle='-')
    control_ax.plot(t, u[:,1], color=color, linestyle=':')

    # Plot batch starting points in the state plot.
    if result_data["batches"] is not None:
        for batch in result_data["batches"]:
            start_time = batch['valid_simulation_data']["t"][0]
            idx = np.where(t == start_time)[0][0]
            velocity_ax.plot(t[idx], v[idx], 'kx', markersize=4)
            headway_ax.plot(t[idx], h[idx], 'kx', markersize=4)
    

if __name__ == "__main__":
  main()
