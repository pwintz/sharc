#! /bin/env python3

import os
import argparse
import math
import json
from scarabintheloop.utils import assertFileExists, readJson, printJson
# Type hinting.
from typing import List, Set, Dict, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_x_vs_t(batches_axs, sim_data):
  x = sim_data['x']
  t = sim_data['t']
  x = np.array(x)
  # print(x)
  # print(t)

  # Plot each column in a separate subplot
  for i in range(x.shape[1]):
      batches_axs[i,0].plot(t, x[:, i], label=f"x_{i+1}")
      batches_axs[i,0].set_xlabel('Time')

def main():
  parser = argparse.ArgumentParser(prog='Generate Plots')
  parser.add_argument('experiment_path')           # positional argument
  args = parser.parse_args()
  experiment_path = os.path.abspath(args.experiment_path)
  assertFileExists(experiment_path)
  experiment_results = readJson(os.path.join(experiment_path, 'experiment_result_list.json'))
  experiment_labels = [result for result in experiment_results]
  print(experiment_labels)

  # Read the dimensions of the system.
  first_experiment = experiment_results[experiment_labels[0]]
  n = first_experiment["experiment config"]["system_parameters"]["state_dimension"]
  m = first_experiment["experiment config"]["system_parameters"]["input_dimension"]
  
  # Create a figure with subplots for each column
  n_plot_columns = 1 # len(experiment_labels)
  fig, batches_axs = plt.subplots(n, n_plot_columns, figsize=(10, 14))
  _, u_and_computations_axs = plt.subplots(2, n_plot_columns, figsize=(8, 8))
  batches_axs            =            batches_axs.reshape(n, n_plot_columns)
  u_and_computations_axs = u_and_computations_axs.reshape(2, n_plot_columns)
  batches_axs[0,0].set_title(f'State')
  x_names = first_experiment["experiment config"]["system_parameters"]['x names']
  

  # For now, we'll just plot the first experiment>>>----vvv
  for i_experiment, label in enumerate([experiment_labels[1]]):
    experiment = experiment_results[label]
    experiment_config = experiment["experiment config"]
    experiment_data = experiment["experiment data"]
    batches = experiment['experiment data']["batches"]
    
    # Check that the dimension of each system is equal.
    assert n == experiment_config["system_parameters"]["state_dimension"]
    assert m == experiment_config["system_parameters"]["input_dimension"]

    # print(batch.keys())
    batches_legend = []
    u_and_computations_legend = []
    for batch in batches:
      all_data_from_batch = batch['all_data_from_batch']
      plot_x_vs_t(batches_axs, all_data_from_batch)
      batches_legend += [f'Batch {batch["batch_init"]["i_batch"]}']
      u_and_computations_legend += [batches_legend[-1] + ' actual', batches_legend[-1] + ' pending']
      pending_computations = all_data_from_batch["pending_computation"]
      t = all_data_from_batch["t"]
      u = all_data_from_batch["u"]
      pc_t     = []
      pc_u     = []
      pc_delay = []
      if all_data_from_batch["pending_computation_prior"]:
        pending_computations = [all_data_from_batch["pending_computation_prior"]] + pending_computations
        # pc_prior = 
        # pc_t += [pc_prior["_t_start"], pc_prior["_t_end"]]
        # pc_u += [pc_prior["_u"], pc_prior["_u"]]
      for pc in pending_computations:
        if pc: 
          pc_t     += [float(pc["_t_start"]), float(pc["_t_end"]), np.nan]
          pc_u     += [float(pc["_u"]),       float(pc["_u"]),     np.nan]
          pc_delay += [float(pc["_delay"]),   float(pc["_delay"]), np.nan]
        else:
          pc_t     += [np.nan, np.nan]
          pc_u     += [np.nan, np.nan]
          pc_delay += [np.nan, np.nan]

      # pending_u = [pc["_u"] for pc in pending_computations]
      # pending_delay = [pc["_delay"] for pc in pending_computations]
      # t_start = [pc["_t_start"] for pc in pending_computations]

      u_line, = u_and_computations_axs[0,0].plot(t, u)
      u_and_computations_axs[0,0].set_title("Control for " + label)
      u_and_computations_axs[0,0].plot(pc_t, pc_u, linestyle=':', color=u_line.get_color(), label='Pending Control')
      u_and_computations_axs[1,0].plot(pc_t, pc_delay)
      u_and_computations_axs[1,0].set_title("Delays")
      # u_and_computations_axs[2].plot(pc_t, pc_u)
      # u_and_computations_axs[2].set_title("Pending Control")

    
  #   print(type(result))
  #   print(result)
    for i in range(n):
      batches_axs[i,0].legend(batches_legend)
      batches_axs[i,0].set_ylabel(x_names[i])
      
    u_and_computations_axs[0,0].legend(u_and_computations_legend, loc=(0.8, 0.50))
    u_and_computations_axs[1,0].legend(batches_legend)
    
  # for key in serial_scarab_experiment_data:
  #   print('key: ', key)
    # printJson('experiment data', experiment['experiment data'])

  # Convert data to numpy array for easier manipulation
 

  # # Plot each column as a separate line
  # for i in range(x.shape[1]):
  #     plt.plot(t, x[:, i], label=f"x_{i+1}")



#   plot_x_vs_t(batches_axs, "Serial Scarab", experiment_results)
#   plot_x_vs_t(batches_axs, "Parallel Scarab", experiment_results)
# 
#   
#   for i in range(n):
#       batches_axs[i].legend(["Parallel Scarab", "Serial Scarab"])

  # Save the plot to a file
  image_path = experiment_path + '/x.png'
  plt.savefig(image_path)
  print('image_path: ', image_path)

if __name__ == '__main__':
  main()