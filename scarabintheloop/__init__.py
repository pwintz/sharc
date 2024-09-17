from __future__ import print_function

import json
import os
import sys
import shutil
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import re
import time
import copy
import argparse
from warnings import warn
from pathlib import Path
import math
import traceback
import json
from contextlib import redirect_stdout
from scarabintheloop.utils import *


# Type hinting.
from typing import List, Set, Dict, Tuple
from typing import Union

from scarab_globals import *
from scarab_globals import scarab_paths

import scarabintheloop.scarabizor as scarabizor

from slugify import slugify

import scarabintheloop.plant_runner as plant_runner

######## Load the controller and plant modules from the current directory #########
root_dir = os.path.abspath(".")

controller_delegator = loadModuleFromWorkingDir("controller_delegator")
plant_dynamics = loadModuleFromWorkingDir("plant_dynamics")

# # Before this point, it is necessary to have added the example root to the Python path.
# import controller_delegator # Import the particular "controller_delegator" from the example directory.
# import plant_dynamics # Import the particular "plant_dynamics" from the example directory.

# Define default debugging levels.
DEBUG_INTERFILE_COMMUNICATION_LEVEL = 0
DEBUG_OPTIMIZER_STATS_LEVEL = 0
DEBUG_DYNAMICS_LEVEL = 0
DEBUG_CONFIGURATION_LEVEL = 0
DEBUG_BUILD_LEVEL = 0

# Add the example's scripts directory to the PATH.
# os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/scripts")
# os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/bin")

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]

# Regex to find text in the form of "--name value"
param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

# Format string to generate text in the form of "--name value"
param_str_fmt = "--{}\t{}\n"

def run(example_dir):
  
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Run a set of examples.")

  parser.add_argument(
      '--config_filename',
      type=str,  # Specify the data type
      default="default.json",
      help="Select the name of a JSON file located in <example_dir>/simulation_configs."
  )

  # Parse the arguments from the command line
  args = parser.parse_args()
  experiments_config_file_path = Path(os.path.abspath(os.path.join(example_dir, "simulation_configs", args.config_filename)))
  base_config_file_path = Path(os.path.abspath(os.path.join(example_dir, 'base_config.json')))

  # If the user gave an output directory, then use it. Otherwise, use "experiments/" within the example folder.
  experiments_dir = os.path.abspath(example_dir) + "/experiments"

  # Read JSON configuration file.
  base_config = readJson(base_config_file_path)

  # Open list of example configurations.
  experiment_config_patches_list = readJson(experiments_config_file_path)
  
  # The experiment config file can either contain a single configuration or a list.
  # In the case where there is only a single configuration, we convert it to a singleton list. 
  if not isinstance(experiment_config_patches_list, list):
    experiment_config_patches_list = [experiment_config_patches_list]

  experiment_list_label = slugify(experiments_config_file_path.stem)
  base_config["experiment_list_label"] = experiment_list_label

  debug_levels = base_config["==== Debgugging Levels ===="]
  DEBUG_INTERFILE_COMMUNICATION_LEVEL = debug_levels["debug_interfile_communication_level"]
  DEBUG_OPTIMIZER_STATS_LEVEL = debug_levels["debug_optimizer_stats_level"]
  DEBUG_DYNAMICS_LEVEL = debug_levels["debug_dynamics_level"]
  DEBUG_CONFIGURATION_LEVEL = debug_levels["debug_configuration_level"]
  DEBUG_BUILD_LEVEL = debug_levels["debug_build_level"]

  experiment_list_dir = experiments_dir + "/" + slugify(experiments_config_file_path.stem) + "--" + time.strftime("%Y-%m-%d--%H-%M-%S") + "/"
  print(f"experiment_list_dir: {experiment_list_dir}")

  # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
  os.makedirs(experiment_list_dir)

  # Create or update the symlink to the latest folder.
  latest_experiment_list_dir_symlink_path = experiments_dir + "/latest"
  if os.path.exists(latest_experiment_list_dir_symlink_path):
    os.remove(latest_experiment_list_dir_symlink_path)
  os.symlink(experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)

  if DEBUG_CONFIGURATION_LEVEL >= 1:
    printJson(f"base_config loaded from {base_config_file_path}", base_config)
    printJson(f"experiment_config_patches_list loaded from {experiments_config_file_path}", experiment_config_patches_list)

  experiment_result_list = {}
  for experiment_config_patches in experiment_config_patches_list:
    os.chdir(experiment_list_dir)
    experiment_result = run_experiment(base_config, experiment_config_patches, example_dir)
    if experiment_result:
      # printJson("experiment_result", experiment_result)
      experiment_result_list[experiment_result["label"]] = experiment_result
      writeJson(experiment_list_dir + "experiment_result_list_incremental.json", experiment_result_list)
      print(f'Added "{experiment_result["label"]}" to list of experiment results')
    else:
      print(f'No result for "{experiment_config_patches["label"]}" experiment.')
  
  experiment_list_json_filename = experiment_list_dir + "experiment_result_list.json"
  print(f"Finished {len(experiment_result_list)} experiments. Output in {experiment_list_json_filename}.")
  writeJson(experiment_list_json_filename, experiment_result_list)

@indented_print
def run_experiment(base_config: dict, experiment_config_patches: dict, example_dir: str):
  """ This function assumes that the working directory is the experiment-list working directory and its parent directory is the example directory, which contains the following:
    - scripts/controller_delegator.py
    - chip_configs/PARAMS.base (and other optional alternative configuration files)
    - simulation_configs/default.json (and other optional alternative configuration files)
  """

  # Load human-readable labels.
  experiment_list_label = base_config["experiment_list_label"]
  experiment_label = experiment_list_label + "/" + slugify(experiment_config_patches["label"])
  
  # Record the start time.
  experiment_start_time = time.time()

  if experiment_config_patches.get("skip", False):
    printHeader1(f'Skipping Experiment: {experiment_label}')
    return
  else:
    printHeader1("Starting Experiment: {experiment_label}")

  example_dir = os.path.abspath('../..') # The "root" of this project.
  experiment_list_dir = os.path.abspath('.') # A path like <example_dir>/experiments/default-2024-08-25-16:17:35
  experiment_dir = os.path.abspath(experiment_list_dir + "/" + slugify(experiment_config_patches['label']))
  simulation_config_path = example_dir + '/simulation_configs'
  chip_configs_path = example_dir + '/chip_configs'

  print("        Example dir: " + example_dir)
  print("Experiment list dir: " + experiment_list_dir)
  print("     Experiment dir: " + experiment_dir)

  # Check that the expected folders exist.
  assertFileExists(chip_configs_path)
  assertFileExists(simulation_config_path)

  ########### Populate the experiment directory ############
  # Make subdirectory for this experiment.
  os.makedirs(experiment_dir)
  os.chdir(experiment_dir)

  # Using the base configuration dictionary (from JSON file), update the values given in the experiment config patch, loaded from
  experiment_config = patch_dictionary(base_config, experiment_config_patches)
  experiment_config["experiment_label"] = experiment_label
  experiment_config["experiment_dir"] = experiment_dir

  if DEBUG_CONFIGURATION_LEVEL >= 2:
    printJson("Experiment configuration", experiment_config)

  # Write the configuration to config.json.
  writeJson("config.json", experiment_config)

  # Create PARAMS.generated file containing chip parameters for Scarab.
  PARAMS_src = chip_configs_path + "/" + experiment_config["PARAMS_base_file"] 
  PARAMS_out = os.path.abspath('PARAMS.generated')
  assertFileExists(PARAMS_src)
  create_patched_PARAMS_file(experiment_config, PARAMS_src, PARAMS_out)
  assertFileExists(PARAMS_out)
  
  if experiment_config["Simulation Options"]["parallel_scarab_simulation"]:
    experiment_data = run_experiment_parallelized(experiment_config, example_dir)
  else:
    experiment_data = run_experiment_sequential(experiment_config, example_dir)

  experiment_end_time = time.time()
  experiment_result = {
    "label": experiment_config["label"],
    "experiment directory": experiment_dir,
    "experiment data": experiment_data,
    "experiment config": experiment_config, 
    "experiment wall time": experiment_start_time - experiment_end_time
  }
  
  writeJson("experiment_result.json", experiment_result)
  return experiment_result
    

def computeTimeStepsDelayed(sample_time: float, t_delay: List[float]):
  """
  Given the sample time <sample_time> and an array of computation times <t_delay>, 
  generate an array with the same length as t_delay that indicates how many time steps control calulations delay updating u. 
  The first entry of t_delay must be None and the coresponding entry in the output is 0.
  """
  time_steps_delayed = t_delay.copy()
  # if t_delay[0] is not None:
    # raise ValueError(f"The first entryf time_steps_delayed must be None. Instead it was {time_steps_delayed[0]}!")
  # time_steps_delayed[0] = 0
  for i in range(len(time_steps_delayed)):
    if time_steps_delayed[i] is None:
      raise ValueError(f"Entry {i} of time_steps_delayed was None. Only the first entry can be None!")
    time_steps_delayed[i] = math.ceil(t_delay[i] / sample_time)
  return time_steps_delayed

def findFirstExcessiveComputationDelay(time_steps_delayed):  
  for (i, steps_delayed) in enumerate(time_steps_delayed):
    if steps_delayed > 1:
      return (i, steps_delayed)
  return None, None

def processBatchSimulationData(batch_config: dict, batch_init:dict, batch_simulation_data: dict):
  checkBatchConfig(batch_config)
  checkSimulationData(batch_simulation_data, batch_config["max_time_steps"])

  # x: The states of the system at each t(0), t(1), etc. 
  # u: The control values applied. The first entry is the u0 previously computed. Subsequent entries are the computed values. The control u[i] is applied from t[i] to t[i+1], with u[-1] not applied during this batch. 
  x = batch_simulation_data["x"]
  u = batch_simulation_data["u"]
  t = batch_simulation_data["t"]
  t_delay = batch_simulation_data["t_delay"]

  sample_time = batch_config["system_parameters"]["sample_time"]
  time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
  first_excessive_delay_local_index, first_excessive_delay_n_of_time_steps = findFirstExcessiveComputationDelay(time_steps_delayed)
  has_missed_computation = first_excessive_delay_local_index is not None

  indices = list(range(batch_config["first_time_index"], batch_config["last_time_index"] + 1))

  print(f'indices: {indices}, batch_config["first_time_index"]: {batch_config["first_time_index"]}, batch_config["last_time_index"]: {batch_config["last_time_index"]}')

  # The values of data from the batch, including values after missed computations.
  all_data_from_batch = {
    "x": x,
    "u": u,
    "t": t, 
    "t_delay": t_delay,
    "first_time_index": batch_config["first_time_index"],
    "last_time_index": batch_config["last_time_index"], 
    "time_steps_delayed": time_steps_delayed,
    "time_indices": indices
  }

  if has_missed_computation:
    first_excessive_delay_time_index = batch_config["first_time_index"] + first_excessive_delay_local_index
    print(f'time_steps_delayed: {time_steps_delayed}')
    print(f'first_time_index: {batch_config["first_time_index"]}')
    print(f'first_excessive_delay_local_index: {first_excessive_delay_local_index}')
    print(f'first_excessive_delay_time_index: {first_excessive_delay_time_index}')
    valid_data_from_batch = {
      "x":             x[0:first_excessive_delay_local_index + 1],
      "u":             u[0:first_excessive_delay_local_index + 1],
      "t":             t[0:first_excessive_delay_local_index + 1], 
      "t_delay": t_delay[0:first_excessive_delay_local_index + 1],
      "time_steps_delayed": time_steps_delayed[0:first_excessive_delay_local_index + 1],
      "first_time_index": batch_config["first_time_index"],
      "last_time_index": first_excessive_delay_time_index
    }

    # Values that are needed to start the next batch.
    next_batch_init = {
      "i_batch": batch_init["i_batch"] + 1,
      "first_time_index": first_excessive_delay_time_index + 1,
      "x0": x[first_excessive_delay_local_index],
      # When there is a missed computation, we continue using u from the prior timestep.
      "u0": u[first_excessive_delay_local_index-1],
      # Record the value of u that is computed late so that the next batch can apply it at the approriate time
      "u_pending": u[first_excessive_delay_local_index],
      "u_pending_time_index": first_excessive_delay_time_index  + first_excessive_delay_n_of_time_steps
    }
  else: # -> No missed computations
    valid_data_from_batch = copy.deepcopy(all_data_from_batch)

    # Values that are needed to start the next batch.
    next_batch_init = {
      "i_batch": batch_init["i_batch"] + 1,
      "first_time_index": batch_config["last_time_index"] + 1,
      "x0": x[-1],
      "u0": u[-1],
      "u_pending": None,
      "u_pending_time_index": None
    }

  # Check that the next_first_time_index doesn't skip over any timesteps.
  if next_batch_init['first_time_index'] > all_data_from_batch['last_time_index'] + 1:
    raise ValueError(F'next first_time_index={next_batch_init["first_time_index"]} > last_time_index_in_batch + 1={all_data_from_batch["last_time_index"] + 1}')

  return {
    "all_data_from_batch": all_data_from_batch, 
    "valid_data_from_batch": valid_data_from_batch, 
    "next_batch_init": next_batch_init
  }

def run_experiment_sequential(experiment_config, example_dir):
  print(f'Running "{experiment_config["experiment_label"]}" sequentially')
  sim_dir = experiment_config["experiment_dir"]
  sim_config = copy.deepcopy(experiment_config)
  sim_config["simulation_label"] = experiment_config["experiment_label"]
  sim_config["simulation_dir"] = experiment_config["experiment_label"]
  simulation_data = runSimulation(sim_config)
  return simulation_data

def run_experiment_parallelized(experiment_config, example_dir):
  print(f'Running "{experiment_config["experiment_label"]}" in parallel')
  
  experiment_dir = experiment_config["experiment_dir"]
  max_time_steps = experiment_config["max_time_steps"]
  max_batch = experiment_config["Simulation Options"]["max_batches"]
  # first_time_index_in_batch = 0
  # i_batch = 0
  # x0 = experiment_config["x0"]
  # u0 = experiment_config["u0"]
  # u_pending = None
  # u_pending_time_index = None
  # sample_time = experiment_config["system_parameters"]["sample_time"] 

  # Create lists for recording the true values (discarding values that erroneously use  missed computations as though were not missed).
  batch_data_list = []
  xs_actual = [experiment_config["x0"]]
  us_actual = [experiment_config["u0"]]
  ts_actual = [0]
  computation_times_actual = [None]

  batch_init = {
    "i_batch": 0,
    "first_time_index": 0,
    "x0": experiment_config["x0"],
    "u0": experiment_config["u0"],
    # There are no pending (previouslly computed) control values because we are at the start of the experiment.
    "u_pending": None,
    "u_pending_time_index": None
  }

  ### Batches Loop ###
  # Loop through batches of time-steps until the start time-step is no longer less than the total number of time-steps desired or until the max number of batches is reached.
  while batch_init["first_time_index"] < max_time_steps and batch_init["i_batch"] < max_batch:
    
    batch_sim_config = create_batch_sim_config(experiment_config, batch_init)

    printHeader2( f'Starting batch: {batch_sim_config["simulation_label"]} ({batch_sim_config["max_time_steps"]} time steps)')
    print(f'↳ dir: {batch_sim_config["simulation_dir"]}')
    print(f'↳                  x0: {batch_sim_config["x0"]}')
    print(f'↳                  u0: {batch_sim_config["u0"]}')
    print(f'↳      "time_indices": {batch_sim_config["time_indices"]}')

    # Run simulation
    batch_sim_data = runSimulation(batch_sim_config)
    batch_data = processBatchSimulationData(batch_sim_config, batch_init, batch_sim_data)
    batch_data['label'] = batch_sim_config["simulation_label"]
    batch_data['batch directory'] = batch_sim_config["simulation_label"]
    batch_data['config'] = batch_sim_config
    batch_data_list.append(batch_data)

    batch_init = batch_data["next_batch_init"]

    # Check that the next_first_time_index doesn't skip over any timesteps.
    if batch_init['first_time_index'] > batch_sim_config["last_time_index"] + 1:
      raise ValueError(F'next first_time_index={batch_init["first_time_index"]} > last_time_index_in_batch + 1={last_time_index_in_batch + 1}')

    # Append all of the valid data (up to the point of he missed computation) except for the first index, which overlaps with the last index of the previous batch.
    xs_actual                += batch_data["valid_data_from_batch"]["x"][1:]
    us_actual                += batch_data["valid_data_from_batch"]["u"][1:]
    ts_actual                += batch_data["valid_data_from_batch"]["t"][1:]
    computation_times_actual += batch_data["valid_data_from_batch"]["t_delay"][1:]

    all_batch_data = batch_data["all_data_from_batch"]
    batch_x = all_batch_data["x"]
    batch_u = all_batch_data["u"]
    batch_t = all_batch_data["t"]

    # if batch_u[0] != batch_init["u0"]:
    #   print("batch_u" + str(batch_u[0]))
    #   raise ValueError(f'batch_u[0] = {batch_u[0]} != batch_init["u0"] = {batch_init["u0"]}')
      
    if batch_u[0] != batch_sim_config["u0"]:
      raise ValueError(f'batch_u[0] = {batch_u[0]} != batch_sim_config["u0"] = {batch_sim_config["u0"]}')

    printHeader2(f'Finished batch {batch_init["i_batch"]}: {batch_sim_config["simulation_label"]}')
    # print(f'↳ dir: {batch_dir}')
    print(f'↳           length of "x": {len(batch_x)}')
    print(f'↳           length of "u": {len(batch_u)}')
    print(f'↳                    "u0": {batch_u[0]}')
    print(f'↳           length of "t": {len(batch_t)}')
    print(f'↳               "t_delay": {all_batch_data["t_delay"]}')
    print(f'↳      "first_time_index": {all_batch_data["first_time_index"]}')
    print(f'↳       "last_time_index": {all_batch_data["last_time_index"]}')
    print(f'↳    "time_steps_delayed": {all_batch_data["time_steps_delayed"]}')
    print(f'↳          "time_indices": {all_batch_data["time_indices"]}')
    print(f'↳          "max_time_steps": {batch_sim_config["max_time_steps"]}')
    print(f'↳ Next "first_time_index": {batch_data["next_batch_init"]["first_time_index"]}')
    print('<<< Batch Time grid >>>')

    experiment_data = {"batches": batch_data_list,
                      "x": xs_actual,
                      "u": us_actual,
                      "t": ts_actual,
                      "computation_times": computation_times_actual,
                      "config": experiment_config}
                      
    writeJson(experiment_dir + "/experiment_data_incremental.json", experiment_data, label="Incremental experiment data")

  # Print out all of the batches!  
  printHeader2('                              All Batches                              ')
  for batch_data in batch_data_list:
    batch_init = batch_data["next_batch_init"]
    all_batch_data = batch_data["all_data_from_batch"]
    batch_x = all_batch_data["x"]
    batch_u = all_batch_data["u"]
    batch_t = all_batch_data["t"]
    def print_sample_time_values(label: str, values: list):
      print(f"{label:>18}: " +  " ------- ".join(f"{v:^7.3g}" for v in values))

    def print_time_step_values(label: str, values: list):
      print(f"{label:>18}:    |    " +  "    |    ".join(f"{v:^7.3g}" for v in values))

    # print(f"             u0: {u0[0]}")
    print_sample_time_values("k", all_batch_data["time_indices"])
    print_time_step_values("time step", range(1, batch_sim_config["max_time_steps"]+1))
    print_sample_time_values("t(k)", batch_t)
    print_sample_time_values("x1(k)", [x[0] for x in batch_x])
    # print_sample_time_values("x2(k)", [x[1] for x in batch_x])
    # print_sample_time_values("x3(k)", [x[2] for x in batch_x])
    print_sample_time_values("x4(k)", [x[3] for x in batch_x])
    print_sample_time_values("u([k, k+1))", [u[0] for u in batch_u])
    print_sample_time_values("t_delay", all_batch_data["t_delay"])
    print_sample_time_values("samples delayed", all_batch_data["time_steps_delayed"])
    print('------------------')


  writeJson(experiment_dir + "/experiment_data.json", experiment_data, label="Experiment data")
  return experiment_data


def create_batch_sim_config(experiment_config, batch_init):
  max_time_steps_per_batch = os.cpu_count()
  
  # Add "max_time_steps_per_batch" steps to the first index to get the last index.
  last_time_index_in_batch = batch_init["first_time_index"] + max_time_steps_per_batch

  # Truncate the index to not extend past the last index
  last_time_index_in_experiment = experiment_config["max_time_steps"]
  last_time_index_in_batch = min(last_time_index_in_batch, last_time_index_in_experiment)

  max_time_steps_per_batch = last_time_index_in_batch - batch_init["first_time_index"]

  batch_label = f'batch{batch_init["i_batch"]}_steps{batch_init["first_time_index"]}-{last_time_index_in_batch}'

  batch_config = copy.deepcopy(experiment_config)
  
  batch_config["simulation_label"] = experiment_config["experiment_label"] + "/" + batch_label
  batch_config["max_time_steps"]   = max_time_steps_per_batch
  batch_config["simulation_dir"]   = experiment_config["experiment_dir"] + "/" + batch_label
  batch_config["x0"]               = batch_init["x0"]
  batch_config["u0"]               = batch_init["u0"]
  batch_config["first_time_index"] = batch_init["first_time_index"]
  batch_config["last_time_index"]  = last_time_index_in_batch
  batch_config["time_indices"]  = list(range(batch_config["first_time_index"], batch_config["last_time_index"]+1))

  # Check that batch data and then write it to "config.json" so it is available in the batch directory with any updated values for the controller to read. 
  checkBatchConfig(batch_config)
  return batch_config


@indented_print
def runSimulation(sim_config: dict):
  sim_dir = sim_config["simulation_dir"]
  simulation_start_time = time.time()

  os.makedirs(sim_dir)
  shutil.copyfile(sim_config["experiment_dir"] + "/PARAMS.generated", sim_dir + "/PARAMS.generated")

  writeJson(sim_config["simulation_dir"] + "/config.json", sim_config)

  os.chdir(sim_dir)

  # Get human-readable label for this simulation.
  simulation_label = sim_config["simulation_label"]

  printHeader2(f"Starting simulation: {simulation_label}")
  print(f'↳ Simulation dir: {sim_dir}')

  chip_params_path = sim_dir + '/PARAMS.generated'
  assertFileExists(chip_params_path)  

  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]
  
  # Open up some loooooooooogs.
  controller_log_path = sim_dir + '/controller.log'
  plant_log_path = sim_dir + '/plant_dynamics.log'
  with openLog(plant_log_path, f'Plant Log for "{simulation_label}"') as plant_log, \
       openLog(controller_log_path, f'Controller Log for "{simulation_label}"') as controller_log:

    simulation_executor = getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)

    # Execute the simulation.
    simulation_data = simulation_executor.run_simulation()

    printHeader2(f'Finished Simulation: "{simulation_label}"')
    print(f'↳ Simulation dir: {sim_dir}')
    print(f'↳ Controller log: {controller_log_path}')
    print(f'↳      Plant log: {plant_log_path}')
    print(f"↳  Data out file: {os.path.abspath('simulation_data.json')}")
    
    simulation_end_time = time.time()
    simulation_data["simulation wall time"] = simulation_start_time - simulation_end_time
    checkSimulationData(simulation_data, sim_config["max_time_steps"])
    return simulation_data

def create_patched_PARAMS_file(sim_config: dict, PARAMS_src_filename: str, PARAMS_out_filename: str):
  """
  Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
  Then, modify the values for keys listed in PARAMS_file_keys to values taken from sim_config. 
  Write the resulting PARAMS data to a file at PARAMS_out_filename in the simulation directory (sim_dir).
  Returns the absolute path to the PARAMS file.
  """
  print(f'Creating chip parameter file.')
  print(f'\tSource: {PARAMS_src_filename}.')
  print(f'\tOutput: {PARAMS_out_filename}.')
  with open(PARAMS_src_filename) as params_out_file:
    PARAM_file_lines = params_out_file.readlines()
  
  for (key, value) in sim_config.items():
    if key in PARAMS_file_keys:
      PARAM_file_lines = changeParamsValue(PARAM_file_lines, key, value)

  # Create PARAMS file with the values from the base file modified
  # based on the values in sim_config.
  with open(PARAMS_out_filename, 'w') as params_out_file:
    params_out_file.writelines(PARAM_file_lines)


# TODO: Write tests for this.
def changeParamsValue(PARAM_lines: List[str], key, value):
  """
  Search through PARAM_lines and modify it in-place by updating the value for the given key. 
  If the key is not found, then an error is raised.
  """
  for line_num, line in enumerate(PARAM_lines):
    regex_match = param_regex_pattern.match(line)
    if not regex_match:
      continue
    if key == regex_match.groupdict()['param_name']:
      # If the regex matches, then we replace the line.
      new_line_text = param_str_fmt.format(key, value)
      PARAM_lines[line_num] = new_line_text
      # print(f"Replaced line number {line_num} with {new_line_text}")
      return PARAM_lines
    
  # After looping through all of the lines, we didn't find the key.
  raise ValueError(f"Key \"{key}\" was not found in PARAM file lines.")


# def get_scarab_in_parallel_cmd(exectuable_path:str, chip_params_path:str, config_json:dict, start_index:int=0, time_horizon:int=10, experiment_name:str="experiment", PARAM_index:str=0, sim_config_path=""):
#   # start_idx=$1 # Typically "0". We can set larger values to start 'halfway' starting midway from a prior experiment.
#   print(f"simulation_executable={exectuable_path}") # The path the executable (already ).
#   print(f"chip_params_path={chip_params_path}") # E.v., "chip_params/raspberry_pi_5"
#   control_sampling_time = config_json["system_parameters"]["sample_time"]
#   total_time_steps = math.ceil(time_horizon / control_sampling_time)
#   if total_time_steps <= 0:
#     raise ValueError(f"total_time_steps={total_time_steps} was not positive!")
#   print(f"total_time_steps={total_time_steps}")
#   cmd = [f'run_in_parallel.sh', 
#             f'{start_index}',            # 1
#             f'{time_horizon}',           # 2
#             f'{chip_params_path}',       # 3
#             f'{control_sampling_time}',  # 4
#             f'{total_time_steps}',        # 5
#             f'{experiment_name}',        # 6
#             f'{0}',                      # 7
#             f'{exectuable_path}',        # 8
#             f'{sim_config_path}',   # 9
#           ] # 8
#   return cmd
#   # print(f"config_json={config_json}") # E.v., "controller_parameters/params.json"
#   # print(f"time_horizon={time_horizon}") # How many timesteps to simulate, in total.
#   # print(f"exp_name={exp_name}") # Define a name for the folders, which are generated.
#   # print(f"PARAM_INDEX={PARAM_INDEX}") # Allows selecting from different PARAMS.generated files from an array of 
 
  
def getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log):
  """ 
  This function implements the "Factory" design pattern, where it returns objects of various classes depending on the imputs.
  """

  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]

  fake_computation_time_simulation = sim_config["Simulation Options"]["use_fake_scarab_computation_times"]

  if use_parallel_scarab_simulation:
    print("Using ParallelSimulationExecutor.")
    return ParallelSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)
  elif fake_computation_time_simulation:
    return ModeledDelaysSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)
  else:
    return SerialSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)


class SimulationExecutor:
  
  def __init__(self, sim_dir, sim_config, controller_log, plant_log):
    self.sim_dir = sim_dir
    self.sim_config = sim_config
    self.controller_log = controller_log
    self.plant_log = plant_log

    # Create the executable.
    with redirect_stdout(controller_log):
      self.controller_executable = controller_delegator.get_controller_executable(sim_config)
    assertFileExists(self.controller_executable)
    
    # Get a function that defines the plant dynamics.
    with redirect_stdout(plant_log):
      self.evolveState_fnc = plant_dynamics.getDynamicsFunction(sim_config)

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log, working_dir=self.sim_dir)
    assertFileExists(sim_dir + "/x_c++_to_py")

  def run_controller(self):
    raise RuntimeError("This function must be implemented by a subclass")

  def _run_controller(self):
    """
    Run the controller via the run_controller() function defined in subclasses, but include some setup and tear-down beforehand.
    """
    print(f'---- Starting controller for {self.sim_config["simulation_label"]} ----\n'\
          f'      Executable: {self.controller_executable} \n'\
          f'  Simulation dir: {self.sim_dir} \n'\
          f'  Controller log: {self.controller_log.name}')
    assertFileExists('config.json') # config.json is read by the controller.
    try:
      self.run_controller()
    except Exception as err:
      print(f'Error: Running the controller failed!')
      print(f"Error: {str(err)}")
      if hasattr(err, 'output') and err.output:
        print(f'Error output: {err.output}')
      print(traceback.format_exc())
      raise err
    print('Controller finished.')

  def _run_plant(self):
    """ 
    Handle setup and exception handling for running the run_plant() functions implemented in subclasses
    """
    print(f'---- Starting plant dynamics for {self.sim_config["simulation_label"]} ----\n' \
          f'\t↳ Simulation dir: {self.sim_dir} \n' \
          f'\t↳      Plant log: {self.plant_log.name}\n')
    self.plant_log.write('Start of thread.\n')
    if DEBUG_CONFIGURATION_LEVEL >= 2:
      printJson(f"Simulation configuration", self.sim_config)

    try:
      with redirect_stdout(self.plant_log):
        plant_result = plant_runner.run(self.sim_dir, self.sim_config, self.evolveState_fnc)
        # assertFileExists(self.sim_dir + '/simulation_data.json')

    except Exception as e:
      print(f'Plant dynamics had an error: {e}')
      print(traceback.format_exc())
      raise e

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if DEBUG_DYNAMICS_LEVEL >= 1:
      # print('-- Plant dynamics finshed -- \n' \
      #       f'\t↳ Simulation directory: {self.sim_dir} \n' \
      #       f'\t↳      Simulation data: {self.sim_dir}/simulation_data.json')
      print('-- Plant dynamics finshed -- \n' \
            f'\t↳ Simulation directory: {self.sim_dir}')
    return plant_result

  def run_simulation(self):
    """ Run everything needed to simulate the plant and dynamics, in parallel."""
    
    # Run plant and controller in parallel, on separate threads.
    N_TASKS = 2
    with ThreadPoolExecutor(max_workers=N_TASKS) as executor:
      # Create two tasks: One for the controller and another for the plant.
      tasks = [executor.submit(self._run_controller), executor.submit(self._run_plant)]
      for future in concurrent.futures.as_completed(tasks):
        if future.exception(): # Waits until finished or failed.
          print(f"Task {future} failed with exception: {future.exception()}")
          raise future.exception()

        # Get the result returned by the _run_plant task.
        simulation_data = tasks[1].result()

    # Do optional postprocessing (no-op if postprocess_simulation_data is not overridden in subclass)
    simulation_data = self.postprocess_simulation_data(simulation_data)

    # Check to make sure the simulation generated data that is well-formed.
    checkSimulationData(simulation_data, self.sim_config["max_time_steps"])
    return simulation_data

  def postprocess_simulation_data(self, simulation_data):
    """ 
    This function can be overridden in subclasses to allow modifications to the simulation data.
    """
    return simulation_data

class ModeledDelaysSimulationExecutor(SimulationExecutor):
  """
  Create a simulation executor that uses a model of the computation delays instead of simulating the controller using Scarab.
  """
  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

class SerialSimulationExecutor(SimulationExecutor):

  def run_controller(self):
    # Run Scarb using execution-driven mode.
    cmd = " ".join([self.controller_executable])
    scarab_cmd_argv = [sys.executable, # The Python executable
                      scarab_paths.bin_dir + '/scarab_launch.py',
                      '--program', cmd,
                      '--param', 'PARAMS.generated',
                      # '--simdir', '.', # Changes the working directory.
                      '--pintool_args',
                      '-fast_forward_to_start_inst 1',
                      '--scarab_args',
                      '--inst_limit 15000000000' # Instruction limit
                      #  '--heartbeat_interval 50', 
                      # '--num_heartbeats 1'
                      # '--power_intf_on 1']
                    ]
    # There is a bug in Scarab that causes it to crash when run in the terminal 
    # when the terminal is resized. To avoid this bug, we run it in a different thread, which appears to fix the problem.
    with ThreadPoolExecutor() as executor:
      future = executor.submit(lambda: run_shell_cmd(scarab_cmd_argv, working_dir=self.sim_dir, log=self.controller_log))

      # Wait for the background task to finish before exiting
      future.result()


class ParallelSimulationExecutor(SimulationExecutor):

  def __init__(self, sim_dir, sim_config, controller_log, plant_log):
    super().__init__(sim_dir, sim_config, controller_log, plant_log)
    self.use_fake_scarab_computation_times = sim_config["Simulation Options"]["use_fake_scarab_computation_times"]

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def postprocess_simulation_data(self, simulation_data):
    """ 
    Runnining the controller exectuable produces DynamoRIO traces but does not give use simulated computation times. In this function, we simulate the traces using Scarab to get the computation times.
    """
    # printJson('simulation_data', simulation_data)
    # Time-steps are the intervals between sampls of x, so there is one less than the number of entries in x.
    max_time_steps = len(simulation_data['x']) - 1 
    # Create a list of the correct length filled with None's.
    computation_time_list = [None] * max_time_steps
    if self.use_fake_scarab_computation_times:
      for i in range(max_time_steps):
        computation_time_list[i] = 0.1 * (i + 1)
    else: 
      (dynamrio_trace_dir_dictionaries, trace_dirs, sorted_indices) = ParallelSimulationExecutor.get_dynamorio_trace_directories()

      # Portabilize the trace files (in parallel) and then simulate with Scarab (in parallel).
      print(f"Starting portabilization of trace directories {trace_dirs} in {self.sim_dir}.")
      with ProcessPoolExecutor(max_workers = os.cpu_count()) as portabalize_executor:
        portabalize_executor.map(ParallelSimulationExecutor.portabalize_trace, trace_dirs)

      print(f"Finished portabilizing traces for each directory in {dynamrio_trace_dir_dictionaries.values()} in {self.sim_dir}.")
      with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
        scarab_data = scarab_executor.map(ParallelSimulationExecutor.simulate_trace_in_scarab, sorted_indices)
        for datum in scarab_data:
          index = datum["index"]
          trace_dir = datum["trace_dir"]
          computation_time = datum["computation_time"]
          computation_time_list[index] = computation_time
    print(f"Finished executing parallel simulations, computation_time_list: {computation_time_list}")

    # simulation_data["t_delay"] = [None] + computation_time_list
    simulation_data["t_delay"] = computation_time_list
    # print(f'simulation_data["t_delay"]: {simulation_data["t_delay"]}')
    return simulation_data
    
  @staticmethod
  def get_dynamorio_trace_directories():
    trace_dir_regex = re.compile(r"dynamorio_trace_(\d+)")
    dynamorio_trace_dirs = glob.glob('dynamorio_trace_*')

    trace_dir_dict = {int(trace_dir_regex.search(dir).group(1)): dir for dir in dynamorio_trace_dirs if trace_dir_regex.match(dir)}
    
    if len(trace_dir_dict) == 0:
      raise ValueError(f"No dynamorio trace directories were found in {os.getcwd()}")

    sorted_indices = sorted(trace_dir_dict)
    sorted_trace_dirs = [trace_dir_dict[dir_ndx] for dir_ndx in sorted(trace_dir_dict)]
    return (trace_dir_dict, sorted_trace_dirs, sorted_indices)

  @staticmethod 
  def portabalize_trace(trace_dir): 
    print(f"Starting portabilization of trace in {trace_dir}.")
    log_path = trace_dir + "/portabilize.log"
    with openLog(log_path, f"Portablize \"{trace_dir}\"") as portabilize_log:
      try:
          run_shell_cmd("run_portabilize_trace.sh", working_dir=trace_dir, log=portabilize_log)
      except Exception as e:
        raise e
    print(f"Finished portabilization of trace in {trace_dir}.")

  @staticmethod
  def simulate_trace_in_scarab(dir_index:int) -> dict:
    trace_dir = f"dynamorio_trace_{dir_index}"

    shutil.copyfile('PARAMS.generated', trace_dir + '/PARAMS.in')

    bin_path = trace_dir + '/bin'
    trace_path = trace_dir + '/trace'
    assertFileExists(trace_dir + '/PARAMS.in')
    assertFileExists(bin_path)
    assertFileExists(trace_path)

    with openLog(f"scarab_simulation_{dir_index}.log") as scarab_log:
      scarab_cmd = ["scarab", # Requires that the Scarab build folder is in PATH
                    "--fdip_enable", "0", 
                    "--frontend", "memtrace", 
                    "--fetch_off_path_ops", "0", 
                    "--cbp_trace_r0=trace", 
                    "--memtrace_modules_log=bin"]
      run_shell_cmd(scarab_cmd, working_dir=trace_dir, log=scarab_log)

    # Read the generated statistics files.
    stats_reader = scarabizor.ScarabStatsReader(trace_dir, is_using_roi=False)
    stats_file_index = 0 # We only simulate one timestep.
    computation_time = stats_reader.readTime(stats_file_index)
    if computation_time == 0:
      raise ValueError(f'The computation time was zero (computation_time = {computation_time}). This typically indicates that Scarab did not find a PARAMS.in file.')
    data = {
            "index": dir_index, 
            "trace_dir": trace_dir, 
            "computation_time": computation_time
            }
    print(f"Finished Scarab simulation. Time to compute controller: {computation_time} seconds.")
    return data

class InternalSimulationExecutor(SimulationExecutor):
  """
  Create a simulation executor that performs all of the plant dynamics calculations internally. This executor does not support simulating computation times with Scarab.
  """
  def __init__(self):
    raise RuntimeError("This class is not tested and is not expected to work correctly as-is.")

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def run_simulation(self):
    # Override run_simulation to just run the controller. 
    self.run_controller()

# Ideas for experiments to run.
# -> Run several iterations of Cache sizes

### Candidate values to modify
# Parameters
  # sample_time
  # lead_car_input
  # x0
# MPC Settings
  # constraints
  # yref (distance buffer)
  # weights
  # warm start
# Optimizer Settings
  # optimizer tolerances
  # osqp_maximum_iteration
  # prediction_horizon
  # control_horizon
# Hardware Settings
  # computation_delay_multiplier  
  # l1_size
  # icache_size
  # dcache_size
  # chip_cycle_time
  # decode_cycles