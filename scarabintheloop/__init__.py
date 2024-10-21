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
from contextlib import redirect_stdout
from scarabintheloop.utils import *

# Type hinting.
from typing import List, Set, Dict, Tuple, Union

import scarabintheloop.scarabizor as scarabizor

from slugify import slugify
import importlib

import scarabintheloop.plant_runner as plant_runner

######## Load the controller and plant modules from the current directory #########
example_dir = os.getcwd()
assertFileExists('controller_delegator.py')
assertFileExists('plant_dynamics.py')
controller_delegator = loadModuleFromWorkingDir("controller_delegator")
plant_dynamics = loadModuleFromWorkingDir("plant_dynamics")

# Define default debugging levels.
debug_interfile_communication_level = 0
debug_optimizer_stats_level = 0
debug_dynamics_level = 0
debug_configuration_level = 0
debug_build_level = 0
debug_batching_level = 0

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]

def run():
  
  print("        Example dir: " + example_dir)

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
  base_config_file_path        = Path(os.path.abspath(os.path.join(example_dir, 'base_config.json')))

  print(f'        Base config: {base_config_file_path}')
  print(f' Experiments config: {experiments_config_file_path}')

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
  debug_interfile_communication_level = debug_levels["debug_interfile_communication_level"]
  debug_optimizer_stats_level         = debug_levels["debug_optimizer_stats_level"]
  debug_dynamics_level                = debug_levels["debug_dynamics_level"]
  debug_configuration_level           = debug_levels["debug_configuration_level"]
  debug_build_level                   = debug_levels["debug_build_level"]
  utils.debug_shell_calls_level       = debug_levels["debug_shell_calls_level"]
  debug_batching_level                = debug_levels["debug_batching_level"]
  scarabizor.debug_scarab_level       = debug_levels["debug_scarab_level"]

  # Set the debugging levels to the plant_runner module.
  plant_runner.debug_interfile_communication_level = debug_interfile_communication_level
  plant_runner.debug_dynamics_level = debug_dynamics_level

  plant_dynamics.debug_interfile_communication_level = debug_interfile_communication_level
  plant_dynamics.debug_dynamics_level = debug_dynamics_level

  experiment_list_dir = experiments_dir + "/" + slugify(experiments_config_file_path.stem) + "--" + time.strftime("%Y-%m-%d--%H-%M-%S") + "/"
  print(f"experiment_list_dir: {experiment_list_dir}")

  # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
  os.makedirs(experiment_list_dir)

  # Create or update the symlink to the latest folder.
  latest_experiment_list_dir_symlink_path = experiments_dir + "/latest"
  try:
    # Try to remove the previous symlink.
    os.remove(latest_experiment_list_dir_symlink_path)
  except FileNotFoundError:
    #  It if fails because it doesn't exist, that's OK.
    pass
  os.symlink(experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)

  # Loop through all of the experiment configurations. For each one, run the experiment and append the result to experiment_result_list. 
  experiment_result_list = {}
  successful_experiment_labels = []
  skipped_experiment_labels = []
  failed_experiment_labels = []

  def print_status():
    n_pending = len(experiment_config_patches_list)  \
                 - len(successful_experiment_labels) \
                 - len(failed_experiment_labels)     \
                 - len(skipped_experiment_labels)
    print(f"Ran {len(experiment_result_list)} experiment{'s' if len(experiment_result_list) > 1 else ''} of {len(experiment_config_patches_list)} experiments.")
    print(f'Successful: {len(successful_experiment_labels):d}. ({successful_experiment_labels})')
    print(f'   Skipped: {len(skipped_experiment_labels):d}. ({skipped_experiment_labels})')
    print(f'    Failed: {len(failed_experiment_labels):d}. ({failed_experiment_labels})')
    print(f'   Pending: {n_pending:d}.')


  for experiment_config_patches in experiment_config_patches_list:
      os.chdir(experiment_list_dir)
      try:
          experiment_result = run_experiment(base_config, experiment_config_patches, example_dir)

          # If no result provided, then the experiment was skipped.
          if not experiment_result:
            print(f'No result for "{experiment_config_patches["label"]}" experiment.')
            skipped_experiment_labels += [experiment_config_patches["label"]]
            print_status()
            continue
      except Exception as err:
          print(f'Running {experiment_config_patches["label"]} failed! Error: {repr(err)}')
          print(traceback.format_exc())
          failed_experiment_labels += [experiment_config_patches["label"]]
          print_status()
          continue
      
      successful_experiment_labels += [experiment_config_patches["label"] + f' ({experiment_result["experiment wall time"]:0.1f} seconds)']
      experiment_result_list[experiment_result["label"]] = experiment_result
      writeJson(experiment_list_dir + "experiment_result_list_incremental.json", experiment_result_list)
      # print(f'Added "{experiment_result["label"]}" to list of experiment results')
      print_status()
      
    
  # Save all of the experiment results to a file.
  experiment_list_json_filename = experiment_list_dir + "experiment_result_list.json"
  writeJson(experiment_list_json_filename, experiment_result_list)
  print(f"Experiment results for \n\t{experiments_config_file_path}\nare in \n\t{experiment_list_json_filename}.")

@indented_print
def run_experiment(base_config: dict, experiment_config_patch: dict, example_dir: str):
  """ This function assumes that the working directory is the experiment-list working directory and its parent directory is the example directory, which contains the following:
    - scripts/controller_delegator.py
    - chip_configs/PARAMS.base (and other optional alternative configuration files)
    - simulation_configs/default.json (and other optional alternative configuration files)
  """

  # Load human-readable labels.
  experiment_list_label = base_config["experiment_list_label"]
  experiment_label = experiment_list_label + "/" + slugify(experiment_config_patch["label"])
  
  # Record the start time.
  experiment_start_time = time.time()

  if experiment_config_patch.get("skip", False):
    printHeader1(f'Skipping Experiment: {experiment_label}')
    return
    
  printHeader1(f"Starting Experiment: {experiment_label}")

  example_dir = os.path.abspath('../..') # The "root" of this project.
  experiment_list_dir = os.path.abspath('.') # A path like <example_dir>/experiments/default-2024-08-25-16:17:35
  experiment_dir = os.path.abspath(experiment_list_dir + "/" + slugify(experiment_config_patch['label']))
  simulation_config_path = example_dir + '/simulation_configs'
  chip_configs_path = example_dir + '/chip_configs'

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
  experiment_config = patch_dictionary(base_config, experiment_config_patch)
  experiment_config["experiment_label"] = experiment_label
  experiment_config["experiment_dir"] = experiment_dir

  if debug_configuration_level >= 2:
    printJson("Experiment configuration", experiment_config)

  # Write the configuration to config.json.
  writeJson("config.json", experiment_config)

  # Create PARAMS.generated file containing chip parameters for Scarab.
  PARAMS_src = chip_configs_path + "/" + experiment_config["PARAMS_base_file"] 
  PARAMS_out = os.path.abspath('PARAMS.generated')
  
  create_patched_PARAMS_file(experiment_config, PARAMS_src, PARAMS_out)
  
  if experiment_config["Simulation Options"]["parallel_scarab_simulation"]:
    experiment_data = run_experiment_parallelized(experiment_config, example_dir)
  else:
    experiment_data = run_experiment_sequential(experiment_config, example_dir)

  # Check the data.
  assert isinstance(experiment_data, dict)
  if experiment_data["batches"]:
    assert isinstance(experiment_data["batches"], list), \
      f'expected experiment_data["batches"] to be a list. Instead it was {type(experiment_data["batches"])}'
  assert isinstance(experiment_data["x"], list)
  assert isinstance(experiment_data["u"], list)
  assert isinstance(experiment_data["t"], list)
  assert isinstance(experiment_data["pending_computations"], list)
  assert isinstance(experiment_data["config"], dict)

  experiment_end_time = time.time()
  experiment_result = {
    "label":                experiment_config["label"],
    "experiment directory": experiment_dir,
    "experiment data":      experiment_data,
    "experiment config":    experiment_config, 
    "experiment wall time": experiment_end_time - experiment_start_time
  }

  writeJson("experiment_result.json", experiment_result)
  return experiment_result
    
def run_experiment_sequential(experiment_config, example_dir):
  print(f'Start of run_experiment_sequential(<{experiment_config["experiment_label"]}>, "{example_dir}")')
  sim_dir = experiment_config["experiment_dir"]
  sim_config = copy.deepcopy(experiment_config)
  sim_config = create_simulation_config(experiment_config)
  simulation_data = run_simulation_common_code(sim_config)
  experiment_data = {"x": simulation_data.x,
                     "u": simulation_data.u,
                     "t": simulation_data.t,
                     "pending_computations": simulation_data.pending_computation,
                     "batches": None,
                     "config": experiment_config}
  return experiment_data

def run_experiment_parallelized(experiment_config, example_dir):
  print(f'Start of run_experiment_parallelized(<{experiment_config["experiment_label"]}>, "{example_dir}")')
  
  experiment_dir = experiment_config["experiment_dir"]
  max_time_steps = experiment_config["max_time_steps"]
  max_batch = experiment_config["Simulation Options"]["max_batches"]

  # Create lists for recording the true values (discarding values that erroneously use  missed computations as though were not missed).
  batch_data_list = []

  k0 = 0
  t0 = 0
  x0 = experiment_config["x0"]
  state_dimension = experiment_config["system_parameters"]["state_dimension"]
  assert len(x0) == state_dimension, f'x0={x0} did not have the expected number of entries: state_dimension={state_dimension}'
  batch_init = {
    "i_batch": 0,
    "first_time_index": 0,
    "x0": x0,
    "u0": experiment_config["u0"],
    # There are no pending (previouslly computed) control values because we are at the start of the experiment.
    "pending_computation": None,
    "Last batch status": "None - first time step"
  }

  actual_time_series = plant_runner.TimeStepSeries(k0, t0, x0, None)

  ### Batches Loop ###
  # Loop through batches of time-steps until the start time-step is no longer less 
  # than the total number of time-steps desired or until the max number of batches is reached.
  try:
    while batch_init["first_time_index"] < max_time_steps and batch_init["i_batch"] < max_batch:
      
      try:
        batch_sim_config = create_simulation_config(experiment_config, batch_init)
      except Exception as err:
        raise Exception(f'Failed to create simulation configuration. Batch init was: {batch_init}') from err

      if debug_batching_level >= 1:
        printHeader2( f'Starting batch: {batch_sim_config["simulation_label"]} ({batch_sim_config["max_time_steps"]} time steps)')
        print(f'↳ dir: {batch_sim_config["simulation_dir"]}')
        print(f'↳  Last batch status: {batch_init["Last batch status"]}')
        print(f'↳                 x0: {batch_sim_config["x0"]}')
        print(f'↳                 u0: {batch_sim_config["u0"]}')
        print(f'↳     batch_init[u0]: {batch_init["u0"]}')
      
      # Run simulation
      batch_sim_data = run_simulation_common_code(batch_sim_config)

      # assert batch_sim_data.u[0] == batch_init["u0"], \
      #   f'batch_sim_data.u[0] = {batch_sim_data.u[0]} must equal batch_init["u0"]={batch_init["u0"]}'

      sample_time = batch_sim_config["system_parameters"]["sample_time"]
      valid_data_from_batch, next_batch_init = processBatchSimulationData(batch_init, batch_sim_data, sample_time)

      batch_data = {
        'label':           batch_sim_config["simulation_label"],
        'batch directory': batch_sim_config["simulation_dir"],
        "batch_init":      batch_init,
        'config':          batch_sim_config,
        "all_data_from_batch":   batch_sim_data, 
        "valid_data_from_batch": valid_data_from_batch, 
        "next_batch_init": next_batch_init
      }    
      batch_data_list.append(batch_data)

      if debug_batching_level >= 1:
        valid_data_from_batch.printTimingData('valid_data')

      # Append all of the valid data (up to the point of rhe missed computation) except for the first index, which overlaps with the last index of the previous batch.
      actual_time_series += valid_data_from_batch
      
      # actual_time_series.printTimingData('actual_time_series after concatenation')

      all_batch_data = batch_data["all_data_from_batch"]
      batch_x = all_batch_data.x
      batch_u = all_batch_data.u
      batch_t = all_batch_data.t

      pending_computation = batch_init["pending_computation"]
      if pending_computation and pending_computation.t_end < batch_t[0]:
        print("USING A PENDING COMPUTATION INSTEAD OF U0")
        expected_u0 = pending_computation.u
      else:
        expected_u0 = batch_init["u0"]

      assert batch_sim_data.u[0] == expected_u0, \
        f'batch_sim_data.u[0] = {batch_sim_data.u[0]} must equal batch_init["u0"]={batch_init["u0"]}'

      assert batch_u[0] == expected_u0, \
        f"""batch_u[0] = {batch_u[0]} must equal expected_u0 = {expected_u0}.
        (Note: batch_init["u0"]={batch_init["u0"]})
        (all_data_from_batch: {batch_data["all_data_from_batch"]})
        (all_data_from_batch.u: {batch_data["all_data_from_batch"].u})
        """

      
      if debug_batching_level >= 1:
        printHeader2(f'Finished batch {batch_init["i_batch"]}: {batch_sim_config["simulation_label"]}')
        print(f'↳           length of x: {len(batch_x)}')
        print(f'↳           length of u: {len(batch_u)}')
        print(f'↳                     u0: {batch_u[0]}')
        print(f'↳           length of "t": {len(batch_t)}')
        print(f'↳         max_time_steps: {batch_sim_config["max_time_steps"]}')
        print(f'↳ Next "first_time_index": {batch_data["next_batch_init"]["first_time_index"]}')

      experiment_data = {"batches": batch_data_list,
                        "x": actual_time_series.x,
                        "u": actual_time_series.u,
                        "t": actual_time_series.t,
                        "pending_computations": actual_time_series.pending_computation,
                        "config": experiment_config}
                        
      writeJson(experiment_dir + "/experiment_data_incremental.json", experiment_data, label="Incremental experiment data")

      # Update values for next iteration of the loop.
      batch_init = next_batch_init
  except Exception as err:
    raise Exception(f'There was an exception when running batch: {batch_init}')
    
  if debug_batching_level >= 1:
    actual_time_series.print('--------------- Actualualized Time Series (Concatenated) ---------------')

  writeJson(experiment_dir + "/experiment_data.json", experiment_data, label="Experiment data")
  return experiment_data


@indented_print
def run_simulation_common_code(sim_config: dict):
  """
  run_simulation_common_code() contains code that us used by both the parallel and serial run_simulation functions.
  """
    
  sim_dir = os.path.abspath(sim_config["simulation_dir"])
  chip_params_path = sim_dir + '/PARAMS.generated'

  if sim_config["experiment_dir"] != sim_config["simulation_dir"]:
    os.makedirs(sim_dir)
    shutil.copyfile(sim_config["experiment_dir"] + "/PARAMS.generated", chip_params_path)
    writeJson(sim_config["simulation_dir"] + "/config.json", sim_config)
  assertFileExists(chip_params_path)  

  # os.chdir(sim_dir)
  # Get human-readable label for this simulation.
  simulation_label = sim_config["simulation_label"]

  printHeader2(f"Starting simulation: {simulation_label}")
  print(f'↳ Simulation dir: {sim_dir}')
  

  # Open up some loooooooooogs.
  controller_log_path = sim_dir + '/controller.log'
  plant_log_path      = sim_dir + '/plant_dynamics.log'
  with openLog(plant_log_path,      f'Plant Log for "{simulation_label}"') as plant_log, \
       openLog(controller_log_path, f'Controller Log for "{simulation_label}"') as controller_log:

    simulation_executor = getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)

    # Execute the simulation.
    simulation_data = simulation_executor.run_simulation()
    printHeader2(f'Finished Simulation: "{simulation_label}"')
    print(f'↳ Simulation dir: {sim_dir}')
    print(f'↳ Controller log: {controller_log_path}')
    print(f'↳      Plant log: {plant_log_path}')
    print(f"↳  Data out file: {os.path.abspath('simulation_data.json')}")
    
    simulation_data.metadata["controller_log_path"] = controller_log_path
    simulation_data.metadata["plant_log_path"]      = plant_log_path

    return simulation_data



def create_simulation_config(experiment_config, batch_init=None):
  # !! We do not, in general, expect experiment_config['u0']==batch_init['u0'] because 
  # !! batch_init overrides the experiment config.

  # printJson("experiment_config in create_simulation_config", experiment_config)
  experiment_max_time_steps = experiment_config["max_time_steps"]
  experiment_label          = experiment_config["label"]
  experiment_directory      = experiment_config["experiment_dir"]
  experiment_x0             = experiment_config["x0"]
  experiment_u0             = experiment_config["u0"]
  last_time_index_in_experiment = experiment_max_time_steps
  experiment_max_batch_size = experiment_config["Simulation Options"]["max_batch_size"]
  
  batch_config = copy.deepcopy(experiment_config)
  experiment_config = None # Clear experiment_config so that we don't accidentally use or modify it anymore in this function

  if batch_init:
    first_time_index    = batch_init["first_time_index"]
    pending_computation = batch_init["pending_computation"]
    x0                  = batch_init["x0"]
    u0                  = batch_init["u0"]
    
    # Truncate the index to not extend past the last index
    max_time_steps_per_batch = min(os.cpu_count(), experiment_max_batch_size, experiment_max_time_steps - first_time_index)

    # Add "max_time_steps_per_batch" steps to the first index to get the last index.
    last_time_index_in_batch = first_time_index + max_time_steps_per_batch

    assert last_time_index_in_batch <= last_time_index_in_experiment, 'last_time_index_in_batch must not be past the end of experiment.'

    batch_label    = f'batch{batch_init["i_batch"]}_steps{first_time_index}-{last_time_index_in_batch}'
    directory      = experiment_directory + "/" + batch_label
    label          = experiment_label     + "/" + batch_label
    max_time_steps = min(experiment_max_time_steps, max_time_steps_per_batch)

  else:
    first_time_index    = 0
    pending_computation = None
    x0                  = experiment_x0
    u0                  = experiment_u0
    directory           = experiment_directory
    label               = experiment_label
    max_time_steps      = experiment_max_time_steps

  last_time_index = first_time_index + max_time_steps
  
  batch_config["simulation_label"] = label
  batch_config["max_time_steps"]   = max_time_steps
  batch_config["simulation_dir"]   = directory
  batch_config["x0"]               = x0
  batch_config["u0"]               = u0
  batch_config["pending_computation"] = pending_computation
  batch_config["first_time_index"] = first_time_index
  batch_config["last_time_index"]  = last_time_index
  batch_config["time_steps"]       = list(range(first_time_index, last_time_index))
  # batch_config["time_indices"]     = list(range(first_time_index, last_time_index+1))
  batch_config["time_indices"]     = batch_config["time_steps"] + [last_time_index]
  
  if batch_init:
    assert batch_init["u0"] == batch_config["u0"], \
              f'batch_u[0] = {batch_u[0]} must equal batch_config["u0"] = {batch_config["u0"]}.'
              # print(f'WARNING: batch_u[0] = {batch_u[0]} != batch_sim_config["u0"] = {batch_sim_config["u0"]}')

  return batch_config



def processBatchSimulationData(batch_init:dict, batch_simulation_data: dict, sample_time):
  first_late_timestep, has_missed_computation = batch_simulation_data.find_first_late_timestep(sample_time)

  if has_missed_computation:
    valid_data_from_batch = batch_simulation_data.truncate(first_late_timestep)
  else: 
    valid_data_from_batch = batch_simulation_data.copy()

  if has_missed_computation:
    valid_data_from_batch.printTimingData(f'valid_data_from_batch (truncated to {first_late_timestep} time steps.)')
  else:
    valid_data_from_batch.printTimingData(f'valid_data_from_batch (no truncated time steps.)')

  # Values that are needed to start the next batch.
  if len(valid_data_from_batch.k) > 0:
    next_batch_init = {
      "i_batch":              batch_init["i_batch"] + 1,
      "first_time_index":     valid_data_from_batch.k[-1] + 1,
      "x0":                   valid_data_from_batch.x[-1],
      "u0":                   valid_data_from_batch.u[-1],
      "pending_computation":  valid_data_from_batch.pending_computation[-1]
    }
    if has_missed_computation:
      next_batch_init["Last batch status"] = f"Missed computation at timestep[{first_late_timestep}]"
    else:
      next_batch_init["Last batch status"] = f"No missed computations."
  else:
    next_batch_init = copy.deepcopy(batch_init)
    next_batch_init["i_batch"] = batch_init["i_batch"] + 1
    next_batch_init["Last batch status"]: f'No valid data. Carried over from previous init, which had Last batch status={batch_init["Last batch status"]}'

  
  # Check that the next_first_time_index doesn't skip over any timesteps.
  last_time_index_in_previous_batch = batch_simulation_data.get_last_sample_time_index()
  assert next_batch_init['first_time_index'] <= last_time_index_in_previous_batch + 1 , \
        f'next first_time_index={next_batch_init["first_time_index"]} must be larger than' + \
        f'(last index in the batch) + 1={batch_simulation_data.i[-1] + 1}'


  return valid_data_from_batch, next_batch_init


def create_patched_PARAMS_file(sim_config: dict, PARAMS_src_filename: str, PARAMS_out_filename: str):
  """
  Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
  Then, modify the values for keys listed in PARAMS_file_keys to values taken from sim_config. 
  Write the resulting PARAMS data to a file at PARAMS_out_filename in the simulation directory (sim_dir).
  Returns the absolute path to the PARAMS file.
  """
  assertFileExists(PARAMS_src_filename)
  if debug_configuration_level > 1:
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
    
  assertFileExists(PARAMS_out_filename)


def getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log):
  """ 
  This function implements the "Factory" design pattern, where it returns objects of various classes depending on the imputs.
  """

  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]
  
  use_fake_delays = sim_config["Simulation Options"]["use_fake_delays"]
  if use_fake_delays:
    # Create some fake delay data.
    max_time_steps = sim_config["max_time_steps"]
    sample_time = sim_config['system_parameters']["sample_time"]
    # Create a list of delays that are shorter than the sample time
    fake_delays = [0.1*sample_time]*max_time_steps

    # Update one of the queued delays to be longer than the sample time
    index_for_delay = 2
    if len(fake_delays) >= index_for_delay+1:
      fake_delays[index_for_delay] = 1.1*sample_time
  # else:
  #   raise ValueError(f'Not using use_fake_delays is currently disabled.')
    
  if use_parallel_scarab_simulation:
    print("Using ParallelSimulationExecutor.")

    if use_fake_delays:
      # Create the mock Scarab runner.
      trace_processor = scarabizor.MockTracesToComputationTimesProcessor(sim_dir, delays=fake_delays)
    else:
      # The "real" trace processor
      trace_processor = scarabizor.ScarabTracesToComputationTimesProcessor(sim_dir)

    return ParallelSimulationExecutor(sim_dir, sim_config, controller_log, plant_log, trace_processor)
  else:
    print("Using SerialSimulationExecutor.")
    executor = SerialSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)
    delay_provider_config = sim_config["Simulation Options"]["in-the-loop_delay_provider"]
    assert delay_provider_config == "execution-driven scarab" or delay_provider_config == "fake execution-driven scarab", \
      f'delay_provider_config={delay_provider_config} must be either "execution-driven scarab" or "fake execution-driven scarab"'
    
    
    if use_fake_delays:
      # Create the mock Scarab runner.
      executor.scarab_runner = scarabizor.MockExecutionDrivenScarabRunner(queued_delays=fake_delays, controller_log=controller_log)
    else:
      executor.scarab_runner = scarabizor.ExecutionDrivenScarabRunner(controller_log)
      
    return executor


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
        # add parent directory to the path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
        # add dynamics directory to the path
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dynamics')))
        dynamics_class = getattr(importlib.import_module(sim_config["dynamics_module_name"]),  sim_config["dynamics_class_name"])
        dynamics_instance = dynamics_class(sim_config)
        self.evolveState_fnc = dynamics_instance.getDynamicsFunction()


    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log, working_dir=self.sim_dir)
    assertFileExists(sim_dir + "/u_c++_to_py")
    assertFileExists(sim_dir + "/x_py_to_c++")

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
      print('Controller finished.') 
    except Exception as err:
      # if hasattr(err, 'output') and err.output:
      #   print(f'Error output: {err.output}')
      raise Exception(f'Running the controller \n\t{type(self.controller_executable)}\nusing {type(self)} failed! \nSee logs: {self.controller_log.name}') from err

  def _run_plant(self):
    """ 
    Handle setup and exception handling for running the run_plant() functions implemented in subclasses
    """
    sim_label = self.sim_config["simulation_label"]
    print(f'---- Starting plant dynamics for {sim_label} ----\n' \
          f'\t↳ Simulation dir: {self.sim_dir} \n' \
          f'\t↳      Plant log: {self.plant_log.name}\n')
    self.plant_log.write('Start of thread.\n')
    if debug_configuration_level >= 2:
      printJson(f"Simulation configuration", self.sim_config)

    try:
      with redirect_stdout(self.plant_log):
        plant_result = plant_runner.run(self.sim_dir, self.sim_config, self.evolveState_fnc)

    except Exception as e:
      raise Exception(f'Plant dynamics for "{sim_label}" had an error. See logs: {self.plant_log.name}') from e

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if debug_dynamics_level >= 1:
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
      plant_task = executor.submit(self._run_plant)
      controller_task = executor.submit(self._run_controller)
      print("Waiting for plant_task and controller_task to run.")
      controller_task.add_done_callback(lambda future: print("Controller done."))
      plant_task.add_done_callback(lambda future: print("Plant done."))
      for future in concurrent.futures.as_completed([plant_task, controller_task]):
        if future.exception():
          print(f'Failed future.')
          
        future.result()  # This will raise the exception if one occurred

      simulation_data = plant_task.result()

    if simulation_data is None:
      raise ValueError(f"simulation_data is None before postprocessing in class {type(self)}")
    
    try:
      # Do optional postprocessing (no-op if postprocess_simulation_data is not overridden in subclass)
      simulation_data = self.postprocess_simulation_data(simulation_data)
    except Exception as err:
      # simulation_data.print("ERROR - Failed to postprocess simulation_data")
      raise Exception(f"""Failed to postprocess simulation_data:\n{simulation_data}
        len(simulation_data.k): {len(simulation_data.k)}
        len(simulation_data.i): {len(simulation_data.i)}
        len(simulation_data.pending_computation): {len(simulation_data.pending_computation)}
      """) from err

    if simulation_data is None:
      raise ValueError(f"simulation_data is None after postprocessing in class {type(self)}")
    
    # Check to make sure the simulation generated data that is well-formed.
    # checkSimulationData(simulation_data, self.sim_config["max_time_steps"])
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
    cmd = " ".join([self.controller_executable])
    print("Starting scarab_runner: ", self.scarab_runner, 'to execute', cmd)
    try:
      self.scarab_runner.run(cmd)
    except Exception as err:
      print(f'Failed to run command with {type(self.scarab_runner)}: "{cmd}"')
      raise err
    
class ParallelSimulationExecutor(SimulationExecutor):

  def __init__(self, sim_dir, sim_config, controller_log, plant_log, trace_processor):
    super().__init__(sim_dir, sim_config, controller_log, plant_log)
    self.trace_processor = trace_processor

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def postprocess_simulation_data(self, simulation_data):
    """ 
    Runnining the controller exectuable produces DynamoRIO traces but does not give 
    use simulated computation times. In this function, we simulate the traces 
    using Scarab to get the computation times.
    """
    computation_times = self.trace_processor.get_all_computation_times()
    expected_n_of_computation_times = simulation_data.n_time_indices() 
    while len(computation_times) < expected_n_of_computation_times:
      computation_times += [None]
    assert len(computation_times) == expected_n_of_computation_times, \
          f"""computation_times={computation_times} must have 
          simulation_data.n_time_indices()={expected_n_of_computation_times} many values, 
          because there is a new computation started at each index. 
          None values represent times when computations are not running.
          """
    simulation_data.overwrite_computation_times(computation_times)
    return simulation_data


# class InternalSimulationExecutor(SimulationExecutor):
#   """
#   Create a simulation executor that performs all of the plant dynamics calculations internally. This executor does not support simulating computation times with Scarab.
#   """
#   def __init__(self):
#     raise RuntimeError("This class is not tested and is not expected to work correctly as-is.")
# 
#   def run_controller(self):
#     run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)
# 
#   def run_simulation(self):
#     # Override run_simulation to just run the controller. 
#     self.run_controller()

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