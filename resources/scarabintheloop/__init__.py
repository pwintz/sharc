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
from enum import Enum
import traceback
from contextlib import redirect_stdout
from scarabintheloop.utils import *
import scarabintheloop.debug_levels as debug_levels

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

class ExperimentList:
  def __init__(self, example_dir, config_filename):
    example_dir = os.path.abspath(example_dir)
    self.example_dir = example_dir

    self.experiments_config_file_path = os.path.join(example_dir, "simulation_configs", config_filename)
    self.base_config_file_path        = os.path.join(example_dir, 'base_config.json')
    
    # If the user gave an output directory, then use it. Otherwise, use "experiments/" within the example folder.
    self.experiments_dir = os.path.join(example_dir, "experiments")

    # Read JSON configuration file.
    self.base_config = readJson(self.base_config_file_path)

    # Open list of example configurations.
    self.experiment_config_patches_list = readJson(self.experiments_config_file_path)

    # The experiment config file can either contain a single configuration or a list.
    # In the case where there is only a single configuration, we convert it to a singleton list. 
    if not isinstance(self.experiment_config_patches_list, list):
      self.experiment_config_patches_list = [self.experiment_config_patches_list]

    self.label = slugify(config_filename)
    self.base_config["experiment_list_label"] = self.label

    self.experiment_list_dir = os.path.join(self.experiments_dir, self.label + "--" + time.strftime("%Y-%m-%d--%H-%M-%S"))
    self.experiment_result_list = {}
    self.successful_experiment_labels = []
    self.skipped_experiment_labels = []
    self.failed_experiment_labels = []

    # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
    os.makedirs(self.experiment_list_dir)

    # Create or update the symlink to the latest folder.
    latest_experiment_list_dir_symlink_path =  os.path.join(self.experiment_list_dir, "latest")
    try:
      # Try to remove the previous symlink.
      os.remove(latest_experiment_list_dir_symlink_path)
    except FileNotFoundError:
      #  It if fails because it doesn't exist, that's OK.
      pass
    os.symlink(self.experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)


    print("         Example dir: " + self.example_dir)
    print(f'        Base config: {self.base_config_file_path}')
    print(f' Experiments config: {self.experiments_config_file_path}')

    self._experiment_list = []

    for experiment_config_patch in self.experiment_config_patches_list:
      # Load human-readable labels.
      experiment_label = self.label + "/" + slugify(experiment_config_patch["label"])

      # example_dir            = os.path.abspath('../..') # The "root" of this project.
      # experiment_list_dir    = os.path.abspath('.') # A path like <example_dir>/experiments/default-2024-08-25-16:17:35
      experiment_dir  = os.path.abspath(self.experiment_list_dir + "/" + slugify(experiment_config_patch['label']))

      experiment_config = patch_dictionary(self.base_config, experiment_config_patch)
      experiment_config["experiment_label"] = experiment_label
      experiment_config["experiment_dir"] = experiment_dir

      chip_configs_path      = self.example_dir + '/chip_configs'
      
      # Check that the expected folders exist.
      assertFileExists(chip_configs_path)

      experiment = Experiment(experiment_config, chip_configs_path)
      self._experiment_list.append(experiment)
    
  def __iter__(self):
    self.iterator_counter = 0
    return self

  def __next__(self): 
    if self.iterator_counter >= len(self.experiment_config_patches_list):
      raise StopIteration
        
    experiment = self._experiment_list[self.iterator_counter]
    # TODO: Make experiment
    self.iterator_counter += 1
    return experiment

  def run_all(self):
    for experiment in self._experiment_list:
      experiment.setup_files()
      experiment_result = experiment.run()
      if experiment.status == ExperimentStatus.FINISHED:
        self.successful_experiment_labels += [repr(experiment)]
        self.experiment_result_list[experiment.label] = experiment.result
        writeJson(self.experiment_list_dir + "experiment_result_list_incremental.json", self.experiment_result_list)
        # print(f'Added "{experiment_result["label"]}" to list of experiment results')
      elif experiment.status == ExperimentStatus.SKIPPED: # No result
        # print(f'Skipped "{experiment.label}" experiment.')
        self.skipped_experiment_labels += [repr(experiment)]
      elif experiment.status == ExperimentStatus.FAILED:
        # print(f'Running {experiment.label} failed! Error: {experiment.exception}')
        self.failed_experiment_labels += [repr(experiment)]
        
      self.print_status()

  def print_status(self):
    n_total      = len(self.experiment_config_patches_list)
    n_successful = len(self.successful_experiment_labels)
    n_failed     = len(self.failed_experiment_labels)
    n_skipped    = len(self.skipped_experiment_labels)
    n_ran        = n_successful + n_failed 
    n_not_pending= n_successful + n_failed + n_skipped
    n_pending    = n_total - n_not_pending
    def plural_suffix(n): 
      return '' if n == 1 else 's'
    print(f"Ran {n_ran} experiment{plural_suffix(n_ran)} of {n_total} experiment{plural_suffix(n_total)}.")
    print(f'Successful: {n_successful}. ({self.successful_experiment_labels})')
    print(f'   Skipped: {n_skipped   }. ({self.skipped_experiment_labels})')
    print(f'    Failed: {n_failed    }. ({self.failed_experiment_labels})')
    print(f'   Pending: {n_pending   }.')

ExperimentStatus = Enum('ExperimentStatus', ['PENDING', 'RUNNING', 'FINISHED', 'SKIPPED', 'FAILED'])

class Experiment:
  def __init__(self, 
              experiment_config: dict, 
              chip_configs_path):
    self.label = experiment_config["experiment_label"]
    self.exception = None
    self.result = None
    self.run_time = None
    self.experiment_config = experiment_config
    self.status = ExperimentStatus.PENDING
    
    ########### Populate the experiment directory ############
    # Make subdirectory for this experiment.
    self.experiment_dir = experiment_config["experiment_dir"]
    
    if debug_levels.debug_configuration_level >= 2:
      printJson("Experiment configuration", experiment_config)
    
    # Create PARAMS.generated file containing chip parameters for Scarab.
    PARAMS_src_filename = os.path.join(chip_configs_path, experiment_config["PARAMS_base_file"])
    
    assertFileExists(PARAMS_src_filename)
    if debug_levels.debug_configuration_level > 1:
      print(f'Creating chip parameter file.')
      print(f'\tSource: {PARAMS_src_filename}.')
      print(f'\tOutput: {PARAMS_out_filename}.')
    
    self.PARAMS_base = scarabizor.ParamsData.from_file(PARAMS_src_filename)

  def setup_files(self):
    os.makedirs(self.experiment_dir)

  @indented_print
  def run(self):
    
    # Record the start time.
    start_time = time.time()
    
    if self.experiment_config.get("skip", False):
      printHeader1(f'Skipping Experiment: {self.label}')
      self.status = ExperimentStatus.SKIPPED
      return
      
    printHeader1(f"Starting Experiment: {self.label}")

    try:
      self.status = ExperimentStatus.RUNNING
      if self.experiment_config["Simulation Options"]["parallel_scarab_simulation"]:
        experiment_data = run_experiment_parallelized(self.experiment_config, self.PARAMS_base)
      else:
        experiment_data = run_experiment_sequential(self.experiment_config, self.PARAMS_base)
        
      self.status = ExperimentStatus.FINISHED
    except Exception as err:
      print(traceback.format_exc())
      self.exception = err
      self.status = ExperimentStatus.FAILED
      return
    finally:
      end_time = time.time()
      self.run_time = end_time - start_time

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

    self.result = {
      "label":                self.label,
      "experiment directory": self.experiment_dir,
      "experiment config":    self.experiment_config, 
      "experiment data":      experiment_data,
      "experiment wall time": end_time - start_time
    }

    writeJson("experiment_result.json", self.result)

  def __repr__(self):
    repr_str = f'Experiment("{self.label}", {self.status.name}'
    if self.status == ExperimentStatus.FAILED:
      repr_str += f', error msg: "{self.exception}"'
    if self.run_time:
      repr_str += f' in {self.run_time:0.1f} s'
    repr_str += ')'
    return repr_str

class Simulation:
  def __init__(self,
               experiment_config: dict,
               PARAMS_base: scarabizor.ParamsData,
               batch_init: dict=None):
    # !! We do not, in general, expect experiment_config['u0']==batch_init['u0'] because 
    # !! batch_init overrides the experiment config.
    assert isinstance(experiment_config, dict)
    assert isinstance(PARAMS_base, scarabizor.ParamsData), f'type(PARAMS_base)={type(PARAMS_base)}'
    if batch_init:
      assert isinstance(batch_init, dict)

    self.experiment_config = copy.deepcopy(experiment_config)
    sim_config = copy.deepcopy(experiment_config)

    # printJson("experiment_config in create_simulation_config", experiment_config)
    experiment_max_time_steps = experiment_config["max_time_steps"]
    experiment_label          = experiment_config["label"]
    experiment_directory      = experiment_config["experiment_dir"]
    experiment_x0             = experiment_config["x0"]
    experiment_u0             = experiment_config["u0"]
    last_time_index_in_experiment = experiment_max_time_steps
    experiment_max_batch_size = experiment_config["Simulation Options"]["max_batch_size"]
    
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
    
    self.label = label
    self.simulation_dir = directory
    self.controller_log_path = os.path.join(self.simulation_dir, 'controller.log')
    self.plant_log_path      = os.path.join(self.simulation_dir, 'plant_dynamics.log')
    # Paths to files needed by controller executable.
    self.params_file_for_controller = os.path.join(self.simulation_dir, 'PARAMS.generated')
    self.config_file_for_controller = os.path.join(self.simulation_dir, 'config.json')
    sim_config["simulation_label"] = label
    sim_config["max_time_steps"]   = max_time_steps
    sim_config["simulation_dir"]   = directory
    sim_config["x0"]               = x0
    sim_config["u0"]               = u0
    sim_config["pending_computation"] = pending_computation
    sim_config["first_time_index"] = first_time_index
    sim_config["last_time_index"]  = last_time_index
    sim_config["time_steps"]       = list(range(first_time_index, last_time_index))
    sim_config["time_indices"]     = sim_config["time_steps"] + [last_time_index]
    
    self.PARAMS = PARAMS_base.copy()
    self.PARAMS = self.PARAMS.patch(sim_config["PARAMS_patch_values"])

    if batch_init:
      assert batch_init["u0"] == sim_config["u0"], \
                f'batch_u[0] = {batch_u[0]} must equal sim_config["u0"] = {sim_config["u0"]}.'

    self.sim_config = sim_config

  def setup_files(self):

    # Create the simulation directory if it does not already exist.
    os.makedirs(self.simulation_dir, exist_ok=True)

    # Move into the simulation directory.
    os.chdir(self.simulation_dir)

    # Write the configuration to config.json.
    writeJson(os.path.join(self.simulation_dir, "config.json"), self.sim_config)

    PARAMS_out_file = os.path.join(self.simulation_dir, 'PARAMS.generated')
    self.PARAMS.to_file(PARAMS_out_file)

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", working_dir=self.simulation_dir)
    assertFileExists(self.simulation_dir + "/u_c++_to_py")
    assertFileExists(self.simulation_dir + "/x_py_to_c++")


  @indented_print
  def run(self):
    printHeader2(f"Starting simulation: {self.label}")
    print(f'↳ Simulation dir: {self.simulation_dir}')
    print(f'↳ Controller log: {self.controller_log_path}')
    print(f'↳      Plant log: {self.plant_log_path}')
    
    assertFileExists(self.params_file_for_controller)
    assertFileExists(self.config_file_for_controller)  

    # Open up some loooooooooogs.
    with openLog(self.plant_log_path,      f'Plant Log for "{self.label}"') as plant_log, \
        openLog(self.controller_log_path, f'Controller Log for "{self.label}"') as controller_log:

      simulation_executor = getSimulationExecutor(self.simulation_dir, 
                                                  self.sim_config, 
                                                  controller_log, 
                                                  plant_log)

      # Execute the simulation.
      simulation_data = simulation_executor.run_simulation()
      printHeader2(f'Finished Simulation: "{self.label}"')
      print(f'↳ Simulation dir: {self.simulation_dir}')
      print(f'↳ Controller log: {self.controller_log_path}')
      print(f'↳      Plant log: {self.plant_log_path}')
      print(f"↳  Data out file: {os.path.abspath('simulation_data.json')}")
      
      simulation_data.metadata["controller_log_path"] = self.controller_log_path
      simulation_data.metadata["plant_log_path"]      = self.plant_log_path

      return simulation_data

def main():
  
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Run a set of experiments.")

  parser.add_argument(
      '--config_filename',
      type   =str,  # Specify the data type
      default="default.json",
      help   ="Select the name of a JSON file located in <example_dir>/simulation_configs."
  )

  # Parse the arguments from the command line
  args = parser.parse_args()
  experiment_list = ExperimentList(example_dir, args.config_filename)

  ##################################################################
  ################### CONFIGURE DEBUGGING LEVELS ###################
  ##################################################################
  debug_levels.set_from_dictionary(experiment_list.base_config["==== Debgugging Levels ===="])

  experiment_list.run_all()
    
  # Save all of the experiment results to a file.
  experiment_list_json_filename = experiment_list.experiment_list_dir + "experiment_result_list.json"
  writeJson(experiment_list_json_filename, experiment_list.experiment_result_list)
  print(f'Experiment results for "{experiment_list.label}" experiments are in \n\t{experiment_list_json_filename}.')

  if experiment_list.failed_experiment_labels:
    exit(1)
    

def run_experiment_sequential(experiment_config, PARAMS_base: list):
  print(f'Start of run_experiment_sequential(<{experiment_config["experiment_label"]}>)')
  sim_dir = experiment_config["experiment_dir"]
  simulation = Simulation(experiment_config, PARAMS_base)
  simulation.setup_files()
  simulation_data = simulation.run()
  experiment_data = {"x": simulation_data.x,
                     "u": simulation_data.u,
                     "t": simulation_data.t,
                     "pending_computations": simulation_data.pending_computation,
                     "batches": None,
                     "config": experiment_config}
  return experiment_data

def run_experiment_parallelized(experiment_config, PARAMS_base: list):
  print(f'Start of run_experiment_parallelized(<{experiment_config["experiment_label"]}>")')
  
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
    
      simulation = Simulation(experiment_config, PARAMS_base, batch_init)
      simulation.setup_files()

      if debug_levels.debug_batching_level >= 1:
        printHeader2( f'Starting batch: {simulation.label} ({simulation.sim_config["max_time_steps"]} time steps)')
        print(f'↳ dir: {batch_sim_config["simulation_dir"]}')
        print(f'↳  Last batch status: {batch_init["Last batch status"]}')
        print(f'↳                 x0: {batch_sim_config["x0"]}')
        print(f'↳                 u0: {batch_sim_config["u0"]}')
        print(f'↳     batch_init[u0]: {batch_init["u0"]}')
      
      # Run simulation
      batch_sim_data = simulation.run()

      # assert batch_sim_data.u[0] == batch_init["u0"], \
      #   f'batch_sim_data.u[0] = {batch_sim_data.u[0]} must equal batch_init["u0"]={batch_init["u0"]}'

      sample_time = simulation.sim_config["system_parameters"]["sample_time"]
      valid_data_from_batch, next_batch_init = processBatchSimulationData(batch_init, batch_sim_data, sample_time)

      batch_data = {
        'label':           simulation.sim_config["simulation_label"],
        'batch directory': simulation.sim_config["simulation_dir"],
        "batch_init":      batch_init,
        'config':          simulation.sim_config,
        "all_data_from_batch":   batch_sim_data, 
        "valid_data_from_batch": valid_data_from_batch, 
        "next_batch_init": next_batch_init
      }    
      batch_data_list.append(batch_data)

      if debug_levels.debug_batching_level >= 1:
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

      if debug_levels.debug_batching_level >= 1:
        printHeader2(f'Finished batch {batch_init["i_batch"]}: {batch_sim_config["simulation_label"]}')
        print(f'↳             length of x: {len(batch_x)}')
        print(f'↳             length of u: {len(batch_u)}')
        print(f'↳                      u0: {batch_u[0]}')
        print(f'↳           length of "t": {len(batch_t)}')
        print(f'↳          max_time_steps: {batch_sim_config["max_time_steps"]}')
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
    
  if debug_levels.debug_batching_level >= 1:
    actual_time_series.print('--------------- Actualualized Time Series (Concatenated) ---------------')

  writeJson(experiment_dir + "/experiment_data.json", experiment_data, label="Experiment data")
  return experiment_data


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


def getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log):
  """ 
  This function implements the "Factory" design pattern, where it returns objects of various classes depending on the imputs.
  """

  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]
  
  use_fake_delays = sim_config["Simulation Options"]["use_fake_delays"]
  if use_fake_delays:
    # Create some fake delay data.
    max_time_steps = sim_config["max_time_steps"]
    sample_time    = sim_config['system_parameters']["sample_time"]
    # Create a list of delays that are shorter than the sample time
    fake_delays = [0.1*sample_time]*max_time_steps

    # Update one of the queued delays to be longer than the sample time
    index_for_delay = 2
    if len(fake_delays) >= index_for_delay+1:
      fake_delays[index_for_delay] = 1.1*sample_time
    
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


# TODO: Maybe rename this as "delegator" or "coordinator".
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
    
    try:
      assertFileExists('config.json') # config.json is read by the controller.
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
    if debug_levels.debug_configuration_level >= 2:
      printJson(f"Simulation configuration", self.sim_config)

    try:
      with redirect_stdout(self.plant_log):
        print('Start of Plant Dynamics thread.')
        plant_result = plant_runner.run(self.sim_dir, self.sim_config, self.evolveState_fnc)
        print('Successfully finished running Plant Dynamics.')

    except Exception as e:
      self.plant_log.write(f'Plant dynamics had an error: {e}')
      raise Exception(f'Plant dynamics for "{sim_label}" had an error. See logs: {self.plant_log.name}') from e

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if debug_levels.debug_dynamics_level >= 1:
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
      plant_task      = executor.submit(self._run_plant)
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