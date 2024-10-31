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
from scarabintheloop.data_types import *
from abc import abstractmethod, ABC

# Type hinting.
from typing import List, Set, Dict, Tuple, Union

import scarabintheloop.scarabizor as scarabizor

from slugify import slugify
import importlib

import scarabintheloop.plant_runner as plant_runner

controller_executable_provider = None

def remove_suffix(string:str, suffix: str):
  """
  Replicate the remove_suffix str method that is added in Python 3.9.
  """
  assert isinstance(string, str)
  assert isinstance(suffix, str)
  if string.endswith(suffix):
    return string[:-len(suffix)]
  else:
    return string

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

    self.label = slugify(remove_suffix(config_filename, ".json"))
    self.base_config["experiment_list_label"] = self.label

    self.experiment_list_dir = os.path.join(self.experiments_dir, self.label + "--" + time.strftime("%Y-%m-%d--%H-%M-%S"))
    self.experiment_result_dict = {}
    self.successful_experiment_labels = []
    self.skipped_experiment_labels = []
    self.failed_experiment_labels = []

    # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
    os.makedirs(self.experiment_list_dir)

    #### CREATE OR UPDATE THE SYMLINK TO THE LATEST FOLDER ####
    latest_experiment_list_dir_symlink_path =  os.path.join(example_dir, "latest")
    try:
      # Try to remove the previous symlink.
      os.remove(latest_experiment_list_dir_symlink_path)
    except FileNotFoundError:
      #  It if fails because it doesn't exist, that's OK.
      pass
    os.symlink(self.experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)

    #### BUILD LIST OF EXPERIMENTS ####
    self._experiment_list = []
    for experiment_config_patch in self.experiment_config_patches_list:
      # Load human-readable labels.
      patch_label = experiment_config_patch["label"]

      experiment_config = patch_dictionary(self.base_config, experiment_config_patch)
      experiment_config["experiment_label"] = f"{self.label}/{slugify(patch_label)}"
      experiment_config["experiment_dir"] = os.path.join(self.experiment_list_dir, slugify(patch_label))

      chip_configs_path      = os.path.join(self.example_dir, 'chip_configs')
      
      # Check that the expected folders exist.
      assertFileExists(chip_configs_path)

      experiment = Experiment(experiment_config, chip_configs_path)
      self._experiment_list.append(experiment)

    printHeader1('Experiment List')
    print("         Example dir: " + self.example_dir)
    print(f'        Base config: {self.base_config_file_path}')
    print(f' Experiments config: {self.experiments_config_file_path}')
    
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
    incremental_data_file_path = os.path.join(self.experiment_list_dir, "experiment_list_data_incremental.json")
    complete_data_file_path = os.path.join(self.experiment_list_dir, "experiment_list_data.json")

    for experiment in self._experiment_list:
      experiment.setup_files()
      
      experiment.run()

      if experiment.status == ExperimentStatus.FINISHED:
        assert experiment.result
        assert isinstance(experiment.result, dict)
        self.successful_experiment_labels += [repr(experiment)]
        self.experiment_result_dict[experiment.label] = experiment.result
      elif experiment.status == ExperimentStatus.SKIPPED: # No result
        self.skipped_experiment_labels += [repr(experiment)]
      elif experiment.status == ExperimentStatus.FAILED:
        self.failed_experiment_labels += [repr(experiment)]

      writeJson(incremental_data_file_path, self.experiment_result_dict, label="Incremental experiment list data")
      
      self.print_status()
    
    print('Finished running experiments from configuration in\n\t' + self.experiments_config_file_path)
    writeJson(complete_data_file_path, self.experiment_result_dict, label="Complete experiment list data")
    
    # Delete the incremental data file.
    os.remove(incremental_data_file_path)

  def get_results(self):
    """
    Return a list of all of the experiment results.
    """
    return list(self.experiment_result_dict.values())

  def n_total(self):
    return len(self.experiment_config_patches_list)

  def n_successful(self):
    return len(self.successful_experiment_labels)

  def n_failed(self):
    return len(self.failed_experiment_labels)

  def n_skipped(self):
    return len(self.skipped_experiment_labels)
    
  def n_ran(self):
    return self.n_successful() + self.n_failed() 
    
  def n_not_pending(self):
    return self.n_successful() + self.n_failed() + self.n_skipped()

  def n_pending(self):
    return self.n_total() - self.n_not_pending()

  def print_status(self):
    n_total      = self.n_total()
    n_successful = self.n_successful()
    n_failed     = self.n_failed()
    n_skipped    = self.n_skipped()
    n_ran        = self.n_ran()
    n_not_pending= self.n_not_pending()
    n_pending    = self.n_pending()
    def plural_suffix(n): 
      return '' if n == 1 else 's'

    def list_to_line_by_line(my_list: list):
      return "\n\t".join([""] + my_list)

    print(f"Ran {n_ran} experiment{plural_suffix(n_ran)} of {n_total} experiment{plural_suffix(n_total)}.")
    print(f'Successful: {n_successful}. {list_to_line_by_line(self.successful_experiment_labels)}')
    print(f'   Skipped: {n_skipped   }. {list_to_line_by_line(self.skipped_experiment_labels)}')
    print(f'    Failed: {n_failed    }. {list_to_line_by_line(self.failed_experiment_labels)}')
    print(f'   Pending: {n_pending   }.')

ExperimentStatus = Enum('ExperimentStatus', ['PENDING', 'RUNNING', 'FINISHED', 'SKIPPED', 'FAILED'])

class Experiment:
  def __init__(self, 
              experiment_config: dict, 
              chip_configs_path):
    self.label = experiment_config["experiment_label"]
    self.exception = None
    self.result    = None
    self.run_time  = None
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
    assert isinstance(experiment_data["k"], list)
    assert isinstance(experiment_data["i"], list)
    assert isinstance(experiment_data["pending_computations"], list)
    assert isinstance(experiment_data["config"], dict)

    self.result = {
      "label":                self.label,
      "experiment directory": self.experiment_dir,
      "experiment config":    self.experiment_config, 
      "experiment data":      experiment_data,
      "experiment wall time": end_time - start_time
    }

    writeJson(os.path.join(self.experiment_dir, "experiment_result.json"), self.result)

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
      assert isinstance(batch_init, dict), f'type(batch_init)={type(batch_init)}'

    self.experiment_config = copy.deepcopy(experiment_config)

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
      time_steps_in_sim = min(os.cpu_count(), \
                              experiment_max_batch_size, \
                              experiment_max_time_steps, \
                              experiment_max_time_steps - first_time_index)
      # print(f'        first_time_index={first_time_index}')
      # print(f'time_steps_in_sim={time_steps_in_sim}')

      # Add "time_steps_in_sim" steps to the first index to get the last index.
      last_time_index_in_batch = first_time_index + time_steps_in_sim

      assert last_time_index_in_batch <= last_time_index_in_experiment, 'last_time_index_in_batch must not be past the end of experiment.'

      batch_label    = f'batch{batch_init["i_batch"]}_steps{first_time_index}-{last_time_index_in_batch}'
      directory      = experiment_directory + "/" + batch_label
      label          = experiment_label     + "/" + batch_label
      # max_time_steps = min(experiment_max_time_steps, max_time_steps_per_batch)

    else:
      first_time_index    = 0
      pending_computation = None
      x0                  = experiment_x0
      u0                  = experiment_u0
      directory           = experiment_directory
      label               = experiment_label
      time_steps_in_sim = experiment_max_time_steps

    last_time_index = first_time_index + time_steps_in_sim
    
    self.label = label
    self.simulation_dir = directory
    self.controller_log_path = os.path.join(self.simulation_dir, 'controller.log')
    self.plant_log_path      = os.path.join(self.simulation_dir, 'plant_dynamics.log')
    # Paths to files needed by controller executable.
    self.params_file_for_controller = os.path.join(self.simulation_dir, 'PARAMS.generated')
    self.config_file_for_controller = os.path.join(self.simulation_dir, 'config.json')
    
    self.sim_config = copy.deepcopy(experiment_config)
    self.sim_config["simulation_label"] = label
    self.sim_config["max_time_steps"]   = time_steps_in_sim
    self.sim_config["simulation_dir"]   = directory
    self.sim_config["x0"]               = x0
    self.sim_config["u0"]               = u0
    self.sim_config["pending_computation"] = pending_computation
    self.sim_config["first_time_index"] = first_time_index
    self.sim_config["last_time_index"]  = last_time_index
    self.sim_config["time_steps"]       = list(range(first_time_index, last_time_index))
    self.sim_config["time_indices"]     = self.sim_config["time_steps"] + [last_time_index]
    
    self.PARAMS = PARAMS_base.patch(self.sim_config["PARAMS_patch_values"])

    if batch_init:
      assert batch_init["u0"] == self.sim_config["u0"], \
                f'batch_u[0] = {batch_u[0]} must equal sim_config["u0"] = {self.sim_config["u0"]}.'


  def setup_files(self):

    # Create the simulation directory if it does not already exist.
    os.makedirs(self.simulation_dir, exist_ok=True)

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

    use_fake_delays = self.sim_config["Simulation Options"]["use_fake_delays"]
    if use_fake_delays:
      # Create some fake delay data.
      max_time_steps = self.sim_config["max_time_steps"]
      sample_time    = self.sim_config['system_parameters']["sample_time"]
      # Create a list of delays that are shorter than the sample time
      fake_delays = [0.1*sample_time]*max_time_steps

      # Update one of the queued delays to be longer than the sample time
      index_for_delay = 2
      if len(fake_delays) >= index_for_delay+1:
        fake_delays[index_for_delay] = 1.1*sample_time
    else:
      fake_delays = None

    # Open up some loooooooooogs.
    with openLog(self.plant_log_path,      f'Plant Log for "{self.label}"')      as plant_log, \
         openLog(self.controller_log_path, f'Controller Log for "{self.label}"') as controller_log:

      controller_interface_selection="pipes"
      in_the_loop_delay_proiver =  self.sim_config["Simulation Options"]["in-the-loop_delay_provider"]
      computation_delay_provider = computation_delay_provider_factory(in_the_loop_delay_proiver, self.simulation_dir, sample_time, fake_delays)
      
      with controller_interface_factory(controller_interface_selection, computation_delay_provider, self.simulation_dir) as controller_interface:


        simulation_executor = getSimulationExecutor(self.simulation_dir, 
                                                    self.sim_config, 
                                                    controller_interface,
                                                    controller_log, 
                                                    plant_log,
                                                    fake_delays = fake_delays)

        # Execute the simulation.
        simulation_data = simulation_executor.run_simulation()
      

      sim_data_path = os.path.join(self.simulation_dir, 'simulation_data.json')
      printHeader2(f'Finished Simulation: "{self.label}"')
      print(f'↳ Simulation dir: {self.simulation_dir}')
      print(f'↳ Controller log: {self.controller_log_path}')
      print(f'↳      Plant log: {self.plant_log_path}')
      print(f"↳  Data out file: {sim_data_path}")
      
      simulation_data.metadata["controller_log_path"] = self.controller_log_path
      simulation_data.metadata["plant_log_path"]      = self.plant_log_path

    return simulation_data

def run(example_dir:str, config_filename:str):
  """
  This function is the entry point for running scarbintheloop 
  from other Python scripts (e.g., in unit testint). 
  When running from command line, the entry point is main(), instead,
  which then calls this function. 
  """
  global controller_executable_provider

  ######## Load the controller and plant modules from the current directory #########
  assert example_dir is not None
  example_dir = os.path.abspath(example_dir)
  assertFileExists(example_dir, f'The current working directory is {os.getcwd()}')
  

  controller_delegator_module = loadModuleInDir(example_dir, "controller_delegator")
  controller_executable_provider = controller_delegator_module.ControllerExecutableProvider(example_dir)
  
  experiment_list = ExperimentList(example_dir, config_filename)

  #----- CONFIGURE DEBUGGING LEVELS -----#
  debug_levels.set_from_dictionary(experiment_list.base_config["==== Debgugging Levels ===="])

  experiment_list.run_all()
  
  assert isinstance(experiment_list, ExperimentList)
  return experiment_list
    
def main(example_dir=None):

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
  example_dir = os.path.abspath('.')
  experiment_list = run(example_dir, args.config_filename)

  if experiment_list.n_failed():
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
                     "k": simulation_data.k, # Time steps
                     "i": simulation_data.i, # Sample time indices
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
                        "k": actual_time_series.k, # Time steps
                        "i": actual_time_series.i, # Sample time indices
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


def getSimulationExecutor(sim_dir, sim_config, controller_interface, controller_log, plant_log, fake_delays=None):
  """ 
  This function implements the "Factory" design pattern, where it returns objects of various classes depending on the imputs.
  """

  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]    
  if use_parallel_scarab_simulation:
    print("Using ParallelSimulationExecutor.")

    if fake_delays:
      # Create the mock Scarab runner.
      trace_processor = scarabizor.MockTracesToComputationTimesProcessor(sim_dir, delays=fake_delays)
    else:
      # The "real" trace processor
      trace_processor = scarabizor.ScarabTracesToComputationTimesProcessor(sim_dir)

    return ParallelSimulationExecutor(sim_dir, sim_config, controller_interface, controller_log, plant_log, trace_processor)
  else:
    print("Using SerialSimulationExecutor.")
    executor = SerialSimulationExecutor(sim_dir, sim_config, controller_interface, controller_log, plant_log)
    delay_provider_config = sim_config["Simulation Options"]["in-the-loop_delay_provider"]
    assert delay_provider_config == "execution-driven scarab", \
      f'delay_provider_config={delay_provider_config} must be "execution-driven scarab" when doing serial execution.'
    
    if fake_delays:
      # Create the mock Scarab runner.
      executor.scarab_runner = scarabizor.MockExecutionDrivenScarabRunner(sim_dir=sim_dir,
                                                                          queued_delays=fake_delays,
                                                                          controller_log=controller_log)
    else:
      executor.scarab_runner = scarabizor.ExecutionDrivenScarabRunner(sim_dir=sim_dir,
                                                                      controller_log=controller_log)
    
    return executor


# TODO: Maybe rename this as "delegator" or "coordinator".
class SimulationExecutor:
  
  def __init__(self, sim_dir, sim_config, controller_interface, controller_log, plant_log):
    global controller_executable_provider
    self.sim_dir = sim_dir
    self.sim_config = sim_config
    self.controller_log = controller_log
    self.plant_log = plant_log
    self.controller_interface = controller_interface

    # Create the executable.
    with redirect_stdout(controller_log):
      self.controller_executable = controller_executable_provider.get_controller_executable(sim_config)
    assertFileExists(self.controller_executable)
    
    # Get a function that defines the plant dynamics.
    with redirect_stdout(plant_log):
        # Add dynamics directory to the path
        sys.path.append(os.environ['DYNAMICS_DIR'])
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
      assertFileExists(os.path.join(self.sim_dir, 'config.json')) # config.json is read by the controller.
      self.run_controller()
    except Exception as err:
      # TODO: When the controller fails, we need to notify the plant_runner to stop running.
      raise Exception(f'Running the controller \n\t{type(self.controller_executable)}\nusing {type(self)} failed! \nSee logs: {self.controller_log.name}') from err
    print('SimulationExecutor._run_controller() finished.') 

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
        print('Start of SimulationExecutor._run_plant() (Plant Dynamics task).')
        plant_result = plant_runner.run(self.sim_dir, self.sim_config, self.evolveState_fnc, self.controller_interface)
        print('Successfully SimulationExecutor._run_plant() (Plant Dynamics task).')

    except Exception as e:
      self.plant_log.write(f'Plant dynamics had an error: {e}')
      raise Exception(f'Plant dynamics for "{sim_label}" had an error. See logs: {self.plant_log.name}') from e

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if debug_levels.debug_dynamics_level >= 1:
      print('-- Plant dynamics finshed -- \n' \
            f'\t↳ Simulation directory: {self.sim_dir}')
    return plant_result

  def run_simulation(self):
    """ Run everything needed to simulate the plant and dynamics, in parallel."""
    # Run the controller in parallel, on a separate thread.
    N_TASKS = 1
    with ThreadPoolExecutor(max_workers=N_TASKS) as executor:
      # Start a separate thread to run the controller.
      print("Starting the controller...")
      controller_task = executor.submit(self._run_controller)
      controller_task.add_done_callback(lambda future: print("Controller task finished."))

      # Start running the plant in the current thread.
      print("Starting the plant...")
      simulation_data = self._run_plant()
      print('Plant finished')

      # Wait for the controller thread to complete its task.
      for future in concurrent.futures.as_completed([controller_task]):
        if future.exception():
          err = future.exception()

          # Create a list of the causes of the exception.
          err_repr_list = []
          while err:
            err_repr_list = [repr(err)] + err_repr_list
            err = err.__cause__
          
          raise Exception('The controller task failed:\n\t' + "\n\t".join(err_repr_list)) from future.exception()
        else:
          print("Controller task was successful.")

    assert simulation_data, f"simulation_data={simulation_data} must not be None"
    
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
      print(f'Failed to run command with {type(self.scarab_runner)}: "{cmd}".\n{type(err)}: {err}\n{traceback.format_exc()}')
      raise err
    
class ParallelSimulationExecutor(SimulationExecutor):

  def __init__(self, sim_dir, sim_config, controller_interface, controller_log, plant_log, trace_processor):
    super().__init__(sim_dir, sim_config, controller_interface, controller_log, plant_log)
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



class DelayProvider(ABC):

  @abstractmethod
  def get_delay(self, metadata):
    """ 
    return t_delay, metadata 
    """
    pass # Abstract method must be overridden.

class ScarabDelayProvider(DelayProvider):
  """
  Read the stats files produced by Execution-driven Scarab to get the delay.
  """

  def __init__(self, sim_dir):
    self._stats_file_number = 0
    self._stats_reader         = scarabizor.ScarabStatsReader(sim_dir)
    self._scarab_params_reader = scarabizor.ScarabPARAMSReader(sim_dir)

  def get_delay(self, metadata):
    if debug_levels.debug_interfile_communication_level >= 2:
      print('Waiting for statistics from Scarab.')

    self._stats_reader.waitForStatsFile(self._stats_file_number)
    t_delay           = self._stats_reader.readTime(self._stats_file_number)
    instruction_count = self._stats_reader.readInstructionCount(self._stats_file_number)
    cycles_count      = self._stats_reader.readCyclesCount(self._stats_file_number)
    self._stats_file_number += 1

    delay_metadata = {}
    params_out = self._scarab_params_reader.params_out_to_dictionary()
    delay_metadata.update(params_out)
    delay_metadata["instruction_count"] = instruction_count
    delay_metadata["cycles_count"] = cycles_count
    return t_delay, delay_metadata

class OneTimeStepDelayProvider(DelayProvider):

  def __init__(self, sample_time, sim_dir, use_fake_scarab):
    self.trace_dir_index = 0
    self.sample_time     = sample_time
    self.sim_dir         = sim_dir
    self.use_fake_scarab = use_fake_scarab
    
  def get_delay(self, metadata):
    t_delay = self.sample_time
    metadata = {}

    if self.use_fake_scarab:
      fake_trace_dir = os.path.join(self.sim_dir, f'dynamorio_trace_{self.trace_dir_index}')
      # Create a fake trace directory.
      os.makedirs(fake_trace_dir)
      # Create a README file in the fake trace directory.
      with open(os.path.join(fake_trace_dir, 'README.txt'), 'w') as file:
        file.write(f'This is a fake trace directory created by {type(self)} for the purpose of faking Scarab data for faster testing.')
      # Increment the trace directory index.
      self.trace_dir_index += 1

    return t_delay, metadata

class NoneDelayProvider(DelayProvider):

  def __init__(self):
    pass

  def get_delay(self, metadata):
    t_delay = None
    metadata = {}
    return t_delay, metadata

class GaussianDelayProvider(DelayProvider):

  def __init__(self, mean, std_dev):
    self.mean = mean
    self.std_dev = std_dev

  def get_delay(self, metadata):
    # Generate a random number from the Gaussian distribution
    t_delay = np.random.normal(self.mean, self.std_dev)
    metadata = {}
    return t_delay, metadata

class LinearBasedOnIteraionsDelayProvider(DelayProvider):
  
  def __init__(self):
    pass

  def get_delay(self, metadata):
    # TODO: Move the computation delays generated by a model out of the plant_runner module.
    iterations = metadata["iterations"]
    delay_model_slope       = config_data["computation_delay_model"]["computation_delay_slope"]
    delay_model_y_intercept = config_data["computation_delay_model"]["computation_delay_y-intercept"]
    if delay_model_slope:
      if not delay_model_y_intercept:
        raise ValueError(f"delay_model_slope was set but delay_model_y_intercept was not.")
      t_delay = delay_model_slope * iterations + delay_model_y_intercept
      print(f"t_delay = {t_delay:.8g} = {delay_model_slope:.8g} * {iterations:.8g} + {delay_model_y_intercept:.8g}")
    else:
      print('Using constant delay times.')
      t_delay = config_data["computation_delay_model"]["fake_computation_delay_times"]
    return t_delay, metadata



class ControllerInterface(ABC):
  """ 
  Create an Abstract Base Class (ABC) for a controller interface. 
  This handles communication to and from the controller, whether that is an executable or (for testing) a mock controller.
  """

  def __init__(self, computational_delay_provider: DelayProvider):
    self.computational_delay_provider = computational_delay_provider

  def open(self):
    """
    Open any resources that need to be closed.
    """
    pass

  def close(self):
    """
    Do cleanup of opened resources.
    """
    pass

  @abstractmethod
  def _write_x(self, x: np.ndarray):
    pass

  @abstractmethod
  def _write_t_delay(self, t: float):
    pass
    
  @abstractmethod
  def _read_u(self) -> np.ndarray:
    return u
    
  @abstractmethod
  def _read_x_prediction(self) -> np.ndarray:
    return x_prediction
    
  @abstractmethod
  def _read_t_prediction(self) -> float:
    return t_prediction

  @abstractmethod
  def _read_iterations(self) -> int:
    return iterations
    
  def _read_metadata(self) -> int:
    x_prediction = self._read_x_prediction()
    t_prediction = self._read_t_prediction()
    iterations = self._read_iterations()
    if isinstance(x_prediction, np.ndarray):
      x_prediction = column_vec_to_list(x_prediction)
    metadata = {
      "x_prediction": x_prediction,
      "t_prediction": t_prediction,
      "iterations": iterations
    }

    return metadata
    

  def get_u(self, t, x, u_before, pending_computation_before: ComputationData):
    """ 
    Get an (possibly) updated value of u. 
    Returns: u, u_delay, u_pending, u_pending_time, metadata,
    where u is the value of u that should start being applied immediately (at t). 
    
    If the updated value requires calling the controller, then u_delay does so.

    This function is tested in <root>/tests/test_scarabintheloop_plant_runner.py/Test_get_u
    """
    
    if debug_levels.debug_dynamics_level >= 1:
      printHeader2('----- get_u (BEFORE) ----- ')
      print(f'u_before[0]: {u_before[0]}')
      printJson("pending_computation_before", pending_computation_before)

    if pending_computation_before is not None and pending_computation_before.t_end > t:
      # If last_computation is provided and the end of the computation is after the current time then we do not update anything. 
      # print(f'Keeping the same pending_computation: {pending_computation_before}')
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = u_before = {u_before}. (Computation pending)')
      u_after = u_before
      pending_computation_after = pending_computation_before
      
      if debug_levels.debug_dynamics_level >= 1:
        printHeader2('----- get_u (AFTER - no update) ----- ')
        print(f'u_after[0]: {u_after[0]}')
        printJson("pending_computation_after", pending_computation_after)
      did_start_computation = False
      return u_after, pending_computation_after, did_start_computation

    if pending_computation_before is not None and pending_computation_before.t_end <= t:
      # If the last computation data is "done", then we set u_after to the pending value of u.
      
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = pending_computation_before.u = {pending_computation_before.u}. (computation finished)')
      u_after = pending_computation_before.u
    elif pending_computation_before is None:
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = u_before = {u_before} (no pending computation).')
      u_after = u_before
    else:
      raise ValueError(f'Unexpected case.')
      

    # If there is no pending_computation_before or the given pending_computation_before finishes before the current time t, then we run the computation of the next control value.
    if debug_levels.debug_dynamics_level >= 2:
      print("About to get the next control value for x = ", repr(x))

    u_pending, u_delay, metadata = self.get_next_control_from_controller(x)
    did_start_computation = True
    pending_computation_after = ComputationData(t, u_delay, u_pending, metadata)

    if debug_levels.debug_dynamics_level >= 1:
      printHeader2('----- get_u (AFTER - With update) ----- ')
      print(f'u_after: {u_after}')
      printJson("pending_computation_after", pending_computation_after)
    # if u_delay == 0:
    #   # If there is no delay, then we update immediately.
    #   return u_pending, None
    assert u_delay > 0, 'Expected a positive value for u_delay.'
      # print(f'Updated pending_computation: {pending_computation_after}')
    return u_after, pending_computation_after, did_start_computation


  def get_next_control_from_controller(self, x: np.ndarray):
    """
    Send the current state to the controller and wait for the responses. 
    Return values: u, x_prediction, t_prediction, iterations.
    """

    # The order of writing and reading to the pipe files must match the order in the controller.
    self._write_x(x)
    u = self._read_u()
    metadata = self._read_metadata()
    u_delay, delay_metadata = self.computational_delay_provider.get_delay(metadata)
    self._write_t_delay(u_delay)
    metadata.update(delay_metadata)

    if debug_levels.debug_interfile_communication_level >= 1:
      print('Input strings from C++:')
      printIndented(f"       u: {u}", 1)
      printIndented(f"metadata: {metadata}", 1)

    return u, u_delay, metadata
    
class PipesControllerInterface(ControllerInterface):

  def __init__(self, computational_delay_provider: DelayProvider, sim_dir):
    self.computational_delay_provider = computational_delay_provider
    self.sim_dir = sim_dir
    assertFileExists(self.sim_dir)

    self.u_reader          = None
    self.x_predict_reader  = None
    self.t_predict_reader  = None
    self.iterations_reader = None
    self.x_outfile         = None
    self.t_delay_outfile   = None

  def open(self):
    """
    Open resources that need to be closed when finished.
    """
    assertFileExists(self.sim_dir + '/u_c++_to_py')
    self.u_reader          = PipeVectorReader(os.path.join(self.sim_dir, 'u_c++_to_py'))
    self.x_predict_reader  = PipeVectorReader(os.path.join(self.sim_dir, 'x_predict_c++_to_py'))
    self.t_predict_reader  = PipeFloatReader(os.path.join(self.sim_dir, 't_predict_c++_to_py'))
    self.iterations_reader = PipeFloatReader(os.path.join(self.sim_dir, 'iterations_c++_to_py'))
    self.x_outfile         = open(os.path.join(self.sim_dir, 'x_py_to_c++'), 'w', buffering=1)
    self.t_delay_outfile   = open(os.path.join(self.sim_dir, 't_delay_py_to_c++'), 'w', buffering=1)
    
    if debug_levels.debug_interfile_communication_level >= 1:
      print('Pipes are open') 

    return self  # Return the instance so that it's accessible as 'as' target

  def close(self):
    """ 
    Close all of the files we opened.
    """
    if self.u_reader:
      self.u_reader.close()
    if self.x_predict_reader:
      self.x_predict_reader.close()
    if self.t_predict_reader:
      self.t_predict_reader.close()
    if self.iterations_reader:
      self.iterations_reader.close()
    if self.x_outfile:
      self.x_outfile.close()
    if self.t_delay_outfile:
      self.t_delay_outfile.close()

  def _write_x(self, x: np.ndarray):
    # Pass the string back to C++.
    x_out_string = nump_vec_to_csv_string(x)
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Writing x output line: {x_out_string} to {self.x_outfile.name}")
    self.x_outfile.write(x_out_string + "\n")# Write to pipe to C++

  def _write_t_delay(self, t_delay: float):
    t_delay_str = f"{t_delay:.8g}"
    
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Writing {t_delay_str} to t_delay file: {self.t_delay_outfile.name}")
    self.t_delay_outfile.write(t_delay_str + "\n")

  def _read_u(self):
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading u file: {self.u_reader.filename}")
    return self.u_reader.read()
    
  def _read_x_prediction(self):
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading x_predict file: {self.x_predict_reader.filename}")
    return self.x_predict_reader.read()
    
  def _read_t_prediction(self):
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading t_predict file: {self.t_predict_reader.filename}")
    return self.t_predict_reader.read()
    
  def _read_iterations(self):
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading iterations file: {self.iterations_reader.filename}")
    return self.iterations_reader.read()
      

@contextmanager
def controller_interface_factory(controller_interface_selection, computation_delay_provider, sim_dir):
  """ 
  Generate the desired ControllerInterface object based on the value of "controller_interface_selection". Typically the value will be the string "pipes", but a ControllerInterface interface object can also be passed in directly to allow for testing.
  
  Example usage:

    with controller_interface_factory("pipes", computation_delay_provider, sim_dir) as controller_interface: 
        <do stuff with controller_interface>
    
  When the "with" block is left, controller_interface.close() is called to clean up resources.
  """

  if isinstance(controller_interface_selection, str):
    controller_interface_selection = controller_interface_selection.lower()

  if controller_interface_selection == "pipes":
    controller_interface = PipesControllerInterface(computation_delay_provider, sim_dir)
  elif isinstance(controller_interface_selection, ControllerInterface):
    controller_interface = controller_interface_selection
  else:
    raise ValueError(f'Unexpected controller_interface: {controller_interface}')

  try:
    yield controller_interface.open()
  finally:
    controller_interface.close()


def computation_delay_provider_factory(computation_delay_name: str, sim_dir, sample_time, use_fake_scarab):
  if isinstance(computation_delay_name, str):
    computation_delay_name = computation_delay_name.lower()

  if computation_delay_name == "none":
    return NoneDelayProvider()
  elif computation_delay_name == "gaussian":
    return GaussianDelayProvider(mean=0.24, std_dev=0.05)
  elif computation_delay_name == "execution-driven scarab":
    return ScarabDelayProvider(sim_dir)
  elif computation_delay_name == "onestep":
    return OneTimeStepDelayProvider(sample_time, sim_dir, use_fake_scarab)
  elif isinstance(computation_delay_name, DelayProvider):
    return computation_delay_name
  else:
    raise ValueError(f'Unexpected computation_delay_name: {computation_delay_name}.')
    

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