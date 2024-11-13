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
from scarabintheloop.controller_interface import DelayProvider, ControllerInterface, PipesControllerInterface
from abc import abstractmethod, ABC
from dataclasses import dataclass


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
  def __init__(self, example_dir, config_filename, fail_fast=False):
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
    # In the case where there is only a single configuration, we convert it to a singleton list 
    # so that it can be handled identically.
    if not isinstance(self.experiment_config_patches_list, list):
      self.experiment_config_patches_list = [self.experiment_config_patches_list]

    self.label = slugify(remove_suffix(config_filename, ".json"))
    self.base_config["experiment_list_label"] = self.label

    self.experiment_list_dir        = os.path.join(self.experiments_dir, self.label + "--" + time.strftime("%Y-%m-%d--%H-%M-%S"))
    self.incremental_data_file_path = os.path.join(self.experiment_list_dir, "experiment_list_data_incremental.json")
    self.complete_data_file_path    = os.path.join(self.experiment_list_dir, "experiment_list_data.json")

    # Collections for Exeperiment Results
    self.experiment_result_dict       = {}
    self.successful_experiment_labels = []
    self.skipped_experiment_labels    = []
    self.failed_experiment_labels     = []

    self.is_fail_fast = fail_fast

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
        

    if debug_levels.debug_program_flow_level >= 1:
      printHeader1('Experiment List')
      print(f'         Example dir: {self.example_dir}')
      print(f'        Base config: {self.base_config_file_path}')
      print(f' Experiments config: {self.experiments_config_file_path}')
    
  def run_all(self):
    """
    Run all of the experiments in the ExperimentList.

    Does not return a valud. The results can be accessed after this method by calling get_results().
    """

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
        
        if self.is_fail_fast:
          raise Exception(f'Experiment Failed: {experiment}')

      writeJson(self.incremental_data_file_path, self.experiment_result_dict, label="Incremental experiment list data")
      
      self.print_status()
    
    if debug_levels.debug_program_flow_level >= 1:
      print('Finished running experiments from configuration in\n\t' + self.experiments_config_file_path)
    writeJson(self.complete_data_file_path, self.experiment_result_dict, label="Complete experiment list data")
    
    # Delete the incremental data file.
    os.remove(self.incremental_data_file_path)

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
    
    if debug_levels.debug_program_flow_level >= 1:
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
      # print(f'\tOutput: {PARAMS_out_filename}.')
    
    self.params_base = scarabizor.ParamsData.from_file(PARAMS_src_filename)

  def setup_files(self):
    os.makedirs(self.experiment_dir)

  @indented_print
  def run(self):
    
    # Record the start time.
    start_time = time.time()
    
    if self.experiment_config.get("skip", False):

      if debug_levels.debug_program_flow_level >= 1:
        printHeader1(f'Skipping Experiment: {self.label}')
      self.status = ExperimentStatus.SKIPPED
      return
      
    if debug_levels.debug_program_flow_level >= 1:
      printHeader1(f"Starting Experiment: {self.label}")

    assert_consistent_simulation_config_dimensions(self.experiment_config)
    try:
      self.status = ExperimentStatus.RUNNING
      if self.experiment_config["Simulation Options"]["parallel_scarab_simulation"]:
        experiment_data = run_experiment_parallelized(self.experiment_config, self.params_base)
      else:
        experiment_data = run_experiment_sequential(self.experiment_config, self.params_base)
        
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
      "experiment wall time": end_time - start_time, 
      "experiment data":      experiment_data,
      "experiment config":    self.experiment_config
    }

    # Check that we have the expected number of time steps. In particular, the last value of k must be one less
    # than n_time_steps because k is zero-indexed.
    assert experiment_data["k"][-1] == self.experiment_config["n_time_steps"] - 1, \
      f'experiment_data["k"][-1]={experiment_data["k"][-1]} must equal n_time_steps-1={self.experiment_config["n_time_steps"] - 1}. ({self.label})'

    writeJson(os.path.join(self.experiment_dir, "experiment_result.json"), self.result)

    for pc in experiment_data["pending_computations"]:
      if pc: 
        print(f'iterations: {pc.metadata["iterations"]:04d} cost: {pc.metadata["cost"]:10.9g}, status: {pc.metadata["status"]}, constraint_error={pc.metadata["constraint_error"]}, dual_residual={pc.metadata["dual_residual"]}')
    printJson('x', experiment_data["x"])
    printJson('u', experiment_data["u"])

  def __repr__(self):
    repr_str = f'Experiment("{self.label}", {self.status.name}'
    if self.status == ExperimentStatus.FAILED:
      repr_str += f', error msg: "{self.exception}"'
    if self.run_time:
      repr_str += f' in {seconds_to_duration_string(self.run_time)}'
    repr_str += ')'
    return repr_str

class Simulation:
  label: str
  x0: np.ndarray
  u0: np.ndarray
  simulation_dir: str
  first_time_index: int
  last_time_index: int
  n_time_steps: int
  n_time_indices: int
  time_steps: list
  time_indices: list
  pending_computation: ComputationData
  params: scarabizor.ParamsData
  controller_log_path: str
  plant_log_path: str
  params_file_for_controller: str
  config_file_for_controller: str
  sim_config: dict

  @staticmethod
  def from_experiment_config_unbatched(experiment_config: dict, params_base: scarabizor.ParamsData, n_time_steps: int):
    label               = experiment_config["label"]
    first_time_index    = 0
    x0                  = list_to_column_vec(experiment_config["x0"])
    u0                  = list_to_column_vec(experiment_config["u0"])
    pending_computation = None
    simulation_dir      = experiment_config["experiment_dir"]
    params = params_base.patch(experiment_config["PARAMS_patch_values"])
    return Simulation(label=label, 
                      first_time_index=first_time_index,
                      n_time_steps=n_time_steps,
                      x0=x0,
                      u0=u0,
                      pending_computation=pending_computation,
                      simulation_dir=simulation_dir,
                      experiment_config=experiment_config,
                      params=params)

  @staticmethod
  def from_experiment_config_batched(experiment_config: dict,
                                     params_base: scarabizor.ParamsData,
                                     batch_init,# : BatchInit,
                                     n_time_steps_in_batch: int):
    # !! We do not, in general, expect experiment_config['u0']==batch_init['u0'] because 
    # !! batch_init overrides the experiment config.
    assert isinstance(batch_init, BatchInit), f'type(batch_init)={type(batch_init)} must be BatchInit.'
    # experiment_max_time_steps       = experiment_config["n_time_steps"]
    # experiment_max_batch_time_steps = experiment_config["Simulation Options"]["max_batch_size"]
    
    first_time_index    = batch_init.k0
    pending_computation = batch_init.pending_computation
    x0                  = list_to_column_vec(batch_init.x0)
    u0                  = list_to_column_vec(batch_init.u0)

    # time_steps_remaining_in_experiment = experiment_max_time_steps - first_time_index
    # assert time_steps_remaining_in_experiment > 0

    # Truncate the index to not extend past the last index
    # time_steps_in_sim = min(os.cpu_count(), \
    #                         experiment_max_batch_time_steps, \
    #                         time_steps_remaining_in_experiment)

    # Add "time_steps_in_sim" steps to the first index to get the last index
    last_time_index = first_time_index + n_time_steps_in_batch

    # assert last_time_index <= experiment_max_time_steps, 'last_time_index_in_batch must not be past the end of experiment.'
    # assert first_time_index < last_time_index, f'first_time_index: {first_time_index}, last_time_index={last_time_index}'
    first_time_step= first_time_index
    last_time_step = last_time_index-1
    batch_label    = f'batch{batch_init.i_batch}_steps{first_time_step}-{last_time_step}'
    simulation_dir = os.path.join(experiment_config["experiment_dir"], batch_label)
    label          = experiment_config["label"] + "/" + batch_label
    
    # Patch the base params object with the values from the experiment configuration.
    params = params_base.patch(experiment_config["PARAMS_patch_values"])
    
    return Simulation(label=label, 
                      first_time_index=first_time_index,
                      n_time_steps=n_time_steps_in_batch,
                      x0=x0,
                      u0=u0,
                      pending_computation=pending_computation,
                      simulation_dir=simulation_dir,
                      experiment_config=experiment_config,
                      params=params)

  def __init__(self, label: str, 
                     first_time_index: int, 
                     n_time_steps: int, 
                     x0: np.ndarray, 
                     u0: np.ndarray, 
                     pending_computation: ComputationData, 
                     simulation_dir: str, 
                     experiment_config: dict, 
                     params: scarabizor.ParamsData):
    assert isinstance(first_time_index, int)
    assert isinstance(n_time_steps, int)
    assert isinstance(x0, np.ndarray)
    assert isinstance(u0, np.ndarray)
    assert pending_computation is None or isinstance(pending_computation, ComputationData)
    assert isinstance(simulation_dir, str)
    assert isinstance(experiment_config, dict)
    assert isinstance(params, scarabizor.ParamsData)
    self.label               = label
    self.x0                  = x0
    self.u0                  = u0
    self.simulation_dir      = simulation_dir
    self.first_time_index    = first_time_index
    self.n_time_steps        = n_time_steps
    self.pending_computation = pending_computation
    self.params              = params

    # Paths for logs.
    self.controller_log_path = os.path.join(self.simulation_dir, 'controller.log')
    self.plant_log_path      = os.path.join(self.simulation_dir, 'dynamics.log')

    # Paths to files needed by controller executable.
    self.params_file_for_controller = os.path.join(self.simulation_dir, 'PARAMS.generated')
    self.config_file_for_controller = os.path.join(self.simulation_dir, 'config.json')
    
    self.sim_config = copy.deepcopy(experiment_config)
    self.sim_config["simulation_label"]    = label
    self.sim_config["first_time_index"]    = first_time_index
    self.sim_config["last_time_index"]     = self.last_time_index
    self.sim_config["n_time_steps"]        = self.n_time_steps
    self.sim_config["x0"]                  = self.x0
    self.sim_config["u0"]                  = self.u0
    self.sim_config["pending_computation"] = pending_computation

  @property
  def last_time_index(self):
    return self.first_time_index + self.n_time_steps 
  
  @property
  def n_time_indices(self):
    return self.n_time_steps + 1
  
  @property
  def time_steps(self):
    return list(range(self.first_time_index, self.last_time_index))
    
  @property 
  def time_indices(self):
    return self.time_steps + [self.last_time_index]

  def setup_files(self):

    # Create the simulation directory if it does not already exist.
    os.makedirs(self.simulation_dir, exist_ok=True)

    # Write the configuration to config.json.
    writeJson(os.path.join(self.simulation_dir, "config.json"), self.sim_config)

    PARAMS_out_file = os.path.join(self.simulation_dir, 'PARAMS.generated')
    self.params.to_file(PARAMS_out_file)

    # # Create the pipe files in the current directory.
    # run_shell_cmd("make_scarabintheloop_pipes.sh", working_dir=self.simulation_dir)
    # assertFileExists(self.simulation_dir + "/u_c++_to_py")
    # assertFileExists(self.simulation_dir + "/x_py_to_c++")

    sample_time = self.sim_config["system_parameters"]["sample_time"]
    use_fake_delays = self.sim_config["Simulation Options"]["use_fake_delays"]
    if use_fake_delays:
      # Create some fake delay data.
      # Create a list of delays that are shorter than the sample time
      fake_delays = [0.1*sample_time]*self.n_time_steps

      # Update one of the queued delays to be longer than the sample time
      index_for_delay = 2
      if len(fake_delays) >= index_for_delay+1:
        fake_delays[index_for_delay] = 1.1*sample_time
    else:
      fake_delays = None

    in_the_loop_delay_provider =  self.sim_config["Simulation Options"]["in-the-loop_delay_provider"]
    # delay_provider_config = sim_config["Simulation Options"]["in-the-loop_delay_provider"]
    # assert in_the_loop_delay_provider == "execution-driven scarab", \
    #   f'delay_provider_config={in_the_loop_delay_provider} must be "execution-driven scarab" when doing serial execution.'
    computation_delay_provider = computation_delay_provider_factory(in_the_loop_delay_provider, self.simulation_dir, sample_time, fake_delays)
    
    controller_interface = PipesControllerInterface(computation_delay_provider, self.simulation_dir)
    # controller_interface_factory("pipes", computation_delay_provider, )

    self.simulation_executor = getSimulationExecutor(
                                                self.simulation_dir, 
                                                self.sim_config, 
                                                controller_interface,
                                                fake_delays = fake_delays)

  @indented_print
  def run(self) -> TimeStepSeries:

    if debug_levels.debug_program_flow_level >= 1:
      printHeader2( f'Starting simulation: {self.label} ({self.sim_config["n_time_steps"]} time steps)')
      print(f'↳ Simulation dir: {self.simulation_dir}')
      print(f'↳ Controller log: {self.controller_log_path}')
      print(f'↳      Plant log: {self.plant_log_path}')

    assertFileExists(self.params_file_for_controller)
    assertFileExists(self.config_file_for_controller)  

    # Open up some loooooooooogs.
    with openLog(self.plant_log_path,           f'Plant Log for "{self.label}"') as plant_log, \
         openLog(self.controller_log_path, f'Controller Log for "{self.label}"') as controller_log:
      self.simulation_executor.set_logs(controller_log, plant_log)

      # Execute the simulation.
      simulation_data = self.simulation_executor.run_simulation()
      
      sim_data_path = os.path.join(self.simulation_dir, 'simulation_data.json')
      if debug_levels.debug_program_flow_level >= 1:
        printHeader2(f'Finished Simulation: "{self.label}"')
        print(f'↳ Simulation dir: {self.simulation_dir}')
        print(f'↳ Controller log: {self.controller_log_path}')
        print(f'↳      Plant log: {self.plant_log_path}')
        print(f"↳  Data out file: {sim_data_path}")
      
      simulation_data.metadata["controller_log_path"] = self.controller_log_path
      simulation_data.metadata["plant_log_path"]      = self.plant_log_path

    return simulation_data

# class DelayMode(Enum):
#   """
#   The DelayMode enum is used to specify the type of computation delay that is used in the simulation.
#   """
#   # For GRID_ALIGNED, the controller delay is always a multiple of the sample time. 
#   # If the computation finishes before the next sample time, the controller will wait until the next sample time to update the control signal. 
#   GRID_ALIGNED = 1
#   # For ASAP, the controller will update the control signal as soon as the computation is finished.
#   ASAP = 2
#   # For NONE, the controller will update the control signal at the time when the computation starts, ignoring the actual computational delay.
#   NONE = 3
# 
# class DelaySource(Enum):
#   SCARAB = 1
#   FAKE   = 2
# 
# class ExecutionMode(Enum):
#   PARALLEL = 1
#   SERIAL = 2


# def get_controller_interface(delay_mode: DelayMode, delay_source: DelaySource, execution_mode: ExecutionMode):
#   """
#   Factory method for creating a controller interface.
#   """
#   if delay_source == DelaySource.SCARAB:
#     if execution_mode == ExecutionMode.PARALLEL:
#       return ParallelControllerInterface(delay_mode)
#     elif execution_mode == ExecutionMode.SERIAL:
#       return SerialControllerInterface(delay_mode)
#     else:
#       raise ValueError(f'execution_mode={execution_mode} is not a valid ExecutionMode.')
#   elif delay_source == DelaySource.FAKE:
#     return FakeDelayControllerInterface(delay_mode)
#   else:
#     raise ValueError(f'delay_source={delay_source} is not a valid DelaySource.')
# 
# class DelaysProvider(ABC):
#   pass
# 
# class FakeDelaysProvider(DelaysProvider):
#   pass
# 
# class ExecutionDrivenScarabDelaysProvider(DelaysProvider):
#   pass
# 
# class TraceBasedScarabDelaysProvider(DelaysProvider):
#   pass
# 
# class ControllerInterface(ABC):
# 
#   def __init__(self, state_dimension: int, input_dimension: int):
#     self.state_dimension = state_dimension
#     self.input_dimension = input_dimension
# 
#   @abstractmethod
#   def get_control_computation(self, x) -> np.ndarray:
#     """
#     Calculate the control signal u given the state x and return a ControlComputation object.
#     The delay time of the ControlComputation object may be a preliminary value that is updated later by the 
#     get_actual_delay() function.
#     """
#     pass
# 
#   @abstractmethod
#   def get_actual_delay() -> float:
#     """
#     Calculate and return the actual delay.
# 
#     Returns:
#       float: The actual delay in seconds. If 
#     """
#     pass
# 
# 
# class MockControllerInterface(ControllerInterface):
#   def __init__(self, delay: float):
#     self.delay = delay
# 
#   def get_control_computation(self, x) -> np.ndarray:
#     return ControlComputation(0, 0, self.delay, 0)
# 
#   def get_actual_delay(self) -> float:
#     return self.delay
# 
def load_controller_delegator(example_dir: str):
  global controller_executable_provider
  controller_delegator_module = loadModuleInDir(example_dir, "controller_delegator")
  controller_executable_provider = controller_delegator_module.ControllerExecutableProvider(example_dir)


def run(example_dir:str, config_filename:str, fail_fast = False):
  """
  This function is the entry point for running scarabintheloop 
  from other Python scripts (e.g., in unit testint). 
  When running from command line, the entry point is main(), instead,
  which then calls this function. 

  Set fail_fast to True to abort after the first failed experiment.
  """

  ######## Load the controller and plant modules from the current directory #########
  assert example_dir is not None
  example_dir = os.path.abspath(example_dir)
  assertFileExists(example_dir)
  
  experiment_list = ExperimentList(example_dir, config_filename, fail_fast)
  #----- CONFIGURE DEBUGGING LEVELS -----#
  debug_levels.set_from_dictionary(experiment_list.base_config["==== Debgugging Levels ===="])

  load_controller_delegator(example_dir)
  
  experiment_list.run_all()
  return experiment_list
    
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
  example_dir = os.path.abspath('.')
  experiment_list = run(example_dir, args.config_filename)

  if experiment_list.n_failed():
    exit(1)

def run_experiment_sequential(experiment_config, params_base: scarabizor.ParamsData):
  print(f'Start of run_experiment_sequential(<{experiment_config["experiment_label"]}>)')
  simulation = Simulation.from_experiment_config_unbatched(experiment_config, params_base, n_time_steps=experiment_config["n_time_steps"])
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

# class BatchStatus(Enum):
#   PENDING = 1
#   RUNNING = 2
#   SUCCESS = 3
#   FAILURE = 4

@dataclass(frozen=True)
class BatchInit:
  i_batch: int
  k0: int
  t0: float
  x0: np.ndarray
  u0: np.ndarray
  pending_computation: ComputationData

  @staticmethod
  def first(x0, u0): # -> BatchInit
    x0 = list_to_column_vec(x0)
    u0 = list_to_column_vec(u0)
    return BatchInit(i_batch=0, k0=0, t0=0.0, x0=x0, u0=u0, pending_computation=None)

  @staticmethod
  def _from_lists(
          i_batch: int,
          k0: int,
          t0: float,
          x0: np.ndarray,
          u0: np.ndarray,
          pending_computation: ComputationData = None
        ):
    """
    To make testing simpler, this function allows 
    for creating a BatchInit object with x0 and u0 
    provided as lists instead of vectors.
    """
    x0 = list_to_column_vec(x0)
    u0 = list_to_column_vec(u0)
    return BatchInit(i_batch, k0, t0, x0, u0, pending_computation)

  def __post_init__(self):
    assert isinstance(self.k0, int)
    assert isinstance(self.t0, float)
    assert isinstance(self.x0, np.ndarray), f'x0={type(self.x0)} must be np.ndarray'
    assert isinstance(self.u0, np.ndarray), f'u0={type(self.u0)} must be np.ndarray'
    if self.pending_computation:
      assert isinstance(self.pending_computation, ComputationData)

  def __eq__(self, other):
    assert isinstance(other, BatchInit), f'Cannot compare equality between this BatchInit and a {type(other)}'
    return self.i_batch == other.i_batch \
       and self.t0 == other.t0 \
       and self.k0 == other.k0 \
       and np.array_equal(self.x0, other.x0) \
       and np.array_equal(self.u0, other.u0) \
       and self.pending_computation == other.pending_computation

  def __str__(self):
    return f'BatchInit(i_batch={self.i_batch}, t0={self.t0}, k0={self.k0}, {"with pending computation" if self.pending_computation else "no pending computation"})'

  def to_dict(self): 
    return  {
                "i_batch": self.i_batch,
                     "t0": self.t0,
                     "k0": self.k0, # First time step (and time index---they are equal).
                     "x0": self.x0,
                     "u0": self.u0,
    "pending_computation": self.pending_computation
  }

class Batch:

  def __init__(self, batch_init: BatchInit, full_simulation_data: TimeStepSeries, sample_time: float):
    assert isinstance(batch_init, BatchInit)
    assert isinstance(full_simulation_data, TimeStepSeries)
    assert isinstance(sample_time, float)
    self.batch_init      = batch_init
    self.full_simulation_data = full_simulation_data.copy()

    first_late_timestep, has_missed_computation = full_simulation_data.find_first_late_timestep(sample_time)

    if has_missed_computation:
      valid_simulation_data = full_simulation_data.truncate(first_late_timestep)
    else: 
      valid_simulation_data = full_simulation_data.copy()

    if debug_levels.debug_batching_level >= 2:
      if has_missed_computation:
        valid_simulation_data.printTimingData(f'valid_simulation_data (truncated to {first_late_timestep} time steps.)')
      else:
        valid_simulation_data.printTimingData(f'valid_simulation_data (no truncated time steps.)')

    # # Check that the next_first_time_index doesn't skip over any timesteps.
    # last_time_index_in_previous_batch = valid_simulation_data.last_time_index
    # assert next_batch_init.k0 <= last_time_index_in_previous_batch + 1 , \
    #       f'next first_time_index={next_batch_init.k0} must be larger than' + \
    #       f'(last index in the batch) + 1={full_simulation_data.i[-1] + 1}'

    if first_late_timestep:
      self.first_late_timestep  = first_late_timestep 
      self.last_valid_timestep  = first_late_timestep
    else:
      self.first_late_timestep  = None
      self.last_valid_timestep  = full_simulation_data.k[-1]
    self.has_missed_computation = has_missed_computation
    self.valid_simulation_data  = valid_simulation_data

  def next_init(self) -> BatchInit:
    assert self.valid_simulation_data is not None

    valid_simulation_data = self.valid_simulation_data
    # Values that are needed to start the next batch.
    if valid_simulation_data.n_time_steps == 0:
      return BatchInit(
          batch_init=self.batch_init.i_batch + 1, 
          k0=self.batch_init.k0 + 1, 
          t0=self.batch_init.t0, 
          x0=list_to_column_vec(self.batch_init.x0), 
          u0=list_to_column_vec(self.batch_init.u0), 
          pending_computation=self.batch_init.pending_computation)
    else:
      return BatchInit( 
          i_batch=self.batch_init.i_batch + 1, 
          k0=valid_simulation_data.k[-1] + 1, 
          t0=valid_simulation_data.t[-1], 
          x0=list_to_column_vec(valid_simulation_data.x[-1]), 
          u0=list_to_column_vec(valid_simulation_data.u[-1]), 
          pending_computation=valid_simulation_data.pending_computation[-1])

  def __eq__(self, other):
    assert isinstance(other, Batch), f'Cannot compare equality between a Batch object and {type(other)}.'
    return self.batch_init             == other.batch_init \
       and self.full_simulation_data   == other.full_simulation_data \
       and self.valid_simulation_data  == other.valid_simulation_data \
       and self.first_late_timestep    == other.first_late_timestep \
       and self.has_missed_computation == other.has_missed_computation \
       and self.last_valid_timestep    == other.last_valid_timestep

  def __repr__(self):
    return f'Batch(init={self.batch_init}, last valid: {self.last_valid_timestep}, first late: {self.first_late_timestep})'

  def to_dict(self): 
    return {
      'batch_init': self.batch_init,
      'valid_simulation_data': self.valid_simulation_data,
      'full_simulation_data': self.full_simulation_data
    }
  
class Batcher:

  def __init__(self, 
               x0, u0, 
               run_batch_fnc, 
               n_time_steps,
               max_batch_size=99999,
               max_batch_count=99999 # setting max # of batches not needed currently.
              ):
    assert callable(run_batch_fnc), f'run_batch_fnc={run_batch_fnc} must be callable (e.g., a function).'
    assert isinstance(max_batch_size, int)
    assert isinstance(max_batch_count, int)
    assert isinstance(n_time_steps, int)
    x0 = list_to_column_vec(x0)
    u0 = list_to_column_vec(u0)
    self._next_batch_init = BatchInit.first(x0, u0)
    self._run_batch_fnc = run_batch_fnc
    self._max_batch_size = max_batch_size
    self._max_batch_count = max_batch_count
    self._n_total_time_steps = n_time_steps
    self._last_batch = None

  def __iter__(self):
    return self

  def __next__(self) -> Batch:
    if self._is_done():
      raise StopIteration
    batch = self._run_batch_fnc(self._next_batch_init, self._next_batch_size())
    self._next_batch_init = batch.next_init()
    self._last_batch = batch

    assert self._next_batch_init is not None
    return batch

  def _next_batch_size(self):
    return min(self._max_batch_size, self._n_time_steps_remaining())

  def _n_time_steps_remaining(self):
    # # "-1" because of zero indexing of the k-values.
    # return self._n_total_time_steps - self._last_valid_k() - 1
    return self._n_total_time_steps - self._next_k0()

  def _last_valid_k(self):
    if self._last_batch is None:
      return None
    else:
      return self._last_batch.last_valid_timestep
    
  def _next_k0(self):
    if self._last_valid_k() is None:
      return 0
    else:
      return self._last_valid_k() + 1
    
  def _is_done(self): 
    assert isinstance(self._next_batch_init, BatchInit), f'self._next_batch_init={self._next_batch_init}'
    return self._next_batch_init.k0 >= self._n_total_time_steps \
        or self._next_batch_init.i_batch >= self._max_batch_count
    

def assert_consistent_simulation_config_dimensions(simulation_config: dict):
  x0 = simulation_config["x0"]
  u0 = simulation_config["u0"]
  state_dimension = simulation_config["system_parameters"]["state_dimension"]
  input_dimension = simulation_config["system_parameters"]["input_dimension"]
  x_names = simulation_config["system_parameters"]["x_names"]
  u_names = simulation_config["system_parameters"]["u_names"]
  assert len(x0) == state_dimension, f'x0={x0} did not have the expected number of entries: state_dimension={state_dimension}'
  assert len(u0) == input_dimension, f'u0={u0} did not have the expected number of entries: input_dimension={input_dimension}'
  assert len(x_names) == state_dimension, \
          f'x_names={x_names} did not have the expected number of entries: state_dimension={state_dimension}'
  assert len(u_names) == input_dimension, \
          f'u_names={simulation_config["u_names"]} did not have the expected number of entries: input_dimension={input_dimension}'
  


def run_experiment_parallelized(experiment_config, params_base: list):
  print(f'Start of run_experiment_parallelized(<{experiment_config["experiment_label"]}>")')
  
  experiment_dir = experiment_config["experiment_dir"]
  n_time_steps = experiment_config["n_time_steps"]
  max_batch      = experiment_config["Simulation Options"]["max_batches"]
  max_batch_size = experiment_config["Simulation Options"]["max_batch_size"]
  sample_time    = experiment_config["system_parameters"]["sample_time"]

  def run_batch(batch_init: BatchInit, n_time_steps) -> Batch:
    simulation = Simulation.from_experiment_config_batched(
                              experiment_config=experiment_config,
                              params_base=params_base,
                              batch_init =batch_init,
                              n_time_steps_in_batch=n_time_steps)
    assert simulation.n_time_steps == n_time_steps
    simulation.setup_files()
    batch_sim_data = simulation.run()

    printJson("simulation config", simulation.sim_config)
    print('batch_init:', batch_init)
    batch_sim_data.printTimingData(f'batch_sim_data from running simulation in "run_batch()" with n_time_steps={n_time_steps}')
    print('batch_sim_data:', batch_sim_data)
    assert batch_sim_data.n_time_steps == n_time_steps, \
      f'batch_sim_data.n_time_steps={batch_sim_data.n_time_steps} must equal n_time_steps={n_time_steps}'
    
    batch = Batch(batch_init, batch_sim_data, sample_time)
    assert batch.valid_simulation_data.n_time_steps <= n_time_steps, \
      f'batch.valid_simulation_data.n_time_steps={batch.valid_simulation_data.n_time_steps} ' \
      + f'must be less than n_time_steps={n_time_steps}.'
    return batch

  ### Batches Loop ###
  # Loop through batches of time-steps until the start time-step is no longer less 
  # than the total number of time-steps desired or until the max number of batches is reached.
  actual_time_series = TimeStepSeries(k0=0, t0=0.0, x0=experiment_config["x0"])
  # Create lists for recording the true values (discarding values that erroneously use  missed computations as though were not missed).
  batch_list = []
  max_batch_size = min(os.cpu_count(), max_batch_size)
  batcher = Batcher(             x0 = experiment_config["x0"], 
                                 u0 = experiment_config["u0"],
                      run_batch_fnc = run_batch, 
                     max_batch_size = max_batch_size,
                    max_batch_count = max_batch,
                       n_time_steps = n_time_steps)

  # try:
  for batch in batcher:
    assert batch, f'batch={batch} should not be empty.'
    batch_list.append(batch)
    # Append all of the valid data (up to the point of the missed computation) except for the first index, 
    # which overlaps with the last index of the previous batch.
    batch.valid_simulation_data.printTimingData(f'Actual time series before appending batch #{actual_time_series}')
    batch.valid_simulation_data.printTimingData(f'Batch #{batch.batch_init.i_batch}')
    actual_time_series += batch.valid_simulation_data
    pending_computation = batch.batch_init.pending_computation
    if pending_computation and pending_computation.t_end < batch.batch_init.t0:
      assert np.array_equal(list_to_column_vec(batch.valid_simulation_data.u[0]), pending_computation.u), \
        f'batch.valid_simulation_data.u[0] = {batch.valid_simulation_data.u[0]} must equal pending_computation.u={pending_computation.u}'
    else:
      assert np.array_equal(list_to_column_vec(batch.valid_simulation_data.u[0]), batch.batch_init.u0), \
      f'batch.valid_simulation_data.u[0] = {batch.valid_simulation_data.u[0]} must equal batch_init.u0={batch.batch_init.u0}'

    experiment_data = {"batches": batch_list,
                        "x": actual_time_series.x,
                        "u": actual_time_series.u,
                        "t": actual_time_series.t,
                        "k": actual_time_series.k, # Time steps
                        "i": actual_time_series.i, # Sample time indices
                        "pending_computations": actual_time_series.pending_computation,
                        "config": experiment_config}

    writeJson(experiment_dir + "/experiment_data_incremental.json", experiment_data, label="Incremental experiment data")

      # # Update values for next iteration of the loop.
      # batch_init = batch.next_batch_init
  # except Exception as err:
  #   raise Exception(f'There was an exception when running batcher: {batcher}')
    
  if debug_levels.debug_batching_level >= 1:
    actual_time_series.print('--------------- Actualualized Time Series (Concatenated) ---------------')

  writeJson(experiment_dir + "/experiment_data.json", experiment_data, label="Experiment data")
  return experiment_data


# def processBatchSimulationData(batch_init:dict, batch_simulation_data: dict, sample_time):
#   first_late_timestep, has_missed_computation = batch_simulation_data.find_first_late_timestep(sample_time)
# 
#   if has_missed_computation:
#     valid_simulation_data = batch_simulation_data.truncate(first_late_timestep)
#   else: 
#     valid_simulation_data = batch_simulation_data.copy()
# 
#   if has_missed_computation:
#     valid_simulation_data.printTimingData(f'valid_simulation_data (truncated to {first_late_timestep} time steps.)')
#   else:
#     valid_simulation_data.printTimingData(f'valid_simulation_data (no truncated time steps.)')
# 
#   # Values that are needed to start the next batch.
#   if len(valid_simulation_data.k) > 0:
#     next_batch_init = {
#       "i_batch":              batch_init["i_batch"] + 1,
#       "first_time_index":     valid_simulation_data.k[-1] + 1,
#       "x0":                   valid_simulation_data.x[-1],
#       "u0":                   valid_simulation_data.u[-1],
#       "pending_computation":  valid_simulation_data.pending_computation[-1]
#     }
#     if has_missed_computation:
#       next_batch_init["Last batch status"] = f"Missed computation at timestep[{first_late_timestep}]"
#     else:
#       next_batch_init["Last batch status"] = f"No missed computations."
#   else:
#     next_batch_init = copy.deepcopy(batch_init)
#     next_batch_init["i_batch"] = batch_init["i_batch"] + 1
#     next_batch_init["Last batch status"]: f'No valid data. Carried over from previous init, which had Last batch status={batch_init["Last batch status"]}'
# 
#   # Check that the next_first_time_index doesn't skip over any timesteps.
#   last_time_index_in_previous_batch = batch_simulation_data.last_time_index
#   assert next_batch_init['first_time_index'] <= last_time_index_in_previous_batch + 1 , \
#         f'next first_time_index={next_batch_init["first_time_index"]} must be larger than' + \
#         f'(last index in the batch) + 1={batch_simulation_data.i[-1] + 1}'
# 
#   return valid_simulation_data, next_batch_init


def getSimulationExecutor(sim_dir, 
                          sim_config, 
                          controller_interface: ControllerInterface, 
                          fake_delays=None):
  """ 
  This function implements the "Factory" design pattern, where it returns objects of various classes depending on the imputs.
  """
  assert isinstance(controller_interface, ControllerInterface)
  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]    
  if use_parallel_scarab_simulation:
    print("Using ParallelSimulationExecutor.")

    if fake_delays:
      # Create the mock Scarab runner.
      trace_processor = scarabizor.MockTracesToComputationTimesProcessor(sim_dir, delays=fake_delays)
    else:
      # The "real" trace processor
      trace_processor = scarabizor.ScarabTracesToComputationTimesProcessor(sim_dir)

    return ParallelSimulationExecutor(sim_dir, sim_config, controller_interface, trace_processor)
  else:
    print("Using SerialSimulationExecutor.")
        
    if fake_delays:
      # Create the mock Scarab runner.
      scarab_runner = scarabizor.MockExecutionDrivenScarabRunner(
                                                       sim_dir=sim_dir,
                                                 queued_delays=fake_delays)
    else:
      scarab_runner = scarabizor.ExecutionDrivenScarabRunner(sim_dir=sim_dir)
    executor = SerialSimulationExecutor(sim_dir, sim_config, controller_interface, scarab_runner)

    return executor

# TODO: Maybe rename this as "delegator" or "coordinator".
class SimulationExecutor(ABC):
  
  def __init__(self, sim_dir, sim_config, controller_interface: ControllerInterface):
    global controller_executable_provider
    self.sim_dir = sim_dir
    self.sim_config = sim_config
    assert controller_interface is not None
    self.controller_interface = controller_interface
    self.sim_config = sim_config

  def set_logs(self, controller_log, plant_log):
    self.controller_log = controller_log
    self.plant_log = plant_log

  @abstractmethod
  def run_controller(self):
    pass

  def _run_controller(self):
    """
    Run the controller via the run_controller() function defined in subclasses, but include some setup and tear-down beforehand.
    """
    # Create the executable.
    try:
      with redirect_stdout(self.controller_log):
        self.controller_executable = controller_executable_provider.get_controller_executable(self.sim_config)
      assertFileExists(self.controller_executable)
      print("The controller_executable is ", self.controller_executable)
    except Exception as err:
      print(f'Failed to run command with {type(self.scarab_runner)}: "{self.controller_executable}".\n{type(err)}: {err}\n{traceback.format_exc()}')
      raise err

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
    Handle setup and exception handling for running the run_plant() functions implemented in subclasses.
    """
    if debug_levels.debug_program_flow_level >= 2:
      print('Start of SimulationExecutor._run_plant() (Plant Dynamics task).')

    # Get a function that defines the plant dynamics.
    with redirect_stdout(self.plant_log):
      # Add dynamics directory to the path
      sys.path.append(os.environ['DYNAMICS_DIR'])
      dynamics_class = getattr(importlib.import_module(self.sim_config["dynamics_module_name"]),  self.sim_config["dynamics_class_name"])
      self.dynamics = dynamics_class(self.sim_config)

    sim_label = self.sim_config["simulation_label"]
    if debug_levels.debug_program_flow_level >= 1:
      print(f'---- Starting plant dynamics for {sim_label} ----\n' \
            f'\t↳ Simulation dir: {self.sim_dir} \n' \
            f'\t↳      Plant log: {self.plant_log.name}\n')
    if debug_levels.debug_configuration_level >= 2:
      printJson(f"Simulation configuration", self.sim_config)

    try:
      self.controller_interface.open()
      if debug_levels.debug_program_flow_level >= 2:
        print('Calling plant_runner.run(...)...')
      with redirect_stdout(self.plant_log):
        plant_result = plant_runner.run(self.sim_dir, self.sim_config, self.dynamics, self.controller_interface)
    except Exception as e:
      self.plant_log.write(f'Plant dynamics had an error: {e}')
      raise Exception(f'Plant dynamics for "{sim_label}" had an error. See logs: {self.plant_log.name}') from e
      
    finally:
      self.controller_interface.close()

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if debug_levels.debug_dynamics_level >= 1:
      print('-- Finished: SimulationExecutor._run_plant() (Plant Dynamics task) -- \n' \
            f'\t↳ Simulation directory: {self.sim_dir}')
    return plant_result

  def run_simulation(self) -> TimeStepSeries:
    """ Run everything needed to simulate the plant and dynamics, in parallel."""
    # Run the controller in parallel, on a separate thread.
    N_TASKS = 1
    with ThreadPoolExecutor(max_workers=N_TASKS) as executor:
      # Start a separate thread to run the controller.
      
      if debug_levels.debug_program_flow_level >= 2:
        print(f"Start of SimulationExecutor.run_simulation()...")
      controller_task = executor.submit(self._run_controller)
      
      if debug_levels.debug_program_flow_level >= 1:
        controller_task.add_done_callback(lambda _: print("controller_task finished."))

      # Start running the plant in the current thread.
      print("Starting the plant...")
      try:
        simulation_data = self._run_plant()
      except Exception as err:
        print(f'Failed to run plant: {err}\n{traceback.format_exc()}')
        raise err
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
    # checkSimulationData(simulation_data, self.sim_config["n_time_steps"])
    return simulation_data

  def postprocess_simulation_data(self, simulation_data: TimeStepSeries) -> TimeStepSeries:
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
  scarab_runner: scarabizor.ExecutionDrivenScarabRunner

  def __init__(self, sim_dir: str, sim_config: dict, controller_interface: ControllerInterface, scarab_runner: scarabizor.ExecutionDrivenScarabRunner):
    super().__init__(sim_dir, sim_config, controller_interface)
    self.scarab_runner = scarab_runner

  def run_controller(self):
    cmd = " ".join([self.controller_executable])
    print("Starting scarab_runner: ", self.scarab_runner, 'to execute', cmd)
    try:
      self.scarab_runner.run(cmd)
    except Exception as err:
      print(f'Failed to run command with {type(self.scarab_runner)}: "{cmd}".\n{type(err)}: {err}\n{traceback.format_exc()}')
      raise err
  
  def set_logs(self, controller_log, plant_log):
    super().set_logs(controller_log, plant_log)
    self.scarab_runner.set_log(controller_log)

class ParallelSimulationExecutor(SimulationExecutor):
  trace_processor: scarabizor.TracesToComputationTimesProcessor

  def __init__(self, sim_dir, sim_config, controller_interface: ControllerInterface, trace_processor: scarabizor.TracesToComputationTimesProcessor):
    super().__init__(sim_dir, sim_config, controller_interface)
    self.trace_processor = trace_processor

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def postprocess_simulation_data(self, simulation_data: TimeStepSeries) -> TimeStepSeries:
    """ 
    Runnining the controller exectuable produces DynamoRIO traces but does not give 
    use simulated computation times. In this function, we simulate the traces 
    using Scarab to get the computation times.
    """
    computation_times = self.trace_processor.get_all_computation_times()
    while len(computation_times) < simulation_data.n_time_steps :
      computation_times += [None]
    assert len(computation_times) == simulation_data.n_time_steps , \
          f"""computation_times={computation_times} must have 
          simulation_data.n_time_steps={simulation_data.n_time_steps } many values, 
          because there is a computation during each time step. 
          None values represent times when computations are not running.
          """
    simulation_data.overwrite_computation_times(computation_times)
    return simulation_data


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

# class LinearBasedOnIteraionsDelayProvider(DelayProvider):
#   
#   def __init__(self):
#     pass
# 
#   def get_delay(self, metadata):
#     iterations = metadata["iterations"]
#     delay_model_slope       = config_data["computation_delay_model"]["computation_delay_slope"]
#     delay_model_y_intercept = config_data["computation_delay_model"]["computation_delay_y-intercept"]
#     if delay_model_slope:
#       if not delay_model_y_intercept:
#         raise ValueError(f"delay_model_slope was set but delay_model_y_intercept was not.")
#       t_delay = delay_model_slope * iterations + delay_model_y_intercept
#       print(f"t_delay = {t_delay:.8g} = {delay_model_slope:.8g} * {iterations:.8g} + {delay_model_y_intercept:.8g}")
#     else:
#       print('Using constant delay times.')
#       t_delay = config_data["computation_delay_model"]["fake_computation_delay_times"]
#     return t_delay, metadata


      
def controller_interface_factory(controller_interface_selection, computation_delay_provider, sim_dir) -> ControllerInterface:
  """ 
  Generate the desired ControllerInterface object based on the value of "controller_interface_selection". Typically the value will be the string "pipes", but a ControllerInterface interface object can also be passed in directly to allow for testing.
  """

  if isinstance(controller_interface_selection, str):
    controller_interface_selection = controller_interface_selection.lower()

  if controller_interface_selection == "pipes":
    return PipesControllerInterface(computation_delay_provider, sim_dir)
  elif isinstance(controller_interface_selection, ControllerInterface):
    return controller_interface_selection
  else:
    raise ValueError(f'Unexpected controller_interface: {controller_interface_selection}')


def computation_delay_provider_factory(computation_delay_name: str, sim_dir, sample_time, use_fake_scarab) -> DelayProvider:
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