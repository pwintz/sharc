#! /bin/env python3
"""
Execute a list of experiments using Scarab-in-the-loop.
"""
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
from scarabintheloop.utils import assertFileExists, writeJson, readJson, in_working_dir, openLog, run_shell_cmd, printJson


# Type hinting.
from typing import List, Set, Dict, Tuple
from typing import Union

from scarab_globals import *
from scarab_globals import scarab_paths

import scarabintheloop.scarabizor as scarabizor

from slugify import slugify

# TODO: Check that we are in a valid example folder

# Update the PYTHONPATH env variable to include the scripts folder.
example_dir = os.path.abspath(".")
delegator_path = example_dir + '/scripts/controller_delegator.py'
if not os.path.exists(delegator_path):
  raise IOError(f'The file "{delegator_path}" does not exist. Did you run this script from the root of an example directory?')
sys.path.append(os.path.abspath(example_dir + "/scripts"))

# Update the enviroment variable so that subshells have the updated path.
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import scarabintheloop.plant_runner as plant_runner
import controller_delegator

# Add the example's scripts directory to the PATH.
# os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/scripts")
os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/bin")

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]

# Regex to find text in the form of "--name value"
param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

# Format string to generate text in the form of "--name value"
param_str_fmt = "--{}\t{}\n"

def main():
  
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

  debug_configuration_level = base_config["==== Debgugging Levels ===="]["debug_configuration_level"]

  experiment_list_dir = experiments_dir + "/" + slugify(experiments_config_file_path.stem) + "--" + time.strftime("%Y-%m-%d--%H-%M-%S") + "/"
  print(f"experiment_list_dir: {experiment_list_dir}")

  # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
  os.makedirs(experiment_list_dir)

  # Create or update the symlink to the latest folder.
  latest_experiment_list_dir_symlink_path = experiments_dir + "/latest"
  if os.path.exists(latest_experiment_list_dir_symlink_path):
    os.remove(latest_experiment_list_dir_symlink_path)
  os.symlink(experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)

  if debug_configuration_level >= 1:
    printJson(f"base_config loaded from {base_config_file_path}", base_config)
    printJson(f"experiment_config_patches_list loaded from {experiments_config_file_path}", experiment_config_patches_list)

  experiment_result_list = {}
  for experiment_config_patches in experiment_config_patches_list:
    os.chdir(experiment_list_dir)
    experiment_result = run_experiment(base_config, experiment_config_patches)
    if experiment_result:
      # printJson("experiment_result", experiment_result)
      print(f'Adding "{experiment_result["label"]}" to list of experiment results')
      experiment_result_list[experiment_result["label"]] = experiment_result
      writeJson(experiment_list_dir + "/experiment_result_list_incremental.json", experiment_result_list)
    else:
      print(f'No result for {experiment_config_patches["label"]}')
  
  print(f"Finished {len(experiment_result_list)} experiments.")
  writeJson(experiment_list_dir + "/experiment_result_list.json", experiment_result_list)

def run_experiment(base_config: dict, experiment_config_patches: dict):
  """ This function assumes that the working directory is the experiment-list working directory and its parent directory is the example directory, which contains the following:
    - scripts/controller_delegator.py
    - chip_configs/PARAMS.base (and other optional configurations)
    - simulation_configs/default.json (and other optional configurations)
  """

  # Load human-readable labels.
  experiment_list_label = base_config["experiment_list_label"]
  experiment_label = experiment_list_label + "/" + slugify(experiment_config_patches["label"])

  if experiment_config_patches.get("skip", False):
    print(f'==== Skipping Experiment: {experiment_label} ====')
    return
  else:
    print(f"====  Running Experiment: {experiment_label} ====")

  debug_configuration_level = base_config["==== Debgugging Levels ===="]["debug_configuration_level"]

  example_dir = os.path.abspath('../..') # The "root" of this project.
  experiment_list_dir = os.path.abspath('.') # A file like <example_dir>/experiments/default-2024-08-25-16:17:35
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

  if debug_configuration_level >= 2:
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
    experiment_data = run_experiment_parallelized(experiment_dir, experiment_config)
  else:
    experiment_data = run_experiment_sequential(experiment_dir, experiment_config)

  experiment_result = {
    "label": experiment_config["label"],
    "dir": experiment_dir,
    "data": experiment_data,
    "config": experiment_config
  }
  writeJson("experiment_result.json", experiment_result)
  return experiment_result
    
def run_experiment_sequential(experiment_dir, experiment_config):
  print(f'Running "{experiment_config["experiment_label"]}" sequentially')
  sim_dir = experiment_dir
  sim_config = copy.deepcopy(experiment_config)
  sim_config["simulation_label"] = experiment_config["experiment_label"]
  simulation_data = run_simulation(sim_dir, sim_config)
  return simulation_data

def run_experiment_parallelized(experiment_dir, experiment_config):
  print(f'Running "{experiment_config["experiment_label"]}" in parallel')
  n_timesteps_in_batch = os.cpu_count()
  n_total_timesteps = experiment_config["n_time_steps"]
  start_timestep_in_batch = 0
  i_batch = 0
  x0 = experiment_config["x0"]
  u0 = experiment_config["u0"]
  sample_time = experiment_config["system_parameters"]["sample_time"] 

  # Loop through batches of time-steps until the start time-step is no longer less than the total number of time-steps desired.
  batch_simulations_data = []
  xs_actual = []
  us_actual = []
  ts_actual = []
  computation_times_actual = []
  while start_timestep_in_batch < n_total_timesteps:
    end_timestep_in_batch = min(start_timestep_in_batch + n_timesteps_in_batch, n_total_timesteps-1)
    n_timesteps_in_batch = end_timestep_in_batch - start_timestep_in_batch + 1
    batch_dir = experiment_dir + f"/batch{i_batch}_steps{start_timestep_in_batch}-{end_timestep_in_batch}"
    os.makedirs(batch_dir)
    shutil.copyfile(experiment_dir + "/PARAMS.generated", batch_dir + "/PARAMS.generated")
    batch_config = copy.deepcopy(experiment_config)
    batch_config["simulation_label"] = experiment_config["experiment_label"] + f"/batch-{i_batch}"
    batch_config["n_time_steps"] = n_timesteps_in_batch
    batch_config["x0"] = x0
    batch_config["u0"] = u0

    # Create "config.json" so it is available in the batch directory with any updated values for the controller to read.
    writeJson(batch_dir + "/config.json", batch_config)
    simulation_data = run_simulation(batch_dir, batch_config)
    # printJson(f"simulation_data from run_simulation in {batch_dir}", simulation_data)

    xs = simulation_data["x"]
    us = simulation_data["u"]
    computation_times = simulation_data["t_delay"]
    xs_actual_in_batch = xs
    us_actual_in_batch = us
    times_actual_in_batch = simulation_data["t"]
    computation_times_actual_in_batch = computation_times
    # Set the values for the next loop under the assumption that no computations were missed. 
    # These will be overwritten if there are late computations.
    next_x0 = simulation_data["x"][-1]
    next_u0 = simulation_data["u"][-1]
    next_start_timestep = start_timestep_in_batch + n_total_timesteps + 1
    next_start_index = n_total_timesteps + 1
    first_late_computation_timestep = None
    first_late_computation_index = None
    for i in range(0, n_total_timesteps):
      # TODO: Work through the indexing in this section to make sure it is all accurate.
      computation_time = computation_times[i]
      if computation_time > sample_time:
        first_late_computation_index = i
        next_start_index = first_late_computation_index + 1
        first_late_computation_timestep = start_timestep_in_batch + first_late_computation_index
        # After a missed computation, we start at next time step using the value of x at that time step but the value of u from the previous timestep because the controller failed to compute an update. 
        next_start_timestep = first_late_computation_timestep + 1
        next_x0 = xs[next_start_index]       # Use 'x'from time step after missed computation
        next_u0 = us[first_late_computation_index] # Continue to use 'u' from time step before missed computation
        xs_actual_in_batch = xs[0:first_late_computation_index]
        us_actual_in_batch = us[0:first_late_computation_index]
        times_actual_in_batch = times[0:first_late_computation_index]
        computation_times_actual_in_batch = computation_times[0:first_late_computation_index]
        print(f"Computation time was too long for i={i} (Computation time={computation_time} > sample_time={sample_time}).")
        print(f'-> next_start_timestep={next_start_timestep}')
        print(f'-> next_u0={next_u0}')
        print(f'-> next_x0={next_x0}')
        break

    simulation_data["next_x0"] = next_x0
    simulation_data["next_u0"] = next_u0
    simulation_data["next_start_timestep"] = next_start_timestep
    simulation_data["first_late_computation_index"] = first_late_computation_index
    simulation_data["next_start_index"] = next_start_index
    simulation_data["first_late_computation_timestep"] = first_late_computation_timestep
    # "Valid" here means that it was not after a failed computation.
    simulation_data["xs_actual_in_batch"] = xs_actual_in_batch
    simulation_data["us_actual_in_batch"] = us_actual_in_batch
    simulation_data["times_actual_in_batch"] = times_actual_in_batch
    simulation_data["computation_times_actual_in_batch"] = computation_times_actual_in_batch
    batch_simulations_data.append(simulation_data)

    xs_actual += xs_actual_in_batch
    us_actual += us_actual_in_batch
    ts_actual += times_actual_in_batch
    computation_times_actual += computation_times_actual_in_batch
    # simulation_data["xs_actual"] = xs_actual
    # simulation_data["us_actual"] = us_actual
    # simulation_data["ts_actual"] = ts_actual
    # simulation_data["computation_times_actual"] = computation_times_actual

    writeJson("simulation_data_incremental.json", batch_simulations_data)

    # Increment Timestep start
    u0 = next_u0
    x0 = next_x0
    start_timestep_in_batch = next_start_timestep
    i_batch += 1

  experiment_data = {"batches": batch_simulations_data, "x": xs_actual, "u": us_actual, "t": ts_actual, "computation_times": computation_times_actual, "config": experiment_config}
  return experiment_data


def run_simulation(sim_dir, sim_config):
  print(f"Start of run_simulation in {sim_dir}")
  os.chdir(sim_dir)

  # Create a human-readable label for this simulation.
  simulation_label = sim_config["simulation_label"]
  print(f"--- Starting simulation: {simulation_label}. ---")

  # sim_config_path = sim_dir + '/config.json'
  # assertFileExists(sim_config_path)

  chip_params_path = sim_dir + '/PARAMS.generated'
  assertFileExists(chip_params_path)  

  controller_log_path = sim_dir + '/controller.log'
  plant_dynamics_log_path = sim_dir + '/plant_dynamics.log'
  
  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]
  debug_configuration_level = sim_config["==== Debgugging Levels ===="]["debug_configuration_level"]
  
  with openLog(plant_dynamics_log_path, f"Plant Log for \"{simulation_label}\"") as plant_dynamics_log, \
       openLog(controller_log_path, f"Controller Log for \"{simulation_label}\"") as controller_log:

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log)
    assertFileExists(sim_dir + "/x_c++_to_py")

    controller_executable = controller_delegator.get_controller_executable(example_dir, sim_config)
    assertFileExists(controller_executable)

    def run_controller():
      print(f'---- Starting controller for {simulation_label} ----\n'\
            f'Simulation dir: {sim_dir} \n'\
            f'Controller log: {controller_log_path}')
      assertFileExists('config.json') # config.json is read by the controller.
        
      try:
        if use_parallel_scarab_simulation or sim_config["Simulation Options"]["use_fake_scarab_computation_times"]:
          # Run the controller.
          print(f"Running controller exectuable {controller_executable}:")
          run_shell_cmd(controller_executable, working_dir=sim_dir, log=controller_log)
        else:
          # Run Scarb using execution-driven mode.
          cmd = " ".join([controller_executable])
          scarab_cmd_argv = [sys.executable,
                            scarab_paths.bin_dir + '/scarab_launch.py',
                            '--program', cmd,
                            '--param', 'PARAMS.generated',
                            # '--simdir', '.', # Changes the working directory.
                            '--pintool_args',
                            '-fast_forward_to_start_inst 1',
                            '--scarab_args',
                            '--inst_limit 15000000000'
                            #  '--heartbeat_interval 50', 
                            # '--num_heartbeats 1'
                            # '--power_intf_on 1']
                          ]
          # There is a bug in Scarab that causes it to crash when run in the terminal 
          # when the terminal is resized. To avoid this bug, we run it in a different thread, 
          # which appears to fix the problem.
          with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: run_shell_cmd(scarab_cmd_argv, working_dir=sim_dir, log=controller_log))

            # Wait for the background task to finish before exiting
            future.result()

      except Exception as err:
        print(f'Error: Running the controller failed!')
        print(f"Error: {str(err)}")
        if hasattr(err, 'output') and err.output:
          warn(f'Error output: {err.output}')
        print(traceback.format_exc())
        raise err
      print('Controller thread finished.')

    def run_plant():
      print(f'---- Starting plant dynamics for {simulation_label} ----\n' \
            f'  Simulation dir: {sim_dir} \n' \
            f'       Plant log: {plant_dynamics_log_path}\n')
      plant_dynamics_log.write('Start of thread.\n')
      if debug_configuration_level >= 2:
        printJson(f"Simulation configuration", sim_config)

      try:
        with redirect_stdout(plant_dynamics_log):
          plant_runner.run(sim_dir, sim_config)
          assertFileExists('simulation_data.json')

      except Exception as e:
        print(f'Plant dynamics had an error: {e}')
        print(traceback.format_exc())
        raise e

      # Print this output in a single call to "print" so that it cannot get interleaved with 
      # print statements in other threads.
      print('Plant thread finshed. \n' \
            f'  Simulation directory: {sim_dir} \n' \
            f'       Simulation data: {sim_dir}/simulation_data.json')

    if not sim_config["Simulation Options"]["use_external_dynamics_computation"]:
      print("Running controller only -- No plant!")
      run_controller()
      data_out = None # No data recorded
      return data_out
    else: # Run plant and controller in parallel.
      with ThreadPoolExecutor(max_workers=2) as executor:
        # Create two tasks: One for the controller and another for the plant.
        tasks = [executor.submit(run_controller), executor.submit(run_plant)]
        for future in concurrent.futures.as_completed(tasks):
          if future.exception(): # Waits until finished or failed.
            print(f"Task {future} failed with exception: {future.exception()}")
            raise future.exception()
      simulation_data = readJson('simulation_data.json')

    if use_parallel_scarab_simulation:
      # Portabilize the trace files (in serial) and then simulate with Scarab (in parallel).

      (dynamrio_trace_dir_dictionaries, trace_dirs, sorted_indices) = get_dynamorio_trace_directories()
      def portabalize_trace(trace_dir): 
        run_shell_cmd("run_portabilize_trace.sh", log=controller_log, working_dir=trace_dir)

      # TODO: Parallelize the portablization
      # with ProcessPoolExecutor(max_workers = os.cpu_count()) as portabalize_executor:
        # scarab_data = portabalize_executor.map(simulate_trace_in_scarab, sorted_indices)

      for dir_index in sorted(dynamrio_trace_dir_dictionaries):
        trace_dir = dynamrio_trace_dir_dictionaries[dir_index]
        result = run_shell_cmd("run_portabilize_trace.sh", log=controller_log, working_dir=trace_dir)
      print(f"Finished portabilizing traces for each directory in {dynamrio_trace_dir_dictionaries.values()} in {sim_dir}.")
      
      computation_time_list = [None] * len(sorted_indices)
      with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
        scarab_data = scarab_executor.map(simulate_trace_in_scarab, sorted_indices)
        
        print(f"Finished executing parallel simulations, scarab_data: {scarab_data}")
        for datum in scarab_data:
          index = datum["index"]
          trace_dir = datum["trace_dir"]
          computation_time = datum["computation_time"]
          computation_time_list[index] = computation_time
          simulation_data["t_delay"][index] = computation_time

      print(f"computation_time list: {computation_time_list}")
      print(f'simulation_data["t_delay"][index]: {simulation_data["t_delay"][index]}')

      # Check that we have loaded all of the computation times correctly.
      for i in sorted_indices:
        if computation_time_list[i] != simulation_data["t_delay"][i]:
          raise ValueError(f'computation_time[{i}]={computation_time[i]} != simulation_data["t_delay"][{i}]={simulation_data["t_delay"][i]}')

    print("End of run_simulation.")
    print(f'  Simulation dir: {sim_dir}')
    print(f'  Controller log: {controller_log_path}')
    print(f'       Plant log: {plant_dynamics_log_path}')
    print(f"   Data out file: {os.path.abspath('simulation_data.json')}")
    return simulation_data


def simulate_trace_in_scarab(dir_index:int) -> dict:
  trace_dir = f"dynamorio_trace_{dir_index}"
  print(f"Simulating trace in {trace_dir}.")

  shutil.copyfile('PARAMS.generated', trace_dir + '/PARAMS.generated')

  bin_path = trace_dir + '/bin'
  trace_path = trace_dir + '/trace'
  assertFileExists(trace_dir + '/PARAMS.generated')
  assertFileExists(bin_path)
  assertFileExists(trace_path)
  
  with openLog(f"scarab_simulation_{dir_index}.log") as scarab_log:
    scarab_cmd = ["scarab", # Requires that the Scarab build folder is in PATH
                  "--fdip_enable", "0", 
                  "--frontend", "memtrace", 
                  "--fetch_off_path_ops", "0", 
                  f"--cbp_trace_r0=trace", 
                  f"--memtrace_modules_log=bin"]
    run_shell_cmd(scarab_cmd, working_dir=trace_dir, log=scarab_log)
  
  # Read the generated statistics files.
  stats_reader = scarabizor.ScarabStatsReader(trace_dir, is_using_roi=False)
  stats_file_index = 0 # We only simulate one timestep.
  computation_time = stats_reader.readTime(stats_file_index)
  data = {"index": dir_index, "trace_dir": trace_dir, "computation_time": computation_time}
  print(f"Finished Scarab simulation. Time to compute controller: {computation_time} seconds.")
  # print(f"Return data for simulate_trace_in_scarab:\n\t{data}")
  return data


def patch_dictionary(base: dict, patch: dict) -> dict:
  """
  Create a new config dictionary by copying the the base dictionary and replacing any keys with the values in the patch dictionary. The base and patch dictionaries are not modified. All keys in the patch dictionary must already exist in base. If any value is itself a dictionary, then patching is done recursively.
  The returned value should have exactly the same hierarchy of keys 
  """
  # Make a copy of the base data so that modifications the base dictionary is not modified.
  patched = copy.deepcopy(base)

  for (key, value) in patch.items():
    if not key in base: 
      raise ValueError(f'The key "{key}" was given in the patch dictionary but not present in the base dictionary.')

    if isinstance(value, dict):
      # If the value is a dictionary, then we need to recursively patch it, so that we don't just replace the whole dictionary resulting in possibly missing values, or 
      value = patch_dictionary(base[key], patch[key]) 
    elif isinstance(base[key], dict):
      raise ValueError(f'The value of base["{key}"] is a dictionary, but patch["{key}"] is not')
    patched[key] = value
  return patched


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
#   total_timesteps = math.ceil(time_horizon / control_sampling_time)
#   if total_timesteps <= 0:
#     raise ValueError(f"total_timesteps={total_timesteps} was not positive!")
#   print(f"total_timesteps={total_timesteps}")
#   cmd = [f'run_in_parallel.sh', 
#             f'{start_index}',            # 1
#             f'{time_horizon}',           # 2
#             f'{chip_params_path}',       # 3
#             f'{control_sampling_time}',  # 4
#             f'{total_timesteps}',        # 5
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
 
  
  
def get_dynamorio_trace_directories():
  trace_dir_regex = re.compile(r"dynamorio_trace_(\d+)")
  dynamorio_trace_dirs = glob.glob('dynamorio_trace_*')

  trace_dir_dict = {int(trace_dir_regex.search(dir).group(1)): dir for dir in dynamorio_trace_dirs if trace_dir_regex.match(dir)}
  
  if len(trace_dir_dict) == 0:
    raise ValueError(f"No dynamorio trace directories were found in {os.getcwd()}")
  # Sort, in place.
  # trace_dir_indices.sort()
  # print("Trace directory indices: " + str(trace_dir_indices))
  sorted_indices = sorted(trace_dir_dict)
  sorted_trace_dirs = [trace_dir_dict[dir_ndx] for dir_ndx in sorted(trace_dir_dict)]
  return (trace_dir_dict, sorted_trace_dirs, sorted_indices)


if __name__ == "__main__":
  main()



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