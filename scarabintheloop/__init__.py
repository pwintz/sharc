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
      print(f'Adding "{experiment_result["label"]}" to list of experiment results')
      experiment_result_list[experiment_result["label"]] = experiment_result
      writeJson(experiment_list_dir + "experiment_result_list_incremental.json", experiment_result_list)
    else:
      print(f'No result for "{experiment_config_patches["label"]}" experiment.')
  
  experiment_list_json_filename = experiment_list_dir + "experiment_result_list.json"
  print(f"Finished {len(experiment_result_list)} experiments. Output in {experiment_list_json_filename}.")
  writeJson(experiment_list_json_filename, experiment_result_list)

def run_experiment(base_config: dict, experiment_config_patches: dict, example_dir: str):
  """ This function assumes that the working directory is the experiment-list working directory and its parent directory is the example directory, which contains the following:
    - scripts/controller_delegator.py
    - chip_configs/PARAMS.base (and other optional configurations)
    - simulation_configs/default.json (and other optional configurations)
  """

  # Load human-readable labels.
  experiment_list_label = base_config["experiment_list_label"]
  experiment_label = experiment_list_label + "/" + slugify(experiment_config_patches["label"])
  
  # Record the start time.
  experiment_start_time = time.time()

  if experiment_config_patches.get("skip", False):
    print(f'==== Skipping Experiment: {experiment_label} ====')
    return
  else:
    print(f"====  Running Experiment: {experiment_label} ====")

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
    experiment_data = run_experiment_parallelized(experiment_dir, experiment_config, example_dir)
  else:
    experiment_data = run_experiment_sequential(experiment_dir, experiment_config, example_dir)

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
  if t_delay[0] is not None:
    raise ValueError(f"The first entryf time_steps_delayed must be None. Instead it was {time_steps_delayed[0]}!")
  time_steps_delayed[0] = 0
  for i in range(1, len(time_steps_delayed)):
    if time_steps_delayed[i] is None:
      raise ValueError(f"Entry {i} of time_steps_delayed was None. Only the first entry can be None!")
    time_steps_delayed[i] = math.ceil(t_delay[i] / sample_time)
  return time_steps_delayed

def findFirstExcessiveComputationDelay(time_steps_delayed):  
  for (i, steps_delayed) in enumerate(time_steps_delayed):
    if steps_delayed > 1:
      return (i, steps_delayed)
  return None, None

def processBatchSimulationData(batch_config: dict, batch_simulation_data: dict):
  checkBatchConfig(batch_config)
  checkBatchSimulationData(batch_simulation_data, batch_config["n_time_steps"])

  # x: The states of the system at each t(0), t(1), etc. 
  # u: The control values applied. The first entry is the u0 previously computed. Subsequent entries are the computed values. The control u[i] is applied from t[i] to t[i+1], with u[-1] not applied during this batch. 
  x = batch_simulation_data["x"]
  u = batch_simulation_data["u"]
  t = batch_simulation_data["t"]
  t_delay = batch_simulation_data["t_delay"]

  # The values of data from the batch, including values after missed computations.
  all_data_from_batch = {
    "x": x,
    "u": u,
    "t": t, 
    "t_delay": t_delay,
    "first_time_step": batch_config["first_time_step"],
    "last_time_step": batch_config["last_time_step"]
  }

  sample_time = batch_config["system_parameters"]["sample_time"]
  time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
  first_excessive_delay_index, first_excessive_delay_n_of_time_steps = findFirstExcessiveComputationDelay(time_steps_delayed)
  has_missed_computation = first_excessive_delay_index is not None

  if has_missed_computation:
    first_excessive_delay_time_step = batch_config["first_time_step"] + first_excessive_delay_index
    valid_data_from_batch = {
      "x":             x[0:first_excessive_delay_index + 1],
      "u":             u[0:first_excessive_delay_index + 1],
      "t":             t[0:first_excessive_delay_index + 1], 
      "t_delay": t_delay[0:first_excessive_delay_index + 1],
      "first_time_step": batch_config["first_time_step"],
      "last_time_step": first_excessive_delay_time_step
    }

    # Values that are needed to start the next batch.
    next_batch_init = {
      "first_time_step": first_excessive_delay_time_step + 1,
      "x0": x[first_excessive_delay_index],
      # When there is a missed computation, we continue using u from the prior timestep.
      "u0": u[first_excessive_delay_index-1],
      # Record the value of u that is computed late so that the next batch can apply it at the approriate time
      "u_pending": u[first_excessive_delay_index],
      "next_u_update_time_step": first_excessive_delay_time_step  + first_excessive_delay_n_of_time_steps
    }
  else: # -> No missed computations
    valid_data_from_batch = copy.deepcopy(all_data_from_batch)

    # Values that are needed to start the next batch.
    next_batch_init = {
      "first_time_step": batch_config["last_time_step"] + 1,
      "x0": x[-1],
      "u0": u[-1],
      "u_pending": None,
      "next_u_update_time_step": 0
    }

  # Check that the next_first_time_step doesn't skip over any timesteps.
  if next_batch_init['first_time_step'] > all_data_from_batch['last_time_step'] + 1:
    raise ValueError(F'next first_time_step={next_batch_init["first_time_step"]} > last_time_step_in_batch + 1={all_data_from_batch["last_time_step"] + 1}')

  return {
    "all_data_from_batch": all_data_from_batch, 
    "valid_data_from_batch": valid_data_from_batch, 
    "next_batch_init": next_batch_init
  }

def run_experiment_sequential(experiment_dir, experiment_config, example_dir):
  print(f'Running "{experiment_config["experiment_label"]}" sequentially')
  sim_dir = experiment_dir
  sim_config = copy.deepcopy(experiment_config)
  sim_config["simulation_label"] = experiment_config["experiment_label"]
  simulation_data = runSimulation(sim_dir, sim_config)
  return simulation_data

def run_experiment_parallelized(experiment_dir, experiment_config, example_dir):
  print(f'Running "{experiment_config["experiment_label"]}" in parallel')
  n_time_steps_in_batch = os.cpu_count()
  n_total_time_steps = experiment_config["n_time_steps"]
  first_time_step_in_batch = 0
  i_batch = 0
  x0 = experiment_config["x0"]
  u0 = experiment_config["u0"]
  u_pending = None
  next_u_update_time_step = 0
  sample_time = experiment_config["system_parameters"]["sample_time"] 

  # Loop through batches of time-steps until the start time-step is no longer less than the total number of time-steps desired.
  batch_data_list = []
  xs_actual = [x0]
  us_actual = [u0]
  ts_actual = [0]
  computation_times_actual = [None]

  ### Batches Loop ###
  while first_time_step_in_batch < n_total_time_steps:

    last_time_step_in_batch = min(first_time_step_in_batch + n_time_steps_in_batch-1, n_total_time_steps-1)
    n_time_steps_in_batch = last_time_step_in_batch - first_time_step_in_batch + 1
    
    batch_label = f"batch{i_batch}_steps{first_time_step_in_batch}-{last_time_step_in_batch}"
    print(f"Starting batch: {batch_label} ({n_time_steps_in_batch} time steps)")
    batch_dir = experiment_dir + "/" + batch_label
    os.makedirs(batch_dir)
    shutil.copyfile(experiment_dir + "/PARAMS.generated", batch_dir + "/PARAMS.generated")
    batch_config = copy.deepcopy(experiment_config)
    batch_config["simulation_label"] = experiment_config["experiment_label"] + "/" + batch_label
    batch_config["n_time_steps"] = n_time_steps_in_batch
    batch_config["x0"] = x0
    batch_config["u0"] = u0
    batch_config["first_time_step"] = first_time_step_in_batch
    batch_config["last_time_step"] = last_time_step_in_batch

    # Check that batch data and then write it to "config.json" so it is available in the batch directory with any updated values for the controller to read. 
    checkBatchConfig(batch_config)
    writeJson(batch_dir + "/config.json", batch_config)

    # Run simulation
    simulation_data = runSimulation(batch_dir, batch_config)
    batch_data = processBatchSimulationData(batch_config, simulation_data)
    batch_data['label'] = batch_label
    batch_data['batch directory'] = batch_dir
    batch_data['config'] = batch_config
    batch_data_list.append(batch_data)

    next_batch_initialization_data = batch_data["next_batch_init"]

    # Check that the next_first_time_step doesn't skip over any timesteps.
    if next_batch_initialization_data['first_time_step'] > last_time_step_in_batch + 1:
      raise ValueError(F'next first_time_step={next_batch_initialization_data["first_time_step"]} > last_time_step_in_batch + 1={last_time_step_in_batch + 1}')

    # Append all of the valid data (up to the point of he missed computation) except for the first index, which overlaps with the last index of the previous batch.
    xs_actual += batch_data["valid_data_from_batch"]["x"][1:]
    us_actual += batch_data["valid_data_from_batch"]["u"][1:]
    ts_actual += batch_data["valid_data_from_batch"]["t"][1:]
    computation_times_actual += batch_data["valid_data_from_batch"]["t_delay"][1:]

    # Increment Timestep start
    u0 = next_batch_initialization_data['u0']
    x0 = next_batch_initialization_data['x0']
    first_time_step_in_batch = next_batch_initialization_data['first_time_step']
    u_pending = next_batch_initialization_data['u_pending']
    next_u_update_time_step = next_batch_initialization_data['next_u_update_time_step']
    
    i_batch += 1
    print(f"Finished batch={first_time_step_in_batch} to last_time_step_in_batch={last_time_step_in_batch}")

    incremental_experiment_data = {"batches": batch_data_list,
                     "x": xs_actual,
                     "u": us_actual,
                     "t": ts_actual,
                     "computation_times": computation_times_actual,
                     "config": experiment_config}

    experiment_data_incremental_path = os.path.abspath(experiment_dir + "/experiment_data_incremental.json")
    writeJson(experiment_data_incremental_path, incremental_experiment_data)
    print('Incremental experiment data: ' + experiment_data_incremental_path)

  experiment_data = {"batches": batch_data_list,
                     "x": xs_actual,
                     "u": us_actual,
                     "t": ts_actual,
                     "computation_times": computation_times_actual,
                     "config": experiment_config}
                     
  writeJson("experiment_data_incremental.json", experiment_data)

  return experiment_data

def portabalize_trace(trace_dir): 
  print(f"Starting portabilization of trace in {trace_dir}.")
  log_path = trace_dir + "/portabilize.log"
  with openLog(log_path, f"Portablize \"{trace_dir}\"") as portabilize_log:
    try:
        run_shell_cmd("run_portabilize_trace.sh", working_dir=trace_dir, log=portabilize_log)
    except Exception as e:
      raise e
  print(f"Finished portabilization of trace in {trace_dir}.")

def runSimulation(sim_dir, sim_config):
  print(f"Start of runSimulation in {sim_dir}")
  simulation_start_time = time.time()
  os.chdir(sim_dir)

  # Create a human-readable label for this simulation.
  simulation_label = sim_config["simulation_label"]
  print(f"----> Starting simulation: {simulation_label} <----")

  chip_params_path = sim_dir + '/PARAMS.generated'
  assertFileExists(chip_params_path)  

  controller_log_path = sim_dir + '/controller.log'
  plant_log_path = sim_dir + '/plant_dynamics.log'
  
  use_parallel_scarab_simulation = sim_config["Simulation Options"]["parallel_scarab_simulation"]
  
  with openLog(plant_log_path, f"Plant Log for \"{simulation_label}\"") as plant_log, \
       openLog(controller_log_path, f"Controller Log for \"{simulation_label}\"") as controller_log:

    # with redirect_stdout(plant_log):
    #   evolveState_fnc = plant_dynamics.getDynamicsFunction(sim_config)

    if "u_pending" in sim_config:
      # A computed control value is pending, so it should not be applied at the start of the simulation, but is instead delayed until a given time-step.
      pass

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log)
    assertFileExists(sim_dir + "/x_c++_to_py")

    controller_executor = getSimulationExecutor(sim_dir, sim_config, controller_log, plant_log)

    # Execute the simulation.
    simulation_data = controller_executor.run_simulation()

#     controller_executable = controller_delegator.get_controller_executable(sim_config)
#     assertFileExists(controller_executable)
# 
#     def run_controller():
#       print(f'---- Starting controller for {simulation_label} ----\n'\
#             f'Simulation dir: {sim_dir} \n'\
#             f'Controller log: {controller_log_path}')
#       assertFileExists('config.json') # config.json is read by the controller.
#         
#       try:
#         if use_parallel_scarab_simulation or sim_config["Simulation Options"]["use_fake_scarab_computation_times"]:
#           # Run the controller.
#           print(f"Running controller exectuable {controller_executable}:")
#           run_shell_cmd(controller_executable, working_dir=sim_dir, log=controller_log)
#         else:
#           # Run Scarb using execution-driven mode.
#           cmd = " ".join([controller_executable])
#           scarab_cmd_argv = [sys.executable, # The Python executable
#                             scarab_paths.bin_dir + '/scarab_launch.py',
#                             '--program', cmd,
#                             '--param', 'PARAMS.generated',
#                             # '--simdir', '.', # Changes the working directory.
#                             '--pintool_args',
#                             '-fast_forward_to_start_inst 1',
#                             '--scarab_args',
#                             '--inst_limit 15000000000' # Instruction limit
#                             #  '--heartbeat_interval 50', 
#                             # '--num_heartbeats 1'
#                             # '--power_intf_on 1']
#                           ]
#           # There is a bug in Scarab that causes it to crash when run in the terminal 
#           # when the terminal is resized. To avoid this bug, we run it in a different thread, which appears to fix the problem.
#           with ThreadPoolExecutor() as executor:
#             future = executor.submit(lambda: run_shell_cmd(scarab_cmd_argv, working_dir=sim_dir, log=controller_log))
# 
#             # Wait for the background task to finish before exiting
#             future.result()
# 
#       except Exception as err:
#         print(f'Error: Running the controller failed!')
#         print(f"Error: {str(err)}")
#         if hasattr(err, 'output') and err.output:
#           warn(f'Error output: {err.output}')
#         print(traceback.format_exc())
#         raise err
#       print('Controller thread finished.')
# 
#     def run_plant():
#       print(f'---- Starting plant dynamics for {simulation_label} ----\n' \
#             f'  Simulation dir: {sim_dir} \n' \
#             f'       Plant log: {plant_log_path}\n')
#       plant_log.write('Start of thread.\n')
#       if DEBUG_CONFIGURATION_LEVEL >= 2:
#         printJson(f"Simulation configuration", sim_config)
# 
#       try:
#         with redirect_stdout(plant_log):
#           plant_runner.run(sim_dir, sim_config, evolveState_fnc)
#           assertFileExists('simulation_data.json')
# 
#       except Exception as e:
#         print(f'Plant dynamics had an error: {e}')
#         print(traceback.format_exc())
#         raise e
# 
#       # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
#       print('Plant thread finshed. \n' \
#             f'  Simulation directory: {sim_dir} \n' \
#             f'       Simulation data: {sim_dir}/simulation_data.json')
# 
    # if not sim_config["Simulation Options"]["use_external_dynamics_computation"]:
    #   print("Running controller only -- No plant!")
    #   run_controller()
    #   return None # No data recorded
    # else: # Run plant and controller in parallel, on separate threads.
    #   N_TASKS = 2
    #   with ThreadPoolExecutor(max_workers=N_TASKS) as executor:
    #     # Create two tasks: One for the controller and another for the plant.
    #     tasks = [executor.submit(run_controller), executor.submit(run_plant)]
    #     for future in concurrent.futures.as_completed(tasks):
    #       if future.exception(): # Waits until finished or failed.
    #         print(f"Task {future} failed with exception: {future.exception()}")
    #         raise future.exception()
    #   simulation_data = readJson('simulation_data.json')
# 
#     if use_parallel_scarab_simulation:
#       # Portabilize the trace files (in parallel) and then simulate with Scarab (in parallel).
# 
#       (dynamrio_trace_dir_dictionaries, trace_dirs, sorted_indices) = get_dynamorio_trace_directories()
# 
#       #? Do we need to portablize if we are just using the trace files in-place? We can save a few seconds on each loop by skipping this step.
#       print(f"Starting portabilization of trace directories {trace_dirs} in {sim_dir}.")
#       with ProcessPoolExecutor(max_workers = os.cpu_count()) as portabalize_executor:
#         portabalize_executor.map(portabalize_trace, trace_dirs)
# 
#       print(f"Finished portabilizing traces for each directory in {dynamrio_trace_dir_dictionaries.values()} in {sim_dir}.")
#       
#       computation_time_list = [None] * len(sorted_indices)
#       with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
#         scarab_data = scarab_executor.map(simulate_trace_in_scarab, sorted_indices)
#         for datum in scarab_data:
#           index = datum["index"]
#           trace_dir = datum["trace_dir"]
#           computation_time = datum["computation_time"]
#           computation_time_list[index] = computation_time
#           simulation_data["t_delay"][index] = computation_time
#         print(f"Finished executing parallel simulations, scarab_data: {scarab_data}")
# 
#       simulation_data["t_delay"] = [None] + simulation_data["t_delay"]
#       print(f"computation_time list: {computation_time_list}")


    print("End of runSimulation.")
    print(f'  Simulation dir: {sim_dir}')
    print(f'  Controller log: {controller_log_path}')
    print(f'       Plant log: {plant_log_path}')
    print(f"   Data out file: {os.path.abspath('simulation_data.json')}")
    
    simulation_end_time = time.time()
    simulation_data["simulation wall time"] = simulation_start_time - simulation_end_time
    return simulation_data


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
    self.controller_executable = controller_delegator.get_controller_executable(sim_config)
    assertFileExists(self.controller_executable)
    
    # Get a function that defines the plant dynamics.
    with redirect_stdout(plant_log):
      self.evolveState_fnc = plant_dynamics.getDynamicsFunction(sim_config)

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log)
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
    print(f'---- Starting plant dynamics for {self.sim_config["simulation_label"]} ----\n' \
          f'  Simulation dir: {self.sim_dir} \n' \
          f'       Plant log: {self.plant_log.name}\n')
    self.plant_log.write('Start of thread.\n')
    if DEBUG_CONFIGURATION_LEVEL >= 2:
      printJson(f"Simulation configuration", self.sim_config)

    try:
      with redirect_stdout(self.plant_log):
        plant_runner.run(self.sim_dir, self.sim_config, self.evolveState_fnc)
        assertFileExists(self.sim_dir + '/simulation_data.json')

    except Exception as e:
      print(f'Plant dynamics had an error: {e}')
      print(traceback.format_exc())
      raise e

    # Print the output in a single call to "print" so that it isn't interleaved with print statements in other threads.
    if DEBUG_DYNAMICS_LEVEL >= 1:
      print('-- Plant dynamics finshed -- \n' \
            f'  Simulation directory: {self.sim_dir} \n' \
            f'       Simulation data: {self.sim_dir}/simulation_data.json')

  def run_simulation(self):
    """ Run everything needed to simulate the plant and dynamics, in parallel."""
    # Run plant and controller in parallel, on separate threads.
    N_TASKS = 2
    with ThreadPoolExecutor(max_workers=N_TASKS) as executor:
      # Create two tasks: One for the controller and another for the plant.
      tasks = [executor.submit(self.run_controller), executor.submit(self._run_plant)]
      for future in concurrent.futures.as_completed(tasks):
        if future.exception(): # Waits until finished or failed.
          print(f"Task {future} failed with exception: {future.exception()}")
          raise future.exception()
    simulation_data = readJson('simulation_data.json')
    simulation_data = self.postprocess_simulation_data(simulation_data)
    return simulation_data

  def postprocess_simulation_data(self, simulation_data):
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

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def postprocess_simulation_data(self, simulation_data):

    # Portabilize the trace files (in parallel) and then simulate with Scarab (in parallel).

    (dynamrio_trace_dir_dictionaries, trace_dirs, sorted_indices) = get_dynamorio_trace_directories()

    #? Do we need to portablize if we are just using the trace files in-place? We can save a few seconds on each loop by skipping this step.
    print(f"Starting portabilization of trace directories {trace_dirs} in {self.sim_dir}.")
    with ProcessPoolExecutor(max_workers = os.cpu_count()) as portabalize_executor:
      portabalize_executor.map(portabalize_trace, trace_dirs)

    print(f"Finished portabilizing traces for each directory in {dynamrio_trace_dir_dictionaries.values()} in {self.sim_dir}.")
    
    computation_time_list = [None] * len(sorted_indices)
    with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
      scarab_data = scarab_executor.map(simulate_trace_in_scarab, sorted_indices)
      for datum in scarab_data:
        index = datum["index"]
        trace_dir = datum["trace_dir"]
        computation_time = datum["computation_time"]
        computation_time_list[index] = computation_time
        simulation_data["t_delay"][index] = computation_time
      print(f"Finished executing parallel simulations, scarab_data: {scarab_data}")

    simulation_data["t_delay"] = [None] + simulation_data["t_delay"]
    print(f"computation_time list: {computation_time_list}")
    return simulation_data
    

class InternalSimulationExecutor(SimulationExecutor):
  """
  Create a simulation executor that performs all of the plant dynamics calculations internally. This executor does not support simulating computation times with Scarab.
  """
  def __init__(self):
    raise RuntimeError("This class is not tested and is not expected to work correctly as-is.")

  def run_controller(self):
    run_shell_cmd(self.controller_executable, working_dir=self.sim_dir, log=self.controller_log)

  def run_simulation(self):
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