#! /bin/env python3

# Run several iterations of Cache sizes

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

from __future__ import print_function

import json
import os
import sys
import shutil
import glob
import concurrent.futures
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

# Add <scarab path>/bin to the Python search pth.
# scarab_root_path = os.environ["SCARAB_ROOT"]
# sys.path.append(scarab_root_path + '/bin')

from scarab_globals import *
from scarab_globals import scarab_paths

from slugify import slugify

# TODO: Check that we are in a valid example folder

# import importlib
# spec = importlib.util.spec_from_file_location("module.name", "scripts/controller_delegator.py")
# controller_delegator_module = importlib.util.module_from_spec(spec)
# sys.modules["controller_delegator"] = controller_delegator_module
# spec.loader.exec_module(controller_delegator_module)

# Update the PYTHONPATH env variable to include the scripts folder.
example_dir = os.path.abspath(".")
delegator_path = example_dir + '/scripts/controller_delegator.py'
if not os.path.exists(delegator_path):
  raise IOError(f'The file "{delegator_path}" does not exist. Did you run this script from the root of an example directory?')
sys.path.append(os.path.abspath(example_dir + "/scripts"))

# Update the enviroment variable so that subshells have the updated path.
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import controller_delegator

# Add the example's scripts directory to the PATH.
# os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/scripts")
os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/bin")

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]
# compile_option_keys = ["prediction_horizon", "control_horizon"]

# Regex to find text in the form of "--name value"
param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

# Format string to generate text in the form of "--name value"
param_str_fmt = "--{}\t{}\n"

def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Run a set of examples.")
  
  # parser.add_argument(
  #     'example_dir',
  #     help="Enter the directory of the example. It must contain all of required example files; see the project README file for a full description of the required files and their format."
  # )

  # parser.add_argument(
  #     '--experiments_dir', 
  #     type=str, # Specify the data type
  #     help="Select the folder where files produced by the simulation are placed."
  # )
  
  parser.add_argument(
      '--config_filename',
      type=str,  # Specify the data type
      default="default.json",
      help="Select the name of a JSON file located in <example_dir>/simulation_configs."
  )

  # Parse the arguments from the command line
  args = parser.parse_args()
  # example_dir = os.path.abspath(args.example_dir)
  config_file_path = Path(os.path.abspath(os.path.join(example_dir, "simulation_configs", args.config_filename)))
  base_config_file_path = Path(os.path.abspath(os.path.join(example_dir, 'base_config.json')))

  # If the user gave an output directory, then use it. Otherwise, use "experiments/" within the example folder.
  # if args.experiments_dir:
  #   experiments_dir = os.path.abspath(args.experiments_dir)
  # else:
  experiments_dir = os.path.abspath(example_dir) + "/experiments"

  try:
    os.chdir(example_dir)  # Change the current working directory
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: The example directory '{example_dir}' does not exist.")
  print(f"Directory changed to: {os.getcwd()}")

  # # Update the PYTHONPATH env variable to include the scripts folder.
  # sys.path.append(os.path.abspath(example_dir + "/scripts"))
  # os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)
  # print("PYTHONPATH: " + os.environ["PYTHONPATH"])

  # Read JSON configuration file.
  with open(base_config_file_path) as json_file:
    base_config = json.load(json_file)

  # Open list of example configurations.
  # with open('example_configs.json') as json_file:
  with open(config_file_path) as config_json_file:
    experiment_config_patches_list = json.load(config_json_file)

  experiment_list_label = slugify(config_file_path.stem)
  base_config["experiment_list_label"] = experiment_list_label

  debug_configuration_level = base_config["==== Debgugging Levels ===="]["debug_configuration_level"]

  experiment_list_dir = experiments_dir + "/" + slugify(config_file_path.stem) + "--" + time.strftime("%Y-%m-%d--%H-%M-%S") + "/"
  print(f"experiment_list_dir: {experiment_list_dir}")

  # Create the experiment_list_dir if it does not exist. An error is created if it already exists.
  os.makedirs(experiment_list_dir)

  # Create or update the symlink to the latest folder.
  latest_experiment_list_dir_symlink_path = experiments_dir + "/latest"
  if os.path.exists(latest_experiment_list_dir_symlink_path):
    os.remove(latest_experiment_list_dir_symlink_path)
  os.symlink(experiment_list_dir, latest_experiment_list_dir_symlink_path, target_is_directory=True)

  # The experiment config file can either contain a single configuration or a list.
  # In the case where there is only a single configuration, we convert it to a singleton list. 
  if not isinstance(experiment_config_patches_list, list):
    experiment_config_patches_list = [experiment_config_patches_list]

  print(f"experiment_config_patches_list:\n{json.dumps(experiment_config_patches_list, indent=2)}")


  for experiment_config_patches in experiment_config_patches_list:
    print(f'======= Starting Experiment: {experiment_config_patches} ========')
    # Make sure we start the experiment_list_dir 
    os.chdir(experiment_list_dir)
    run_experiment(base_config, experiment_config_patches)

def run_experiment(base_config: dict, experiment_config_patches: dict):
  """ This function assumes that the working directory is the experiment-list working directory and its parent directory is the example directory, which contains the following:
    - scripts/controller_delegator.py
    - chip_configs/PARAMS.base (and other optional configurations)
    - simulation_configs/default.json (and other optional configurations)
  """
  if experiment_config_patches.get("skip", False):
    print(f'Skipping example: {experiment_config_patches["label"]}')
    return

  # Load human-readable labels.
  experiment_list_label = base_config["experiment_list_label"]
  experiment_label = experiment_list_label + "/" + experiment_config_patches["label"]
  print(f"==== Running Experiment: {experiment_label} ==== ")

  example_dir = os.path.abspath('../..') # The "root" of this project.
  experiment_list_dir = os.path.abspath('.') # A file like <example_dir>/experiments/default-2024-08-25-16:17:35
  experiment_dir = os.path.abspath(experiment_list_dir + "/" + slugify(experiment_config_patches['label']))
  simulation_config_path = example_dir + '/simulation_configs'
  chip_configs_path = example_dir + '/chip_configs'

  print("        example_dir: " + example_dir)
  print("experiment_list_dir: " + experiment_list_dir)
  print("     experiment_dir: " + experiment_dir)

  # Check that the expected folders exist.
  assertFileExists(chip_configs_path)
  assertFileExists(simulation_config_path)

  ########### Populate the experiment directory ############
  # Make subdirectory for this experiment.
  print(f"Experiment directory: {experiment_dir}")
  os.makedirs(experiment_dir)
  os.chdir(experiment_dir)

  # Using the base configuration dictionary (from JSON file), update the values given in the experiment config patch, loaded from
  experiment_config = patch_dictionary(base_config, experiment_config_patches)
  experiment_config["experiment_label"] = experiment_label

  # Write the configuration to config.json.
  with open("config.json", 'w') as experiment_config_file:
    experiment_config_file.write(json.dumps(experiment_config, indent=2))

  # Create PARAMS.generated file containing chip parameters for Scarab.
  PARAMS_src = chip_configs_path + "/" + experiment_config["PARAMS_base_file"] 
  PARAMS_out = os.path.abspath('PARAMS.generated')
  assertFileExists(PARAMS_src)
  create_patched_PARAMS_file(experiment_config, PARAMS_src, PARAMS_out)
  assertFileExists(PARAMS_out)
  
  if experiment_config["use_fake_scarab_computation_times"]:
    run_experiment_no_scarab(experiment_dir, experiment_config)
  elif experiment_config["parallel_scarab_simulation"]:
    run_experiment_parallelized(experiment_dir, experiment_config)
  else:
    run_experiment_sequential(experiment_dir, experiment_config)
    
def run_experiment_no_scarab(experiment_dir, experiment_config):
  sim_dir = experiment_dir
  sim_config = copy.deepcopy(experiment_config)
  sim_config["simulation_label"] = experiment_config["experiment_label"]
  data_out = run_simulation(sim_dir, sim_config)

def run_experiment_sequential(experiment_dir, experiment_config):
  sim_dir = experiment_dir
  sim_config = copy.deepcopy(experiment_config)
  sim_config["simulation_label"] = experiment_config["experiment_label"]
  data_out = run_simulation(sim_dir, sim_config)

def run_experiment_parallelized(experiment_dir, experiment_config):
  n_timesteps_per_batch = 2 #TODO (restore): os.cpu_count()
  sim_config = copy.deepcopy(experiment_config)
  for i in range(4):
    batch_dir = experiment_dir + f"/batch_{i}"
    print(batch_dir)
    os.makedirs(batch_dir)
    os.chdir(batch_dir)
    shutil.copyfile(experiment_dir + "/PARAMS.generated", batch_dir + "/PARAMS.generated")
    batch_config = copy.deepcopy(sim_config)
    batch_config["simulation_label"] = experiment_config["experiment_label"] + f"-batch-{i}"
    batch_config["n_time_steps"] = n_timesteps_per_batch
    
    with open("config.json", 'w') as batch_config_file:
      batch_config_file.write(json.dumps(batch_config, indent=2))
    data_out = run_simulation(batch_dir, batch_config)

  # raise ValueError("Not implemented.")

  # for sim_dir in simulation_list:
  # os.chdir(sim_dir)
  # assertFileExists(sim_dir + "/PARAMS.generated")

  # # Write the configuration to config.json.
  # with open(sim_config_path, 'w') as sim_config_file:
  #     sim_config_file.write(json.dumps(sim_config, indent=2))

def run_simulation(sim_dir, sim_config):
  os.chdir(sim_dir)

  # Create a human-readable label for this simulation.
  simulation_label = sim_config["simulation_label"]
  print(f"--- Starting simulation: {simulation_label}. ---")

  sim_config_path = sim_dir + '/config.json'
  chip_params_path = sim_dir + '/PARAMS.generated'
  assertFileExists(sim_config_path)
  assertFileExists(chip_params_path)  

  controller_log_path = sim_dir + '/controller.log'
  plant_dynamics_log_path = sim_dir + '/plant_dynamics.log'

  with open(controller_log_path, 'w+') as controller_log, \
        open(plant_dynamics_log_path, 'w+') as plant_dynamics_log:
    writeHeader(controller_log,     f"Simulation: \"{simulation_label}\" (Controller Log)")
    writeHeader(plant_dynamics_log, f"Simulation: \"{simulation_label}\" (Plant Log)")

    #   if sim_config["backup_previous_data"]:
    #     print('Creating new data_out.json file (backing up old file).')
    #   else:
    #     print('Appending to existing data_out.json')
    # if sim_config["use_fake_scarab_computation_times"]:
    #   print('Using fake Scarab data.')

    # Create a list of all the key-value pairs that are marked "Keys to Pass to Makefile".
    # make_parameters = [key + '=' + str(experiment_config[key]) for key in experiment_config["Keys to Pass to Makefile"]]

    # Create a list of all the key-value pairs that are marked "Keys to Pass to Makefile".
    # build_config: dict = {key : experiment_config[key] for key in experiment_config["Keys to Pass to Makefile"]}

#         def run_make(task=None):
#           """ Run a Makefile task in the example directory with the parameters from "make_parameters". """
#           if task: 
#             make_cmd = ['make', task] + make_parameters
#           else:
#              make_cmd = ['make'] + make_parameters
#           print(f">> " + " ".join(make_cmd))
#           # Run the make command from the root directory of the example.
#           result = subprocess.check_output(make_cmd, cwd=example_dir)
#           return result.decode("utf-8")
# 
#         # TODO: Controller executable
#         executable = example_dir + "/bin/" + run_make('print_executable').strip()
#         
#         # Build the executable, via Make.
#         run_make()

    # Create the pipe files in the current directory.
    run_shell_cmd("make_scarabintheloop_pipes.sh", log=controller_log)
    assertFileExists(sim_dir + "/x_c++_to_py")

    controller_executable = controller_delegator.get_controller_executable(example_dir, sim_config)
    assertFileExists(controller_executable)

    def run_controller():
      print(f'---- Starting controller for {simulation_label} ----')
      print(f'  Simulation dir: {sim_dir}')
      print(f'  Controller log: {controller_log_path}')
      assertFileExists('config.json')
        
      try:
        if sim_config["parallel_scarab_simulation"]:
          print("Running Scarab in Parallel.")
          
          # Run the controller.
          run_shell_cmd(controller_executable, log=controller_log)

          dynamorio_trace_dirs = glob.glob('dynamorio_trace_*')
          for trace_dir in dynamorio_trace_dirs:
            print("Running run_portabilize_trace in " + trace_dir)
            run_shell_cmd("run_portabilize_trace.sh", log=controller_log, working_dir=trace_dir)
          print(f"Finished portabilizing traces in {sim_dir}.")

          def simulate_trace(dir:str):
            print(f"Simulating trace in {dir}.")
            bin_dir   = dir + "/bin"
            trace_dir = dir + "/trace"
            scarab_cmd = ["scarab", "--fdip_enable", "0", "--frontend", "memtrace", "--fetch_off_path_ops", "0", f"--cbp_trace_r0={trace_dir}", f"--memtrace_modules_log={bin_dir}"]
            run_shell_cmd(scarab_cmd, log=controller_log, working_dir=dir)

          with concurrent.futures.ThreadPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
            tasks = [scarab_executor.submit(simulate_trace(trace_dir)) for trace_dir in dynamorio_trace_dirs]
            print("tasks:" + str(tasks))
            for future in concurrent.futures.as_completed(tasks):
              if future.exception():
                print(f"Task {future} failed with exception: {future.exception()}")
                raise future.exception()
              data = future.result()
              print(f"Task {future} finished")
            # with open('data_out.json') as json_file:
            #   data_out = json.load(json_file)

        elif sim_config["use_fake_scarab_computation_times"]:
          run_shell_cmd([controller_executable], log=controller_log)
        else:
            cmd = " ".join([controller_executable])
            scarab_cmd_argv = [sys.executable,
                     scarab_paths.bin_dir + '/scarab_launch.py',
                     '--program', cmd,
                     '--param', 'PARAMS.generated',
                    #  '--simdir', '.', # Changes the working directory.
                     '--pintool_args',
                     '-fast_forward_to_start_inst 1',
                     '--scarab_args',
                     '--inst_limit 15000000000'
                     #  '--heartbeat_interval 50', 
                     # '--num_heartbeats 1'
                     # '--power_intf_on 1']
                     ]
            run_shell_cmd(scarab_cmd_argv, log=controller_log)
            # subprocess.check_call(scarab_cmd_argv, stdout=stdout, stderr=stderr)

      except Exception as err:
        print(f'Error: Failed when using Scarab to simulate "{controller_executable}"!')
        print(f"Error: {str(err)}")
        if hasattr(err, 'output') and err.output:
          warn(f'Error output: {err.output}')
        print(traceback.format_exc())
        raise err
      print('Controller thread finished.')

    def run_plant():
      print(f'---- Starting plant dynamics for {simulation_label} ----')
      print(f'  Simulation dir: {sim_dir}')
      print(f'       Plant log: {plant_dynamics_log_path}')
      plant_dynamics_log.write('Start of thread.\n')
      assertFileExists('config.json')

      try:
        run_shell_cmd('run_plant.py', log=plant_dynamics_log)
      except Exception as e:
        warn(f'Plant dynamics had an error: {str(e)}')
        print(traceback.format_exc())
        raise e
      print('Plant thread finshed.')
      print(f'  Simulation directory: {sim_dir}')
      print(f'       Simulation data: {sim_dir}/data_out.json')

    if not sim_config["use_external_dynamics_computation"]:
      print("Running controller only -- No plant!")
      run_controller()
      data_out = None # No data recorded
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(run_controller), executor.submit(run_plant)]
        for future in concurrent.futures.as_completed(tasks):
          if future.exception():
            print(f"Task {future} failed with exception: {future.exception()}")
            raise future.exception()
          data = future.result()
        with open('data_out.json') as json_file:
          data_out = json.load(json_file)
    
    print("End of run_simulation.")
    print(f'  Simulation dir: {sim_dir}')
    print(f'  Controller log: {controller_log_path}')
    print(f'       Plant log: {plant_dynamics_log_path}')
    print(f"        Data out: {os.path.abspath('data_out.json')}")
    return data_out


def patch_dictionary(base: dict, patch: dict) -> dict:
  """
  Create a new config dictionary by copying the the base dictionary and replacing any keys with the values in the patch dictionary. The base and patch dictionaries are not modified.
  """
  # Make a copy of the base data so that modifications are not persisted 
  # between examples.
  patched = copy.deepcopy(base);

  for (key, value) in patch.items():
    # if debug_configuration_level >= 2:
    #   print(f"Updating patching configuration data to {key}: {value}")
    patched[key] = value
  return patched


def create_patched_PARAMS_file(sim_config, PARAMS_src, PARAMS_out):
  """
  Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
  Then, modify the values for keys listed in PARAMS_file_keys to values taken from sim_config. 
  Write the resulting PARAMS data to PARAMS.generated in the simulation directory (sim_dir).
  Returns the absolute path to the PARAMS file.
  """
  print(f'Creating chip parameter file.')
  print(f'\tSource: {PARAMS_src}.')
  print(f'\tOutput: {PARAMS_out}.')
  with open(PARAMS_src) as params_out_file:
    PARAM_file_lines = params_out_file.readlines()
  
  for (key, value) in sim_config.items():
    if key in PARAMS_file_keys:
      PARAM_file_lines = changeParamsValue(PARAM_file_lines, key, value)

  # Create PARAMS.generated with the values from the base file modified
  # based on the values in example_configs.json.
  with open(PARAMS_out, 'w') as params_out_file:
    params_out_file.writelines(PARAM_file_lines)


def changeParamsValue(PARAM_lines, key, value):
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


def get_scarab_in_parallel_cmd(exectuable_path:str, chip_params_path:str, config_json:dict, start_index:int=0, time_horizon:int=10, experiment_name:str="experiment", PARAM_index:str=0, sim_config_path=""):
  # start_idx=$1 # Typically "0". We can set larger values to start 'halfway' starting midway from a prior experiment.
  print(f"simulation_executable={exectuable_path}") # The path the executable (already ).
  print(f"chip_params_path={chip_params_path}") # E.v., "chip_params/raspberry_pi_5"
  control_sampling_time = config_json["system_parameters"]["sample_time"]
  total_timesteps = math.ceil(time_horizon / control_sampling_time)
  if total_timesteps <= 0:
    raise ValueError(f"total_timesteps={total_timesteps} was not positive!")
  print(f"total_timesteps={total_timesteps}")
  cmd = [f'run_in_parallel.sh', 
            f'{start_index}',            # 1
            f'{time_horizon}',           # 2
            f'{chip_params_path}',       # 3
            f'{control_sampling_time}',  # 4
            f'{total_timesteps}',        # 5
            f'{experiment_name}',        # 6
            f'{0}',                      # 7
            f'{exectuable_path}',        # 8
            f'{sim_config_path}',   # 9
          ] # 8
  return cmd
  # print(f"config_json={config_json}") # E.v., "controller_parameters/params.json"
  # print(f"time_horizon={time_horizon}") # How many timesteps to simulate, in total.
  # print(f"exp_name={exp_name}") # Define a name for the folders, which are generated.
  # print(f"PARAM_INDEX={PARAM_INDEX}") # Allows selecting from different PARAMS.in files from an array of 
 

def run_shell_cmd(cmd, log=None, working_dir=None):
  if isinstance(cmd, str):
    cmd_print_string = ">> " + cmd
  else:
    cmd_print_string = ">> " + " ".join(cmd)
  print(cmd_print_string)
  if log:
    log.write(cmd_print_string + "\n")
    log.flush()
  
  try:
    subprocess.check_call(cmd, stdout=log, stderr=log, cwd=working_dir)
  except Exception as e:
    print(f'ERROR: The command "{cmd}" failed!')
    if log: 
      # If using an external log file, print the contents.
      log.flush()
      log.seek(0)
      print("(log) ".join([''] + log.readlines()))
    raise e
  

def writeHeader(log, headerText):
  log.write(f'===={"="*len(headerText)}==== \n')
  log.write(f'=== {headerText} === \n')
  log.write(f'===={"="*len(headerText)}==== \n')
  log.flush()

def assertFileExists(path:str):
  if not os.path.exists(path):
      raise IOError(f'Expected {path} to exist but it does not.')

if __name__ == "__main__":
  main()
