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
import concurrent.futures
import subprocess
import re
import time
import copy
import argparse
from warnings import warn
from pathlib import Path
import sys

import json

# Add <scarab path>/bin to the Python search pth.
# scarab_root_path = os.environ["SCARAB_ROOT"]
# sys.path.append(scarab_root_path + '/bin')

from scarab_globals import *

from slugify import slugify

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]
# compile_option_keys = ["prediction_horizon", "control_horizon"]

# Regex to find text in the form of "--name value"
param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

# Format string to generate text in the form of "--name value"
param_str_fmt = "--{}\t{}\n"

def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Run a set of examples.")
  
  parser.add_argument(
      'example_dir',
      help="Enter the directory of the example. It must contain all of required example files; see the project README file for a full description of the required files and their format."
  )

  # parser.add_argument(
  #     '--output_dir', 
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
  example_dir = os.path.abspath(args.example_dir)
  config_file_path = Path(os.path.abspath(os.path.join(example_dir, "simulation_configs", args.config_filename)))
  base_config_file_path = Path(os.path.abspath(os.path.join(example_dir, 'base_config.json')))

  # If the user gave an output directory, then use it. Otherwise, use "out/" within the example folder.
  # if args.output_dir:
  #   output_dir = os.path.abspath(args.output_dir)
  # else:
  output_dir = os.path.abspath(example_dir) + "/out"

  try:
    os.chdir(example_dir)  # Change the current working directory
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: The example directory '{example_dir}' does not exist.")
  print(f"Directory changed to: {os.getcwd()}")

  # Add the example's scripts directory to the PATH.
  os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/scripts")
  os.environ["PATH"] += os.pathsep + os.path.abspath(example_dir + "/bin")


  # Read JSON configuration file.
  with open(base_config_file_path) as json_file:
    base_config = json.load(json_file)

  # Open list of example configurations.
  # with open('example_configs.json') as json_file:
  with open(config_file_path) as config_json_file:
    sim_config_patches_list = json.load(config_json_file)

  debug_configuration_level = base_config["==== Debgugging Levels ===="]["debug_configuration_level"]

  time_in_filename_format = "%Y-%m-%d-%H:%M:%S"
  sim_list_dir = output_dir + "/" + slugify(config_file_path.stem) + "-" + time.strftime(time_in_filename_format) + "/"
  print(f"sim_list_dir: {sim_list_dir}")

  # if not os.path.exists(sim_list_dir):
    # Create the sim_list_dir if it does not exist.
  os.makedirs(sim_list_dir)

  # Create a symlink to the latest folder.
  latest_sim_list_dir_symlink_path = output_dir + "/latest"
  if os.path.exists(latest_sim_list_dir_symlink_path):
    os.remove(latest_sim_list_dir_symlink_path)
  os.symlink(sim_list_dir, latest_sim_list_dir_symlink_path, target_is_directory=True)

  # if base_config["backup_previous_data"]:
  #   # Create a backup of the exising data_out.json file.
  #   os.makedirs("data_out_backups", exist_ok=True)
  #   timestr = time.strftime("%Y%m%d-%H%M%S")
  #   try:
  #     backup_file_name = "data_out_backups/data_out-" + timestr + "-backup.json"
  #     os.rename(sim_dir + 'data_out.json', backup_file_name)
  #     print(f"data_out.json backed up to {backup_file_name}")
  #   except FileNotFoundError as err:
  #     # Not creating backup because data_out.json does not exist.
  #     pass

  # The simulation config file can either contain a single configuration or a list.
  # In the case where there is only a single configuration, we convert it to a singleton list. 
  if not isinstance(sim_config_patches_list, list):
    sim_config_patches_list = [sim_config_patches_list]

  for sim_config_patches in sim_config_patches_list:

    if sim_config_patches.get("skip", False):
      print(f'Skipping example: {sim_label}')
      continue

    # Make subdirectory for this simulation.
    sim_dir = os.path.abspath(sim_list_dir + "/" + slugify(sim_config_patches['label']) )
    os.makedirs(sim_dir)
    os.chdir(sim_dir)
    print(f"Simulation directory: {sim_dir}")

    sim_label = config_file_path.stem + "/" + sim_config_patches["label"]
    print(f"==== {sim_label} ==== ")

    controller_log_filename = sim_dir + '/controller.log'
    plant_dynamics_log_filename = sim_dir + '/plant_dynamics.log'

    with open(controller_log_filename, 'w') as controller_log, \
          open(plant_dynamics_log_filename, 'w') as plant_dynamics_log:

        writeHeader(controller_log,     f"Simulation: \"{sim_label}\" (Controller Log)")
        writeHeader(plant_dynamics_log, f"Simulation: \"{sim_label}\" (Plant Log)")

        def run_shell_cmd_for_controller(cmd):
          print(">> " + cmd)
          subprocess.check_call(cmd, stdout=controller_log, stderr=controller_log)

        def run_shell_cmd_for_plant(cmd):
          print(">> " + cmd)
          subprocess.check_call(cmd, stdout=plant_dynamics_log, stderr=plant_dynamics_log)

        # Using the base configuration dictionary (from JSON file), update the values fiven in the simulation config patch, loaded from
        sim_config = patch_dictionary(base_config, sim_config_patches)

        # Create PARAMS.generated file.
        read_patch_and_write_PARAMS(sim_config, sim_dir, example_dir)

        # Write the configuration to config.json.
        with open(sim_dir + '/config.json', 'w') as config_file:
            config_file.write(json.dumps(sim_config, indent=2))

        if not sim_config["record_data_out"]:
          print("!!! Not recording data !!!")
        else:
          if sim_config["backup_previous_data"]:
            print('Creating new data_out.json file (backing up old file).')
          else:
            print('Appending to existing data_out.json')
        if sim_config["use_fake_scarab_computation_times"]:
          print('Using fake Scarab data.')

        # Create a list of all the key-value pairs that are marked "Keys to Pass to Makefile".
        make_parameters = [key + '=' + str(sim_config[key]) for key in sim_config["Keys to Pass to Makefile"]]

        def run_make(task=None):
          """ Run a Makefile task in the example directory with the parameters from "make_parameters". """
          if task: 
            make_cmd = ['make', task] + make_parameters
          else:
             make_cmd = ['make'] + make_parameters
          print(f">> " + " ".join(make_cmd))
          result = subprocess.check_output(make_cmd, cwd=example_dir)
          return result.decode("utf-8")

        executable = example_dir + "/bin/" + run_make('print_executable').strip()
        
        # Build the executable, via Make.
        run_make()

        # Create the pipe files.
        run_shell_cmd_for_controller("make_scarabintheloop_pipes.sh")

        def run_controller():
          print(f'Starting controller for {sim_label} example...  (See {controller_log_filename})')
          try:
            if sim_config["parallel_scarab_simulation"]:
              run_shell_cmd_for_controller(executable)

            if sim_config["use_fake_scarab_computation_times"]:
              run_shell_cmd_for_controller(executable)
            else:
              run_scarab(
                executable, 
                sim_dir=sim_dir, 
                stdout=controller_log, 
                stderr=controller_log
              )
          except subprocess.CalledProcessError as err:
            warn(f'Running Scarab to simulate "{executable}" Failed: \n {err.output}')
            raise err
          print('Scarab thread finished.')

        def run_plant():
          print(f'Starting plant dynamics for {sim_label} example...  (See {plant_dynamics_log_filename})')
          plant_dynamics_log.write('Start of thread.\n')
          plant_dynamics_log.flush()

          try:
            run_shell_cmd_for_plant('plant_dynamics.py')
          except Exception as e:
            print('plant dynamics failed')
            raise e
          else:
            print('Plant dynamics thread finshed succesfully.')

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            tasks = [executor.submit(run_controller), executor.submit(run_plant)]
            for future in concurrent.futures.as_completed(tasks):
              try:
                print(f"Task {future} finished? {future.exception()}")
                data = future.result()
              except Exception as e:
                print('Exception!')
                executor.shutdown()
                raise e
  # try:
  #   example_list = sim_config[example_list_name]
  # except KeyError as e:
  #   raise ValueError(f"There is no set of examples named '{example_list_name}' in example_configs.json.")
 

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


def read_patch_and_write_PARAMS(sim_config, sim_dir, example_dir):
  """
  Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
  Then, modify the values for keys listed in PARAMS_file_keys to values taken from sim_config. 
  Write the resulting PARAMS data to PARAMS.generated in the simulation directory (sim_dir).
  """
  PARAMS_base_filename = example_dir + "/" + sim_config["PARAMS_base_file"]
  print(f'Using parameter file: {PARAMS_base_filename}.')
  with open(PARAMS_base_filename) as params_out_file:
    PARAM_file_lines = params_out_file.readlines()
  
  for (key, value) in sim_config.items():
    if key in PARAMS_file_keys:
      PARAM_file_lines = changeParamsValue(PARAM_file_lines, key, value)
    # elif key in compile_option_keys:
    #   changeCompilationOption(key, value)
    # else:
    #   sim_config[key] = value
    # parameter_key_handler(key, value)
  # print(sim_config)

  # Create PARAMS.generated with the values from the base file modified
  # based on the values in example_configs.json.
  with open(sim_dir + '/PARAMS.generated', 'w') as params_out_file:
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


def run_scarab(cmd_to_simulate, sim_dir=None, stdout=None, stderr=None):
  scarab_cmd_argv = [sys.executable,
                     scarab_paths.bin_dir + '/scarab_launch.py',
                     '--program', cmd_to_simulate,
                     '--param', 'PARAMS.generated',
                    #  '--simdir', '.', # Changes the working directory.
                     '--pintool_args',
                     '-fast_forward_to_start_inst 1',
                     '--scarab_args',
                     '--inst_limit 15000000000',
                    #  '--heartbeat_interval 50', 
                     # '--num_heartbeats 1'
                     # '--power_intf_on 1']
                     ]
  print('Scarab cmd:', ' '.join(scarab_cmd_argv))
  subprocess.check_call(scarab_cmd_argv, stdout=stdout, stderr=stderr)

def writeHeader(log, headerText):
  log.write(f'===={"="*len(headerText)}==== \n')
  log.write(f'=== {headerText} === \n')
  log.write(f'===={"="*len(headerText)}==== \n')
  log.flush()

if __name__ == "__main__":
  main()
