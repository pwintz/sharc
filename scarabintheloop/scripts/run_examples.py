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

import json
import os
import concurrent.futures
import subprocess
import re
import time
import copy
import argparse
from warnings import warn

from slugify import slugify

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]
compile_option_keys = ["prediction_horizon", "control_horizon"]

# Regex to find text in the form of "--name value"
param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

# Format string to generate text in the form of "--name value"
param_str_fmt = "--{}\t{}\n"

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


def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Run a set of examples.")
  
  parser.add_argument(
      'example_dir',
      help="Enter the directory of the example. It must contain all of required example files; see the project README file for a full description of the required files and their format."
  )

  parser.add_argument(
      '--example_set_name',  # Another argument name
      type=str,  # Specify the data type
      default="Default",  # Provide a default value
      help="Select the set of examples (default: 'Default')"
  )
  
  parser.add_argument(
      '--output_dir', 
      type=str, # Specify the data type
      help="Select the folder where files produced by the simulation are placed."
  )
  
  parser.add_argument(
      '--config_filename',
      type=str,  # Specify the data type
      default= ,
      help="Select the name of a JSON file located in <example_dir>/simulation_configs."
  )

  # Parse the arguments from the command line
  args = parser.parse_args()
  example_dir = args.example_dir
  example_set_name = args.example_set_name
  config_filename = args.config_filename

  # If the user gave an output directory, then use it. Otherwise, use "out/" within the example folder.
  if args.output_dir:
    output_dir = os.path.abspath(args.output_dir)
  else:
    output_dir = os.path.abspath(example_dir) + "/out"

  try:
    os.chdir(example_dir)  # Change the current working directory
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: The example directory '{example_dir}' does not exist.")
  print(f"Directory changed to: {os.getcwd()}")

  # Read JSON configuration file.
  with open('config_base.json') as json_file:
    config_base_data = json.load(json_file)

  # Open list of example configurations.
  with open('example_configs.json') as json_file:
    example_config_data = json.load(json_file)

  debug_configuration_level = config_base_data["==== Debgugging Levels ===="]["debug_configuration_level"]

  time_in_filename_format = "%Y-%m-%d-%H:%M:%S"
  sim_dir = output_dir + "/" + slugify(example_set_name) + "-" + time.strftime(time_in_filename_format) + "/"
  print(f"sim_dir: {sim_dir}")

  latest_sim_dir_path = output_dir + "/latest"

  if not os.path.exists(sim_dir):
    # Create the sim_dir if it does not exist.
    os.makedirs(sim_dir)

  if os.path.exists(latest_sim_dir_path):
    os.remove(latest_sim_dir_path)

  os.symlink(sim_dir, latest_sim_dir_path, target_is_directory=True)

  if config_base_data["backup_previous_data"]:
    # Create a backup of the exising data_out.json file.
    os.makedirs("data_out_backups", exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    try:
      backup_file_name = "data_out_backups/data_out-" + timestr + "-backup.json"
      os.rename(sim_dir + 'data_out.json', backup_file_name)
      print(f"data_out.json backed up to {backup_file_name}")
    except FileNotFoundError as err:
      # Not creating backup because data_out.json does not exist.
      pass

  # example_set_name = config_base_data["example_set_name"]
  try:
    example_set = example_config_data[example_set_name]
  except KeyError as e:
    raise ValueError(f"There is no set of examples named '{example_set_name}' in example_configs.json.")
 
  print(f"==== {example_set_name} ==== ")
  
  controller_log_filename = sim_dir + 'controller.log'
  plant_dynamics_log_filename = sim_dir + 'plant_dynamics.log'
  with open(controller_log_filename, 'w') as controller_log, \
        open(plant_dynamics_log_filename, 'w') as plant_dynamics_log:

    for example in example_set:
      if example.get("skip", False):
        print(f'Skipping example: {example["label"]}')
        continue

      controller_log.write(f'\n============================================== \n')
      controller_log.write(f'===== Start of {example["label"]} example ==== \n')
      controller_log.write(f'============================================== \n')
      controller_log.flush()
      plant_dynamics_log.write(f'\n============================================== \n')
      plant_dynamics_log.write(f'===== Start of {example["label"]} example ==== \n')
      plant_dynamics_log.write(f'============================================== \n')
      plant_dynamics_log.flush()

      # Make a copy of the base data so that modifications are not persisted 
      # between examples.
      example_config_data = copy.deepcopy(config_base_data);

      for (key, value) in example.items():
        if debug_configuration_level >= 2:
          print(f"Updating example configuration data to {key}: {value}")
        example_config_data[key] = value

      PARAMS_base_filename = example_config_data["PARAMS_base_file"]
      print(f'Using parameter file: {PARAMS_base_filename}.')
      with open(PARAMS_base_filename) as params_out_file:
        PARAM_file_lines = params_out_file.readlines()
      
      for (key, value) in example_config_data.items():
        if key in PARAMS_file_keys:
          PARAM_file_lines = changeParamsValue(PARAM_file_lines, key, value)
        # elif key in compile_option_keys:
        #   changeCompilationOption(key, value)
        # else:
        #   example_config_data[key] = value
        # parameter_key_handler(key, value)
      # print(example_config_data)

      # Create PARAMS.generated with the values from the base file modified
      # based on the values in example_configs.json.
      with open(sim_dir + 'PARAMS.generated', 'w') as params_out_file:
        params_out_file.writelines(PARAM_file_lines)

      # Write the configuration to config.json.
      with open(sim_dir + 'config.json', 'w') as config_file:
          config_file.write(json.dumps(example_config_data))

      if not example_config_data["record_data_out"]:
        print("!!! Not recording data !!!")
      else:
        if example_config_data["backup_previous_data"]:
          print('Creating new data_out.json file (backing up old file).')
        else:
          print('Appending to existing data_out.json')
      if example_config_data["use_fake_scarab_computation_times"]:
        print('Using fake Scarab data.')

      make_cmd = ['make', 'simulate', 'PREDICTION_HORIZON=' + str(example_config_data["prediction_horizon"]), 'CONTROL_HORIZON='+ str(example_config_data["control_horizon"])]
      if debug_configuration_level >= 1:
        print("Make command: " + " ".join(make_cmd))

      def run_scarab_simulation():
        print(f'Starting controller Scarab for {example["label"]} example...  (See {controller_log_filename})')
        # subprocess.check_call(['make', 'simulate', ], stdout=controller_log, stderr=controller_log)
        
        try:
          subprocess.check_call(make_cmd, stdout=controller_log, stderr=controller_log)
        except subprocess.CalledProcessError as err:
          warn(f'Running "{" ".join(make_cmd)}" Failed')
          warn(err.output)
          raise err
        print('Scarab thread finished.')

      def run_plant():
        print(f'Starting plant dynamics for {example["label"]} example...  (See {plant_dynamics_log_filename})')
        subprocess.check_call(['make', 'run_plant'], stdout=plant_dynamics_log, stderr=plant_dynamics_log)
        print('Plant dynamics thread finshed.')

      with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
          tasks = [executor.submit(run_scarab_simulation), executor.submit(run_plant)]
          for future in concurrent.futures.as_completed(tasks):
            data = future.result()

if __name__ == "__main__":
  main()
