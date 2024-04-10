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

import sys
import json
import os
import concurrent.futures
import subprocess
import re
import time

PARAMS_file_keys = ["chip_cycle_time", "l1_size", "icache_size", "dcache_size"]
compile_option_keys = ["prediction_horizon", "control_horizon"]
config_json_keys = ["enable_mpc_warm_start", "osqp_maximum_iteration"]

def main():

  # Create a backup of the exising data_out.json file.
  os.makedirs("data_out_backups", exist_ok=True)
  timestr = time.strftime("%Y%m%d-%H%M%S")
  try:
    os.rename('data_out.json', "data_out_backups/data_out-" + timestr + "-backup.json")
  except FileNotFoundError as err:
    # Not creating backup because data_out.json does not exist.
    pass

  # Read JSON files.
  with open('config_base.json') as json_file:
      config_base_data = json.load(json_file)
  with open('example_configs.json') as json_file:
      example_config_data = json.load(json_file)

  param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")
  param_str_fmt = "--{}\t{}"
  with open('PARAMS.base') as params_out_file:
    PARAM_file_lines = params_out_file.readlines()

  # print(PARAM_file_lines[0])
  # print(param_str_fmt.format("hello", 0))

  def changeParamsValue(key, value):
    for index, line in enumerate(PARAM_file_lines):
      regex_match = param_regex_pattern.match(line)
      if not regex_match:
        continue
      if key == regex_match.groupdict()['param_name']:
        # If the regex matches, then we replace the line.
        PARAM_file_lines[index] = param_str_fmt.format(key, value)
        return
      raise ValueError(f"Key \"{key}\" was not found.")
    
    def changeCompilationOption(key, value):
      pass

  def parameter_key_handler(key, value):
    if key in PARAMS_file_keys:
      changeParamsValue(key, value)
    elif key in compile_option_keys:
      changeCompilationOption(key, option)
    else:
      example_config_data[key] = value
    # elif key in config_json_keys:
    # else: 
    #   raise ValueError(f"Unknown key: {key}")

  examples = example_config_data["Clock speed"]
  for example in examples:
    example_config_data = config_base_data;
    for (key, value) in example.items():
      parameter_key_handler(key, value)
    # print(example_config_data)

    with open('PARAMS.generated', 'w') as params_out_file:
      params_out_file.writelines(PARAM_file_lines)

    with open('config.json', 'w') as config_file:
        config_file.write(json.dumps(example_config_data))

    def run_scarab_simulation():
      subprocess.check_call(['make', 'simulate'])
      # subprocess.check_call(['make', 'simulate', '-B', 'PREDICTION_HORIZON=4' + str(example_config_data["prediction_horizon"]), 'CONTROL_HORIZON='+str(example_config_data["control_horizon"])])

    def run_plant():
        subprocess.check_call(['make', 'run_plant'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(run_scarab_simulation), executor.submit(run_plant)]

        for future in concurrent.futures.as_completed(tasks):
          # try:
            data = future.result()
          # except Exception as exc:
          #   print('error:' + repr(exc))
          #   raise exc
          # else:
            # print('OK')

if __name__ == "__main__":
  main()
