#!/usr/bin/env python3
""" 
This script computes the evolution of a system (plant) using control imputs received from a controller executable. 
The communication is done via pipe files contained in the simdir directory.
Instead of calling this script directly, call run_scarabintheloop.py
"""
import numpy as np
from scipy.integrate import ode
import scipy.signal
import numpy.linalg as linalg
import argparse # Parsing of input args.
import plant_dynamics

import math
import time
import datetime
import traceback # Provides pretty printing of exceptions (https://stackoverflow.com/a/1483494/6651650)

import scarabintheloop.scarabizor as scarabizor

import json
import csv
import re
import os

class DataNotRecievedViaFileError(IOError):
  pass

def main():
  print("Start of run_plant.py")
  parser = argparse.ArgumentParser(description=f'Run the plant dynamics, which communicates with the controller via pipe files.')
  args = parser.parse_args()

  sim_dir:str = os.path.abspath('.')
  if not sim_dir.endswith('/'):
    sim_dir += "/"
  
  print(f"Runnning run_plant.py in {sim_dir}")
  
  # Read config.json.
  with open(sim_dir + 'config.json') as json_data_file:
    config_data = json.load(json_data_file)
    print("Read JSON data from 'config.json'.")

#   # Read system_dynamics.json.
#   remaining_timeout = 60 # seconds
#   if not os.path.exists(sim_dir + 'system_dynamics.json.lock'):
#     raise IOError(f"The file {sim_dir + 'system_dynamics.json.lock'} does not exist.")
#   while not os.path.exists(sim_dir + 'system_dynamics.json') or os.path.exists(sim_dir + 'system_dynamics.json.lock'):
#     print(f'Waiting for system_dynamics.json to be created ({remaining_timeout:0.1f} sec until timeout).')
#     time.sleep(0.1)
#     remaining_timeout -= 0.1
#     if remaining_timeout <= 0:
#       raise IOError(f'system_dynamics.json was not created within the time limit.')
#   # Sleep a moment to make sure system_dynamics.json is done being written.
#   time.sleep(1)
# 
#   with open(sim_dir + 'system_dynamics.json') as json_data_file:
#     system_dynamics_data = json.load(json_data_file)
#     print("Read JSON data from 'system_dynamics.json'")

  stats_reader = scarabizor.ScarabStatsReader(sim_dir)

  # Debugging levels
  debug_config = config_data["==== Debgugging Levels ===="]
  debug_interfile_communication_level = debug_config["debug_interfile_communication_level"]
  debug_dynamics_level = debug_config["debug_dynamics_level"]

  # Settings
  use_scarab_delays = not config_data["use_fake_scarab_computation_times"]
  only_update_control_at_sample_times = config_data["only_update_control_at_sample_times"]
  is_parallelized = config_data["parallel_scarab_simulation"]
  if is_parallelized:
    if use_scarab_delays:
      print(f"Changing use_scarab_delays={use_scarab_delays} to False because parallel_scarab_simulation=True.")
      use_scarab_delays = False
    if not only_update_control_at_sample_times:
      print(f"Changing only_update_control_at_sample_times={only_update_control_at_sample_times} to True because parallel_scarab_simulation=True.")
      only_update_control_at_sample_times = True


  #  File names for the pipes we use for communicating with C++.
  x_in_filename = sim_dir + 'x_c++_to_py'
  u_in_filename = sim_dir + 'u_c++_to_py'
  x_predict_in_filename = sim_dir + 'x_predict_c++_to_py'
  t_predict_in_filename = sim_dir + 't_predict_c++_to_py'
  iterations_in_filename = sim_dir + 'iterations_c++_to_py'
  x_out_filename = sim_dir + 'x_py_to_c++'
  t_delay_out_filename = sim_dir + 't_delay_py_to_c++'

  # File names for recording values, to use in plotting.
  x_data_filename = sim_dir + 'x_data.csv'
  u_data_filename = sim_dir + 'u_data.csv'
  t_data_filename = sim_dir + 't_data.csv'

  n_time_steps = config_data["n_time_steps"]
  sample_time = config_data["system_parameters"]["sample_time"] # Discrete sample period.
  fake_computation_delay_times = config_data["fake_computation_delay_times"]

  # Get the dynamics definition.
  n = config_data["system_parameters"]["state_dimension"]
  m = config_data["system_parameters"]["input_dimension"]
  evolveState = plant_dynamics.getDynamicsFunction(config_data)

  xs_list = []
  us_list = []
  ts_list = []
  x_predictions_list = []
  t_predictions_list = [] # Absolute time
  t_delays_list = [] # Relative time interval
  simulated_computation_times_list = []
  instruction_counts_list = []
  cycles_counts_list = []
  walltimes_list = []

  u_prev = np.zeros((m, 1)) 

  def printIndented(string_to_print:str, indent: int=1):
    indent_str = '\t' * indent
    indented_line_break = "\n" + indent_str
    string_to_print = string_to_print.replace('\n', indented_line_break)
    print(indent_str + string_to_print)


  def convertStringToVector(vector_str: str):
      vector_str_list = vector_str.split(',') #.strip().split("\t")

      # Convert the list of strings to a list of floats.
      chars_to_strip = ' []\n'
      v = np.array([[np.float64(x.strip(chars_to_strip)),] for x in vector_str_list])
      
      # Make into a column vector;
      # v.transpose()
    
      if debug_interfile_communication_level >= 3:
        print('convertStringToVector():')
        printIndented('vector_str:', 1)
        printIndented(repr(vector_str), 1)
        printIndented('v:', 1)
        printIndented(repr(v), 1)
        # print('\tvector_str:\n' + repr(vector_str).replace('\n','\n\t'), end='\n\t')
        # print('\t    v:\n' + repr(v).replace('\n','\n\t'))

      return v

  def numpyArrayToCsv(array: np.array) -> str:
    string = ', '.join(map(str, array.flatten()))
    # print("In numpyArrayToCsv")
    # printIndented("array", 1)
    # printIndented(repr(array), 2)
    # printIndented('String: ' + string, 1)
    return string


  def checkAndStripInputLoopNumber(expected_k, input_line):
    """ Check that an input line, formatted as "Loop <k>: <data>" 
    has the expected value of k (given by the argument "expected_k") """

    split_input_line = input_line.split(':')
    loop_input_str = split_input_line[0]

    # Check that the input line is in the expected format. 
    if not loop_input_str.startswith('Loop '):
      raise ValueError(f'The input_line "{input_line}" did not start with a loop label.')
    
    # Extract the loop number and convert it to an integer.
    input_loop_number = int(loop_input_str[len('Loop:'):])

    if input_loop_number != k:
      raise ValueError(f'The input_loop_number="{input_loop_number}" does not match expected_k={expected_k}.')
      # If the first piece of the input line doesn't give the correct loop numb
      
    # Return the part of the input line that doesn't contain the loop info.
    return split_input_line[1]

  def waitForLineFromFile(file):
    # max_loops = 
    if debug_interfile_communication_level >= 1:
          print(f"Waiting for input_line from {file.name}.")
    for i in range(0, 10):
      input_line = file.readline()
      # print(f'input line read i:{i}, input_line:{input_line}')
      # print()
      if input_line:
        if debug_interfile_communication_level >= 1:
          print(f"Recieved input_line from {file.name} on loop #{i}:")
          print(input_line)
        return input_line

      # time.sleep(0.1) # Sleep for X seconds

    raise DataNotRecievedViaFileError(f"No input recieved from {file.name}.")

  def generateDataOut():
    # Create a JSON object for storing the data out, using all the values in config_data.
    data_out_json = config_data;
    data_out_json["datetime_of_run"] = datetime.datetime.now().isoformat()
    data_out_json["cause_of_termination"] = cause_of_termination
    data_out_json["x"] = xs_list;
    data_out_json["x_prediction"] = x_predictions_list;
    data_out_json["t_prediction"] = t_predictions_list;
    data_out_json["t_delay"] = t_delays_list;
    data_out_json["u"] = us_list;
    data_out_json["t"] = ts_list;
    data_out_json["computation_delays"] = simulated_computation_times_list;
    data_out_json["instruction_counts"] = instruction_counts_list;
    data_out_json["cycles_counts"] = cycles_counts_list
    data_out_json["execution_walltime"] = walltimes_list
    
    # Remove all of the phony JSON "header" entries, such as "==== Settings ====".
    # for key in  list(data_out_json.keys()):
    #   if key.startswith("="):
    #     del data_out_json[key]

    if use_scarab_delays:
      # Save Scarab parameters from PARAMS.out into the data_out JSON object. 
      param_keys_to_save_in_data_out = ["chip_cycle_time", "l1_size", "dcache_size", "icache_size", "decode_cycles"]
      param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")
      with open(sim_dir + 'PARAMS.out') as params_out_file:
        f_lines = params_out_file.readlines()
        for line in f_lines: 
          regex_match = param_regex_pattern.match(line)
          if regex_match:
            param_name = regex_match.groupdict()['param_name']
            param_value = regex_match.groupdict()['param_value']
            if param_name in param_keys_to_save_in_data_out:
              data_out_json[param_name] = int(param_value)
        
    # Read optimizer info.
    with open(sim_dir + 'optimizer_info.csv', newline='') as csvfile:
      csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      
      num_iterationss_list = []
      costs_list = []
      primal_residuals_list = []
      dual_residuals_list = []
      is_header = True
      for row in csv_reader:
          if is_header:
            num_iterations_ndx = row.index('num_iterations')
            cost_ndx = row.index('cost')
            primal_residual_ndx = row.index('primal_residual')
            dual_residual_ndx = row.index('dual_residual')
            is_header = False
          else:
            num_iterationss_list.append(int(row[num_iterations_ndx]))
            costs_list.append(float(row[cost_ndx]))
            primal_residuals_list.append(float(row[primal_residual_ndx]))
            dual_residuals_list.append(float(row[dual_residual_ndx]))
        
    data_out_json["num_iterations"] = num_iterationss_list
    data_out_json["cost"] = costs_list
    data_out_json["primal_residual"] = primal_residuals_list
    data_out_json["dual_residual"] = dual_residuals_list
      
    # Read current data_out.json. If it exists, append the new data. Otherwise, create it.
    try:
      with open(sim_dir + 'data_out.json', 'r') as json_data_file:
        data_out_list = json.load(json_data_file)
        data_out_list.append(data_out_json)
    except FileNotFoundError as err:
      # Create a new data_out_list containing one element, the data_out_json from this run.
      print('data_out.json not found. Creating a new data_out_list.')
      data_out_list = [data_out_json]

    return data_out_list

  LINE_BUFFERING = 1
  if debug_interfile_communication_level >= 1:
    print("About to open files for interprocess communication. Will wait for readers...")
  with open(x_in_filename, 'r', buffering= LINE_BUFFERING) as x_infile, \
      open(u_in_filename, 'r', buffering= LINE_BUFFERING) as u_infile, \
      open(x_predict_in_filename, 'r', buffering= LINE_BUFFERING) as x_predict_infile, \
      open(t_predict_in_filename, 'r', buffering= LINE_BUFFERING) as t_predict_infile, \
      open(iterations_in_filename, 'r', buffering= LINE_BUFFERING) as iterations_infile, \
      open(x_out_filename, 'w', buffering= LINE_BUFFERING) as x_outfile, \
      open(t_delay_out_filename, 'w', buffering= LINE_BUFFERING) as t_delay_outfile, \
      open(x_data_filename, 'w', buffering= LINE_BUFFERING) as x_datafile, \
      open(u_data_filename, 'w', buffering= LINE_BUFFERING) as u_datafile, \
      open(t_data_filename, 'w', buffering= LINE_BUFFERING) as t_datafile :
    print('Pipes are open')

    def snapshotState(k: int, t: float, xarray: np.ndarray, uarray: np.ndarray, description: str):
      """ Define a function for printing the current state of the system to 
      the console and saving the state to the data files. """
      x_str = numpyArrayToCsv(xarray)
      u_str = numpyArrayToCsv(uarray)
      t_str = str(t)

      if debug_interfile_communication_level >= 1:
        print(f'Snapshotting State: t = {t_str} ({description})')
        printIndented(f'x: {x_str}', 1)
        printIndented(f'u: {u_str}', 1)
        printIndented(f't: {t_str}', 1)

      # x_datafile.write(x_str + "\n")
      # u_datafile.write(u_str + "\n")
      # t_datafile.write(t_str + "\n")

      # Add xarray and uarray to the cumulative lists. 
      # We want xarray to be stored as a single list of numbers, but the 
      # tolist() function creates a nested lists, with the outter list containing
      # a single list. Thus, we use "[0]" to reference the inner list. 
      xs_list.append(xarray.transpose().tolist()[0])
      us_list.append(uarray.transpose().tolist()[0])
      ts_list.append(t)

      
    def snapshotPrediction(t_prediction: float, x_prediction_array: np.ndarray, description: str):
      """ Define a function for printing the prediction to 
      the console and saving the state to the data files. """
      x_predict_str = numpyArrayToCsv(x_prediction_array)
      t_predict_str = f"{t_prediction:.8}"

      if debug_interfile_communication_level >= 1:
        print(f'Snapshotting Prediction ({description})')
        printIndented(f'x prediction: {x_predict_str}', 1)
        printIndented(f't prediction: {t_predict_str}', 1)

      x_datafile.write(x_predict_str + "\n")
      t_datafile.write(t_predict_str + "\n")

      # Add xarray and uarray to the cumulative lists. 
      # We want xarray to be stored as a single list of numbers, but the 
      # tolist() function creates a nested lists, with the outter list containing
      # a single list. Thus, we use "[0]" to reference the inner list. 
      x_predictions_list.append(x_prediction_array.transpose().tolist()[0])
      t_predictions_list.append(t_prediction)

    t = 0.0
    cause_of_termination = "In progress"
    try:
      for k in range(0, n_time_steps+1):
        print(f'(Loop {k}) - Waiting for state \'x\' from file.')
        t_start_of_loop = t
        walltime_start_of_loop = time.time()

        # Get the 'x' input line from the file.
        x_input_line = waitForLineFromFile(x_infile)
        
        if x_input_line.startswith("Done"):
          print("Received 'DONE' from controller.")
          walltimes_list.append(time.time() - walltime_start_of_loop)
          break

        u_input_line = waitForLineFromFile(u_infile)
        x_predict_input_line = waitForLineFromFile(x_predict_infile)
        t_predict_input_line = waitForLineFromFile(t_predict_infile)
        iterations_input_line = waitForLineFromFile(iterations_infile)
          
        x_input_line = checkAndStripInputLoopNumber(k, x_input_line)
        u_input_line = checkAndStripInputLoopNumber(k, u_input_line)
        x_predict_input_line = checkAndStripInputLoopNumber(k, x_predict_input_line)
        t_predict_input_line = checkAndStripInputLoopNumber(k, t_predict_input_line)
        iterations_input_line = checkAndStripInputLoopNumber(k, iterations_input_line) 

        if debug_interfile_communication_level >= 1:
          print('Input strings from C++:')
          printIndented("           x input line: " + x_input_line.strip(), 1)
          printIndented("           u input line: " + u_input_line.strip(), 1)
          printIndented("x prediction input line: " + x_predict_input_line.strip(), 1)
          printIndented("t prediction input line: " + t_predict_input_line.strip(), 1)
          printIndented("  iterations input line: " + iterations_input_line.strip(), 1)

        x = convertStringToVector(x_input_line)
        u = convertStringToVector(u_input_line)
        x_prediction = convertStringToVector(x_predict_input_line)
        t_prediction = t_start_of_loop + float(t_predict_input_line)
        iterations = float(iterations_input_line)

        # Get the statistics input line.
        if use_scarab_delays: 
          if debug_interfile_communication_level >= 2:
            print('Waiting for statistics from Scarab.')
          stats_reader.waitForStatsFile(k)
          simulated_computation_time = stats_reader.readTime(k)
          instruction_count = stats_reader.readInstructionCount(k)
          cycles_count = stats_reader.readCyclesCount(k)
        else:
          delay_model_slope = config_data["computation_delay_slope"]
          delay_model_y_intercept = config_data["computation_delay_y-intercept"]
          if delay_model_slope:
            if not delay_model_y_intercept:
              raise ValueError(f"delay_model_slope was set but delay_model_y_intercept was not.")
            simulated_computation_time = delay_model_slope * iterations + delay_model_y_intercept
            print(f"simulated_computation_time = {simulated_computation_time:.8g} = {delay_model_slope:.8g} * {iterations:.8g} + {delay_model_y_intercept:.8g}")
          else:
            print('Using constant delay times.')
            simulated_computation_time = fake_computation_delay_times
          instruction_count = 0
          cycles_count = 0

        simulated_computation_times_list.append(simulated_computation_time)
        instruction_counts_list.append(instruction_count)
        cycles_counts_list.append(cycles_count)
        
        simulated_computation_time_str = f"{simulated_computation_time:.8g}"
        print(f"Delay time: {simulated_computation_time_str} seconds ({100*simulated_computation_time/sample_time:.3g}% of sample time).")

        # Write the data for the first time step to the CSV files.
        if k == 0:
          snapshotState(k, t, x, u_prev, 'Initial condition')

        print("t_predict_input_line: " + t_predict_input_line)
        snapshotPrediction(t_prediction, x_prediction, "Prediction")

        n_samples_without_finishing_computation = math.floor(simulated_computation_time/sample_time)
        n_samples_until_next_sample_after_computation_finishes = n_samples_without_finishing_computation + 1

        if debug_dynamics_level >= 1:
          print(f"simulated_computation_time={simulated_computation_time_str}, sample_time={sample_time:.2f}.")

        if only_update_control_at_sample_times:
          t_start = t
          t_computation_update = t + (n_samples_without_finishing_computation+1)*sample_time
          t_end = t_computation_update
          # TODO: This model of the delay is designed to match the one used by Yasin's parallelized scheme, but we should check that his model uses u_prev until the computation time passes.
          (t, x) = evolveState(t_start, x, u_prev, t_end)
          snapshotState(k, t, x, u_prev, f"After full sample interval with {n_samples_without_finishing_computation} missed computation updates.")
          u_prev = u
        else: # Update when the computation finishes, without waiting for the next sample time.
          t_start = t
          t_computation_update = t + simulated_computation_time
          t_end = t + (n_samples_without_finishing_computation + 1)*sample_time

          # Evolve x over the one or more sample periods where the control is not updated.
          (t, x) = evolveState(t_start, x, u_prev, t_computation_update)

          # Snapshot at the computation time with the old control value.
          snapshotState(k, t, x, u_prev, f"After delay - Previous control. Missed {n_samples_without_finishing_computation} samples without updating controller.")
          
          # Snapshot at the computation time with the new control value.
          snapshotState(k, t, x, u, f"After delay - New control. Missed {n_samples_without_finishing_computation} samples without updating controller.")

          (t, x) = evolveState(t_computation_update, x, u, t_end)

          # Save the state at the end of the sample period.
          snapshotState(k, t, x, u, "End of sample period. Missed {n_samples_without_finishing_computation} samples without updating controller.")
        
          u_prev = u

        # Pass the string back to C++.
        x_out_string = numpyArrayToCsv(x)
        if debug_interfile_communication_level >= 1:
          print("x output line:" + repr(x_out_string))
        x_outfile.write(x_out_string + "\n")# Write to pipe to C++
        t_delay_outfile.write(f"{simulated_computation_time_str} \n")
        walltimes_list.append(time.time() - walltime_start_of_loop)
        t_delays_list.append(simulated_computation_time)
        
        data_out_intermediate_list = generateDataOut()
        with open(sim_dir + 'data_out_intermediate.json', 'w') as json_data_file:
          json_data_file.write(json.dumps(data_out_intermediate_list, allow_nan=False, indent=4))
        print('\n=====\n')
    except NameError as err:
      raise err
    except (DataNotRecievedViaFileError, BrokenPipeError) as err:
      cause_of_termination = repr(err)
      walltimes_list.append(time.time() - walltime_start_of_loop)
      traceback.print_exc()
    else:
      cause_of_termination = "Finished."

  data_out_list = generateDataOut()
  # Save the data_out to file (if enabled)
  with open(sim_dir + 'data_out.json', 'w') as json_data_file:
    json_data_file.write(json.dumps(data_out_list, allow_nan=False, indent=4))

  print(f"There are now {len(data_out_list)} entries in data_out.json")


if __name__ == "__main__":
  main()
