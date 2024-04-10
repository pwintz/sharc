#!/usr/bin/env python3
import numpy as np
from scipy.integrate import ode

import time
import datetime
import traceback # Provides pretty printing of exceptions (https://stackoverflow.com/a/1483494/6651650)


import sys
sys.path.append('/workspaces/ros-docker/')

import os
os.chdir('/workspaces/ros-docker/libmpc_example')
import scarabizor

import json
import csv
import re

class DataNotRecievedViaFileError(IOError):
  pass

# Read config.json.
with open('config.json') as json_data_file:
    config_data = json.load(json_data_file)

# Read system_dynamics.json.
with open('system_dynamics.json') as json_data_file:
    system_dynamics_data = json.load(json_data_file)

stats_reader = scarabizor.ScarabStatsReader('sim_dir')

# Settings
debug_level = 1
use_scarab_delays = not config_data["use_fake_scarab_computation_times"]

#  File names for the pipes we use for communicating with C++.
x_out_filename = 'sim_dir/x_py_to_c++'
u_in_filename = 'sim_dir/u_c++_to_py'
x_in_filename = 'sim_dir/x_c++_to_py'
# File names for recording values, to use in plotting.
x_data_filename = 'sim_dir/x_data.csv'
u_data_filename = 'sim_dir/u_data.csv'
t_data_filename = 'sim_dir/t_data.csv'

n_time_steps = config_data["n_time_steps"]
sample_time = config_data["sample_time"] # Discrete sample period.
fake_computation_delay_times = config_data["fake_computation_delay_times"]

# Read the state and input matrices from the JSON files, reshaping them according to the given 
# values of m and n (they are stored as 1D vectors).
n = system_dynamics_data["state_dimension"]
m = system_dynamics_data["input_dimension"]
A = np.array(system_dynamics_data["Ac_entries"]).reshape(n, n)
B = np.array(system_dynamics_data["Bc_entries"]).reshape(n, m)
# print(Ac)
# print(Bc)

x_cumul = []
u_cumul = []
t_cumul = []
simulated_computation_time_cumul = []
instruction_count_cumul = []
cycles_count_cumul = []
walltime_cumul = []

u_prev = np.zeros((m, 1)) 

def convertStringToVector(v_str: str):
    v_str_list = v_str.split(',') #.strip().split("\t")

    # Convert the list of strings to a list of floats.
    v = np.array([[np.float64(x),] for x in v_str_list])
    
    # Make into a column vector;
    # v.transpose()
   
    if debug_level >= 3:
      print('\tv: ' + repr(v))

    return v

def numpyArrayToCsv(array: np.array) -> str:
  return ', '.join(map(str, array.flatten()));
  # ','.join(map(str, x.flatten())_

def system_derivative(t, x, u):
    # Convert x into a 2D (nx1) array instead of 1D array (length n).
    x = x[:,None]

    # Compute the system dynamics.
    xdot = np.matmul(A, x) +  np.matmul(B, u);
    # print(f"x={x}, u={u}, xdot={xdot}.")
    # current_position, current_velocity = x
    # return [current_velocity, u]
    if debug_level >= 3:
      print('Values when calculating system xdot:')
      print('x = ' + repr(x))
      print('u = ' + repr(u))
      print('Ax = ' + repr( np.dot(A, x)))
      print('Bu = ' + repr( np.dot(B, u)))
      print('xdot = ' + repr(xdot))
    return xdot

solver = ode(system_derivative).set_integrator('vode', method='bdf')
def evolveState(t0, x0, u, tf):
  # Create an ODE solver
  solver.set_initial_value(x0, t=t0)
  solver.set_f_params(u)
  # Simulate the system until the computation ends, using the previous control value.
  solver.integrate(tf)
  x = solver.y

  if debug_level >= 2:
    print(f'ODE solver run over interval t=[{t0}, {tf}] with x0={x0}, u={u}')
  if debug_level >= 3:
    print('Solver values at end of delay.')
    print(f"\tsolver.u = {repr(solver.f_params)}")
    print(f"\tsolver.y = {repr(solver.y)}")
    print(f"\tsolver.t = {repr(solver.t)}")
    # print(f"x = {y}")
  return x

def checkAndStripInputLoopNumber(expected_k, input_line):
  """ Check that an input line, formatted as "Loop <k>: <data>" 
  has the expected value of k (given by the argument "expected_k") """

  split_input_line = input_line.split(':')
  # if debug_level >= 1:
  #   print(f"Reading an input line for {split_input_line[0]}.")

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
  if debug_level >= 1:
        print(f"Waiting for input_line from {file.name}.")
  for i in range(0, 10):
    input_line = file.readline()
    # print(f'input line read i:{i}, input_line:{input_line}')
    # print()
    if input_line:
      if debug_level >= 1:
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
  data_out_json["x"] = x_cumul;
  data_out_json["u"] = u_cumul;
  data_out_json["t"] = t_cumul;
  data_out_json["computation_delays"] = simulated_computation_time_cumul;
  data_out_json["instruction_counts"] = instruction_count_cumul;
  data_out_json["cycles_counts"] = cycles_count_cumul
  data_out_json["execution_walltime"] = walltime_cumul
  data_out_json["prediction_horizon"] = system_dynamics_data["prediction_horizon"]
  data_out_json["control_horizon"] = system_dynamics_data["control_horizon"]

  # Remove all of the phony JSON "header" entries, such as "==== Settings ====".
  for key in  list(data_out_json.keys()):
    if key.startswith("="):
      del data_out_json[key]

  if use_scarab_delays:
    # Save Scarab parameters from PARAMS.out into the data_out JSON object. 
    param_keys_to_save_in_data_out = ["chip_cycle_time", "l1_size", "dcache_size", "icache_size", "decode_cycles"]
    param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")
    with open('sim_dir/PARAMS.out') as params_out_file:
      f_lines = params_out_file.readlines()
      for line in f_lines: 
        regex_match = param_regex_pattern.match(line)
        if regex_match:
          param_name = regex_match.groupdict()['param_name']
          param_value = regex_match.groupdict()['param_value']
          if param_name in param_keys_to_save_in_data_out:
            data_out_json[param_name] = int(param_value)
      
  # Read optimizer info.
  with open('sim_dir/optimizer_info.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    num_iterations_cumul = []
    cost_cumul = []
    primal_residual_cumul = []
    dual_residual_cumul = []
    is_header = True
    for row in csv_reader:
        if is_header:
          num_iterations_ndx = row.index('num_iterations')
          cost_ndx = row.index('cost')
          primal_residual_ndx = row.index('primal_residual')
          dual_residual_ndx = row.index('dual_residual')
          is_header = False
        else:
          num_iterations_cumul.append(int(row[num_iterations_ndx]))
          cost_cumul.append(float(row[cost_ndx]))
          primal_residual_cumul.append(float(row[primal_residual_ndx]))
          dual_residual_cumul.append(float(row[dual_residual_ndx]))
      
  data_out_json["num_iterations"] = num_iterations_cumul
  data_out_json["cost"] = cost_cumul
  data_out_json["primal_residual"] = primal_residual_cumul
  data_out_json["dual_residual"] = dual_residual_cumul
    
  # Read current data_out.json. If it exists, append the new data. Otherwise, create it.
  try:
    with open('data_out.json', 'r') as json_data_file:
      data_out_list = json.load(json_data_file)
      data_out_list.append(data_out_json)
  except FileNotFoundError as err:
    # Create a new data_out_list containing one element, the data_out_json from this run.
    print('data_out.json not found. Creating a new data_out_list.')
    data_out_list = [data_out_json]

  return data_out_list

LINE_BUFFERING = 1
with open(x_in_filename, 'r', buffering= LINE_BUFFERING) as x_infile, \
     open(u_in_filename, 'r', buffering= LINE_BUFFERING) as u_infile, \
     open(x_out_filename, 'w', buffering= LINE_BUFFERING) as x_outfile, \
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

    print(f't = {t_str} ({description})')
    print(f'\t x: {x_str}')
    print(f'\t u: {u_str}')
    print(f'\t t: {t_str}')

    x_datafile.write(x_str + "\n")
    u_datafile.write(u_str + "\n")
    t_datafile.write(t_str + "\n")

    # Add xarray and uarray to the cumulative lists. 
    # We want xarray to be stored as a single list of numbers, but the 
    # tolist() function creates a nested lists, with the outter list containing
    # a single list. Thus, we use "[0]" to reference the inner list. 
    x_cumul.append(xarray.transpose().tolist()[0])
    u_cumul.append(uarray.transpose().tolist()[0])
    t_cumul.append(t)

  t = 0.0
  cause_of_termination = "In progress"
  try:
    for k in range(0,n_time_steps+1):
      print(f'(Loop {k})')
      t_start_of_loop = t
      walltime_start_of_loop = time.time()

      # Get the 'x' input line from the file.
      x_input_line = waitForLineFromFile(x_infile)
      
      if x_input_line.startswith("Done"):
        print("Received 'DONE' from controller.")
        walltime_cumul.append(time.time() - walltime_start_of_loop)
        break

      u_input_line = waitForLineFromFile(u_infile)
        
      x_input_line = checkAndStripInputLoopNumber(k, x_input_line)
      u_input_line = checkAndStripInputLoopNumber(k, u_input_line)

      # Get the statistics input line.
      if use_scarab_delays: 
        stats_reader.waitForStatsFile(k)
        simulated_computation_time = stats_reader.readTime(k)
        instruction_count = stats_reader.readInstructionCount(k)
        cycles_count = stats_reader.readCyclesCount(k)
      else:
        simulated_computation_time = fake_computation_delay_times
        instruction_count = 0
        cycles_count = 0

      simulated_computation_time_cumul.append(simulated_computation_time)
      instruction_count_cumul.append(instruction_count)
      cycles_count_cumul.append(cycles_count)
      
      print(f"Delay time: {simulated_computation_time:.3g} seconds ({100*simulated_computation_time/sample_time:.3g}% of sample time).")

      # if debug_level >= 1:
      #   print('Input strings:')
      #   print("x input line: " + repr(x_input_line))
      #   print("u input line: " + repr(u_input_line))

      x = convertStringToVector(x_input_line)
      u = convertStringToVector(u_input_line)

      # Write the data for the first time step to the CSV files.
      if k == 0:
        snapshotState(k, t, x, u_prev, 'Initial condition')

      # If the controller computed the update in less than the sample time, then 
      # we compute the remainder of the sample time period using the new control value. 
      # If it was not computed in time, then u_prev will not be updated, so the previous 
      # value will be used again in the next interval. 
      if simulated_computation_time >= sample_time:
        print(f"simulated_computation_time={simulated_computation_time:.2f} >= sample_time={sample_time:.2f} . Control not being updated.")
        x = evolveState(t, x, u_prev, t + sample_time)
        t = t_start_of_loop + sample_time

        # Save values. We use 'u' instead of 'u_prev' because this is the point where we update the input.
        snapshotState(k, t, x, u_prev, "After full sample interval - No control update.")
      else:
        print(f"simulated_computation_time={simulated_computation_time:.2f} < sample_time={sample_time:.2f}. Control to be updated.")
        x = evolveState(t, x, u_prev, t + simulated_computation_time)
        t = t_start_of_loop + simulated_computation_time

        # Save values. We use 'u' instead of 'u_prev' because this is the point where we update the input.
        snapshotState(k, t, x, u_prev, "After delay - Previous control")

        snapshotState(k, t, x, u, "After delay - New control")

        x = evolveState(t, x, u, t_start_of_loop+sample_time)
        t = t_start_of_loop + sample_time
        u_prev = u

        snapshotState(k, t, x, u, "End of sample")

      # Pass the string back to C++.
      x_out_string = numpyArrayToCsv(x)
      print("x output line:" + repr(x_out_string))
      x_outfile.write(x_out_string + "\n")# Write to pipe to C++
      walltime_cumul.append(time.time() - walltime_start_of_loop)
      
      data_out_list = generateDataOut()
      with open('data_out_intermediate.json', 'w') as json_data_file:
        json_data_file.write(json.dumps(data_out_list, allow_nan=False, indent=4))
      print('\n=====\n')
  except NameError as err:
    raise err
  except (DataNotRecievedViaFileError, BrokenPipeError) as err:
    cause_of_termination = repr(err)
    walltime_cumul.append(time.time() - walltime_start_of_loop)
    traceback.print_exc()
  else:
    cause_of_termination = "Finished."

data_out_list = generateDataOut()
# Save the data_out to file (if enabled)
if config_data["record_data_out"]:
  with open('data_out.json', 'w') as json_data_file:
    json_data_file.write(json.dumps(data_out_list, allow_nan=False, indent=4))
else:
  # print(json.dumps(data_out_list))
  print("JSON Data not recorded because config_data[\"record_data_out\"] is False.")


print(f"There are now {len(data_out_list)} entries in data_out.json")