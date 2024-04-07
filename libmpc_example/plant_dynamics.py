#!/usr/bin/env python3
import numpy as np
from scipy.integrate import ode

import time

import sys
sys.path.append('/workspaces/ros-docker/')

import os
os.chdir('/workspaces/ros-docker/libmpc_example')
# os.environ['TRACES_PATH'] = '/path/to/traces'
# os.environ['DYNAMORIO_ROOT'] = '/path/to/dynamorio'
# os.environ['SCARAB_ROOT'] = '/path/to/scarab'
# os.environ['SCARAB_OUT_PATH'] = '/path/to/scarab/out'
import scarabizor

import json

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
computation_delay_scalar = config_data["computation_delay_scalar"]

# Read the state and input matrices from the JSON files, reshaping them according to the given 
# values of m and n (they are stored as 1D vectors).
n = system_dynamics_data["state_dimension"]
m = system_dynamics_data["input_dimension"]
A = np.array(system_dynamics_data["Ac_entries"]).reshape(n, n)
B = np.array(system_dynamics_data["Bc_entries"]).reshape(n, m)
# print(Ac)
# print(Bc)

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
  for i in range(0, 3):
    input_line = file.readline()
    print(f'input line read i:{i}, input_line:{input_line}')
    # print()
    if input_line:
      if debug_level >= 1:
        print(f"Recieved input_line from {file.name} on loop #{i}:")
        print(input_line)
      return input_line

    # time.sleep(0.01) # Sleep for X seconds

  raise Error(f"No input recieved from {file.name}.")


LINE_BUFFERING = 1
with open(x_in_filename, 'r', buffering= LINE_BUFFERING) as x_infile, \
     open(u_in_filename, 'r', buffering= LINE_BUFFERING) as u_infile, \
     open(x_out_filename, 'w', buffering= LINE_BUFFERING) as x_outfile, \
     open(x_data_filename, 'w', buffering= LINE_BUFFERING) as x_datafile, \
     open(u_data_filename, 'w', buffering= LINE_BUFFERING) as u_datafile, \
     open(t_data_filename, 'w', buffering= LINE_BUFFERING) as t_datafile :
  print('Pipe is open')

  def snapshotState(k, t, xarray, uarray, description):
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
    # END: snapshotState

  t = 0.0
  for k in range(0,n_time_steps):
    print(f'(Loop {k})')
    t_start_of_loop = t

    # Get the 'x' input line from the file.
    x_input_line = waitForLineFromFile(x_infile)

    x_input_line = checkAndStripInputLoopNumber(k, x_input_line)
    
    # Get the 'u' input line from the file.
    u_input_line = waitForLineFromFile(u_infile)

    u_input_line = checkAndStripInputLoopNumber(k, u_input_line)

    # Get the statistics input line.
    if use_scarab_delays: 
      stats_reader.waitForStatsFile(k)
      simulated_time = stats_reader.readTime(k)
      simulated_time *= computation_delay_scalar
    else:
      simulated_time = 0.01 * sample_time
      if computation_delay_scalar == 0:
        simulated_time = 0
    # print(f"(Loop {k}). Simulated computation time: {simulated_time} sec.")
    # print(f"(Loop {k}). Current t: {t} sec.") 

    if simulated_time > sample_time:
      raise ValueError(f"The simulated time {simulated_time} is longer than the sample time {sample_time}.")

    print(f"Delay time: {simulated_time:.3g} seconds ({100*simulated_time/sample_time:.3g}% of sample time).")

    if debug_level >= 1:
      print('Input strings:')
      print("x input line: " + repr(x_input_line))
      print("u input line: " + repr(u_input_line))

    x_entries = x_input_line.split(',')
    x = convertStringToVector(x_input_line)
    u_entries = u_input_line.split(',')
    u = convertStringToVector(u_input_line)

    # Write the data for the first time step to the CSV files.
    if k == 0:
      snapshotState(k, t, x, u_prev, 'Initial condition')
      # x_datafile.write(numpyArrayToCsv(x) + "\n")
      # u_datafile.write(numpyArrayToCsv(u_prev) + "\n")
      # t_datafile.write(str(0) + "\n")

    if debug_level >= 3:
      print('Numpy array representations of input strings:')
      print("x = " + repr(x))
      # print("x' = " + repr(np.transpose(x)))
      print("u = " + repr(u))
    # for entry in x_entries:
    #   print(entry + ",",end="\t") 
    # if len(x_input_line) == 1:
    #   print('line length = ' + str(len(x_input_line)))

    # Create an ODE solver
    solver = ode(system_derivative).set_integrator('vode', method='bdf')
    solver.set_initial_value(x, t=0)
    solver.set_f_params(u_prev)

    delay_time = simulated_time
    solver.integrate(delay_time)
    x_after_delay = solver.y

    t = t_start_of_loop + delay_time

    # Save values to CSV files.
    # We use 'u' instead of 'u_prev' because this is the point where
    snapshotState(k, t, x_after_delay, u_prev, "After delay - Previous control")
    snapshotState(k, t, x_after_delay, u, "After delay - New control")
    # x_datafile.write(numpyArrayToCsv(x_after_delay) + "\n")
    # u_datafile.write(numpyArrayToCsv(u_prev) + "\n")
    # t_datafile.write(str(t) + "\n")
    

    if debug_level >= 2:
      print('Solver values at end of delay.')
      print(f"\tsolver.u = {repr(solver.f_params)}")
      print(f"\tsolver.y = {repr(solver.y)}")
      print(f"\tsolver.t = {repr(solver.t)}")
      # print(f"x = {y}")

    solver.set_initial_value(x_after_delay, t=delay_time)
    solver.set_f_params(u)
    u_prev = u
    t = t_start_of_loop + sample_time
    
    # Integrate from the initial time 'delay_time' to the final time 'sample_time'.
    solver.integrate(t=sample_time)
    x = solver.y


    snapshotState(k, t, x, u_prev, "End of sample")
    # # Save values to CSV files.
    # x_datafile.write(numpyArrayToCsv(x) + "\n")
    # u_datafile.write(numpyArrayToCsv(u) + "\n")
    # t_datafile.write(str(t) + "\n")

    if debug_level >= 2:
      print('Solver values at end of sample time.')
      print(f"\tsolver.u = {repr(solver.f_params)}")
      print(f"\tsolver.y = {repr(solver.y)}")
      print(f"\tsolver.t = {repr(solver.t)}")
      # print(f"x = {y}")

    # Put the Numpy array into a comma-separated string.
    # x_out_string = ', '.join(map(str, x.flatten()));
    x_out_string = numpyArrayToCsv(x)
    u_out_string = numpyArrayToCsv(u)
    t_out_string = str(t)

    # x_out_string = np.array2string(x, separator=',').replace('\n', '')

    # Pass the string back to C++.
    print("x output line:" + repr(x_out_string))
    x_outfile.write(x_out_string + "\n")# Write to pipe to C++

    print('\n=====\n')
