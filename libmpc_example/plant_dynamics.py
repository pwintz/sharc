#!/usr/bin/env python3
import numpy as np
from scipy.integrate import ode

import sys
sys.path.append('/workspaces/ros-docker/')



import os
os.chdir('/workspaces/ros-docker/libmpc_example')
# os.environ['TRACES_PATH'] = '/path/to/traces'
# os.environ['DYNAMORIO_ROOT'] = '/path/to/dynamorio'
# os.environ['SCARAB_ROOT'] = '/path/to/scarab'
# os.environ['SCARAB_OUT_PATH'] = '/path/to/scarab/out'
import scarabizor

stats_reader = scarabizor.ScarabStatsReader('sim_dir')

# Settings
debug_level = 1

#  File names for the pipes we use for communicating with C++.
x_out_filename = 'sim_dir/x_py_to_c++'
u_in_filename = 'sim_dir/u_c++_to_py'
x_in_filename = 'sim_dir/x_c++_to_py'
x_data_filename = 'sim_dir/x_data.csv'
u_data_filename = 'sim_dir/u_data.csv'
t_data_filename = 'sim_dir/t_data.csv'

n = 1e-3  # For low-Earth orbit.

# Define the dynamics of the system
A = np.array([
    [0,      0,      1,     0],
    [0,      0,      0,     1],
    [3*n**2, 0,      0, 2 * n],
    [0,      0, -2 * n,     0]
])
# A = np.eye(4)

# Define matrix B
B = np.vstack([np.zeros((2, 2)), np.eye(2)])

# Define 
u_prev = np.zeros((2, 1))

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
    xdot =  np.matmul(A, x) +  np.matmul(B, u);
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
  print(f"Input loop number: {split_input_line[0]}")
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

LINE_BUFFERING = 1
with open(x_in_filename, 'r', buffering= LINE_BUFFERING) as x_infile,     \
     open(u_in_filename, 'r', buffering= LINE_BUFFERING) as u_infile,     \
     open(x_out_filename, 'w', buffering= LINE_BUFFERING) as x_outfile,   \
     open(x_data_filename, 'w', buffering= LINE_BUFFERING) as x_datafile, \
     open(u_data_filename, 'w', buffering= LINE_BUFFERING) as u_datafile, \
     open(t_data_filename, 'w', buffering= LINE_BUFFERING) as t_datafile :
  print('Pipe is open')

  def snapshotState(k, t, xarray, uarray, description):
    """ Define a function for printing the current state of the system to 
    the console and saving the state to the data files."""
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

  k = 0
  t = 0.0
  
  while True:
    print(f'(Loop {k})')
    t_start_of_loop = t

    # Get the x input line.
    x_input_line = x_infile.readline()
    while not x_input_line:
      x_input_line = x_infile.readline()
      continue

    x_input_line = checkAndStripInputLoopNumber(k, x_input_line)
    
    # Get the u input line.
    u_input_line = u_infile.readline()      
    while not u_input_line:
      u_input_line = u_infile.readline()
      continue

    u_input_line = checkAndStripInputLoopNumber(k, u_input_line)

    # Get the statistics input line.
    stats_reader.waitForStatsFile(k)
    simulated_time = stats_reader.readTime(k)
    # print(f"(Loop {k}). Simulated computation time: {simulated_time} sec.")
    # print(f"(Loop {k}). Current t: {t} sec.")

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

    sample_time = 0.1
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

    k = k+1