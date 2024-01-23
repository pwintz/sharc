#!/usr/bin/env python3

import time
import numpy as np
from scipy.integrate import ode

# Settings
debug = False

#  File names for the pipes we use for communicating with C++.
x_out_filename = 'sim_dir/x_py_to_c++'
u_in_filename = 'sim_dir/u_c++_to_py'
x_in_filename = 'sim_dir/x_c++_to_py'

n = 1e-3  # For low-Earth orbit.

# Define the dynamics of the system
A = np.array([
    [0,      0,      1, 0],
    [0,      0,      0, 1],
    [3*n**2, 0,      0, 2 * n],
    [0,      0, -2 * n, 0]
])

# Define matrix B
B = np.vstack([np.zeros((2, 2)), np.eye(2)])

def convertStringToVector(v_str: str):
    v_str_list = v_str.split(',') #.strip().split("\t")

    # Convert the list of strings to a list of floats.
    v = np.array([np.float64(x) for x in v_str_list])

    return v

def system_derivative(t, x, u):
    xdot =  np.dot(A, x) +  np.dot(B, u);
    # print(f"x={x}, u={u}, xdot={xdot}.")
    # current_position, current_velocity = x
    # return [current_velocity, u]
    return xdot

LINE_BUFFERING = 1
with open(x_in_filename, 'r', buffering= LINE_BUFFERING) as x_infile,  \
     open(u_in_filename, 'r', buffering= LINE_BUFFERING) as u_infile,  \
     open(x_out_filename, 'w', buffering= LINE_BUFFERING) as x_outfile:
  print('Pipe is open')
  while True:
    x_input_line = x_infile.readline()
    if x_input_line:
      u_input_line = u_infile.readline()
      x_entries = x_input_line.split(',')
      x = convertStringToVector(x_input_line)
      u_entries = u_input_line.split(',')
      u = convertStringToVector(u_input_line)

      if debug:
        print("x input line: " + x_input_line )
        print("u input line: " + u_input_line )
        print("x = " + str(x))
        print("u = " + str(u))

      # for entry in x_entries:
      #   print(entry + ",",end="\t") 
      # if len(x_input_line) == 1:
      #   print('line length = ' + str(len(x_input_line)))

      # Create an ODE solver
      solver = ode(system_derivative).set_integrator('vode', method='bdf')
      solver.set_initial_value(x, t=0)
      solver.set_f_params(u)

      sample_time = 1293.23
      solver.integrate(sample_time)
      x = solver.y

      # print(f"solver.y = {solver.y}")

      # Put the Numpy array into a comma-separated string.
      x_out_string = ', '.join(map(str, x));

      # Pass the string back to C++.
      print(f"Writing to x_outfile: {x_out_string}")
      x_outfile.write(x_out_string + "\n")