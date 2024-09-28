"""
This module defines a function that defines the evolution of the plant of a given system. 
It is loaded by the run_plant.py script.
"""

import numpy as np
from scipy.integrate import ode
import scipy.signal
import numpy.linalg as linalg
from scarabintheloop.utils import printIndented

debug_interfile_communication_level = 0
debug_dynamics_level = 0

# TODO: Implement "appendComputedParameters" to generate computed parameters that need to be accessible from both the plant and controller. Currently, these values are hard coded in both the C++ controller and the Python plant.
# def appendComputedParameters(config_data: dict):
#   """
#   For the given config_data, compute any additional system parameters that are dependent on the given data.
#   """
#   # config_data["system_dynamics"]["Ac_entriesa"] = 
#   # system_dynamics_data["Ac_entries"]).reshape(n, n
#   A 
#   B = np.array(system_dynamics_data["Bc_entries"]).reshape(n, m)
#   B_dist = np.array(system_dynamics_data["Bc_disturbances_entries"])

## TODO Use OOP to create base classes for different types of plant dynamics.
# class BasePlantDynamics():
#   def __init__(self, config_data: dict):
#     pass
# 
#   def evolve_state(...):
#     raise Error("Must be implemented by user")
# 
# class OdePlantDynamics(BasePlantDynamics):
#   def __init__(self, config_data: dict):
#     A = config_data["A"]
#   
#   def system_derivative(t, x, params):
#     raise Error("Must be implemented by user")
# 
#   def evolve_state(self, ...):
#     
#     return (tf, xf)
# 
# class LTIPlantDynamics(OdePlantDynamics):
#   def __init__(self, config_data: dict):
#     A = config_data["A"]
#   
#   def system_derivative(self, t, x, params):
#     u = params["u"]
#     w = params["w"]
#     return self.A * x + self.B * u + self.B_dist * w


def getDynamicsFunction(config_data: dict):

  ######################################
  ######## Define the Dynamics #########
  ######################################

  # Read the state and input matrices from the JSON files, reshaping them according to the given 
  # values of m and n (they are stored as 1D vectors).
  n = config_data["system_parameters"]["state_dimension"]
  m = config_data["system_parameters"]["input_dimension"]
  if not n == 5:
      raise ValueError('Unsupported state dimension') 
  if not m == 1:
      raise ValueError('Unsupported input dimension') 
  # A = np.array(config_data["system_parameters"]["Ac_entries"]).reshape(n, n)
  # B = np.array(config_data["system_parameters"]["Bc_entries"]).reshape(n, m)
  # B_dist = np.array(config_data["system_parameters"]["Bc_disturbances_entries"]).reshape(n, m)

  tau = config_data["system_parameters"]["tau"]
  A = np.array([[0,      1, 0,  0,      0],  # v1
                [0, -1/tau, 0,  0,      0],  # a1
                [1,      0, 0, -1,      0],  # d2
                [0,      0, 0,  0,      1],  # v2
                [0,      0, 0,  0, -1/tau]]) # a2

  # Define continuous-time input matrix B_c
  B = np.array([[0],
                [0],
                [0], 
                [0], 
                [-1/tau]]);
  B_dist = np.array([[0],
                     [1/tau],
                     [0], 
                     [0], 
                     [0]]);
  

  # State to output matrix
  C = np.array([[0, 0, 1, 0, 0], 
                [1, 0, 0,-1, 0]]);

  D = np.zeros([5, 1], float)

  if debug_dynamics_level >= 1:
    fake_computation_delay_times = config_data["Simulation Options"]["use_fake_scarab_computation_times"]
    sample_time = config_data["system_parameters"]["sample_time"]
    (A_to_delay, B_to_delay, C_to_delay, D_to_delay, dt) = scipy.signal.cont2discrete((A, B, C, D), fake_computation_delay_times)
    (A_delay_to_sample, B_delay_to_sample, C_delay_to_sample, D_delay_to_sample, dt) = scipy.signal.cont2discrete((A, B, C, D), sample_time - fake_computation_delay_times)
    print("A_to_delay")
    print(A_to_delay)
    print("A_delay_to_sample")
    print(A_delay_to_sample)

  if debug_dynamics_level >= 2 or debug_interfile_communication_level >= 2:
    print("A from C++:")
    print(A)
    print("B from C++:")
    print(B)

  def system_derivative(t, x, params):
    """ Give the value of f(x) for the system \dot{x} = f(x). """
    # Convert x into a 2D (nx1) array instead of 1D array (length n).
    x = x[:, None]
    u = params["u"]
    w = params["w"]

    if debug_dynamics_level >= 3:
      print('Values when calculating system xdot:')
      printIndented('x = ' + str(x), 1)
      printIndented('u = ' + str(u), 1)
      printIndented('w = ' + str(w), 1)

    # Compute the system dynamics.
    xdot = np.matmul(A, x) +  np.matmul(B, u) + B_dist * w;

    if debug_dynamics_level >= 3:
      print('Values when calculating system xdot:')
      printIndented('  A*x = ' + str( np.matmul(A, x)), 1)
      printIndented('  B*u = ' + str( np.matmul(B, u)), 1)
      printIndented('B_d*w = ' + str((B_dist * w)), 1)
      printIndented(' xdot = ' + str(xdot), 1)
    return xdot
  

  solver = ode(system_derivative).set_integrator('vode', method='bdf')
  def evolveState(t0, x0, u, tf):

    if debug_dynamics_level >= 1:
      print(f"== evolveState(t=[{t0:.2}, {tf:.2}]) ==")
      print("x0': " + str(x0.transpose()))
      print(" u': " + str(u))

    # Create an ODE solver
    params = {"u": u, "w": 0}
    solver.set_f_params(params)
    solver.set_initial_value(x0, t=t0)
    # Simulate the system until the computation ends, using the previous control value.
    solver.integrate(tf)
    xf = solver.y

    # As an alternative method to compute the evolution of the state, use the discretized dynamics.
    Ad, Bd = scipy.signal.cont2discrete((A, B, C, D), tf - t0)[0:2]
    xf_alt = np.matmul(Ad, x0) + np.matmul(Bd, u)
    
    if debug_dynamics_level >= 2:
      print("Ad")
      printIndented(str(Ad), 1)
      print("Bd")
      printIndented(str(Bd), 1)
      print(f"    xf={xf.transpose()}.")
      print(f"xf_alt={xf_alt.transpose()}.")

    error_metric_between_methods = linalg.norm(xf - xf_alt) / max(linalg.norm(xf), 1)
    if error_metric_between_methods > 1e-2:
      raise ValueError(f"|xf - xf_alt| = {linalg.norm(xf - xf_alt)} is larger than tolerance")
    xf = xf_alt

    if debug_dynamics_level >= 2:
      print(f'ODE solver run over interval t=[{t0}, {tf}] with x0={x0.transpose().tolist()[0]}, u={u}')
    if debug_dynamics_level >= 3:
      print('Solver values at end of delay.')
      print(f"\tsolver.u = {repr(solver.f_params)}")
      print(f"\tsolver.y = {repr(solver.y)}")
      print(f"\tsolver.t = {repr(solver.t)}")
    return (tf, xf)

  return evolveState