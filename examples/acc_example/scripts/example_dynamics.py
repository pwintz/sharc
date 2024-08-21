
import numpy as np
from scipy.integrate import ode
import scipy.signal
import numpy.linalg as linalg

def getDynamicsFunction(config_data, system_dynamics_data):

  # Debugging levels
  debug_config = config_data["==== Debgugging Levels ===="]
  debug_interfile_communication_level = debug_config["debug_interfile_communication_level"]
  debug_dynamics_level = debug_config["debug_dynamics_level"]

  ######################################
  ######## Define the Dynamics #########
  ######################################

  # Read the state and input matrices from the JSON files, reshaping them according to the given 
  # values of m and n (they are stored as 1D vectors).
  n = system_dynamics_data["state_dimension"]
  m = system_dynamics_data["input_dimension"]
  A = np.array(system_dynamics_data["Ac_entries"]).reshape(n, n)
  B = np.array(system_dynamics_data["Bc_entries"]).reshape(n, m)
  B_dist = np.array(system_dynamics_data["Bc_disturbances_entries"]).reshape(n, m)

  C = np.identity(5, float)
  D = np.zeros([5, 1], float)

  if debug_dynamics_level >= 1:
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
    x = x[:,None]
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


# COPIED from plant_dynamics.py.
# TODO: place it in a single place accesible from both python files.
def printIndented(string_to_print:str, indent: int=1):
  indent_str = '\t' * indent
  indented_line_break = "\n" + indent_str
  string_to_print = string_to_print.replace('\n', indented_line_break)
  print(indent_str + string_to_print)