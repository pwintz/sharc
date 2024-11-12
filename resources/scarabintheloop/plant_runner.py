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
import copy
# import argparse # Parsing of input args.

# Import contextmanager to allow defining commands to be used to create "with" blocks.
from contextlib import contextmanager 

import math
import time
import traceback # Provides pretty printing of exceptions (https://stackoverflow.com/a/1483494/6651650)
from abc import abstractmethod, ABC

# from scarabintheloop import ControllerInterface
from scarabintheloop.controller_interface import ControllerInterface
from scarabintheloop.utils import *
import scarabintheloop.debug_levels as debug_levels
from scarabintheloop.data_types import *
from scarabintheloop.dynamics_base import Dynamics

from typing import Callable, List, Set, Dict, Tuple, Union

try:
  # Ensure that assertions are enabled.
  assert False
  raise Exception('Python assertions are not working. Scarab-in-the-loop relies on Python assertions to fail quickly when problems arise. Possible causes for assertions being disabled are running with the "-O" flag or running a precompiled (".pyo" or ".pyc") module.')
except AssertionError:
    pass

def run(sim_dir: str, config_data: dict, dynamics: Dynamics, controller_interface: ControllerInterface) -> dict:
  if not sim_dir.endswith("/"):
    sim_dir += "/"
  print(f"Start of plant_runner.run()  in {sim_dir}")

  n_time_steps = config_data["n_time_steps"]
  sample_time = config_data["system_parameters"]["sample_time"] # Discrete sample period.
  n = config_data["system_parameters"]["state_dimension"] 
  m = config_data["system_parameters"]["input_dimension"] 
  first_time_index = config_data['first_time_index']
  only_update_control_at_sample_times = config_data["only_update_control_at_sample_times"]
  
  # Read the initial control value.
  x0 = list_to_column_vec(config_data['x0'])
  u0 = list_to_column_vec(config_data['u0'])
  
  assert x0.shape == (n, 1), f'x0={x0} must have the shape: {(n, 1)}'
  assert u0.shape == (m, 1), f'u0={u0} must have the shape: {(m, 1)}'

  # Note: pending_computation may be None.
  pending_computation0 = config_data["pending_computation"]

  try:
    x_start = x0
    u_before = u0
    pending_computation_before = pending_computation0
    
    assert x_start.shape  == (n, 1), f'x_start={x_start} did not have the expected shape: {(n, 1)}'
    assert u_before.shape == (m, 1), f'u_before={u_before} did not have the expected shape: {(m, 1)}'

    time_step_series = TimeStepSeries(k0=first_time_index, 
                                      t0=first_time_index * sample_time, 
                                      x0=x0, 
                                      pending_computation_prior=pending_computation0)
    
    controller_interface.post_simulator_running() # Post the simulator status for the controller to access.

    simulation_time_steps = list(range(first_time_index, first_time_index + n_time_steps))
    assert len(simulation_time_steps) == n_time_steps
    for k_time_step in simulation_time_steps:
      printHeader2(f'Starting time step #{time_step_series.n_time_steps} of {n_time_steps} steps.')
      t_start = k_time_step * sample_time
      t_mid   = None
      t_end   = (k_time_step + 1) * sample_time
      x_mid   = None

      w = dynamics.get_exogenous_input(t_start)
      u_after, pending_computation_after, _ = controller_interface.get_u(k_time_step, t_start, x_start, u_before, w, pending_computation_before)
      
      if pending_computation_before is None:
        assert np.array_equal(u_before, u_after), f'When there is no pending computation, u cannot change from u_before={u_before} to u_after={u_after}.'
        
      assert pending_computation_after is not None, f'pending_computation_after is None'

      # if pending_computation_after.delay:
      #   assert pending_computation_after.delay < 2*sample_time, \
      #     "Missing a computation by multiple time steps is not supported, yet."

      if only_update_control_at_sample_times or pending_computation_after.delay >= sample_time:
        # Evolve state for entire time step and then update u_after
        (t_end, x_end) = dynamics.evolve_state(t_start, x_start, u_after, w, t_end)
      else:
        # Evolve the state halfway and then update u.
        t_mid = pending_computation_after.t_end
        u_mid = pending_computation_after.u
        (t_mid, x_mid) = dynamics.evolve_state(t_start, x_start, u_after, w, t_mid)
        (t_end, x_end) = dynamics.evolve_state(t_mid,   x_mid,     u_mid, w, t_end)
        assert pending_computation_after.delay < sample_time

      # Check that the output of x_end is the expected type and shape.
      assert_is_column_vector(x_end)
      assert x_end.shape == (n, 1), f'x_end={x_end} did not have the expected shape: {(n, 1)}'

      if debug_levels.debug_dynamics_level >= 2:
        print(f'k_time_step={k_time_step}, t_start={t_start}, t_mid={t_mid}, t_end={t_end}, pending_computation_after={pending_computation_after}')
        print(f'u_before={u_before}, u_after={u_after}')

      # Save the data.
      time_step_series.append(t_end=t_end, 
                              x_end=x_end, 
                              u    =u_after, 
                              pending_computation=pending_computation_after, 
                              t_mid=t_mid, 
                              x_mid=x_mid)
      
      assert time_step_series.last_time_step == k_time_step
      if debug_levels.debug_dynamics_level >= 1:
        time_step_series.printTimingData(f"time_step_series after calculating {time_step_series.n_time_steps} of {n_time_steps} control values")

      writeJson(sim_dir + "simulation_data_incremental.json", time_step_series)

      # Update values:
      u_before = u_after
      x_start  = x_end
      pending_computation_before = pending_computation_after
    
    controller_interface.post_simulator_finished()
    time_step_series.finish()
  except (DataNotRecievedViaFileError, BrokenPipeError) as err:
    controller_interface.post_simulator_errored()
    time_step_series.finish(repr(err))
    traceback.print_exc()
    raise err
  finally:
    controller_interface.close()


  print(f'time_step_series at end of run(): {time_step_series}')
  return time_step_series




if __name__ == "__main__":
  raise ValueError("Not intended to be run directly")
