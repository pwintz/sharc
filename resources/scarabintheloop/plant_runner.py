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

from scarabintheloop.utils import *
import scarabintheloop.debug_levels as debug_levels
from scarabintheloop.data_types import *
import warnings

import json
import csv
import re
import os
import select # Used to check if a pipe is ready.

from typing import List, Set, Dict, Tuple, Union

try:
  # Ensure that assertions are enabled.
  assert False
  raise Exception('Python assertions are not working. This tool relies on Python assertions to fail quickly when problems arise. Possible causes for assertions being disabled are running with the "-O" flag or running a precompiled (".pyo" or ".pyc") module.')
except AssertionError:
    pass


def run(sim_dir: str, config_data: dict, evolveState, controller_interface) -> dict:
  if not sim_dir.endswith("/"):
    sim_dir += "/"
  print(f"Start of plant_runner.run()  in {sim_dir}")

  # use_fake_scarab_computation_times = config_data["Simulation Options"]["use_fake_scarab_computation_times"]
  max_time_steps = config_data["max_time_steps"]
  sample_time = config_data["system_parameters"]["sample_time"] # Discrete sample period.
  n = config_data["system_parameters"]["state_dimension"] 
  m = config_data["system_parameters"]["input_dimension"] 
  first_time_index = config_data['first_time_index']
  use_fake_scarab = config_data['Simulation Options']["use_fake_delays"]
  only_update_control_at_sample_times = config_data["only_update_control_at_sample_times"]
  

  # Read the initial control value.
  x0 = list_to_column_vec(config_data['x0'])
  u0 = list_to_column_vec(config_data['u0'])
  
  assert x0.shape == (n, 1), f'x0={x0} must have the shape: {(n, 1)}'
  assert u0.shape == (m, 1), f'u0={u0} must have the shape: {(m, 1)}'

  # pending_computation may be None.
  pending_computation0 = config_data["pending_computation"]
  # print(f"                  u0: {u0}")
  # if pending_computation0:
  #   printJson(f"pending_computation0", pending_computation0)


  try:
    x_start = x0
    u_before = u0
    u_before_str = repr(u_before)
    pending_computation_before = pending_computation0
    time_index = first_time_index
    
    assert x_start.shape  == (n, 1), f'x_start={x_start} did not have the expected shape: {(n, 1)}'
    assert u_before.shape == (m, 1), f'u_before={u_before} did not have the expected shape: {(m, 1)}'

    # The number of time steps that the controller has been computed. 
    # When there is a pending 'u_before' value, there may be several time steps that don't compute new control values.
    n_new_u_values_computed = 0

    time_step_series = TimeStepSeries(k0=first_time_index, t0=time_index * sample_time, x0=x0, pending_computation_prior=pending_computation0)
    while n_new_u_values_computed < max_time_steps:
      t_start = time_index * sample_time
      t_mid = None
      t_end = (time_index + 1) * sample_time
      x_mid = None

      u_after, pending_computation_after, did_start_computation = controller_interface.get_u(t_start, x_start, u_before, pending_computation_before)
      
      if pending_computation_before is None:
        assert u_before == u_after, f'When there is no pending computation, u cannot change from u_before={u_before} to u_after={u_after}.'
        
      assert pending_computation_after is not None, f'pending_computation_after is None'

      if pending_computation_after.delay and pending_computation_after.delay > 2*sample_time:
        raise ValueError("Missing computation by multiple time steps is not supported, yet.")

      if only_update_control_at_sample_times or pending_computation_after.delay >= sample_time:
        # Evolve state for entire time step and then update u_after
        (t_end, x_end) = evolveState(t_start, x_start, u_after, t_end)
      else:
        # Evolve the state halfway and then update u.
        t_mid = pending_computation_after.t_end
        u_mid = pending_computation_after.u
        (t_mid, x_mid) = evolveState(t_start, x_start, u_after, t_mid)
        (t_end, x_end) = evolveState(t_mid,   x_mid,     u_mid, t_end)
        assert pending_computation_after.delay < sample_time

      # Check that the output of x_end is the expected type and shape.
      assert_is_column_vector(x_end)
      assert x_end.shape == (n, 1), f'x_end={x_end} did not have the expected shape: {(n, 1)}'

      if debug_levels.debug_dynamics_level >= 2:
        print(f'time_index={time_index}, t_start={t_start}, t_mid={t_mid}, t_end={t_end}, pending_computation_after={pending_computation_after}')
        print(f'u_before={u_before}, u_after={u_after}')

      # Save the data.
      time_step_series.append(t_end=t_end, 
                              x_end=x_end, 
                              u    =u_after, 
                              pending_computation=pending_computation_after, 
                              t_mid=t_mid, 
                              x_mid=x_mid)

      if did_start_computation:
        n_new_u_values_computed += 1

      time_index += 1
      # assert time_index < 10000:, 'time_index > 10000'
        
      if debug_levels.debug_dynamics_level >= 1:
        print(f'time_step_series: {time_step_series}')
      writeJson(sim_dir + "simulation_data_incremental.json", time_step_series)
      print('\n=====\n')

      # Update values:
      u_before = u_after
      x_start  = x_end
      pending_computation_before = pending_computation_after
    
    time_step_series.finish()
  except (DataNotRecievedViaFileError, BrokenPipeError) as err:
    time_step_series.finish(repr(err))
    traceback.print_exc()
    raise err

  print(f'time_step_series at end of run(): {time_step_series}')
  return time_step_series




if __name__ == "__main__":
  raise ValueError("Not intended to be run directly")
