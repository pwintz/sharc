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
import datetime
import traceback # Provides pretty printing of exceptions (https://stackoverflow.com/a/1483494/6651650)
from abc import abstractmethod, ABC

import scarabintheloop.scarabizor as scarabizor
from scarabintheloop.scarabizor import ScarabPARAMSReader
from scarabintheloop.utils import *
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


class DataNotRecievedViaFileError(IOError):
  pass

debug_interfile_communication_level = 0
debug_dynamics_level = 0

class PipeReader:
  def __init__(self, filename: str):
    self.filename = filename
    if debug_interfile_communication_level >= 1:
      print(f"About to open reader for {filename}. Waiting for it to available...")
    self.file = open(filename, 'r', buffering=1)

  def close(self):
    if debug_interfile_communication_level >= 1:
      print(f'Closing {self}')
    self.file.close()

  def read(self):
    self._wait_for_pipe_content()
    input_line = self._waitForLineFromFile()
    input_line = checkAndStripInputLoopNumber(input_line)
    return input_line

  def __repr__(self):
    return f'PipeReader(file={self.filename}. Is closed? {self.file.closed})'

  def _waitForLineFromFile(self):
    if debug_interfile_communication_level >= 1:
      print(f"Waiting for input_line from {self.file.name}.")

    input_line = ""
    while not input_line.endswith("\n"):
      #!! Caution, printing out everytime we read a line will cause massive log 
      #!! files that can (will) grind my system to a hault.
      input_line += self.file.readline()

    if debug_interfile_communication_level >= 1:
      print(f'Received input_line from {os.path.basename(self.filename)}: {repr(input_line)}.')
    return input_line

  def _wait_for_pipe_content(self):

    # Wait for the pip
    stat_info = os.stat(self.filename)
  
    # Check if the file size is greater than zero (some data has been written)
    return stat_info.st_size > 0

class PipeFloatReader(PipeReader):
  
  def __init__(self, filename: str):
    super().__init__(filename)

  def read(self):
    return float(super().read())

class PipeVectorReader(PipeReader):

  def __init__(self, filename: str):
    super().__init__(filename)

  def read(self):
    return convertStringToVector(super().read())

class DelayProvider(ABC):

  @abstractmethod
  def get_delay(self, metadata):
    """ 
    return t_delay, metadata 
    """
    pass # Abstract method must be overridden.

class ScarabDelayProvider(DelayProvider):

  def __init__(self, sim_dir):
    self._stats_file_number = 0
    self.stats_reader = scarabizor.ScarabStatsReader(sim_dir)
    self.scarab_params_reader = ScarabPARAMSReader(sim_dir)

  def get_delay(self, metadata):
    if debug_interfile_communication_level >= 2:
      print('Waiting for statistics from Scarab.')

    self.stats_reader.waitForStatsFile(self._stats_file_number)
    t_delay = self.stats_reader.readTime(self._stats_file_number)
    instruction_count = self.stats_reader.readInstructionCount(self._stats_file_number)
    cycles_count = self.stats_reader.readCyclesCount(self._stats_file_number)
    self._stats_file_number += 1

    delay_metadata = {}
    params_out = self.scarab_params_reader.params_out_to_dictionary()
    delay_metadata.update(params_out)
    delay_metadata["instruction_count"] = instruction_count
    delay_metadata["cycles_count"] = cycles_count
    return t_delay, delay_metadata

class OneTimeStepDelayProvider(DelayProvider):

  def __init__(self, sample_time, sim_dir, use_fake_scarab):
    self.trace_dir_index = 0
    self.sample_time     = sample_time
    self.sim_dir         = sim_dir
    self.use_fake_scarab = use_fake_scarab
    
  def get_delay(self, metadata):
    t_delay = self.sample_time
    metadata = {}

    if self.use_fake_scarab:
      fake_trace_dir = os.path.join(self.sim_dir + f'dynamorio_trace_{self.trace_dir_index}')
      os.makedirs(fake_trace_dir)
      self.trace_dir_index += 1
      with open(os.path.join(fake_trace_dir, 'README.txt'), 'w') as file:
        file.write(f'This is a fake trace directory created by {type(self)} for the purpose of faking Scarab data for faster testing.')

    return t_delay, metadata

class NoneDelayProvider(DelayProvider):

  def __init__(self):
    pass

  def get_delay(self, metadata):
    t_delay = None
    metadata = {}
    return t_delay, metadata

class GaussianDelayProvider(DelayProvider):

  def __init__(self, mean, std_dev):
    self.mean = mean
    self.std_dev = std_dev

  def get_delay(self, metadata):
    # Generate a random number from the Gaussian distribution
    t_delay = np.random.normal(self.mean, self.std_dev)
    metadata = {}
    return t_delay, metadata

class LinearBasedOnIteraionsDelayProvider(DelayProvider):
  
  def __init__(self):
    pass

  def get_delay(self, metadata):
    # TODO: Move the computation delays generated by a model out of the plant_runner module.
    iterations = metadata["iterations"]
    delay_model_slope       = config_data["computation_delay_model"]["computation_delay_slope"]
    delay_model_y_intercept = config_data["computation_delay_model"]["computation_delay_y-intercept"]
    if delay_model_slope:
      if not delay_model_y_intercept:
        raise ValueError(f"delay_model_slope was set but delay_model_y_intercept was not.")
      t_delay = delay_model_slope * iterations + delay_model_y_intercept
      print(f"t_delay = {t_delay:.8g} = {delay_model_slope:.8g} * {iterations:.8g} + {delay_model_y_intercept:.8g}")
    else:
      print('Using constant delay times.')
      t_delay = config_data["computation_delay_model"]["fake_computation_delay_times"]
    return t_delay, metadata


class ControllerInterface(ABC):
  """ 
  Create an Abstract Base Class (ABC) for a controller interface. 
  This handles communication to and from the controller, whether that is an executable or (for testing) a mock controller.
  """

  def __init__(self, computational_delay_provider: DelayProvider):
    self.computational_delay_provider = computational_delay_provider

  def open(self):
    """
    Open any resources that need to be closed.
    """
    pass

  def close(self):
    """
    Do cleanup of opened resources.
    """
    pass

  @abstractmethod
  def _write_x(self, x: np.ndarray):
    pass

  @abstractmethod
  def _write_t_delay(self, t: float):
    pass
    
  @abstractmethod
  def _read_u(self) -> np.ndarray:
    return u
    
  # TODO: Instead of having separate data streams for things like x_prediction and iterations, which 
  # TODO: will depend on the given system, write all of the system-specific data to a JSON file.
  # def _read_metadata(self, k) -> dict:
  #   pass

  @abstractmethod
  def _read_x_prediction(self) -> np.ndarray:
    return x_prediction
    
  @abstractmethod
  def _read_t_prediction(self) -> float:
    return t_prediction

  @abstractmethod
  def _read_iterations(self) -> int:
    return iterations
    
  def _read_metadata(self) -> int:
    x_prediction = self._read_x_prediction()
    t_prediction = self._read_t_prediction()
    iterations = self._read_iterations()
    if isinstance(x_prediction, np.ndarray):
      x_prediction = column_vec_to_list(x_prediction)
    metadata = {
      "x_prediction": x_prediction,
      "t_prediction": t_prediction,
      "iterations": iterations
    }

    return metadata
    
  def get_next_control_from_controller(self, x: np.ndarray):
    """
    Send the current state to the controller and wait for the responses. 
    Return values: u, x_prediction, t_prediction, iterations.
    """

    # The order of writing and reading to the pipe files must match the order in the controller.
    self._write_x(x)
    u = self._read_u()
    metadata = self._read_metadata()
    u_delay, delay_metadata = self.computational_delay_provider.get_delay(metadata)
    self._write_t_delay(u_delay)
    metadata.update(delay_metadata)

    if debug_interfile_communication_level >= 1:
      print('Input strings from C++:')
      printIndented(f"       u: {u}", 1)
      printIndented(f"metadata: {metadata}", 1)

    return u, u_delay, metadata
    
class PipesControllerInterface(ControllerInterface):

  def __init__(self, computational_delay_provider: DelayProvider, sim_dir):
    self.computational_delay_provider = computational_delay_provider
    self.sim_dir = sim_dir

  def open(self):
    """
    Open resources that need to be closed when finished.
    """
    self.u_reader         = PipeVectorReader(self.sim_dir + '/u_c++_to_py') 
    self.x_predict_reader = PipeVectorReader(self.sim_dir + '/x_predict_c++_to_py') 
    self.t_predict_reader  = PipeFloatReader(self.sim_dir + '/t_predict_c++_to_py') 
    self.iterations_reader = PipeFloatReader(self.sim_dir + '/iterations_c++_to_py') 
    self.x_outfile         = open(self.sim_dir + 'x_py_to_c++', 'w', buffering=1)
    self.t_delay_outfile   = open(self.sim_dir + 't_delay_py_to_c++', 'w', buffering=1)
    
    if debug_interfile_communication_level >= 1:
      print('Pipes are open') 

    return self  # Return the instance so that it's accessible as 'as' target

  def close(self):
    """ 
    Close all of the files we opened.
    """
    # Code to release the resource (e.g., close a file, database connection)
    self.u_reader.close()
    self.x_predict_reader.close()
    self.t_predict_reader.close()
    self.iterations_reader.close()
    self.x_outfile.close()
    self.t_delay_outfile.close()

  def _write_x(self, x: np.ndarray):
    # Pass the string back to C++.
    x_out_string = nump_vec_to_csv_string(x)
    if debug_interfile_communication_level >= 2:
      print(f"Writing x output line: {x_out_string} to {self.x_outfile.name}")
    self.x_outfile.write(x_out_string + "\n")# Write to pipe to C++

  def _write_t_delay(self, t_delay: float):
    t_delay_str = f"{t_delay:.8g}"
    
    if debug_interfile_communication_level >= 2:
      print(f"Writing {t_delay_str} to t_delay file: {self.t_delay_outfile.name}")
    self.t_delay_outfile.write(t_delay_str + "\n")

  def _read_u(self):
    if debug_interfile_communication_level >= 2:
      print(f"Reading u file: {self.u_reader.filename}")
    return self.u_reader.read()
    
  def _read_x_prediction(self):
    if debug_interfile_communication_level >= 2:
      print(f"Reading x_predict file: {self.x_predict_reader.filename}")
    return self.x_predict_reader.read()
    
  def _read_t_prediction(self):
    if debug_interfile_communication_level >= 2:
      print(f"Reading t_predict file: {self.t_predict_reader.filename}")
    return self.t_predict_reader.read()
    
  def _read_iterations(self):
    if debug_interfile_communication_level >= 2:
      print(f"Reading iterations file: {self.iterations_reader.filename}")
    return self.iterations_reader.read()
      




class ComputationData:
  def __init__(self, t_start: float, delay: float, u: float, metadata: dict = None):
      # Using setters to assign initial values, enforcing validation
      self.t_start = t_start
      self.delay = delay
      self.u = u
      self.metadata = metadata

  @property
  def t_start(self) -> float:
      return self._t_start

  @t_start.setter
  def t_start(self, value: float):
      if not isinstance(value, (float, int)):
          raise ValueError("t_start must be a float or int.")
      self._t_start = float(value)

  @property
  def delay(self) -> float:
      return self._delay

  @delay.setter
  def delay(self, value: float):
      if not isinstance(value, (float, int)):
          raise ValueError(f"delay must be a float or int. instead it was {type(value)}")
      self._delay = float(value)
      self._t_end = self._t_start + self._delay  # Update t_end whenever delay changes

  @property
  def u(self) -> Union[float, np.ndarray]:
      return self._u

  @u.setter
  def u(self, value: float):
      if not isinstance(value, (float, int, np.ndarray, list)):
          raise ValueError(f"u must be a float, int, or numpy array. instead it was {type(value)}")
      self._u = value

  @property
  def metadata(self) -> dict:
      return self._metadata

  @metadata.setter
  def metadata(self, value: dict):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a dictionary. instead it was {type(value)}")
    self._metadata = value

  @property
  def t_end(self) -> float:
      return self.t_start + self.delay

  def __str__(self):
    if self.metadata:
      # return f'ComputationData(t_start={self.t_start}, delay={self.delay}, t_end={self.t_end}, u<{type(self.u)}>={self.u}, metadata={self.metadata})'
      return f'ComputationData(t_start={self.t_start}, delay={self.delay})'
    else:
      # return f'ComputationData(t_start={self.t_start}, delay={self.delay}, t_end={self.t_end}, u<{type(self.u)}>={self.u})'
      return f'ComputationData(t_start={self.t_start}, delay={self.delay})'


  def __repr__(self):
    # if self.metadata:
      # return f'ComputationData(t_start={self.t_start}, delay={self.delay}, t_end={self.t_end}, u<{type(self.u)}>={self.u}, metadata={self.metadata})'
      # return f'ComputationData(t_start={self.t_start}, delay={self.delay})'
    # else:
      # return f'ComputationData(t_start={self.t_start}, delay={self.delay}, t_end={self.t_end}, u<{type(self.u)}>={self.u})'
    return f'ComputationData([{self.t_start:.3f}, {self.t_end:.3f}])'
  

  def copy(self):
    return ComputationData(self.t_start, self.delay, copy.deepcopy(self.u), copy.deepcopy(self.metadata))

  def __eq__(self, other):
    """ Check the equality of the *data* EXCLUDING metadata. """
    if not isinstance(other, ComputationData):
      return False
    return self.t_start == other.t_start \
       and self.delay == other.delay     \
       and self.t_end == other.t_end     \
       and self.u == other.u #            \
      #  and self.metadata == other.metadata 

  def next_batch_initialization_summary(self) -> str:
    if isinstance(self.u, np.ndarray):
      return f't_end={self.t_end}, u[0]={self.u[0][0]}'
    else:
      return f't_end={self.t_end}, u[0]={self.u[0]}'


class TimeStepSeries:

  def __init__(self, k0, t0, x0, pending_computation_prior=None):
    assert k0 is not None, f"k0={k0} must not be None"
    assert t0 is not None, f"t0={t0} must not be None"
    assert x0 is not None, f"x0={x0} must not be None"

    k0 = int(k0)
    t0 = float(t0)
    assert isinstance(k0, int), f'k_0={k_0} had an unexpected type: {type(k_0)}. Must be an int.'
    assert isinstance(t0, (float, int)), f't0={t0} had an unexpected type: {type(t0)}.'
    assert isinstance(x0, (list, np.ndarray)), f'x0={x0} had an unexpected type: {type(x0)}.'
    
    # Initial values
    self.k0 = k0 # Initial time step number
    self.i0 = k0 # Intial sample index equals the initial time step number.
    self.t0 = t0
    self.x0 = TimeStepSeries.cast_vector(x0)

    # Unlike k0, t0, and x0, "pending_computation_prior" is not stored in the data arrays. 
    # Instead, it simple is used to record what the 'incoming' computation was, if any.
    self.pending_computation_prior = pending_computation_prior
    
    ### Time-index aligned values ###
    self.x = []
    self.u = []
    self.t = []
    self.k = [] # Time step number
    self.i = [] # Sample index number
    self.metadata = {}

    # Note: self.pending_computation[j] is the computation that was either started at self.t[j] 
    #       or previously started and is still continuing at self.t[j].
    self.pending_computation = []
    
    # Some Metadata for the simulation
    self._walltime_start: float = time.time()
    self.walltime = None
    self.cause_of_termination: str = "In Progress"
    self.datetime_of_run = datetime.datetime.now().isoformat()


  def finish(self, cause_of_termination=None):
    self.walltime = time.time() - self._walltime_start
    if not cause_of_termination:
      self.cause_of_termination = "Finished"
    else:
      self.cause_of_termination = repr(cause_of_termination)

  def get_pending_computation_for_time_step(self, k):
    ndx = self.k.index(k)
    return self.pending_computation[ndx]

  def get_pending_computation_started_at_sample_time_index(self, i):
    ndx = self.i.index(i)
    if ndx > 0:
      # If not the very first entry, then we need to increment the ndx by one because the value of 'i' increments (in the array) on entry before pending_computation "increments" (gets a new value).
      ndx += 1
    # print(f'ndx from i={i}: {ndx}. (self.i: {self.i})')
    # if self.pending_computation[ndx] is None:
      # return self.pending_computation[ndx + 1]
    if self.i[ndx] != i:
      raise ValueError("self.i[ndx] != i")
    
    return self.pending_computation[ndx]


  def __repr__(self):

      # Create lists of fixed-width columns.
      i_str = [f'{i_val:13d}' for i_val in self.i]
      k_str = [f'{k_val:13d}' for k_val in self.k]
      t_str = [f'{t_val:13.2f}' for t_val in self.t]
      pending_computations_str = [f'[{pc.t_start:5.2f},{pc.t_end:5.2f}]' if pc else f'         None' for pc in self.pending_computation]

      # Truncate 
      if self.n_time_steps() > 8:
        n_kept_on_ends = 4
        n_omitted = self.n_time_steps()
        omitted_text = f'[{n_omitted} of {n_omitted + 2*n_kept_on_ends} entries hidden]'
        k_str = k_str[:n_kept_on_ends] + [omitted_text] + k_str[-n_kept_on_ends:]
        i_str = i_str[:n_kept_on_ends] + [omitted_text] + i_str[-n_kept_on_ends:]
        t_str = t_str[:n_kept_on_ends] + [omitted_text] + t_str[-n_kept_on_ends:]
        pending_computations_str = pending_computations_str[:n_kept_on_ends] + [omitted_text] + pending_computations_str[-n_kept_on_ends:]

      k_str = ", ".join(k_str)
      i_str = ", ".join(i_str)
      t_str = ", ".join(t_str)
      pending_computations_str = ", ".join(pending_computations_str)

      return (f"{self.__class__.__name__}(k0={self.k0}, t0={self.t0}, x0={self.x0}, pending_computation_prior={self.pending_computation_prior},\n"
              f"\t     k (time steps)={k_str}\n "
              f"\t     i (time indxs)={i_str}\n "
              f"\t     t             ={t_str}\n "
              # f" x              = {self.x},\n "
              # f" u              = {self.u},\n "
              f"\tpending_computation={pending_computations_str})")

  def __eq__(self, other):
    """ Check the equality of the time series EXCLUDING metadata. """
    return self.k0 == other.k0 \
       and self.t0 == other.t0 \
       and self.x0 == other.x0 \
       and self.x  == other.x  \
       and self.u  == other.u  \
       and self.t  == other.t  \
       and self.k  == other.k  \
       and self.i  == other.i  \
       and self.pending_computation_prior == other.pending_computation_prior \
       and self.pending_computation       == other.pending_computation

  def copy(self):
    copied = TimeStepSeries(self.k0, self.t0, self.x0, self.pending_computation_prior)

    # Copy values.
    copied.x                   = self.x.copy()
    copied.u                   = self.u.copy()
    copied.t                   = self.t.copy()
    copied.k                   = self.k.copy()
    copied.i                   = self.i.copy()
    copied.pending_computation = self.pending_computation.copy()
    return copied

  def append(self, t_end, x_end, u, pending_computation: ComputationData, t_mid=None, x_mid=None):

    assert isinstance(u, (np.ndarray, list)), f'u={u} had an unexpected type: {type(u)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
    assert isinstance(x_end, (np.ndarray, list)), f'x_end={x_end} had an unexpected type: {type(x_end)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
    assert isinstance(t_end, (float, int)), f't_end={t_end} had an unexpected type: {type(t_end)}.'
    assert (t_mid is None) == (x_mid is None),  f'It must be that t_mid={t_mid} is None if and only if x_mid={x_mid} is None.'
    if t_mid:
      assert isinstance(t_mid, (float, int)), f't_mid={t_mid} had an unexpected type: {type(t_mid)}.'
      assert isinstance(x_mid, (np.ndarray, list)), f'x_mid={x_mid} had an unexpected type: {type(x_mid)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
      assert len(x_end) == len(x_mid)

    # is_x_end_None = x_end is None
    x_end     = TimeStepSeries.cast_vector(x_end)
    u         = TimeStepSeries.cast_vector(u)
    x_mid     = TimeStepSeries.cast_vector(x_mid)

    if x_end is None:
      raise ValueError(f'x_end is None')

    if pending_computation and pending_computation.t_end > t_end:
      # raise ValueError(f"pending_computation.t_end = {pending_computation.t_end:.3} > t_end = {t_end:.3}")
      warnings.warn("pending_computation.t_end = {pending_computation.t_end:.3} > t_end = {t_end:.3}", Warning)

    if len(self.t) == 0:
      k       = self.k0
      t_start = self.t0
      x_start = self.x0
      pending_computation_prev = self.pending_computation_prior
    else:
      k       = self.k[-1] + 1
      t_start = self.t[-1]
      x_start = self.x[-1]
      pending_computation_prev = self.pending_computation[-1]

    # If there is a pending computation, then it must have started before the time "t_start".
    if pending_computation and pending_computation.t_start > t_start:
      # warnings.warn(f'pending_computation.t_start = {pending_computation.t_start} != t_start = {t_start}')
      raise ValueError(f'TimeStepSeries.append(): pending_computation.t_start = {pending_computation.t_start} != t_start = {t_start}')
  
    # If there is a pending computation, and the computation start time is before the start time for this interval, 
    # then the new pending computation must equal the previous pending computation.
    if pending_computation and pending_computation.t_start < t_start:
      assert pending_computation_prev == pending_computation
    

    assert t_start < t_end,                f't_start={t_start} >= t_end={t_end}. self:{self}'
    if t_mid: 
      assert t_start <= t_mid and t_mid <= t_end, f'We must have t_start={t_start} <= t_mid={t_mid} <= t_end={t_end}'

    if t_mid is None:
      assert len(x_start) == len(x_end)
      #                           [ Start of time step,  End of time step  ]
      self.k                   += [                  k,                   k] # Step number
      self.i                   += [                  k,               k + 1] # Sample index
      self.t                   += [            t_start,               t_end]
      self.x                   += [            x_start,               x_end]
      self.u                   += [                  u,                   u]
      self.pending_computation += [pending_computation, pending_computation]

    else: # A mid point was given, indicating an update to u part of the way through.
      assert len(x_start) == len(x_mid)
      assert len(x_start) == len(x_end)
      u_mid = TimeStepSeries.cast_vector(pending_computation.u)
      assert len(u) == len(u_mid)
      self.t += [t_start, t_mid, t_mid, t_end]
      self.x += [x_start, x_mid, x_mid, x_end]
      self.u += [       u,    u, u_mid, u_mid]
      self.k += [       k,    k,     k,     k]
      self.i += [       k,    k,     k, k + 1] # Sample index
      # The value of u_pending was applied, starting at t_mid, so it is no longer pending 
      # at the end of the sample. Thus, we set it to None.
      self.pending_computation += [pending_computation, pending_computation, None, None]

  def append_multiple(self, 
                      t_end: List, 
                      x_end: List, 
                      u: List, 
                      pending_computation: List, 
                      t_mid=None, 
                      x_mid=None):
    if len(t_end) != len(x_end):
      raise ValueError("len(t_end) != len(x_end)")

    if len(t_end) != len(u):
      raise ValueError("len(t_end) != len(u)")
    
    if len(t_end) != len(pending_computation):
      raise ValueError("len(t_end) != len(pending_computation)")

    for i in range(len(t_end)):
      if t_mid is None:
        i_t_mid = None
      else:
        i_t_mid = t_mid[i]

      if x_mid is None:
        i_x_mid = None
      else:
        i_x_mid = x_mid[i]

      self.append(t_end[i], x_end[i], u[i], pending_computation=pending_computation[i], t_mid=i_t_mid, x_mid=i_x_mid)

  def overwrite_computation_times(self, computation_times):
    """
    This function overwrites the computation times recorded in the time series.
    """
    
    # assert not any(delay is None for delay in computation_times), f"computation_times={computation_times} must not contain any None values."

    if len(computation_times) != self.n_time_indices():
      raise ValueError(f"len(computation_times) = {len(computation_times)} != self.n_time_indices = {self.n_time_indices()}.\nself: {self}.\ncomputation_times: {computation_times}")

    if len(self.pending_computation) != 2*len(computation_times):
      raise ValueError(f'len(self.pending_computation)={len(self.pending_computation)} != 2*len(computation_times) = {2*len(computation_times)}')
      
      
    # print(f'Overwriting computation times. computation_times={computation_times}, self.pending_computation={self.pending_computation}.')
    i_pending_computation = 0
    i_computation_time = 0
    while i_pending_computation < len(self.pending_computation):
      pc = self.pending_computation[i_pending_computation]
      delay = computation_times[i_computation_time]
      if delay is not None:
        new_pending_computation = ComputationData(t_start=pc.t_start, delay=delay, u=pc.u, metadata=pc.metadata)
      else:
        new_pending_computation = None
      
      self.pending_computation[i_pending_computation]     = new_pending_computation
      self.pending_computation[i_pending_computation + 1] = new_pending_computation
      i_pending_computation += 2
      i_computation_time    += 1

  # Override the + operator
  def __add__(self, other):
    if not isinstance(other, TimeStepSeries):
      raise ValueError(f'Cannot concatenate this TimeStepSeries with a {type(other)} object.')
    
    # if self.is_empty():
    #   raise ValueError('Cannot concatenate onto an empty TimeStepSeries')
    if other.is_empty():
      return self

    if not self.is_empty():
      if self.k[-1] + 1 != other.k0:
        raise ValueError(f"Initial k does not match: self.k[-1] + 1 = {self.k[-1] + 1} != other.k0 = {other.k0}")
      if self.t[-1] != other.t0:
        raise ValueError("Initial time does not match: self.t[-1] != other.t0")
      if self.x[-1] != other.x0:
        raise ValueError("Initial state does not match: self.x[-1] != other.x0")

    concatenated = self.copy()
    concatenated.x                   += other.x
    concatenated.u                   += other.u
    concatenated.t                   += other.t
    concatenated.k                   += other.k # Time step index
    concatenated.i                   += other.i # Sample time index
    concatenated.pending_computation += other.pending_computation

    return concatenated

  def truncate(self, last_k):
    """
    Truncate this series to end at the last index in time step "last_k".
    """
    assert not self.is_empty(), "Cannot truncate an empty time series."
    assert last_k in self.k, f'last_k={last_k} must be in self.k={self.k}'
    # assert last_k >= self.k[0], "last_k must not be less than the first time step."

    n_time_steps = last_k - self.k0

    if self.n_time_steps() <= n_time_steps:
      return self.copy()
    
    truncated = self.copy()

    # Time-index aligned values
    last_index_of_last_k = max([row for row, k in enumerate(self.k) if k == last_k])
    ndxs = slice(0, last_index_of_last_k+1)
    truncated.x                    = self.x[ndxs]
    truncated.u                    = self.u[ndxs]
    truncated.t                    = self.t[ndxs]
    truncated.k                    = self.k[ndxs]
    truncated.i                    = self.i[ndxs]
    truncated.pending_computation  = self.pending_computation[ndxs]

    if truncated.n_time_steps() > 0:
      assert truncated.k[-2] == truncated.k[-1]
      assert truncated.t[-2] < truncated.t[-1]

    return truncated 

  def is_empty(self):
    return len(self.t) == 0

  def n_time_indices(self):
    if self.is_empty():
      return 0
    return self.i[-1] - self.i[0]

  def n_time_steps(self):
    if self.is_empty():
      return 0
    return self.k[-1] - self.k[0]

  def get_first_sample_time_index(self):
    if len(self.i) == 0:
      return self.i0
    else:
      return self.i[0]
    
  def get_last_sample_time_index(self):
    if len(self.i) == 0:
      return self.i0
    else:
      return self.i[-1]
    
  def get_first_time_step_index(self):
    if len(self.k) == 0:
      return self.k0
    else:
      return self.k[0]
    
  def get_last_time_step_index(self):
    if len(self.k) == 0:
      return self.k0
    else:
      return self.k[-1]

  def find_first_late_timestep(self, sample_time):
    for (i_step, pc) in enumerate(self.pending_computation):
      if pc is not None and pc.delay > sample_time:
        has_missed_computation_time = True
        first_late_timestep = self.k[i_step]
        print(f'first_late_timestep: {first_late_timestep}')
        return first_late_timestep, has_missed_computation_time 
    
    has_missed_computation_time = False
    return None, has_missed_computation_time

  def printTimingData(self, label):
    timing_data = {
      "t0": self.t0,
      "k0": self.k0,
      "t":  self.t,
      "k":  self.k,
      "i":  self.i
    }
    printJson(label + ' (timing data)', timing_data)
    
    print(f'{label}')
    TimeStepSeries.print_sample_time_values("k_time_step", self.k)
    TimeStepSeries.print_sample_time_values("i_sample_ndx", self.k)
    TimeStepSeries.print_sample_time_values("t(k)", self.t)

  def print(self, label):
    print(f'{label}:')
    if self.pending_computation_prior:
      print("pending_computation_prior", self.pending_computation_prior.next_batch_initialization_summary())
    else:
      print("No prior pending computation.")
    TimeStepSeries.print_sample_time_values("k", self.k)
    # TimeStepSeries.print_time_step_values("time step", self.step_k)
    TimeStepSeries.print_sample_time_values("t(k)", self.t)

    for i in range(len(self.x[0])):
      TimeStepSeries.print_sample_time_values(f"x_{i}(k)", [x[i] for x in self.x])
    TimeStepSeries.print_sample_time_values("u([k, k+1))", [u[0] for u in self.u])
    TimeStepSeries.print_sample_time_values("comp delay", [(pc.delay if pc else pc) for pc in self.pending_computation])
    TimeStepSeries.print_sample_time_values("comp u", [(pc.u[0][0] if pc else pc) for pc in self.pending_computation])
    TimeStepSeries.print_sample_time_values("comp t_end", [(pc.t_end if pc else pc) for pc in self.pending_computation])

  @staticmethod
  def cast_vector(x):
    if x is None:
      return None
    if isinstance(x, (float, int)):
      return [x]
    if isinstance(x, list):
      return x
    elif isinstance(x, np.ndarray):
      return column_vec_to_list(x)
    raise ValueError(f'cast_vector(): Unexpected type: {type(x)}')
    
  @staticmethod
  def print_sample_time_values(label: str, values: list):
    values = [float('nan') if v is None else v for v in values]
    print(f"{label:>18}: " +  " --- ".join(f"{v:^7.3g}" for v in values))

  @staticmethod
  def print_time_step_values(label: str, values: list):
    try:
      str_list = []
      for v in values:
        if v is None:
          str_list += ["  None "] 
        elif isinstance(v, (float, int)) :
          str_list += [f"{v:^7.3g}"]
        else:
          raise TypeError(f'Unxpected type: {type(v)}')
          
      print(f"{label:>18}:    |  " + "  |           |  ".join(str_list) + "  |")
    except Exception as err:
      print(f'Failed to print values in {values}. Error: {err}')
      raise err


#     # Read optimizer info.
#     # # TODO: Move the processing of the optimizer info out of "run_plant.py"
#     # with open(sim_dir + 'optimizer_info.csv', newline='') as csvfile:
#     #   csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     #   
#     #   num_iterationss_list = []
#     #   costs_list = []
#     #   primal_residuals_list = []
#     #   dual_residuals_list = []
#     #   is_header = True
#     #   for row in csv_reader:
#     #     if is_header:
#     #       num_iterations_ndx = row.index('num_iterations')
#     #       cost_ndx = row.index('cost')
#     #       primal_residual_ndx = row.index('primal_residual')
#     #       dual_residual_ndx = row.index('dual_residual')
#     #       is_header = False
#     #     else:
#     #       num_iterationss_list.append(int(row[num_iterations_ndx]))
#     #       costs_list.append(float(row[cost_ndx]))
#     #       primal_residuals_list.append(float(row[primal_residual_ndx]))
#     #       dual_residuals_list.append(float(row[dual_residual_ndx]))
#     #     
#     # simulation_dictionary["num_iterations"]  = num_iterationss_list
#     # simulation_dictionary["cost"]            = costs_list
#     # simulation_dictionary["primal_residual"] = primal_residuals_list
#     # simulation_dictionary["dual_residual"]   = dual_residuals_list

def get_u(t, x, u_before, pending_computation_before: ComputationData, controller_interface):
  """ 
  Get an (possibly) updated value of u. 
  Returns: u, u_delay, u_pending, u_pending_time, metadata,
  where u is the value of u that should start being applied immediately (at t). 
  
  If the updated value requires calling the controller, then u_delay does so.

  This function is tested in <root>/tests/test_scarabintheloop_plant_runner.py/Test_get_u
  """
  
  if debug_dynamics_level >= 1:
    printHeader2('----- get_u (BEFORE) ----- ')
    print(f'u_before[0]: {u_before[0]}')
    printJson("pending_computation_before", pending_computation_before)

  if pending_computation_before is not None and pending_computation_before.t_end > t:
    # If last_computation is provided and the end of the computation is after the current time then we do not update anything. 
    # print(f'Keeping the same pending_computation: {pending_computation_before}')
    if debug_dynamics_level >= 2:
      print(f'Set u_after = u_before = {u_before}. (Computation pending)')
    u_after = u_before
    pending_computation_after = pending_computation_before
    
    if debug_dynamics_level >= 1:
      printHeader2('----- get_u (AFTER - no update) ----- ')
      print(f'u_after[0]: {u_after[0]}')
      printJson("pending_computation_after", pending_computation_after)
    did_start_computation = False
    return u_after, pending_computation_after, did_start_computation

  if pending_computation_before is not None and pending_computation_before.t_end <= t:
    # If the last computation data is "done", then we set u_after to the pending value of u.
    
    if debug_dynamics_level >= 2:
      print(f'Set u_after = pending_computation_before.u = {pending_computation_before.u}. (computation finished)')
    u_after = pending_computation_before.u
  elif pending_computation_before is None:
    if debug_dynamics_level >= 2:
      print(f'Set u_after = u_before = {u_before} (no pending computation).')
    u_after = u_before
  else:
    raise ValueError(f'Unexpected case.')
    

  # If there is not pen         ding_computation_before or the given pending_computation_before finishes before the current time t, then we run the computation of the next control value.
  if debug_dynamics_level >= 2:
    print("About to get the next control value for x = ", repr(x))

  u_pending, u_delay, metadata = controller_interface.get_next_control_from_controller(x)
  did_start_computation = True
  pending_computation_after = ComputationData(t, u_delay, u_pending, metadata)


  printHeader2('----- get_u (AFTER - With update) ----- ')
  print(f'u_after[0]: {u_after[0]}')
  printJson("pending_computation_after", pending_computation_after)
  # if u_delay == 0:
  #   # If there is no delay, then we update immediately.
  #   return u_pending, None
  assert u_delay > 0, 'Expected a positive value for u_delay.'
    # print(f'Updated pending_computation: {pending_computation_after}')
  return u_after, pending_computation_after, did_start_computation
  

def computation_delay_provider_factory(computation_delay_name: str, sim_dir, sample_time, use_fake_scarab):
  if isinstance(computation_delay_name, str):
    computation_delay_name = computation_delay_name.lower()
  if computation_delay_name == "none":
    return NoneDelayProvider()
  elif computation_delay_name == "gaussian":
    computational_delay_provider = GaussianDelayProvider(mean=0.24, std_dev=0.05)
  elif computation_delay_name == "execution-driven scarab" or computation_delay_name == "fake execution-driven scarab":
    return ScarabDelayProvider(sim_dir)
  elif computation_delay_name == "onestep":
    return OneTimeStepDelayProvider(sample_time, sim_dir, use_fake_scarab)
  elif isinstance(computation_delay_name, DelayProvider):
    return computation_delay_name
  else:
    raise ValueError(f'Unexpected computation_delay_name: {computation_delay_name}.')
    
@contextmanager
def controller_interface_factory(controller_interface_selection, computation_delay_provider, sim_dir):
  """ 
  Generate the desired ControllerInterface object based on the value of "controller_interface_selection". Typically the value will be the string "pipes", but a ControllerInterface interface object can also be passed in directly to allow for testing.
  
  Example usage:

    with controller_interface_factory("pipes", computation_delay_provider, sim_dir) as controller_interface: 
        <do stuff with controller_interface>
    
  When the "with" block is left, controller_interface.close() is called to clean up resources.
  """

  if isinstance(controller_interface_selection, str):
    controller_interface_selection = controller_interface_selection.lower()

  if controller_interface_selection == "pipes":
    controller_interface = PipesControllerInterface(computation_delay_provider, sim_dir)
  elif isinstance(controller_interface_selection, ControllerInterface):
    controller_interface = controller_interface_selection
  else:
    raise ValueError(f'Unexpected controller_interface: {controller_interface}')

  try:
    yield controller_interface.open()
  finally:
    controller_interface.close()

def run(sim_dir: str, config_data: dict, evolveState) -> dict:
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

  computation_delay_provider = computation_delay_provider_factory(config_data["Simulation Options"]["in-the-loop_delay_provider"], sim_dir, sample_time, use_fake_scarab)

  # Read the initial control value.
  x0 = list_to_column_vec(config_data['x0'])
  u0 = list_to_column_vec(config_data['u0'])
  
  assert x0.shape == (n, 1), f'x0={x0} did not have the expected shape: {(n, 1)}'
  assert u0.shape == (m, 1), f'u0={u0} did not have the expected shape: {(m, 1)}'

  # pending_computation may be None.
  pending_computation0 = config_data["pending_computation"]
  print(f"                  u0: {u0}")
  printJson(f"pending_computation0", pending_computation0)

  with controller_interface_factory("pipes", computation_delay_provider, sim_dir) as controller_interface:

    # computational_delay_provider = ScarabDelayProvider(sim_dir)
    # computational_delay_provider = OneTimeStepDelayProvider(sample_time)
    # computational_delay_provider = GaussianDelayProvider(mean=0.24, std_dev=0.05)

    try:
      x_start = x0
      u_before = u0
      u_before_str = repr(u_before)
      pending_computation_before = pending_computation0
      time_index = first_time_index
      
      assert x_start.shape == (n, 1), f'x_start={x_start} did not have the expected shape: {(n, 1)}'
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

        u_after, pending_computation_after, did_start_computation = get_u(t_start, x_start, u_before, pending_computation_before, controller_interface)
        
        if pending_computation_before is None:
          assert u_before == u_after, f'When there is no pending computation, u cannot from u_before={u_before} to u_after={u_after}.'
          
        assert pending_computation_after is not None, f'pending_computation_after is None'

        if pending_computation_after.delay and pending_computation_after.delay > 2*sample_time:
          raise ValueError("Missing computation by multiple time steps is not supported, yet.")

        if pending_computation_after.delay < sample_time:
          # Evolve the state halfway and then update u.
          t_mid = pending_computation_after.t_end
          u_mid = pending_computation_after.u
          (t_mid, x_mid) = evolveState(t_start, x_start, u_after, t_mid)
          (t_end, x_end) = evolveState(t_mid,   x_mid,     u_mid, t_end)
        else: #u_delay and u_delay == sample_time:
          # Evolve state for entire time step and then update u_after
          (t_end, x_end) = evolveState(t_start, x_start, u_after, t_end)

        # Check that the output of x_end is the expected type and shape.
        assert_is_column_vector(x_end)
        assert x_end.shape == (n, 1), f'x_end={x_end} did not have the expected shape: {(n, 1)}'

        if debug_dynamics_level >= 2:
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
          
        if debug_dynamics_level >= 1:
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

def convertStringToVector(vector_str: str):
  vector_str_list = vector_str.split(',') #.strip().split("\t")

  # Convert the list of strings to a list of floats.
  chars_to_strip = ' []\n'
  v = np.array([[np.float64(x.strip(chars_to_strip)),] for x in vector_str_list])
  
  if debug_interfile_communication_level >= 3:
    print('convertStringToVector():')
    printIndented('vector_str:', 1)
    printIndented(repr(vector_str), 1)
    printIndented('v:', 1)
    printIndented(repr(v), 1)

  return v


def checkAndStripInputLoopNumber(input_line):
  """ Check that an input line, formatted as "Loop <k>: <data>" 
  has the expected value of k (given by the argument "expected_k") """

  split_input_line = input_line.split(':')
  loop_input_str = split_input_line[0]

  # Check that the input line is in the expected format. 
  if not loop_input_str.startswith('Loop '):
    raise ValueError(f'The input_line "{input_line}" did not start with a loop label.')
  
  # Extract the loop number and convert it to an integer.
  input_loop_number = int(loop_input_str[len('Loop:'):])

  # if input_loop_number != expected_k:
  #   # If the first piece of the input line doesn't give the correct loop number.
  #   raise ValueError(f'The input_loop_number="{input_loop_number}" does not match expected_k={expected_k}.')
  #   
  # Return the part of the input line that doesn't contain the loop info.
  return split_input_line[1]

if __name__ == "__main__":
  raise ValueError("Not intended to be run directly")
