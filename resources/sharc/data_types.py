from typing import List, Set, Dict, Tuple, Union
import numpy as np
import time
import datetime
import warnings

from sharc.utils import *


class DataNotRecievedViaFileError(IOError):
  pass

class ComputationData:

  def __init__(self, t_start: float, delay: float, u: float, metadata: dict = None):
      # Using setters to assign initial values, enforcing validation.
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

  @property
  def u(self) -> np.ndarray:
      return self._u

  @u.setter
  def u(self, value: float):
      if not isinstance(value, (float, int, np.ndarray, list)):
          raise ValueError(f"u must be a float, int, or numpy array. type(value)={type(value)}")
      self._u = list_to_column_vec(value)

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
  
  def to_dict(self):
    return {
      "t_start": self.t_start,
      "delay": self.delay,
      "u": self.u,
      "metadata": self.metadata
    }

  def copy(self):
    return ComputationData(self.t_start, self.delay, copy.deepcopy(self.u), copy.deepcopy(self.metadata))

  def __eq__(self, other):
    """ Check the equality of the *data* EXCLUDING metadata. """
    if not isinstance(other, ComputationData):
      return False
    return self.t_start == other.t_start \
       and self.delay == other.delay     \
       and self.t_end == other.t_end     \
       and np.array_equal(self.u, other.u) #            \
      #  and self.metadata == other.metadata 

  def next_batch_initialization_summary(self) -> str:
    if isinstance(self.u, np.ndarray):
      return f't_end={self.t_end}, u[0]={self.u[0][0]}'
    else:
      return f't_end={self.t_end}, u[0]={self.u[0]}'

class TimeStepSeries:
  k0: int
  i0: int
  t0: float
  x0: List[float]
  x: List[List[float]]
  u: List[List[float]]
  w: List[List[float]]
  t: List[float] # time  
  k: List[int]   # k_time_step 
  i: List[int]   # i_time_index
  metadata: dict
  pending_computation_prior: Union[ComputationData, None]
  pending_computation: List[Union[ComputationData, None]]
  cause_of_termination: str

  @staticmethod
  def _from_lists(k0: int, 
                  t_list: List[float], 
                  x_list: List[np.ndarray], 
                  u_list: List[np.ndarray], 
                  w_list: List[np.ndarray], 
                  delay_list: List[Union[float, None]], # "None" indicates to use the previous computation.
                  pending_computation_prior: Union[ComputationData, None] = None): 
    # Make copies of the lists so that modificaitons don't effect the 
    # lists where this function is called.
    t_list     = t_list.copy()    
    x_list     = x_list.copy()
    u_list     = u_list.copy()
    w_list     = w_list.copy()
    delay_list = delay_list.copy()

    assert isinstance(k0, int)
    assert isinstance(t_list, list)
    assert isinstance(x_list, list)
    assert isinstance(u_list, list)
    assert isinstance(delay_list, list)
    assert len(t_list) > 1
    assert len(t_list) == len(x_list), f'len(t_list)={len(t_list)} must equal = len(x_list) = {len(x_list)}'
    assert len(t_list) == len(u_list) + 1 , f'len(t_list)={len(t_list)} must equal = len(u_list) + 1 = {len(u_list) + 1}'
    
    pc_list = []
    for i, delay in enumerate(delay_list):
      if delay is not None: 
        pc_list.append(ComputationData(t_list[i], delay, 0.0))
      elif len(pc_list) > 0:
        pc_list.append(pc_list[-1])
      else:
        assert pending_computation_prior is not None
        pc_list.append(pending_computation_prior)

    t0 = t_list.pop(0) # Pop first element.
    x0 = x_list.pop(0) # Pop first element.
    time_series = TimeStepSeries(k0, t0, x0, pending_computation_prior)
    time_series.append_multiple(t_list, x_list, u_list, w_list, pc_list)
    return time_series

  def __init__(self, k0, t0, x0, pending_computation_prior: Union[ComputationData, None]=None):
    assert k0 is not None, f"k0={k0} must not be None"
    assert t0 is not None, f"t0={t0} must not be None"
    assert x0 is not None, f"x0={x0} must not be None"

    k0 = int(k0)
    t0 = float(t0)
    assert isinstance(k0, int), f'k_0={k0} had an unexpected type: {type(k0)}. Must be an int.'
    assert isinstance(t0, (float, int)), f't0={t0} had an unexpected type: {type(t0)}.'
    assert isinstance(x0, (list, np.ndarray)), f'x0={x0} had an unexpected type: {type(x0)}.'
    
    # Initial values
    self.k0 = k0 # Initial time step number
    self.i0 = k0 # Intial sample index equals the initial time step number.
    self.t0 = t0
    self.x0 = TimeStepSeries.cast_vector(x0)

    # Unlike k0, t0, and x0, "pending_computation_prior" is not stored in the data arrays. 
    # Instead, it is used to record what the 'incoming' computation was, if any.
    self.pending_computation_prior = pending_computation_prior
    
    ### Time-index aligned values ###
    self.x = []
    self.u = []
    self.w = []
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

  def finish(self, cause_of_termination: Exception = None):
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
      i_str = [f'{i_val:13d}'   for i_val in self.i]
      k_str = [f'{k_val:13d}'   for k_val in self.k]
      t_str = [f'{t_val:13.2f}' for t_val in self.t]
      pending_computations_str = [f'[{pc.t_start:5.2f},{pc.t_end:5.2f}]' if pc else f'         None' for pc in self.pending_computation]

      # Truncate the string.
      if self.n_time_steps > 8:
        n_kept_on_ends = 4
        n_omitted = self.n_time_steps
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
    if not isinstance(other, TimeStepSeries):
      raise ValueError(f'Cannot compare this TimeStepSeries to an object of type {type}')
      
    return self.k0 == other.k0 \
       and self.t0 == other.t0 \
       and self.x0 == other.x0 \
       and self.x  == other.x  \
       and self.u  == other.u  \
       and self.w  == other.w  \
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
    copied.w                   = self.w.copy()
    copied.t                   = self.t.copy()
    copied.k                   = self.k.copy()
    copied.i                   = self.i.copy()
    copied.pending_computation = self.pending_computation.copy()
    return copied

  def append(self, t_end, x_end, u, w, pending_computation: ComputationData, t_mid=None, x_mid=None, w_mid=None):

    assert isinstance(u, (np.ndarray, list)),     f'u={u} had an unexpected type: {type(u)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
    assert isinstance(w, (np.ndarray, list)),     f'w={w} had an unexpected type: {type(w)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
    assert isinstance(x_end, (np.ndarray, list)), f'x_end={x_end} had an unexpected type: {type(x_end)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
    assert isinstance(t_end, (float, int)),       f't_end={t_end} had an unexpected type: {type(t_end)}.'
    assert isinstance(pending_computation, ComputationData), f'pending_computation={pending_computation} had an unexpected type: {type(pending_computation)}. Must be ComputationData.'
    assert (t_mid is None) == (x_mid is None),  f'It must be that t_mid={t_mid} is None if and only if x_mid={x_mid} is None.'
    if t_mid:
      assert isinstance(t_mid, (float, int)), f't_mid={t_mid} had an unexpected type: {type(t_mid)}.'
      assert isinstance(x_mid, (np.ndarray, list)), f'x_mid={x_mid} had an unexpected type: {type(x_mid)}. Must be a list or numpy array. For scalar values, make it a list of length 1.'
      assert len(x_end) == len(x_mid)

    # is_x_end_None = x_end is None
    x_end     = TimeStepSeries.cast_vector(x_end)
    u         = TimeStepSeries.cast_vector(u)
    w         = TimeStepSeries.cast_vector(w)
    x_mid     = TimeStepSeries.cast_vector(x_mid)

    if x_end is None:
      raise ValueError(f'x_end is None')

    # if pending_computation and pending_computation.t_end > t_end:
    #   # raise ValueError(f"pending_computation.t_end = {pending_computation.t_end:.3f} > t_end = {t_end:.3f}")
    #   warnings.warn(f"pending_computation.t_end = {pending_computation.t_end:.3f} > t_end = {t_end:.3f}", Warning)

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
      raise ValueError(f'TimeStepSeries.append(): pending_computation.t_start = {pending_computation.t_start} must be before t_start = {t_start}')
  
    # If there is a pending computation, and the computation start time is before the start time for this interval, 
    # then the new pending computation must equal the previous pending computation.
    if pending_computation and pending_computation.t_start < t_start:
      assert pending_computation_prev == pending_computation

    if pending_computation_prev and pending_computation and (pending_computation_prev is not pending_computation):
      if pending_computation_prev.t_end > pending_computation.t_start:
        raise ValueError(f'The value t_end={pending_computation_prev.t_end} of the previous computation is after the t_start={pending_computation.t_start} of the pending computation that is being appended. Time step is k={k}.')
        

    
    assert t_start < t_end, f't_start={t_start} must be greater than or equal to t_end={t_end}. self:{self}'
    if t_mid: 
      assert t_start <= t_mid and t_mid <= t_end, f'We must have t_start={t_start} <= t_mid={t_mid} <= t_end={t_end}'

    if t_mid is None:
      assert len(x_start) == len(x_end), f'len(x_start)={len(x_start)} must equal len(x_end)={len(x_end)}'
      #                           [ Start of time step,  End of time step  ]
      self.k                   += [                  k,                   k] # Step number
      self.i                   += [                  k,               k + 1] # Sample index
      self.t                   += [            t_start,               t_end]
      self.x                   += [            x_start,               x_end]
      self.u                   += [                  u,                   u]
      self.w                   += [                  w,                   w]
      self.pending_computation += [pending_computation, pending_computation]

    else: # A mid point was given, indicating an update to u part of the way through.
      assert len(x_start) == len(x_mid)
      assert len(x_start) == len(x_end)
      u_mid = TimeStepSeries.cast_vector(pending_computation.u)
      assert len(u) == len(u_mid)
      assert len(w) == len(w_mid)
      self.t += [t_start, t_mid, t_mid, t_end]
      self.x += [x_start, x_mid, x_mid, x_end]
      self.u += [       u,    u, u_mid, u_mid]
      self.w += [       w,    w, w_mid, w_mid]
      self.k += [       k,    k,     k,     k]
      self.i += [       k,    k,     k, k + 1] # Sample index
      # The value of u_pending was applied, starting at t_mid, so it is no longer pending 
      # at the end of the sample. Thus, we set it to None.
      self.pending_computation += [pending_computation, pending_computation, None, None]
    assert self.pending_computation[0] is not None

  def append_multiple(self, 
                      t_end: List, 
                      x_end: List, 
                      u: List, 
                      w: List, 
                      pending_computation: List, 
                      t_mid=None, 
                      x_mid=None, 
                      w_mid=None):
    assert len(t_end) == len(x_end), \
      f"len(t_end)={len(t_end)} must equal len(x_end)={len(x_end)}"

    assert len(t_end) == len(u), \
      f"len(t_end)={len(t_end)} must equal len(u)={len(u)}"
    
    assert len(t_end) == len(w), \
      f"len(t_end)={len(t_end)} must equal len(w)={len(w)}"
    
    assert len(t_end) == len(pending_computation), \
      f"len(t_end)={len(t_end)} must equal len(pending_computation)={len(pending_computation)}"

    for i in range(len(t_end)):
      assert (t_mid is None) == (x_mid is None), f"t_mid={t_mid} is None iff x_mid={x_mid} is None." 
      assert (t_mid is None) == (w_mid is None), f"t_mid={t_mid} is None iff w_mid={w_mid} is None." 
      
      if t_mid is None:
        i_t_mid = None
        i_x_mid = None
        i_w_mid = None
      else:
        i_t_mid = t_mid[i]
        i_x_mid = x_mid[i]
        i_w_mid = w_mid[i]

      self.append(t_end = t_end[i], x_end = x_end[i], u = u[i],     w = w[i], pending_computation=pending_computation[i], 
                  t_mid = i_t_mid, x_mid = i_x_mid,            w_mid = i_w_mid)

  def to_dict(self):
    return {
      "x":self.x,
      "u":self.u,
      "w":self.w,
      "t":self.t,
      "k":self.k,
      "i":self.i,
      "pending_computation":self.pending_computation
    }

  def overwrite_computation_times(self, computation_for_k_dict: dict) -> None:
    """
    This function overwrites the computation times recorded in the time series.
    """
    assert isinstance(computation_for_k_dict, dict)
    for (i, k) in enumerate(self.k):
      if k in computation_for_k_dict:
        current_pc = self.pending_computation[i]
        delay = computation_for_k_dict[k]
        assert delay is not None
        assert delay >= 0
        self.pending_computation[i] = ComputationData(t_start=current_pc.t_start, 
                                                      delay=delay, # <- Update!
                                                      u=current_pc.u, 
                                                      metadata=current_pc.metadata)

  # Override the + operator to define concatenation of TimeStepSeries.
  def __add__(self, other):
    if not isinstance(other, TimeStepSeries):
      raise ValueError(f'Cannot concatenate this TimeStepSeries with a {type(other)} object.')
    
    if other.is_empty:
      return self

    if not self.is_empty:
      if self.k[-1] + 1 != other.k0:
        raise ValueError(f"Initial k does not match: self.k[-1] + 1 = {self.k[-1] + 1} != other.k0 = {other.k0}")
      if self.t[-1] != other.t0:
        raise ValueError("Initial time does not match: self.t[-1] != other.t0")
      if self.x[-1] != other.x0:
        raise ValueError("Initial state does not match: self.x[-1] != other.x0")

    concatenated = self.copy()
    concatenated.x                   += other.x
    concatenated.u                   += other.u
    concatenated.w                   += other.w
    concatenated.t                   += other.t
    concatenated.k                   += other.k # Time step index
    concatenated.i                   += other.i # Sample time index
    concatenated.pending_computation += other.pending_computation

    return concatenated

  def truncate(self, last_k):
    """
    Truncate this series to end at the last index in time step "last_k".
    """
    assert not self.is_empty, "Cannot truncate an empty time series."
    assert last_k in self.k, f'last_k={last_k} must be in self.k={self.k}'
    # assert last_k >= self.k[0], "last_k must not be less than the first time step."

    n_time_steps = last_k - self.k0

    if self.n_time_steps <= n_time_steps:
      return self.copy()
    
    truncated = self.copy()

    # Time-index aligned values
    last_index_of_last_k = max([row for row, k in enumerate(self.k) if k == last_k])
    ndxs = slice(0, last_index_of_last_k+1)
    truncated.x                    = self.x[ndxs]
    truncated.u                    = self.u[ndxs]
    truncated.w                    = self.w[ndxs]
    truncated.t                    = self.t[ndxs]
    truncated.k                    = self.k[ndxs]
    truncated.i                    = self.i[ndxs]
    truncated.pending_computation  = self.pending_computation[ndxs]

    if truncated.n_time_steps > 0:
      assert truncated.k[-2] == truncated.k[-1]
      assert truncated.t[-2]  < truncated.t[-1]

    return truncated 

  @property
  def is_empty(self):
    return len(self.t) == 0

  @property
  def n_time_indices(self):
    return self.n_time_steps + 1

  @property
  def n_time_steps(self):
    if self.is_empty:
      return 0
    return self.k[-1] - self.k[0] + 1

  @property
  def first_time_index(self):
    if len(self.i) == 0:
      return self.i0
    else:
      return self.i[0]
    
  @property
  def last_time_index(self):
    if len(self.i) == 0:
      return self.i0
    else:
      return self.i[-1]
    
  @property
  def first_time_step(self):
    if len(self.k) == 0:
      return self.k0
    else:
      return self.k[0]
    
  @property
  def last_time_step(self):
    if len(self.k) == 0:
      return self.k0
    else:
      return self.k[-1]

  # def find_first_late_timestep(self, sample_time):
  #   for (i_step, t) in enumerate(self.t):
  #     if i_step == 0 or self.pending_computation[i_step-1] is None:
  #       continue # Skip first step.
  #     if self.pending_computation[i_step-1].t_end > t:
  #       has_missed_computation_time = True
  #       first_late_timestep = self.k[i_step]
  #       return first_late_timestep, has_missed_computation_time 
  #   
  #   has_missed_computation_time = False
  #   return None, has_missed_computation_time
  
#   def t_at(self, i=None, k_start=None, k_end=None):
#     if i:
#       assert k_start is None and k_end is None
#       row = self.i.index(i)
#     elif k_start:
#       assert k_start is None and i is None
#     elif k_end:
#       assert k_start is None and i is None
# 
#     return self.t[row] 
  
  def find_first_missed_computation_started_during_series(self):
    """
    Search through the time series to find 
    """
    # Check all of the pending computations, except the one in the last row (which 
    # should equal the pending computation in the penultimate row).
    for (row, pc) in enumerate(self.pending_computation[:-1]):
      # If there is no computation going at the current row, then it cannot be late, so skip.
      if pc is None:
        continue
      # If the current computation equals self.pending_computation_prior, then it was not 
      # started during this time series, so we skip
      if pc == self.pending_computation_prior:
        continue

      # If the end of this pending computation is after the next time
      if pc.t_end > self.t[row+1]:
        first_late_timestep = self.k[row]
        return first_late_timestep 
    
    # has_missed_computation_time = False
    return None
  
    # for (i_step, pc) in enumerate(self.pending_computation):
    #   if pc is not None and pc.delay > sample_time:
    #     has_missed_computation_time = True
    #     first_late_timestep = self.k[i_step]
    #     return first_late_timestep, has_missed_computation_time 
    # 
    # has_missed_computation_time = False
    # return None, has_missed_computation_time

  def printTimingData(self, label):
    print(f'{label} (timing data):')
    TimeStepSeries.print_sample_time_values("k_time_step", self.k)
    TimeStepSeries.print_sample_time_values("i_sample_ndx", self.i)
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
    TimeStepSeries.print_sample_time_values("w([k, k+1))", [w[0] for w in self.w])
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
