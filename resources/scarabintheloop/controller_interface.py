


from abc import ABC, abstractmethod
from enum import Enum
from scarabintheloop.utils import *
from scarabintheloop.data_types import ComputationData
import numpy as np

class SimulatorStatus(Enum):
  """
  Indicators to alert the C++ code the status of the (Python) simulation.
  """
  PENDING  = 0
  RUNNING  = 1
  FINISHED = 2
  ERRORED  = 3

class DelayProvider(ABC):

  @abstractmethod
  def get_delay(self, metadata):
    """ 
    return t_delay, metadata 
    """
    pass # Abstract method must be overridden.
  
  
class ControllerInterface(ABC):
  """ 
  Create an Abstract Base Class (ABC) for a controller interface. 
  This handles communication to and from the controller, whether that is an executable or (for testing) a mock controller define in Python.
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

  # def post_simulator_pending(self):
  #   self._write_simulator_status(SimulatorStatus.PENDING)

  def post_simulator_running(self):
    self._write_simulator_status(SimulatorStatus.RUNNING)

  def post_simulator_finished(self):
    self._write_simulator_status(SimulatorStatus.FINISHED)

  def post_simulator_errored(self):
    self._write_simulator_status(SimulatorStatus.ERRORED)

  @abstractmethod
  def _write_k(self, k: int):
    pass

  @abstractmethod
  def _write_t(self, t: float):
    pass

  @abstractmethod
  def _write_x(self, x: np.ndarray):
    pass

  @abstractmethod
  def _write_w(self, w: np.ndarray):
    pass

  @abstractmethod
  def _write_t_delay(self, t: float):
    pass
    
  @abstractmethod
  def _write_simulator_status(self, status: SimulatorStatus):
    pass
    
  @abstractmethod
  def _read_u(self) -> np.ndarray:
    pass
    
#   @abstractmethod
#   def _read_x_prediction(self) -> np.ndarray:
#     pass
#     
#   @abstractmethod
#   def _read_t_prediction(self) -> float:
#     pass
# 
#   @abstractmethod
#   def _read_iterations(self) -> int:
#     pass
    
  @abstractmethod
  def _read_metadata(self) -> int:
#     x_prediction = self._read_x_prediction()
#     t_prediction = self._read_t_prediction()
#     iterations = self._read_iterations()
#     if isinstance(x_prediction, np.ndarray):
#       x_prediction = column_vec_to_list(x_prediction)
#     metadata = {
#       "x_prediction": x_prediction,
#       "t_prediction": t_prediction,
#       "iterations": iterations
#     }
# 
#     return metadata
    pass

  def get_u(self, k: int, t: float, x: np.ndarray, u_before: np.ndarray, w: np.ndarray, pending_computation_before: ComputationData):
    """ 
    Get an (possibly) updated value of u. 
    Returns: u, u_delay, u_pending, u_pending_time, metadata,
    where u is the value of u that should start being applied immediately (at t). 
    
    If the updated value requires calling the controller, then u_delay does so.

    This function is tested in <root>/tests/test_scarabintheloop_plant_runner.py/Test_get_u
    """
    
    if debug_levels.debug_dynamics_level >= 1:
      printHeader2('----- get_u (BEFORE) ----- ')
      print(f'u_before[0]: {u_before[0]}')
      printJson("pending_computation_before", pending_computation_before)

    if pending_computation_before is not None and pending_computation_before.t_end > t:
      # If last_computation is provided and the end of the computation is after the current time then we do not update anything. 
      # print(f'Keeping the same pending_computation: {pending_computation_before}')
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = u_before = {u_before}. (Computation pending)')
      u_after = u_before
      pending_computation_after = pending_computation_before
      
      if debug_levels.debug_dynamics_level >= 1:
        printHeader2('----- get_u (AFTER - no update) ----- ')
        print(f'u_after[0]: {u_after[0]}')
        printJson("pending_computation_after", pending_computation_after)
      did_start_computation = False
      return u_after, pending_computation_after, did_start_computation

    if pending_computation_before is not None and pending_computation_before.t_end <= t:
      # If the last computation data is "done", then we set u_after to the pending value of u.
      
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = pending_computation_before.u = {pending_computation_before.u}. (computation finished)')
      u_after = pending_computation_before.u
    elif pending_computation_before is None:
      if debug_levels.debug_dynamics_level >= 2:
        print(f'Set u_after = u_before = {u_before} (no pending computation).')
      u_after = u_before
    else:
      raise ValueError(f'Unexpected case.')
      
    # If there is no pending_computation_before or the given pending_computation_before finishes before the current time t, then we run the computation of the next control value.
    if debug_levels.debug_dynamics_level >= 2:
      print(f"About to get the next control value for x = {x}, w={w}")

    u_pending, u_delay, metadata = self.get_next_control_from_controller(k, t, x, w)
    did_start_computation = True
    pending_computation_after = ComputationData(t, u_delay, u_pending, metadata)

    if debug_levels.debug_dynamics_level >= 1:
      printHeader2('----- get_u (AFTER - With update) ----- ')
      print(f'u_after: {u_after}')
      printJson("pending_computation_after", pending_computation_after)
    # if u_delay == 0:
    #   # If there is no delay, then we update immediately.
    #   return u_pending, None
    assert u_delay > 0, 'Expected a positive value for u_delay.'
      # print(f'Updated pending_computation: {pending_computation_after}')
    return u_after, pending_computation_after, did_start_computation

  def get_next_control_from_controller(self, k:int, t:float, x: np.ndarray, w:np.ndarray) -> Tuple[np.ndarray, float, dict]:
    """
    Send the current state to the controller and wait for the responses. 
    Return values: u, x_prediction, t_prediction, iterations.
    """

    # The order of writing and reading to the pipe files must match the order in the controller.
    self._write_k(k)
    self._write_t(t)
    self._write_x(x)
    self._write_w(w)
    u = self._read_u()
    metadata = self._read_metadata()
    u_delay, delay_metadata = self.computational_delay_provider.get_delay(metadata)
    self._write_t_delay(u_delay)
    metadata.update(delay_metadata)

    if debug_levels.debug_interfile_communication_level >= 1:
      print('Input strings from C++:')
      printIndented(f"       u: {u}", 1)
      printIndented(f"metadata: {metadata}", 1)

    return u, u_delay, metadata
  

  
class PipesControllerInterface(ControllerInterface):

  def __init__(self, computational_delay_provider: DelayProvider, sim_dir: str):
    self.computational_delay_provider = computational_delay_provider
    self.sim_dir = sim_dir
    assertFileExists(self.sim_dir)

    # Readers
    self.u_reader          = PipeVectorReader(os.path.join(self.sim_dir, 'u_c++_to_py'))
    # self.x_predict_reader  = PipeVectorReader(os.path.join(self.sim_dir, 'x_predict_c++_to_py'))
    # self.t_predict_reader  = PipeFloatReader( os.path.join(self.sim_dir, 't_predict_c++_to_py'))
    # self.iterations_reader = PipeFloatReader( os.path.join(self.sim_dir, 'iterations_c++_to_py'))
    self.metadata_reader   = PipeJsonReader(  os.path.join(self.sim_dir, 'metadata_c++_to_py')) 

    # Writers
    self.k_writer         = PipeIntWriter(    os.path.join(self.sim_dir, 'k_py_to_c++'))
    self.t_writer         = PipeFloatWriter(  os.path.join(self.sim_dir, 't_py_to_c++'))
    self.x_writer         = PipeVectorWriter( os.path.join(self.sim_dir, 'x_py_to_c++'))
    self.w_writer         = PipeVectorWriter( os.path.join(self.sim_dir, 'w_py_to_c++'))
    self.t_delay_writer   = PipeFloatWriter(  os.path.join(self.sim_dir, 't_delay_py_to_c++'))
    self.post_simulator_running()

  def open(self):
    """
    Open resources that need to be closed when finished.
    """
    assertFileExists(os.path.join(self.sim_dir, 'u_c++_to_py'))
    self.u_reader.open()
    # self.x_predict_reader.open()
    # self.t_predict_reader.open()
    # self.iterations_reader.open()
    self.metadata_reader.open()
    self.k_writer.open()
    self.t_writer.open()
    self.x_writer.open()
    self.w_writer.open()
    self.t_delay_writer.open()
    
    if debug_levels.debug_interfile_communication_level >= 1:
      print('PipesControllerInterface is open.') 


  def close(self):
    """ 
    Close all of the files we opened.
    """
    # CLose readers.
    self.u_reader.close()
    # self.x_predict_reader.close()
    # self.t_predict_reader.close()
    # self.iterations_reader.close()
    self.metadata_reader.close()
    # Close writers.
    self.k_writer.close()
    self.t_writer.close()
    self.x_writer.close()
    self.w_writer.close()
    self.t_delay_writer.close()

  def _write_k(self, k: int):
    self.k_writer.write(k)# Write to pipe to C++

  def _write_t(self, t: float):
    self.t_writer.write(t)# Write to pipe to C++
    
  def _write_x(self, x: np.ndarray):
    # Pass the string back to C++.
    self.x_writer.write(x)# Write to pipe to C++

  def _write_w(self, w: np.ndarray):
    # Pass the string back to C++.
    self.w_writer.write(w)# Write to pipe to C++

  def _write_simulator_status(self, status: SimulatorStatus):
    status_out_path = os.path.join(self.sim_dir, 'status_py_to_c++')
    with open(status_out_path, "w") as status_file:
      status_file.write(status.name)
    print(f'Wrote status "{status.name}" to {status_out_path}')

  def _write_t_delay(self, t_delay: float):
    self.t_delay_writer.write(t_delay)

  def _read_u(self) -> np.ndarray:
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading u file: {self.u_reader.filename}")
    return self.u_reader.read()
    
  def _read_metadata(self) -> dict:
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f"Reading metadata file: {self.metadata_reader.filename}")
    return self.metadata_reader.read()

  # def _read_x_prediction(self):
  #   if debug_levels.debug_interfile_communication_level >= 2:
  #     print(f"Reading x_predict file: {self.x_predict_reader.filename}")
  #   return self.x_predict_reader.read()
  #   
  # def _read_t_prediction(self):
  #   if debug_levels.debug_interfile_communication_level >= 2:
  #     print(f"Reading t_predict file: {self.t_predict_reader.filename}")
  #   return self.t_predict_reader.read()
  #   
  # def _read_iterations(self):
  #   if debug_levels.debug_interfile_communication_level >= 2:
  #     print(f"Reading iterations file: {self.iterations_reader.filename}")
  #   return self.iterations_reader.read()