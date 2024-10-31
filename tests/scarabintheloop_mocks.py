from scarabintheloop import ControllerInterface, DelayProvider
import queue


class MockDelayProvider(DelayProvider):

  def __init__(self):
    self.delay_queue = queue.Queue()

  def put_delay(self, delay):
    self.delay_queue.put(delay)

  def get_delay(self, metadata):
    if self.delay_queue.qsize() == 0:
      raise ValueError(f'Delay queue is empty!')
      
    t_delay = self.delay_queue.get()
    metadata = {}
    return t_delay, metadata


class MockControllerInterface(ControllerInterface):
  
  def __init__(self, computational_delay_provider):
    super().__init__(computational_delay_provider)
    self.x_last_write = None
    self.u = 0
    self.t_delay_last_write = None
    self.x_prediction = 0
    self.t_prediction = 0
    self.iterations = 0

  def get_last_x_written(self):
    return self.x_last_write

  def get_last_t_delay_written(self):
    return self.t_delay_last_write
    
  def set_next_u(self, u):
    self.u = u
    
  def set_next_x_prediction(self, x_prediction):
    self.x_prediction = x_prediction
    
  def set_next_t_prediction(self, t_prediction):
    self.t_prediction = t_prediction
    
  def set_next_iterations(self, iterations):
    self.iterations = iterations

  # Override abstract superclass function 
  def _write_x(self, x):
    self.x_last_write = x
    
  # Override abstract superclass function 
  def _write_t_delay(self, t_delay: float):
    self.t_delay_last_write = t_delay
    
  # Override abstract superclass function 
  def _read_u(self):
    return self.u
    
  # Override abstract superclass function 
  def _read_x_prediction(self):
    return self.x_prediction
    
  # Override abstract superclass function 
  def _read_t_prediction(self):
    return self.t_prediction
    
  # Override abstract superclass function 
  def _read_iterations(self):
    return self.iterations