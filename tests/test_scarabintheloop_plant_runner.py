#! /bin/env python3

import unittest
import scarabintheloop
from scarabintheloop.plant_runner import DelayProvider, ControllerInterface
import scarabintheloop.plant_runner as plant_runner
import copy
import queue

class MockDelayProvider(DelayProvider):

  def __init__(self):
    self.delay_queue = queue.Queue()

  def put_delay(self, delay):
    self.delay_queue.put(delay)

  def get_delay(self, metadata):
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


class Test_get_u(unittest.TestCase):

  def setUp(self):
    self.computational_delay_interface = MockDelayProvider()
    self.controller_interface = MockControllerInterface(self.computational_delay_interface)

  def test_with_non_pending(self):
    fake_delay = 0.2
    self.computational_delay_interface.put_delay(fake_delay)
    self.controller_interface.set_next_u(3.4)

    u_current, u_delay, u_pending, u_pending_time, metadata = plant_runner.get_u(
                  t = 0.2,
                  x = 2.3,
                  u_current = 1.1,
                  u_pending = None,
                  u_pending_time = None,
                  controller_interface = self.controller_interface
                )
    self.assertEqual(u_current,  1.1)
    self.assertEqual(u_pending,  3.4)
    self.assertEqual(u_delay,    fake_delay)
    self.assertEqual(u_pending_time, 0.2+fake_delay)
    self.assertEqual(self.controller_interface.get_last_t_delay_written(), fake_delay)

    
  def test_with_pending_before_current_time(self):
    self.computational_delay_interface.put_delay(0.2)
    self.controller_interface.set_next_u(3.4)

    u_current, u_delay, u_pending, u_pending_time, metadata = plant_runner.get_u(
      t = 1.0,
      u_pending_time = 0.8, # Less than t
      x = 2.3,
      u_current = 1.1,
      u_pending = 3.865,
      controller_interface = self.controller_interface)
    self.assertEqual(u_current,  3.865)
    self.assertEqual(u_delay,    None)
    self.assertEqual(u_pending,  None)
    self.assertEqual(u_pending_time,    None)

  def test_with_pending_after_current_time(self):
    u_current, u_delay, u_pending, u_pending_time, metadata = plant_runner.get_u(
      t = 1.0,
      u_pending_time = 1.8, # Greater than t
      x = 2.3,
      u_current = 1.1,
      u_pending = 3.865,
      controller_interface = self.controller_interface)
    self.assertEqual(u_current,  1.1)
    self.assertEqual(u_delay,    None)
    self.assertEqual(u_pending,  3.865)
    self.assertEqual(u_pending_time, 1.8)




if __name__ == '__main__':
  unittest.main()