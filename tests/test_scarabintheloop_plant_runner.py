#! /bin/env python3

import unittest
import scarabintheloop
import copy
from scarabintheloop.plant_runner import ComputationData
from scarabintheloop_mocks import MockDelayProvider, MockControllerInterface
from scarabintheloop.utils import printJson


class Test_get_u(unittest.TestCase):

  def setUp(self):
    self.computational_delay_interface = MockDelayProvider()
    self.controller_interface = MockControllerInterface(self.computational_delay_interface)
  
  def test_without_pending(self):
    t0 = 0.111
    u0 = [-12.3]
    delay = 1.1
    u_pending  = [2.2222]
    self.computational_delay_interface.put_delay(delay)
    self.controller_interface.set_next_u(u_pending)
    expected_pending_computation = ComputationData(t_start=t0, delay=delay, u=u_pending, metadata={})

    u_new, pending_computation_new, did_start_computation = self.controller_interface.get_u(
                                    t        = t0,
                                    x        = 2.3,
                                    u_before = u0,
                                    pending_computation_before = None)
      
    self.assertTrue(did_start_computation)
    self.assertEqual(u_new,  u0)
    self.assertEqual(pending_computation_new, expected_pending_computation)


#   def test_with_non_pending(self):
#     fake_delay = 0.2
#     self.computational_delay_interface.put_delay(fake_delay)
#     self.controller_interface.set_next_u(3.4)
# 
#     u_current, u_delay, u_pending, u_pending_time, metadata = plant_runner.get_u(
#                   t = 0.2,
#                   x = 2.3,
#                   u_current = 1.1,
#                   u_pending = None,
#                   u_pending_time = None,
#                   controller_interface = self.controller_interface
#                 )
#     self.assertEqual(u_current,  1.1)
#     self.assertEqual(u_pending,  3.4)
#     self.assertEqual(u_delay,    fake_delay)
#     self.assertEqual(u_pending_time, 0.2+fake_delay)
#     self.assertEqual(self.controller_interface.get_last_t_delay_written(), fake_delay)

    
  def test_with_pending_before_current_time(self):

    expected_u_new = [1.1]
    expected_delay = 0.8
    expected_u_pending = [2.2]
    self.computational_delay_interface.put_delay(expected_delay)
    self.controller_interface.set_next_u(expected_u_pending)
    pending_computation_prev = ComputationData(t_start=0, delay=0.9, u=expected_u_new)
    u_new, pending_computation_new, did_start_computation = self.controller_interface.get_u(
                                    t = 1.0,
                                    x = 2.3,
                                    u_before = [0.0],
                                    pending_computation_before = pending_computation_prev)
    self.assertTrue(did_start_computation)
    self.assertEqual(u_new, expected_u_new)
    self.assertEqual(pending_computation_new.delay, expected_delay)
    self.assertEqual(pending_computation_new.u, expected_u_pending)
    self.assertEqual(pending_computation_new.t_end, 1.0 + expected_delay)# t + delay

  def test_with_pending_after_current_time(self):
    u0 = [-12.3]
    delay = 1.1
    u_pending_time = 1.1 # Greaterthan t = 1.0
    u_pending      = 2.2222
    expected_u_new = u0
    expected_delay = 0.8
    expected_u_pending = [2.2]
    self.computational_delay_interface.put_delay(expected_delay)
    self.controller_interface.set_next_u(expected_u_pending)

    pending_computation_prev = ComputationData(t_start=0, delay=10000.9, u=u_pending)
    expected_pending_computation = pending_computation_prev.copy()

    u_new, pending_computation_new, did_start_computation = self.controller_interface.get_u(
                                    t = 1.0,
                                    x = 2.3,
                                    u_before = u0,
                                    pending_computation_before = pending_computation_prev)
      
    self.assertFalse(did_start_computation)
    self.assertEqual(u_new,  u0)
    self.assertEqual(pending_computation_new, expected_pending_computation)


if __name__ == '__main__':
  unittest.main()