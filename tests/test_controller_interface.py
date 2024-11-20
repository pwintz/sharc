#! /bin/env python3

import unittest
import sharc
# from sharc.plant_runner import  EchoerControllerInterface
import copy
from sharc_mocks import MockControllerInterface, MockDelayProvider
import sharc.plant_runner as plant_runner

### Enable more debugging.
# plant_runner.debug_interfile_communication_level = 2

class TestControllerInterface(unittest.TestCase):

  def setUp(self):
    self.computational_delay_interface = MockDelayProvider()
    self.controller_interface = MockControllerInterface(self.computational_delay_interface)

  def test_set_and_read_fake_data(self):
    self.controller_interface.set_next_u(3.4)
    self.assertEqual(self.controller_interface._read_u(), 3.4)

  def test_get_next_control_from_controller(self):
    k = 45
    t = 10.23
    x_given = 304.1
    u_given = 3984
    w_given = 123.02
    x_prediction_given = 19.23
    t_prediction_given = 123.83
    iterations_given = 12
    k_given = 0
    self.computational_delay_interface.put_delay(0.4)
    self.controller_interface.set_next_u(u_given)
    u, u_delay, metadata  = self.controller_interface.get_next_control_from_controller(
                                                        k, t, x_given, w_given)
    self.assertEqual(u, u_given)
    self.assertEqual(u_delay, 0.4)
    # self.assertEqual(metadata["x_prediction"], x_prediction_given)
    # self.assertEqual(metadata["t_prediction"], t_prediction_given)
    # self.assertEqual(metadata["iterations"], iterations_given)


if __name__ == '__main__':
  unittest.main()