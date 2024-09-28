#! /bin/env python3

import unittest
import scarabintheloop
from scarabintheloop.plant_runner import FakeComputationalDelayInterface, EchoerControllerInterface
import copy

class TestControllerInterface(unittest.TestCase):

  def setUp(self):
    self.computational_delay_interface = FakeComputationalDelayInterface()
    self.controller_interface = EchoerControllerInterface(self.computational_delay_interface)

  def test_set_and_read_fake_data(self):
    self.controller_interface.set_next_u(3.4)
    self.assertEqual(self.controller_interface._read_u(0), 3.4)

  def test_get_next_control_from_controller(self):
    x_given = 304.1
    u_given = 3984
    x_prediction_given = 19.23
    t_prediction_given = 123.83
    iterations_given = 12
    k_given = 0
    self.controller_interface.set_next_u(u_given)
    u, x_prediction, t_prediction, iterations, t_delay, instruction_count, cycles_count = self.controller_interface.get_next_control_from_controller(k_given, x_given)
    self.assertEqual(u, u_given)
    self.assertEqual(x_prediction, x_prediction_given)
    self.assertEqual(t_prediction, t_prediction_given)
    self.assertEqual(iterations, iterations_given)


if __name__ == '__main__':
  unittest.main()