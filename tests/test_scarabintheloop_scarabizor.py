#!/bin/env python3

import unittest
import copy
from scarabintheloop.scarabizor import ScarabPARAMSReader

class TestScarabPARAMSReader(unittest.TestCase):
  def test_load(self):
    scarab_params_reader = ScarabPARAMSReader('.')
    params_in = scarab_params_reader.params_in_to_dictionary()
    self.assertTrue(params_in is not None)
    
  def test_read_params_file(self):
    scarab_params_reader = ScarabPARAMSReader()
    params_lines = scarab_params_reader.read_params_file('PARAMS.out')
    self.assertTrue(len(params_lines) > 100)

  def test_params_lines_to_dict(self):
    params_lines = ["## Simulation Parameters",
      "--mode                          full",
      "--model                         cmp",
      "# Length of each chip cycle in femtosecond (10^-15 sec).",
      "--chip_cycle_time               100000000"]
    scarab_params_reader = ScarabPARAMSReader()
    params_dict = scarab_params_reader.params_lines_to_dict(params_lines)
    self.assertEqual(params_dict["chip_cycle_time"], 100000000)

if __name__ == '__main__':
    unittest.main()