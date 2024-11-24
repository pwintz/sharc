#!/bin/env python3

import unittest
import os
import copy
from sharc.scarabizor import ScarabTracesToComputationTimesProcessor, MockTracesToComputationTimesProcessor
from unittest.mock import mock_open, patch

@unittest.skip
class Test_ScarabTracesToComputationTimesProcessor(unittest.TestCase):

  # @unittest.skip
  def test_simulate_a_trace(self):
    # ---- Setup ---- 
    sim_dir = 'test_traces_dir_single'
    trace_processor = ScarabTracesToComputationTimesProcessor(sim_dir)
    trace_index = 0

    # ---- Execution ----
    computation_time = trace_processor.get_computation_time_from_trace(trace_index)

    # ---- Assertions ---- 
    # self.assertEqual(data["index"], trace_index)
    
    # The output trace directory sohuld be given as an absolute path.
    # expected_trace_dir = os.path.abspath(sim_dir + '/dynamorio_trace_0')
    # self.assertEqual(data["trace_dir"], expected_trace_dir)

    # This computation time is just taken from one of the previous executions. 
    # Having a fixed value is usful to identify if there is an unexpected change in the 
    # outputs, but it doesn't actual verify that this number is correct.
    self.assertEqual(computation_time, 0.15558460000000002)

    # ---- Clean up ---- 
    trace_processor._clean_statistics()

  def test_get_all_computation_times(self):
    # ---- Setup ---- 
    sim_dir = 'test_traces_dir_all'
    trace_processor = ScarabTracesToComputationTimesProcessor(sim_dir)

    # ---- Execution ----
    computation_times = trace_processor.get_all_computation_times()
    
    # ---- Assertions ---- 
    # There are two trace directories, so the length of the computation times must be 2.
    self.assertEqual(len(computation_times), 2)

    # Check that we consistently get the same computation times.
    self.assertEqual(computation_times[0], 0.15558460000000002)
    self.assertEqual(computation_times[1], 0.1505806)

    # ---- Clean up ----
    trace_processor._clean_statistics()


# @unittest.skip
class Test_MockTracesToComputationTimesProcessor(unittest.TestCase):

  def test_simulate_a_trace(self):
    # ---- Setup ---- 
    delays = [0.0, 0.1, 0.2, 0.3]
    sim_dir = 'test_traces_dir_mock_8'
    trace_processor = MockTracesToComputationTimesProcessor(sim_dir, delays=delays)
    trace_index = 3

    # ---- Execution ----
    computation_time = trace_processor.get_computation_time_from_trace(trace_index)

    # ---- Assertions ---- 
    # self.assertEqual(data["index"], trace_index)
    
    # The output trace directory sohuld be given as an absolute path.
    # expected_trace_dir = os.path.abspath(sim_dir + '/dynamorio_trace_3')
    # self.assertEqual(data["trace_dir"], expected_trace_dir)

    # We are reading index 3, which has a delay set to "0.3"
    self.assertEqual(computation_time, 0.3)

  def test_get_all_computation_times(self):
    # ---- Setup ---- 
    sim_dir = 'test_traces_dir_mock_8'
    delays = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0]
    trace_processor = MockTracesToComputationTimesProcessor(sim_dir, delays=delays)

    # We assume that the number of delays math
    n_trace_directories = 8 # len(os.listdir(sim_dir))
    self.assertLessEqual(n_trace_directories, len(delays))

    # ---- Execution ----
    computation_times = trace_processor.get_all_computation_times()
    
    # ---- Assertions ---- 
    self.assertEqual(len(computation_times), n_trace_directories)
    self.assertEqual(computation_times[0], 0.0)
    self.assertEqual(computation_times[1], 0.1)
    self.assertEqual(computation_times[2], 0.2)
    self.assertEqual(computation_times[3], 0.3)
    self.assertEqual(computation_times[4], 0.4)
    self.assertEqual(computation_times[5], 0.5)
    self.assertEqual(computation_times[6], 0.6)
    self.assertEqual(computation_times[7], 0.7)


if __name__ == '__main__':
    unittest.main()