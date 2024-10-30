#! /bin/env python3

import unittest
import scarabintheloop
import copy
import os

class TestConsistency(unittest.TestCase):

  def test_working_dir_unchanged_by_scarabintheloop(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'fake_delays.json'
    working_dir_before = os.getcwd()

    # ---- Execution ----
    experiment_list = scarabintheloop.run(example_dir, config_file)

    # ---- Assertions ---- 
    self.assertEqual(os.getcwd(), working_dir_before)

  def test_fake_delays(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_serial_consistency_with_fake_delays.json'
    # ---- Execution ----
  
    experiment_list = scarabintheloop.run(example_dir, config_file)

    # ---- Assertions ---- 
    self.assert_all_results_equal(experiment_list)

  def test_serial_vs_parallel_with_fake_delays(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_serial_vs_parallel_with_fake_delays.json'
    # ---- Execution ----
  
    experiment_list = scarabintheloop.run(example_dir, config_file)

    # ---- Assertions ---- 
    self.maxDiff = 1000
    self.assert_all_results_almost_equal(experiment_list)

  @unittest.skip # SLOW test.
  def test_serial_vs_parallel(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_consistency_serial_vs_parallel.json'
    
    # ---- Execution ----
    experiment_list = scarabintheloop.run(example_dir, config_file)

    # ---- Assertions ---- 
    self.assert_all_results_almost_equal(experiment_list)

  def assert_all_results_equal(self, experiment_list):
    results = experiment_list.get_results()
    baseline_result = results[0]
    baseline_data = baseline_result["experiment data"]
    for result in results:
      this_data = result["experiment data"]
      self.assertEqual(this_data['x'], baseline_data['x'])
      self.assertEqual(this_data['u'], baseline_data['u'])
      self.assertEqual(this_data['t'], baseline_data['t'])
      self.assertEqual(this_data['k'], baseline_data['k'])
      self.assertEqual(this_data['i'], baseline_data['i'])
      self.assertEqual(this_data['pending_computations'], baseline_data['pending_computations'])

  def assert_all_results_almost_equal(self, experiment_list):
    results = experiment_list.get_results()
    baseline_result = results[0]
    baseline_data = baseline_result["experiment data"]
    for result in results:
      this_data = result["experiment data"]
      # self.assertEqual(len(this_data['x']), len(baseline_data['x']))
      # self.assertEqual(len(this_data['u']), len(baseline_data['u']))
      # self.assertEqual(len(this_data['t']), len(baseline_data['t']))
      # self.assertEqual(len(this_data['k']), len(baseline_data['k']))
      # self.assertEqual(len(this_data['i']), len(baseline_data['i']))
      self.assertEqual(this_data['x'], baseline_data['x'])
      self.assertAlmostEqual(this_data['u'], baseline_data['u'])
      self.assertEqual(this_data['t'], baseline_data['t'])
      self.assertEqual(this_data['k'], baseline_data['k'])
      self.assertEqual(this_data['i'], baseline_data['i'])
      self.assertAlmostEqual(this_data['pending_computations'], baseline_data['pending_computations'])


if __name__ == '__main__':
  # Bcause some of the tests are very slow, we want to fail fast in case earlier tests have an error.
  unittest.main(failfast=True)