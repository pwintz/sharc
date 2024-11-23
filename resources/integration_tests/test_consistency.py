#! /bin/env python3

from typing import List
import unittest
import sharc
import sharc.testing
import os
import numpy as np

from sharc.data_types import ComputationData

class TestConsistency(sharc.testing.TestCase):

  def test_working_dir_unchanged_by_sharc(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'fake_delays.json'
    working_dir_before = os.getcwd()

    # ---- Execution ----
    experiment_list = sharc.run(example_dir, config_file, fail_fast=True)

    # ---- Assertions ---- 
    self.assertEqual(os.getcwd(), working_dir_before)

  def test_fake_delays(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_serial_consistency_with_fake_delays.json'

    # ---- Execution ----
    experiment_list = sharc.run(example_dir, config_file, fail_fast=True)

    # ---- Assertions ---- 
    self.assert_all_results_equal(experiment_list)

  def test_serial_vs_parallel_with_fake_delays(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_serial_vs_parallel_with_fake_delays.json'
    
    # ---- Execution ----
    experiment_list = sharc.run(example_dir, config_file, fail_fast=True)

    # ---- Assertions ---- 
    self.maxDiff = 1000
    self.assertEqual(experiment_list.n_failed(), 0)
    self.assert_all_results_almost_equal(experiment_list)

  # @unittest.skip # SLOW test.
  def test_serial_vs_parallel(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'test_consistency_serial_vs_parallel.json'
    
    # ---- Execution ----
    experiment_list = sharc.run(example_dir, config_file, fail_fast=True)

    # ---- Assertions ---- 
    self.assertEqual(experiment_list.n_failed(), 0)
    self.assert_all_results_almost_equal(experiment_list)

  def assert_all_results_equal(self, experiment_list):
    self.assertEqual(experiment_list.n_failed(), 0)
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
      self.assertEqual(this_data['k'], baseline_data['k'], \
        f'Result: {result["label"]} (max_steps={result["experiment config"]["n_time_steps"]}), baseline result: {baseline_result["label"]}  (max_steps={baseline_result["experiment config"]["n_time_steps"]})')
      self.assertEqual(this_data['i'], baseline_data['i'])
      self.assertEqual(len(this_data['x']), len(baseline_data['x']), f'The length of x is different')
      self.assertEqual(len(this_data['u']), len(baseline_data['u']))
      self.assertEqual(len(this_data['t']), len(baseline_data['t']))
      self.assertEqual(len(this_data['k']), len(baseline_data['k']))
      self.assertEqual(len(this_data['i']), len(baseline_data['i']))
      self.assertEqual(len(this_data['pending_computations']), len(baseline_data['pending_computations']))
      self.asssert_vector_list_almost_equal(this_data['x'], baseline_data['x'], places=1)
      self.asssert_vector_list_almost_equal(this_data['u'], baseline_data['u'], places=0)
      self.assertEqual(this_data['t'], baseline_data['t'])
      self.assertEqual(len(this_data['pending_computations']), len(baseline_data['pending_computations']))
      this_pending_computations:     List[ComputationData] = this_data['pending_computations']
      baseline_pending_computations: List[ComputationData] = baseline_data['pending_computations']
      self.assertEqual(len(this_pending_computations), len(baseline_pending_computations))
      for k, _ in enumerate(baseline_pending_computations[:-2]):
        self.assertEqual(this_pending_computations[k] is None, baseline_pending_computations[k] is None, 
                         f'k={k}, {result["label"]} pending_computations[k]={this_pending_computations[k]}, {baseline_result["label"]} pending_computations[k]={baseline_pending_computations[k]}')
        self.assertAlmostEqual(this_pending_computations[k]._t_start, baseline_pending_computations[k]._t_start)
        # self.assertAlmostEqual(this_pending_computations[k]._delay, baseline_pending_computations[k]._delay)
        np.testing.assert_array_almost_equal(this_pending_computations[k]._u, baseline_pending_computations[k]._u)
        # self.assertAlmostEqual(this_pending_computations[k]._t_end, baseline_pending_computations[k]._t_end)

  def asssert_vector_list_almost_equal(self, list1, list2, places=6):
    self.assertEqual(len(list1), len(list2))
    for k_time_step, _ in enumerate(list1):
        x1_k = list1[k_time_step]
        x2_k = list2[k_time_step]
        self.assertEqual(len(x1_k), len(x2_k))
        for i, _ in enumerate(x1_k):
          self.assertAlmostEqual(x1_k[i], x1_k[i], places=places, msg=f'k={k_time_step}, i={i}')

if __name__ == '__main__':
  # Because some of the tests are very slow, we use failfast=True so that fail when the first test fails. In general, the fast tests are run first.
  unittest.main(failfast=True)