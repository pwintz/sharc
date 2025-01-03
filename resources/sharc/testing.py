import unittest
import numpy as np

from sharc.data_types import ComputationData
from sharc.utils import *
from typing import List


class TestCase(unittest.TestCase):
  
  def assert_all_experiment_list_results_equal(self, experiment_list):
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

  def assert_all_experiment_list_results_almost_equal(self, experiment_list):
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
        self.assertAlmostEqual(this_pending_computations[k]._u, baseline_pending_computations[k]._u)
        # self.assertAlmostEqual(this_pending_computations[k]._t_end, baseline_pending_computations[k]._t_end)

  def asssert_vector_list_almost_equal(self, list1, list2, places=6):
    
    self.assertEqual(len(list1), len(list2))
    for k_time_step, _ in enumerate(list1):
        x1_k = list1[k_time_step]
        x2_k = list2[k_time_step]
        self.assertEqual(len(x1_k), len(x2_k))
        for i, _ in enumerate(x1_k):
          self.assertAlmostEqual(x1_k[i], x1_k[i], places=places, msg=f'k={k_time_step}, i={i}')
