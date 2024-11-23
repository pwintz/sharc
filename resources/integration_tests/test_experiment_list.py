#! /bin/env python3

from typing import List
import unittest
import sharc
import sharc.testing
from sharc.utils import assertFileExists
import os

from sharc.data_types import ComputationData

class TestExperimentList(sharc.testing.TestCase):

  def test_complete_data_file_is_created(self):
    # ---- Setup ---- 
    example_dir = os.path.abspath('../examples/acc_example')
    config_file = 'fake_delays.json'

    # ---- Execution ----
    experiment_list = sharc.run(example_dir, config_file, fail_fast=True)

    # ---- Assertions ---- 
    # assertFileExists(experiment_list.incremental_data_file_path)
    assertFileExists(experiment_list.complete_data_file_path)
    # Delete file
    os.remove(experiment_list.complete_data_file_path)


if __name__ == '__main__':
  # Because some of the tests are very slow, we use failfast=True so that fail when the first test fails. In general, the fast tests are run first.
  unittest.main(failfast=True)