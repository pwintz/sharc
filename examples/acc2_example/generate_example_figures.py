#! /bin/env python3

import unittest
import sharc
from sharc import Simulation, BatchInit, Batch
import sharc.scarabizor as scarabizor
from sharc.data_types import *
import numpy as np
import copy
import os
import sharc.make_plots

def main():
  example_dir = "/dev-workspace/examples/acc2_example"
  experiment_dir = "/dev-workspace/examples/acc2_example/latest"
  experiment_incremental_results_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
  experiment_final_results_path = os.path.join(experiment_dir, "experiment_list_data_incremental.json")
  out_dir = os.path.join(example_dir, "images")
  # try:
  experiment_results = readJson(experiment_incremental_results_path)
  
  sharc.make_plots.run(experiment_results, out_dir)
  pass

if __name__ == "__main__":
  main()
