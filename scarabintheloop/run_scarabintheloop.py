#! /bin/env python3
"""
Execute a list of experiments using Scarab-in-the-loop.
"""

import os
import sys

from scarabintheloop.utils import assertFileExists

# Update the PYTHONPATH env variable to include the scripts folder.
example_dir = os.path.abspath(".")
delegator_path = example_dir + '/controller_delegator.py'

assertFileExists(delegator_path, 'Did you run this script from the root of an example directory?')
# if not os.path.exists(delegator_path):
#   raise IOError(f'The file "{delegator_path}" does not exist. Did you run this script from the root of an example directory?')
sys.path.append(os.path.abspath(example_dir))

# Update the enviroment variable so that subshells have the updated path.
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

import scarabintheloop

def main():
  example_dir = os.path.abspath(".")
  scarabintheloop.run(example_dir)


if __name__ == "__main__":
  main()



# Ideas for experiments to run.
# -> Run several iterations of Cache sizes

### Candidate values to modify
# Parameters
  # sample_time
  # lead_car_input
  # x0
# MPC Settings
  # constraints
  # yref (distance buffer)
  # weights
  # warm start
# Optimizer Settings
  # optimizer tolerances
  # osqp_maximum_iteration
  # prediction_horizon
  # control_horizon
# Hardware Settings
  # computation_delay_multiplier  
  # l1_size
  # icache_size
  # dcache_size
  # chip_cycle_time
  # decode_cycles