#!/usr/bin/env python3

#  Copyright 2020 HPS/SAFARI Research Groups
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""
> python ./bin/scarab_test_qsort.py path_to_results_directory
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

import json

# Read config.json.
sim_dir = 'sim_dir/'
with open(sim_dir + 'config.json') as json_data:
    config_data = json.load(json_data)

example_dir = os.path.dirname(os.path.abspath(__file__))

# Add <scarab path>/bin to the Python search pth.
# scarab_root_path = os.environ["SCARAB_ROOT"]
# sys.path.append(scarab_root_path + '/bin')

from scarab_globals import *

def run_scarab(cmd_to_simulate, use_fake_scarab_computation_times=False, sim_dir="sim_dir"):
  if use_fake_scarab_computation_times:
      subprocess.check_call(example_dir + '/' + cmd_to_simulate)
      return

  scarab_cmd_argv = [sys.executable,
                     scarab_paths.bin_dir + '/scarab_launch.py',
                     '--program', example_dir + '/' + cmd_to_simulate,
                     '--param', example_dir + '/PARAMS.generated',
                     '--simdir', sim_dir, # Changes the working directory.
                     '--pintool_args',
                     '-fast_forward_to_start_inst 1',
                     '--scarab_args',
                     '--inst_limit 15000000000',
                    #  '--heartbeat_interval 50', 
                     # '--num_heartbeats 1'
                     # '--power_intf_on 1']
                     ]
  print ('Scarab cmd:', ' '.join(scarab_cmd_argv))
  subprocess.check_call(scarab_cmd_argv)


def __main():
  global args #?? Is this needed??

  parser = argparse.ArgumentParser(description=f'Run a command using Scarab.')
  parser.add_argument('cmd', nargs=1, help='Command to build with Make (if needed) and then simulate with Scarab.')
  parser.add_argument('--sim_dir', nargs='?', help='Path to the simulation directory.', default="sim_dir")
  args = parser.parse_args()

  cmd_to_simulate = args.cmd[0]
  run_scarab(
    cmd_to_simulate, 
    use_fake_scarab_computation_times=config_data["use_fake_scarab_computation_times"],
    sim_dir=args.sim_dir
  )


if __name__ == "__main__":
  __main()
