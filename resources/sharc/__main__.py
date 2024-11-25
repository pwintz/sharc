from sharc import *
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(prog="sharc", description="Run a set of experiments using the SHARC simulator.")

parser.add_argument(
    '--config_filename',
    type   =str,  # Specify the data type
    default="default.json",
    help   ="Select the name of a JSON file located in <example_dir>/simulation_configs."
)

parser.add_argument(
  '--failfast',   # on/off flag
  action = 'store_true',   # Deafult to 'fals'
  help   = "Make SHARC terminate after the first failed experiment."
)

# Parse the arguments from the command line
args = parser.parse_args()
example_dir = os.path.abspath('.')
is_fail_fast = args.failfast
experiment_list = run(example_dir, args.config_filename, fail_fast=is_fail_fast)

if experiment_list.n_failed():
  exit(1)