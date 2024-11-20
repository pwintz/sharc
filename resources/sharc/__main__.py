from sharc import *
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run a set of experiments.")

parser.add_argument(
    '--config_filename',
    type   =str,  # Specify the data type
    default="default.json",
    help   ="Select the name of a JSON file located in <example_dir>/simulation_configs."
)

# Parse the arguments from the command line
args = parser.parse_args()
example_dir = os.path.abspath('.')
experiment_list = run(example_dir, args.config_filename)

if experiment_list.n_failed():
  exit(1)