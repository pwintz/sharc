#! /usr/bin/bash

# Make the script fail if any commands fail.
set -Eeuo pipefail

./run_example_in_container.sh acc_example parallel_vs_serial.json