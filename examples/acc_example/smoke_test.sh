#!/bin/bash
## Execute this script in the examples/acc_example directory to run a relatively quick (~5 mins) check that simulations can run without errors in serial and parallel modes and using fake delays and Scarab delays.

# Make the script fail if any commands fail.
set -Eeuo pipefail

time sharc --failfast --config_file smoke_test.json 