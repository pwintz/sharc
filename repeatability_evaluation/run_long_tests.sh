#! /usr/bin/bash
# Run SHARC's automated take several minutes to complete, as well as unit tests.
# The tests are run within a Docker container. 

# Make the script fail if any commands fail.
set -Eeuo pipefail

docker run --rm sharc:latest bash -c "cd /home/dcuser/resources/tests && ./run_all.sh"

docker run --rm sharc:latest bash -c "cd /home/dcuser/examples/acc_example && sharc --config_file fake_delays.json"

docker run --rm sharc:latest bash -c "cd /home/dcuser/examples/acc_example && sharc --config_file smoke_test.json"

echo "Tests succeeded."