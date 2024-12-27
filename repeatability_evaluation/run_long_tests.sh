#! /usr/bin/bash
# Run SHARC's automated tests that take several minutes to complete, as well as unit tests.
# The tests are run within a Docker container which is deleted upon completion. 

# Make the script fail if any commands fail.
set -Eeuo pipefail

# Run unit tests.
docker run --rm sharc:latest bash -c "cd /home/dcuser/resources/tests && ./run_all.sh"

docker run --privileged --rm sharc:latest bash -c "cd /home/dcuser/examples/acc_example && sharc --config_file smoke_test.json"

# Run the acc example using fake delays. This include testing the figure generation script, so it is slightly different that running the smoke test.
./run_acc_example_with_fake_delays.sh

echo "Tests succeeded."
