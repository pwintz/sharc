#! /usr/bin/bash
# Run SHARC's automated unit tests. 
# The tests are run within a Docker container. 

# Make the script fail if any commands fail.
set -Eeuo pipefail

docker run --rm sharc:latest bash -c "cd /home/dcuser/resources/tests && ./run_all.sh"

echo "Tests succeeded."