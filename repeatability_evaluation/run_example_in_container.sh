#! /usr/bin/bash

# Make the script fail if any commands fail.
set -Eeuo pipefail

EXAMPLE_NAME=$1
CONFIG_FILE=$2

# Make the experiments folder in the host environment if it does not already exist.
mkdir  -p $(pwd)/${EXAMPLE_NAME}_experiments
echo "Experiment results will be available on the host machine in $(pwd)/${EXAMPLE_NAME}_experiments/"

# Run the experiment in docker, including generating plots.
# Explanation of arguments:
  #         --rm: Remove the container when it exits, so we don't fill up the hard drive. \
  # --privileged: Run the container with elevated permissions so that Scarab can use "setarch x86_64 -R". 
  #           -v: Mount a volume so that the results of experiments persist on the host machine.
docker run --rm --privileged \
  -v "$(pwd)/${EXAMPLE_NAME}_experiments:/home/dcuser/examples/${EXAMPLE_NAME}/experiments" \
  sharc:latest \
  bash -c "cd /home/dcuser/examples/${EXAMPLE_NAME} \
        && sharc --config_filename ${CONFIG_FILE} --failfast \
        && ./generate_example_figures.py" 

echo "Experiments finished. See results in $(pwd)/${EXAMPLE_NAME}_experiments."