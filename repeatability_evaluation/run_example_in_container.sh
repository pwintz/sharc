#! /usr/bin/bash

# Make the script fail if any commands fail.
set -Eeuo pipefail

EXAMPLE_NAME=$1
CONFIG_FILE=$2

# Make the experiments folder in the host environment if it does not already exist.
mkdir  -p $(pwd)/${EXAMPLE_NAME}_experiments
echo "Experiment results will be available on the host machine in $(pwd)/${EXAMPLE_NAME}_experiments/"

# Run the experiment in docker, including generating plots.
docker run --rm \
  -v "$(pwd)/${EXAMPLE_NAME}_experiments:/home/dcuser/examples/${EXAMPLE_NAME}/experiments" \
  sharc:latest \
  bash -c "cd /home/dcuser/examples/${EXAMPLE_NAME} \
        && sharc --config_filename ${CONFIG_FILE} \
        && ./generate_example_figures.py" 

echo "Experiments finished. See results in $(pwd)/${EXAMPLE_NAME}_experiments."