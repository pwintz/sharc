#! /usr/bin/bash

EXAMPLE_NAME=$1
CONFIG_FILE=$2

mkdir $(pwd)/${EXAMPLE_NAME}_experiments
echo "Experiment results will be available on the host machine in $(pwd)/${EXAMPLE_NAME}_experiments/"

docker run --rm \
  -v "$(pwd)/${EXAMPLE_NAME}_experiments:/home/dcuser/examples/${EXAMPLE_NAME}/experiments" \
  sharc:latest \
  bash -c "cd /home/dcuser/examples/${EXAMPLE_NAME} && sharc --config_filename ${CONFIG_FILE}" 

echo "Experiments finished. See results in $(pwd)/${EXAMPLE_NAME}_experiments."