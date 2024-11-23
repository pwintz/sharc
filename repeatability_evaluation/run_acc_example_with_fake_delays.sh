#! /usr/bin/bash

EXAMPLE_NAME=$1
CONFIG_FILE=$2

docker run --rm \
  -v "$(pwd)/${EXAMPLE_NAME}_experiments:/home/dcuser/${EXAMPLE_NAME}/experiments" \
  sharc:latest \
  bash -c "cd /home/dcuser/examples/${EXAMPLE_NAME} && sharc --config_filename ${CONFIG_FILE}" 
