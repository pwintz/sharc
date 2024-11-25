#!/bin/bash
# Create and enter an interactive SHARC Docker container.
# The container is removed when you close it, and any changes are lost.


# Name of the Docker image
IMAGE_NAME=sharc:latest

if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
  # It is necessary to run in a priviliaged container so that Scarab can use "setarch x86_64 -R". 
  docker run --rm -it --privileged $IMAGE_NAME bash -c \
  "echo && echo ====================== && echo == \"Welcome to SHARC\" == && echo ====================== && exec /bin/bash"
else
  echo "The image $IMAGE_NAME is not available."
  echo "Run ./setup_sharc.sh in the root of the sharc repository to generate a Docker image."
  exit 1
fi
