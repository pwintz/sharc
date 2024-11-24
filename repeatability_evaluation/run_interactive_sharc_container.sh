#!/bin/bash
# Create and enter an interactive SHARC Docker container.
# The container is removed when you close it, and any changes are lost.


# Name of the Docker image
IMAGE_NAME="sharc:latest"

if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
  docker run --rm -it $IMAGE_NAME # bash -c "echo \"Welcome to a SHARC container.\" "
else
  echo "The image $IMAGE_NAME is not available."
  echo "Run ./setup_sharc.sh in the root of the sharc repository to generate a Docker image."
  exit 1
fi