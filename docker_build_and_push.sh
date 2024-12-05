#!/bin/bash

# SHARC Setup Script
# This script builds the Docker image and pushes it to DockerHub. 
# Tests are run to make sure that the image is OK. 

# Make the script fail if any commands fail.
set -Eeuo pipefail

# Main Script Execution
echo "SHARC Build and Push Script"
echo "---------------------------------"
echo "This script will:"
echo "1. Build a SHARC Docker image."
echo "2. Push the image to DockerHub."
echo ""


# Repopository Name
REPO_NAME="pwintz"

# Name of the Docker image
IMAGE_NAME="${REPO_NAME}/sharc:latest"

push_docker_image() {
  echo "Pushing the SHARC Docker image..."
  docker push $IMAGE_NAME

  # Rename the image from "pwintz/sharc:latest" to "sharc:latest"
  # so that the naming is consistent regardless of whether the 
  # image was pulled or built.
  # docker image tag $REPO_NAME/$IMAGE_NAME $IMAGE_NAME 

  # Check if the build was successful
  if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' pushed successfully."
  else
    echo "Failed to push Docker image '$IMAGE_NAME'."
    exit 1
  fi
}

build_docker_image() {
    # Path to the Dockerfile (adjust as needed)
    DOCKERFILE_PATH="."

    # Build the Docker image
    docker build -t $IMAGE_NAME $DOCKERFILE_PATH

    # Check if the build was successful
    if [ $? -eq 0 ]; then
      echo "Docker image '$IMAGE_NAME' built successfully."
    else
      echo "Failed to build Docker image '$IMAGE_NAME'."
      echo "This script must be run from the repeatability_evaluation/ folder."
      exit 1
    fi
}

if [ "$(git status --porcelain)" ]; then 
  read -p "The Git directory has uncommitted changes. Do you want to continue? " choice
  if [[ $choice == "y" || $choice == "Y" ]]; then
    # OK
    echo "Continuing..."
  else
    exit 1
  fi
fi

# Working directory clean -> Build the image (regardless of whether it already exists.)
build_docker_image

echo "Running Tests..."
time ./repeatability_evaluation/run_long_tests.sh

echo ""
echo "All tests passed!"

read -p "Are you ready to push the image to DockerHub? " choice
if [[ $choice == "y" || $choice == "Y" ]]; then
    echo "Running container...."
    push_docker_image
    if [ $? -eq 0 ]; then 
      echo "The image was pushed to DockerHub."
      exit 0
    else
      echo "Something went wrong."
      exit 1
    fi
else
  echo "Image not pushed to DockerHub."
  exit 1
fi
