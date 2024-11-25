#!/bin/bash

# SHARC Setup Script
# This script pulls or builds the Docker image and runs a suite of tests 
# in a Docker container to check that the image is OK.

# Main Script Execution
echo "SHARC Setup Script"
echo "---------------------------------"
echo "This script will:"
echo "1. Pull or build the SHARC Docker image."
echo "2. Create a container and tests to check image is OK."
echo ""


# Repopository Name
REPO_NAME="pwintz"

# Name of the Docker image
IMAGE_NAME="sharc:latest"

pull_docker_image() {
  echo "Pulling the SHARC Docker image..."
  docker pull $REPO_NAME/$IMAGE_NAME

  # Rename the image from "pwintz/sharc:latest" to "sharc:latest"
  # so that the naming is consistent regardless of whether the 
  # image was pulled or built.
  docker image tag $REPO_NAME/$IMAGE_NAME $IMAGE_NAME 

  # Check if the build was successful
  if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' pulled successfully."
  else
    echo "Failed to pull Docker image '$IMAGE_NAME'."
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


# Check if the image already exists.
if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
  echo "1. Docker image '$IMAGE_NAME' is already available."
  echo "If you want a new image, first run"
  echo "   docker image rm sharc:latest"
  read -p "Do you want to remove the existing SHARC image? [y/n]: " choice

  if [[ $choice == "y" || $choice == "Y" ]]; then
      docker image rm $IMAGE_NAME
      if [ $? -eq 0 ]; then 
        echo "The image has be removed. Run this script again to complete setup."
        exit 0
      else
        echo "Removing the image failed. There might be a running container using the image. Here is a list of all containers using the image:"
        echo 
        docker container ls --all --filter=ancestor=$IMAGE_NAME
        echo
        echo "To stop the container, run "
        echo "   docker container kill $(docker ps -a -q  --filter ancestor=$IMAGE_NAME)"
        
        read -p "Do you want to stop the running container and delete the image? [y/n]: " killchoice
        if [[ $killchoice == "y" || $killchoice == "Y" ]]; then
          docker container kill $(docker ps -a -q  --filter ancestor=$IMAGE_NAME)
          docker image rm $IMAGE_NAME
          if [ $? -eq 0 ]; then 
            echo "The container has be stopped and the image deleted. Run ./setup_sharc.sh again to setup SHARC."
            exit 0
          else
            echo "Something went wrong. The image was not deleted."
          fi
        fi
      fi
  fi
  echo "Error: Unexpected case"
  exit 1
else
  # If the image doesn't exist, ask the user if they want to pull it 
  # from Docker Hub or build it.
  read -p "1. Do you want to pull or build a SHARC image? [p(ull)/b(uild)/c(ancel)]: " choice

  if [[ $choice == "p" || $choice == "pull" ]]; then
      pull_docker_image
  elif [[ $choice == "b" || $choice == "build" ]]; then
      build_docker_image
  else
      echo "Setup canceled. Exiting."
      exit 0
  fi
fi

echo "2. Running Tests..."
./repeatability_evaluation/run_tests.sh

echo "Setup complete."
echo ""

read -p "Do you want to run the container and enter it interactively (changes made in the container will not persist after exiting)? [y/n]: " choice
if [[ $choice == "y" || $choice == "Y" ]]; then
    echo "Running container...."
    ./repeatability_evaluation/run_interactive_sharc_container.sh
    exit 0
fi