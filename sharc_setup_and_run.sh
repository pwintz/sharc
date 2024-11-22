#!/bin/bash

# SHARC Setup and Execution Script
# This script pulls the Docker image and runs the ACC example, binding directories to save results locally.

# Step 1: Pull SHARC Docker Image
pull_docker_image() {
    echo "Pulling the SHARC Docker image..."
    docker pull pwintz/sharc:latest
    if [ $? -eq 0 ]; then
        echo "SHARC Docker image pulled successfully."
    else
        echo "Failed to pull the SHARC Docker image. Check your Docker setup."
        exit 1
    fi
}

# Step 2: Run the ACC Example
run_acc_example() {
    echo "Running the Adaptive Cruise Control (ACC) example..."
    docker run --rm \
        -v "$(pwd)/resources:/sharc/resources" \
        -v "$(pwd)/examples:/sharc/examples" \
        pwintz/sharc:latest \
        bash -c "cd /sharc/examples/acc_example && sharc --config_file fake_delays.json"
    if [ $? -eq 0 ]; then
        echo "Simulation completed successfully."
        echo "Results are saved in 'examples/acc_example/experiments/'."
    else
        echo "Simulation failed. Please check the logs above for details."
        exit 1
    fi
}

# Main Script Execution
echo "SHARC Setup and Execution Script"
echo "---------------------------------"
echo "This script will:"
echo "1. Pull the SHARC Docker image."
echo "2. Run the Adaptive Cruise Control (ACC) example by binding the 'resources' and 'examples' directories to the container."
echo ""
read -p "Proceed with setup and execution? [y/n]: " choice

if [[ $choice == "y" || $choice == "Y" ]]; then
    pull_docker_image
    run_acc_example
    echo "All steps completed successfully."
else
    echo "Setup aborted. Exiting."
fi