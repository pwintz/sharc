#!/bin/bash

# SHARC Apptainer Setup Script
# Sets up SHARC using Apptainer for HPC systems (e.g., Savio)

set -e  # Exit on error

echo "SHARC Apptainer Setup Script"
echo "=============================="
echo ""

# Configuration
IMAGE_NAME="sharc_latest.sif"
DOCKER_IMAGE="pwintz/sharc:latest"

# Check if Apptainer is installed
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not installed or not in PATH"
    echo "On Savio, load it with: module load apptainer"
    exit 1
fi

# Check if image already exists
if [ -f "$IMAGE_NAME" ]; then
    echo "Apptainer image '$IMAGE_NAME' already exists."
    read -p "Do you want to remove and rebuild it? [y/n]: " choice
    if [[ $choice == "y" || $choice == "Y" ]]; then
        rm "$IMAGE_NAME"
        echo "Existing image removed."
    else
        echo "Using existing image."
        exit 0
    fi
fi

# Pull Docker image and convert to Apptainer
echo "Pulling Docker image and converting to Apptainer format..."
echo "This may take several minutes..."
apptainer pull "$IMAGE_NAME" docker://"$DOCKER_IMAGE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Setup complete!"
    echo "✓ Apptainer image created: $IMAGE_NAME"
    echo ""
    echo "Next steps:"
    echo "  - Test locally: ./run_example_apptainer.sh"
    echo "  - Submit job: sbatch submit_job.sh"
else
    echo "✗ Failed to pull and convert image"
    exit 1
fi

