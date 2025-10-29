#!/bin/bash

# Run SHARC examples locally using Apptainer
# Usage: ./run_example_apptainer.sh [example] [config]
#   example: acc_example or cartpole (default: acc_example)
#   config: configuration file name (default: default.json)

set -e  # Exit on error

# Configuration
IMAGE="sharc_latest.sif"
EXAMPLE=${1:-"Rocket_example"}
CONFIG=${2:-"default.json"}

# Check if image exists
if [ ! -f "$IMAGE" ]; then
    echo "Error: Apptainer image '$IMAGE' not found"
    echo "Run './setup_sharc_apptainer.sh' first to create the image"
    exit 1
fi

# Check if example exists
if [ ! -d "examples/$EXAMPLE" ]; then
    echo "Error: Example directory 'examples/$EXAMPLE' not found"
    echo "Available examples:"
    ls -1 examples/
    exit 1
fi

echo "Running SHARC Example"
echo "===================="
echo "Example: $EXAMPLE"
echo "Config:  $CONFIG"
echo "===================="
echo ""

# Run simulation
apptainer exec \
    --bind "$(pwd)/resources:/home/dcuser/resources" \
    --bind "$(pwd)/examples:/examples" \
    "$IMAGE" \
    bash -c "cd /examples/$EXAMPLE && sharc --config_filename $CONFIG"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Simulation completed successfully"
    echo "✓ Results saved to: examples/$EXAMPLE/experiments/"
else
    echo ""
    echo "✗ Simulation failed"
    exit 1
fi

