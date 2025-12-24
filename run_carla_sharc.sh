#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
#  CARLA + SHARC Docker Runner
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Activate docker group to avoid sudo
# If docker doesn't work, run: newgrp docker
# Then run this script again
if ! docker ps > /dev/null 2>&1; then
    echo "Docker not accessible. Run: newgrp docker"
    echo "Then run this script again."
    exit 1
fi

# Configuration
IMAGE_NAME="carla-sharc"
CONTAINER_NAME="carla-sharc"
WORKSPACE_DIR="${1:-$(pwd)/workspace}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CARLA + SHARC Docker Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# Create workspace if it doesn't exist
mkdir -p "$WORKSPACE_DIR"
echo -e "${GREEN}✓${NC} Workspace: $WORKSPACE_DIR"

# Allow X11 forwarding
xhost +local:docker > /dev/null 2>&1 || true
echo -e "${GREEN}✓${NC} X11 forwarding enabled"

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" > /dev/null
fi

# Remove existing container
docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true

echo -e "${GREEN}✓${NC} Starting container..."
echo ""

# Run container with NVIDIA GPU support
docker run -it --rm \
    --gpus all \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged \
    --network host \
    --name "$CONTAINER_NAME" \
    -e DISPLAY="$DISPLAY" \
    -e QT_X11_NO_MITSHM=1 \
    -e SDL_VIDEODRIVER=x11 \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$WORKSPACE_DIR":/home/workspace/my_files \
    "$IMAGE_NAME"

