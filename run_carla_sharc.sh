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
CONTAINER_NAME="carla-sharc-yasin5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CARLA + SHARC Docker Environment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# Allow X11 forwarding
if [ -z "$DISPLAY" ]; then
    # Scan for the first available X11 socket
    for socket in /tmp/.X11-unix/X*; do
        if [ -S "$socket" ]; then
            display_num=$(echo "$socket" | sed 's/.*X//')
            export DISPLAY=":$display_num"
            echo -e "${YELLOW}WARNING: DISPLAY was empty. Detected X11 socket at $DISPLAY${NC}"
            break
        fi
    done
fi

if [ -n "$DISPLAY" ]; then
    xhost +local:docker > /dev/null 2>&1 || true
    echo -e "${GREEN}✓${NC} X11 forwarding enabled (DISPLAY=$DISPLAY)"
else
    echo -e "${YELLOW}⚠ WARNING: No X11 display detected. GUI applications will fail.${NC}"
    echo -e "  If you are on a remote server, use 'ssh -X' or 'ssh -Y'."
fi

# Check if container exists and is running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${YELLOW}Container '$CONTAINER_NAME' is already running. Attaching to it...${NC}"
    echo ""
    docker exec -it -u "$(id -u):$(id -g)" "$CONTAINER_NAME" /bin/bash
    exit 0
fi

# Check if container exists but is stopped
if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${YELLOW}Container '$CONTAINER_NAME' exists but is stopped. Starting it...${NC}"
    docker start "$CONTAINER_NAME" > /dev/null
    echo -e "${GREEN}✓${NC} Container started."
    echo -e "${GREEN}✓${NC} Attaching to it..."
    echo ""
    docker exec -it -u "$(id -u):$(id -g)" "$CONTAINER_NAME" /bin/bash
    exit 0
fi

# Container doesn't exist, create a new one
echo -e "${GREEN}✓${NC} Creating new container..."
echo ""

# Prepare XAUTHORITY mount if it exists and is a file
XAUTH_MOUNT=""
if [ -n "$XAUTHORITY" ] && [ -f "$XAUTHORITY" ]; then
    XAUTH_MOUNT="-v $XAUTHORITY:$XAUTHORITY -e XAUTHORITY=$XAUTHORITY"
elif [ -f "$HOME/.Xauthority" ]; then
    XAUTH_MOUNT="-v $HOME/.Xauthority:/home/admin/.Xauthority -e XAUTHORITY=/home/admin/.Xauthority"
elif [ -n "$DISPLAY" ]; then
    echo -e "${YELLOW}⚠ WARNING: No valid .Xauthority file found. GUI applications may fail.${NC}"
fi

# Run container with NVIDIA GPU support (without --rm so it persists)
# HOST_UID/HOST_GID tell the entrypoint to realign the container user
docker run -it \
    --gpus all \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --privileged \
    --network host \
    --name "$CONTAINER_NAME" \
    --shm-size=8gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e DISPLAY="$DISPLAY" \
    $XAUTH_MOUNT \
    -e QT_X11_NO_MITSHM=1 \
    -e SDL_VIDEODRIVER=x11 \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$SCRIPT_DIR/resources":/home/workspace/sharc/resources \
    -v "$SCRIPT_DIR/examples":/home/workspace/sharc/examples \
    "$IMAGE_NAME" 

