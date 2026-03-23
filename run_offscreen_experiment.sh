#!/bin/bash
# CARLA + SHARC Headless Experiment Runner
# Run from the HOST machine — no monitor or X display required.
# Usage: ./run_offscreen_experiment.sh [--example NAME] [--config FILE]
#        [--container NAME] [--user NAME] [--timeout SECS] [--log FILE]

set -euo pipefail

CONTAINER="carla-sharc-yasin5"
CONTAINER_USER="admin"
EXAMPLE_NAME="MPC_example"
CONFIG_NAME="obstacle_constraint.json"
CARLA_PORT=2000
TIMEOUT=180
LOG_FILE=""

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)   EXAMPLE_NAME="$2";   shift 2 ;;
        --config)    CONFIG_NAME="$2";    shift 2 ;;
        --container) CONTAINER="$2";      shift 2 ;;
        --user)      CONTAINER_USER="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2";        shift 2 ;;
        --log)       LOG_FILE="$2";       shift 2 ;;
        -h|--help)
            sed -n '2,5p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true \
    || die "Container '$CONTAINER' is not running. Start it with: docker start $CONTAINER"

[[ -n "$LOG_FILE" ]] && exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Container: $CONTAINER (user: $CONTAINER_USER)"
echo "==> Example:   $EXAMPLE_NAME / $CONFIG_NAME"
echo ""

docker exec -i \
    -u "$CONTAINER_USER" \
    -e DISPLAY= \
    -e SDL_VIDEODRIVER=offscreen \
    -e _EXP_EXAMPLE="$EXAMPLE_NAME" \
    -e _EXP_CONFIG="$CONFIG_NAME" \
    -e _EXP_PORT="$CARLA_PORT" \
    -e _EXP_TIMEOUT="$TIMEOUT" \
    "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

CARLA_ROOT=/home/workspace/carla_0.9.16
EXAMPLE_DIR="/home/workspace/sharc/examples/${_EXP_EXAMPLE}"
CARLA_LOG="${EXAMPLE_DIR}/carla_server.log"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true

echo "Running as: $(whoami)"

# Kill stale CARLA
pkill -f CarlaUE4 2>/dev/null || true
sleep 2

# [1/3] Start CARLA
echo ""
echo "=== [1/3] Starting CARLA (headless, -nullrhi) ==="
"$CARLA_ROOT/CarlaUE4.sh" -nullrhi -RenderOffScreen -nosound \
    -carla-rpc-port="${_EXP_PORT}" > "$CARLA_LOG" 2>&1 &
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID | log: $CARLA_LOG"

trap 'echo "Stopping CARLA..."; kill "$CARLA_PID" 2>/dev/null; pkill -f CarlaUE4 2>/dev/null; wait "$CARLA_PID" 2>/dev/null' EXIT INT TERM

# [2/3] Wait for RPC port
echo ""
echo "=== [2/3] Waiting for CARLA on port ${_EXP_PORT} (timeout ${_EXP_TIMEOUT}s) ==="
elapsed=0
until python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_EXP_PORT})); s.close()" 2>/dev/null; do
    kill -0 "$CARLA_PID" 2>/dev/null || { echo "ERROR: CARLA died. Log:"; tail -40 "$CARLA_LOG"; exit 1; }
    [[ $elapsed -lt $_EXP_TIMEOUT ]] || { echo "ERROR: Timeout after ${_EXP_TIMEOUT}s. Log:"; tail -40 "$CARLA_LOG"; exit 1; }
    sleep 3; elapsed=$((elapsed+3)); echo "  ${elapsed}s..."
done
echo "CARLA ready (${elapsed}s)"

# [3/3] Run SHARC
echo ""
echo "=== [3/3] Running SHARC: ${_EXP_CONFIG} ==="
cd "$EXAMPLE_DIR"
sharc --config_filename "${_EXP_CONFIG}"

echo ""
echo "=== Saving dashboard image ==="
python3 -m sharc.dashboard --save "$EXAMPLE_DIR"

echo ""
echo "=== DONE ==="
[[ -L latest ]] && echo "Results: $EXAMPLE_DIR/$(readlink latest)"
CONTAINER_SCRIPT
