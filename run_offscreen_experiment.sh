#!/bin/bash
# CARLA + SHARC Headless Experiment Runner
# Run from the HOST machine — no monitor or X display required.
# Starts a fresh CARLA server (ensuring deterministic physics), runs the
# experiment, optionally records a video, and saves a dashboard image.
#
# Usage: ./run_offscreen_experiment.sh [--example NAME] [--config FILE]
#        [--video] [--highres] [--container NAME] [--user NAME] [--timeout SECS] [--log FILE]
#
# Examples:
#   ./run_offscreen_experiment.sh --config obstacle_constraint.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_dense_traffic.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_lead_vehicle_braking.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_pedestrian_crossing.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_vehicle_cutin.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_high_speed_avoidance.json --video
#
# Leaderboard scenarios (in examples/MPC_example/simulation_configs/):
#   leaderboard_dense_traffic.json          - 20 vehicles, 5 walkers, heavy obstruction
#   leaderboard_lead_vehicle_braking.json   - Lead vehicle hard braking, long horizon
#   leaderboard_pedestrian_crossing.json    - 10 walkers, slow cautious driving
#   leaderboard_vehicle_cutin.json          - 15 vehicles, lane-change cut-ins
#   leaderboard_high_speed_avoidance.json   - High-speed obstacle avoidance, 25 m/s target

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_EXAMPLES_DIR="$SCRIPT_DIR/examples"

CONTAINER="Tyler-container"
CONTAINER_USER="admin"
EXAMPLE_NAME="CarCarlaMPC_example"
CONFIG_NAME="base_config.json"
CARLA_PORT=2000
TIMEOUT=180
LOG_FILE=""
RECORD_VIDEO=0
HIGHRES=0

die() { echo "ERROR: $*" >&2; exit 1; }

verify_container_mounts_current_repo() {
    local container_name="$1"
    local host_repo="$2"

    local mounted_repo_source
    mounted_repo_source="$(docker inspect "$container_name" \
        --format '{{range .Mounts}}{{if eq .Destination "/home/workspace/sharc"}}{{.Source}}{{end}}{{end}}' \
        2>/dev/null || true)"

    if [[ -n "$mounted_repo_source" ]]; then
        if [[ "$mounted_repo_source" != "$host_repo" ]]; then
            die "Container '$container_name' is mounted to '$mounted_repo_source', but this script is running from '$host_repo'. Start/update the container so /home/workspace/sharc points at '$host_repo'."
        fi
        return 0
    fi

    local mounted_examples_source
    mounted_examples_source="$(docker inspect "$container_name" \
        --format '{{range .Mounts}}{{if eq .Destination "/home/workspace/sharc/examples"}}{{.Source}}{{end}}{{end}}' \
        2>/dev/null || true)"

    if [[ -z "$mounted_examples_source" ]]; then
        echo "WARNING: Could not determine the source mounted at /home/workspace/sharc/examples in $container_name"
        return 0
    fi

    local expected_examples_source="$host_repo/examples"
    if [[ "$mounted_examples_source" != "$expected_examples_source" ]]; then
        die "Container '$container_name' is mounted to '$mounted_examples_source', but this script is running from '$expected_examples_source'. Start/update the container so /home/workspace/sharc points at '$host_repo'."
    fi
}

sync_results_from_container() {
    local container_name="$1"
    local example_name="$2"
    local host_examples_dir="$3"

    local container_latest
    container_latest="$(docker exec -u "$CONTAINER_USER" "$container_name" \
        bash -lc "readlink -f /home/workspace/sharc/examples/${example_name}/latest 2>/dev/null || true")"
    if [[ -z "$container_latest" ]]; then
        echo "WARNING: Could not locate latest experiment dir in container for ${example_name}"
        return 0
    fi

    local result_basename
    result_basename="$(basename "$container_latest")"
    local host_example_dir="$host_examples_dir/$example_name"
    local host_experiments_dir="$host_example_dir/experiments"
    local host_result_dir="$host_experiments_dir/$result_basename"
    local mounted_repo_source

    mounted_repo_source="$(docker inspect "$container_name" \
        --format '{{range .Mounts}}{{if eq .Destination "/home/workspace/sharc"}}{{.Source}}{{end}}{{end}}' \
        2>/dev/null || true)"

    if [[ -n "$mounted_repo_source" && "$mounted_repo_source" == "$SCRIPT_DIR" ]]; then
        if [[ ! -d "$host_result_dir" ]]; then
            echo "WARNING: Expected host result dir to exist via bind mount, but it was not found: $host_result_dir"
            return 0
        fi
    else
        mkdir -p "$host_experiments_dir"
        rm -rf "$host_result_dir"
        docker cp "${container_name}:${container_latest}" "$host_result_dir"
    fi

    ln -sfn "$host_result_dir" "$host_example_dir/latest"
    echo "==> Synced results to: $host_result_dir"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)   EXAMPLE_NAME="$2";   shift 2 ;;
        --config)    CONFIG_NAME="$2";    shift 2 ;;
        --container) CONTAINER="$2";      shift 2 ;;
        --user)      CONTAINER_USER="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2";        shift 2 ;;
        --log)       LOG_FILE="$2";       shift 2 ;;
        --video)     RECORD_VIDEO=1;      shift ;;
        --highres)   HIGHRES=1;            shift ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true \
    || die "Container '$CONTAINER' is not running. Start it with: docker start $CONTAINER"

verify_container_mounts_current_repo "$CONTAINER" "$SCRIPT_DIR"

[[ -n "$LOG_FILE" ]] && exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Container: $CONTAINER (user: $CONTAINER_USER)"
echo "==> Example:   $EXAMPLE_NAME / $CONFIG_NAME"
echo "==> Video:     $([ $RECORD_VIDEO -eq 1 ] && echo 'ENABLED (GPU)' || echo 'disabled')$([ $HIGHRES -eq 1 ] && echo ' [HIGH-RES 1280x720]' || true)"
echo ""

set +e
docker exec -i \
    -u "$CONTAINER_USER" \
    -e DISPLAY= \
    -e _EXP_EXAMPLE="$EXAMPLE_NAME" \
    -e _EXP_CONFIG="$CONFIG_NAME" \
    -e _EXP_PORT="$CARLA_PORT" \
    -e _EXP_TIMEOUT="$TIMEOUT" \
    -e _EXP_VIDEO="$RECORD_VIDEO" \
    -e _EXP_HIGHRES="$HIGHRES" \
    "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

CARLA_ROOT=/home/workspace/carla_0.9.16
if [ ! -f "$CARLA_ROOT/CarlaUE4.sh" ]; then
    CARLA_ROOT=/workspace
fi
SHARC_ROOT=/home/workspace/sharc
EXAMPLE_DIR="${SHARC_ROOT}/examples/${_EXP_EXAMPLE}"
CARLA_LOG="/tmp/carla_server.log"

# Verify the example directory exists
if [ ! -d "$EXAMPLE_DIR" ]; then
    echo "ERROR: Example directory does not exist: $EXAMPLE_DIR"
    exit 1
fi

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true

echo "Running as: $(whoami)"

# ═══════════════════════════════════════════════════════════════════════
# [1/5] Kill stale CARLA — fresh restart guarantees clean physics state
#       (invalid substep settings from prior runs permanently corrupt
#       CARLA's PhysX engine; only a restart fixes it)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [1/5] Stopping any existing CARLA instance ==="
pkill -9 -f CarlaUE4 2>/dev/null || true
pkill -9 -f Xvfb 2>/dev/null || true
sleep 5
# Ensure port is free
elapsed_kill=0
while python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_EXP_PORT})); s.close()" 2>/dev/null; do
    sleep 2; elapsed_kill=$((elapsed_kill+2))
    if [ $elapsed_kill -ge 30 ]; then
        echo "WARNING: Port ${_EXP_PORT} still in use after 30s"
        break
    fi
    echo "  Waiting for port ${_EXP_PORT} to be released..."
done

# ═══════════════════════════════════════════════════════════════════════
# [2/5] Start CARLA
#   --video mode: xvfb-run + RenderOffScreen (GPU rendering for cameras)
#   default:      -nullrhi (physics only, faster startup)
# ═══════════════════════════════════════════════════════════════════════
echo ""
if [ "${_EXP_VIDEO}" = "1" ]; then
    if ! command -v xvfb-run >/dev/null 2>&1; then
        echo "ERROR: --video requires 'xvfb-run', but it is not installed in the container."
            echo "Install package 'xvfb' in this container (e.g. apt-get install -y xvfb), then rerun."
        exit 1
    fi
    echo "=== [2/5] Starting CARLA (GPU via Xvfb — video recording enabled) ==="
    xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24 +extension GLX" \
        "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound \
        -carla-rpc-port="${_EXP_PORT}" > "$CARLA_LOG" 2>&1 &
else
    echo "=== [2/5] Starting CARLA (headless, -nullrhi) ==="
    DISPLAY= "$CARLA_ROOT/CarlaUE4.sh" -nullrhi -RenderOffScreen -nosound \
        -carla-rpc-port="${_EXP_PORT}" > "$CARLA_LOG" 2>&1 &
fi
CARLA_PID=$!
echo "CARLA PID: $CARLA_PID | log: $CARLA_LOG"

trap 'echo "Stopping CARLA..."; kill "$CARLA_PID" 2>/dev/null; pkill -f CarlaUE4 2>/dev/null; pkill -f Xvfb 2>/dev/null; wait "$CARLA_PID" 2>/dev/null' EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════
# [3/5] Wait for CARLA RPC port + GPU warmup
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [3/5] Waiting for CARLA on port ${_EXP_PORT} (timeout ${_EXP_TIMEOUT}s) ==="
elapsed=0
until python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_EXP_PORT})); s.close()" 2>/dev/null; do
    # Check if CARLA process tree is still alive (xvfb-run -> CarlaUE4.sh -> CarlaUE4-Linux-Shipping)
    if ! kill -0 "$CARLA_PID" 2>/dev/null && ! pgrep -f CarlaUE4 >/dev/null 2>&1; then
        echo "ERROR: CARLA died. Log:"
        tail -40 "$CARLA_LOG"
        exit 1
    fi
    if [ $elapsed -ge $_EXP_TIMEOUT ]; then
        echo "ERROR: Timeout after ${_EXP_TIMEOUT}s. Log:"
        tail -40 "$CARLA_LOG"
        exit 1
    fi
    sleep 3; elapsed=$((elapsed+3)); echo "  ${elapsed}s..."
done
echo "CARLA port ready (${elapsed}s)"

rpc_ready=0
for attempt in $(seq 1 20); do
    if python3 - <<PY >/dev/null 2>&1
import carla
client = carla.Client("localhost", ${_EXP_PORT})
client.set_timeout(10.0)
client.get_world()
PY
    then
        rpc_ready=1
        echo "CARLA RPC ready (attempt ${attempt})"
        break
    fi
    sleep 2
done
if [ $rpc_ready -ne 1 ]; then
    echo "ERROR: CARLA RPC never became ready. Log:"
    tail -40 "$CARLA_LOG"
    exit 1
fi

# GPU warmup: compile shaders + cache textures so first experiment run is deterministic
if [ "${_EXP_VIDEO}" = "1" ]; then
    echo "  Running GPU warmup..."
    python3 "${EXAMPLE_DIR}/warmup_carla.py" </dev/null 2>&1 || echo "  (warmup skipped)"
fi

# ═══════════════════════════════════════════════════════════════════════
# [4/5] Run SHARC experiment
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [4/5] Running SHARC: ${_EXP_CONFIG} ==="
cd "$EXAMPLE_DIR"
sharc --config_filename "${_EXP_CONFIG}" </dev/null

echo ""
echo "=== Saving dashboard image ==="
python3 -m sharc.dashboard --save "$EXAMPLE_DIR" </dev/null

# ═══════════════════════════════════════════════════════════════════════
# [4b] Compute experiment metrics
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== Computing experiment metrics ==="
LATEST_DIR="$EXAMPLE_DIR/latest"
if [ -L "$LATEST_DIR" ]; then
    METRICS_RESULT_DIR="$(readlink -f "$LATEST_DIR")"
    # Find the sim sub-directory (serial-with-scarab, etc.)
    METRICS_SIM_DIR="$(find "$METRICS_RESULT_DIR" -name 'experiment_data.json' -printf '%h\n' 2>/dev/null | head -1)"
    if [ -n "$METRICS_SIM_DIR" ]; then
        python3 "$EXAMPLE_DIR/compute_metrics.py" "$METRICS_SIM_DIR" </dev/null
    else
        echo "WARNING: No experiment_data.json found for metrics"
    fi
else
    echo "WARNING: No 'latest' symlink — skipping metrics"
fi

# ═══════════════════════════════════════════════════════════════════════
# [5/5] Record video (if --video enabled)
# ═══════════════════════════════════════════════════════════════════════
if [ "${_EXP_VIDEO}" = "1" ]; then
    echo ""
    echo "=== [5/5] Recording experiment video ==="
    LATEST_DIR="$EXAMPLE_DIR/latest"
    if [ -L "$LATEST_DIR" ]; then
        RESULT_DIR="$(readlink -f "$LATEST_DIR")"
        VIDEO_ARGS="--fps 20"
        if [ "${_EXP_HIGHRES}" = "1" ]; then
            VIDEO_ARGS="$VIDEO_ARGS --highres"
        fi
        python3 "${EXAMPLE_DIR}/record_experiment_video.py" "$RESULT_DIR" \
            $VIDEO_ARGS </dev/null 2>&1 || echo "WARNING: Video recording failed"
    else
        echo "WARNING: No 'latest' symlink found — skipping video"
    fi
else
    echo ""
    echo "=== [5/5] Video recording skipped (use --video to enable) ==="
fi

echo ""
echo "=== DONE ==="
if [ -L latest ]; then
    echo "Results: $(readlink -f latest)"
fi
CONTAINER_SCRIPT
docker_status=$?
set -e

sync_results_from_container "$CONTAINER" "$EXAMPLE_NAME" "$HOST_EXAMPLES_DIR"
exit "$docker_status"
