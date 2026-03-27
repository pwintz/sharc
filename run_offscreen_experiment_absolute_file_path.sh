#!/bin/bash
# CARLA + SHARC Headless Experiment Runner
# Run from the HOST machine — no monitor or X display required.
# Starts a fresh CARLA server, runs the experiment, optionally records a video,
# and saves a dashboard image.
#
# This version:
# - does NOT use EXAMPLE_DIR/latest
# - stores the real SHARC experiment folder path in a per-run file
# - does NOT kill other CARLA/Xvfb instances
# - only checks whether the requested CARLA port is already in use
# - stores per-run metadata/logs under runs/<run-id>/
#
# IMPORTANT:
# You must also patch SHARC Python so it writes experiment_list.experiment_list_dir
# to the file path given by SHARC_RUN_DIR_FILE.
#
# Usage:
#   ./run_offscreen_experiment.sh [--example NAME] [--config FILE]
#       [--video] [--highres] [--container NAME] [--user NAME]
#       [--timeout SECS] [--log FILE] [--run-id ID]
#       [--port PORT] [--rpc-port PORT]
#
# Examples:
#   ./run_offscreen_experiment.sh --config obstacle_constraint.json --video
#   ./run_offscreen_experiment.sh --config leaderboard_dense_traffic.json --video --run-id dense1 --port 2010 --rpc-port 8100
#   ./run_offscreen_experiment.sh --config leaderboard_vehicle_cutin.json --run-id cutin2 --port 2012 --rpc-port 8102

set -euo pipefail

CONTAINER="carla-sharc-shengmin3"
CONTAINER_USER="admin"
EXAMPLE_NAME="MPC_example"
CONFIG_NAME="lead_vehicle_stop.json"
CARLA_PORT=2010
RPC_PORT=8100
TIMEOUT=180
LOG_FILE=""
RECORD_VIDEO=0
HIGHRES=0
RUN_ID=""

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --example)   EXAMPLE_NAME="$2";   shift 2 ;;
        --config)    CONFIG_NAME="$2";    shift 2 ;;
        --container) CONTAINER="$2";      shift 2 ;;
        --user)      CONTAINER_USER="$2"; shift 2 ;;
        --timeout)   TIMEOUT="$2";        shift 2 ;;
        --port)      CARLA_PORT="$2";     shift 2 ;;
        --rpc-port)  RPC_PORT="$2";       shift 2 ;;
        --log)       LOG_FILE="$2";       shift 2 ;;
        --run-id)    RUN_ID="$2";         shift 2 ;;
        --video)     RECORD_VIDEO=1;      shift ;;
        --highres)   HIGHRES=1;           shift ;;
        -h|--help)
            sed -n '2,34p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(basename "${CONFIG_NAME%.json}")_$(date +%Y%m%d_%H%M%S)_$$"
fi

docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true \
    || die "Container '$CONTAINER' is not running. Start it with: docker start $CONTAINER"

[[ -n "$LOG_FILE" ]] && exec > >(tee -a "$LOG_FILE") 2>&1

echo "==> Container: $CONTAINER (user: $CONTAINER_USER)"
echo "==> Example:   $EXAMPLE_NAME / $CONFIG_NAME"
echo "==> Run ID:    $RUN_ID"
echo "==> CARLA:     port=$CARLA_PORT rpc-port=$RPC_PORT"
echo "==> Video:     $([ $RECORD_VIDEO -eq 1 ] && echo 'ENABLED (GPU)' || echo 'disabled')$([ $HIGHRES -eq 1 ] && echo ' [HIGH-RES 1280x720]' || true)"
echo ""

docker exec -i \
    -u "$CONTAINER_USER" \
    -e DISPLAY= \
    -e _EXP_EXAMPLE="$EXAMPLE_NAME" \
    -e _EXP_CONFIG="$CONFIG_NAME" \
    -e _EXP_PORT="$CARLA_PORT" \
    -e _EXP_RPC_PORT="$RPC_PORT" \
    -e _EXP_TIMEOUT="$TIMEOUT" \
    -e _EXP_VIDEO="$RECORD_VIDEO" \
    -e _EXP_HIGHRES="$HIGHRES" \
    -e _EXP_RUN_ID="$RUN_ID" \
    "$CONTAINER" bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

CARLA_ROOT=/home/workspace/carla_0.9.16
if [ ! -f "$CARLA_ROOT/CarlaUE4.sh" ]; then
    CARLA_ROOT=/workspace
fi

SHARC_ROOT=/home/workspace/sharc
EXAMPLE_DIR="${SHARC_ROOT}/examples/${_EXP_EXAMPLE}"

RUN_META_DIR="${EXAMPLE_DIR}/runs/${_EXP_RUN_ID}"
mkdir -p "$RUN_META_DIR"

RUN_DIR_FILE="${RUN_META_DIR}/experiment_list_dir.txt"
CARLA_LOG="${RUN_META_DIR}/carla_server.log"
RUN_INFO_FILE="${RUN_META_DIR}/run_info.txt"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true

cat > "$RUN_INFO_FILE" <<EOF
run_id=${_EXP_RUN_ID}
example=${_EXP_EXAMPLE}
config=${_EXP_CONFIG}
carla_port=${_EXP_PORT}
rpc_port=${_EXP_RPC_PORT}
video=${_EXP_VIDEO}
highres=${_EXP_HIGHRES}
start_time=$(date)
run_meta_dir=${RUN_META_DIR}
run_dir_file=${RUN_DIR_FILE}
carla_log=${CARLA_LOG}
EOF

echo "Running as: $(whoami)"
echo "Run metadata dir: $RUN_META_DIR"
echo "Run dir file:     $RUN_DIR_FILE"
echo "CARLA log:        $CARLA_LOG"

# ═══════════════════════════════════════════════════════════════════════
# [1/5] Check target CARLA port only
#   Do NOT kill any existing CARLA/Xvfb processes.
#   Just fail if the chosen port is already in use.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [1/5] Checking CARLA port ${_EXP_PORT} ==="

if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_EXP_PORT})); s.close()" 2>/dev/null; then
    echo "ERROR: Port ${_EXP_PORT} is already in use."
    echo "Choose a different --port for this run."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════
# [2/5] Start CARLA
#   --video mode: xvfb-run + RenderOffScreen
#   default:      -nullrhi (physics only, faster startup)
# ═══════════════════════════════════════════════════════════════════════
echo ""
if [ "${_EXP_VIDEO}" = "1" ]; then
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

trap 'echo "Stopping CARLA for this run..."; kill "$CARLA_PID" 2>/dev/null || true; wait "$CARLA_PID" 2>/dev/null || true' EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════
# [3/5] Wait for CARLA RPC port + GPU warmup
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [3/5] Waiting for CARLA on port ${_EXP_PORT} (timeout ${_EXP_TIMEOUT}s) ==="
elapsed=0
until python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_EXP_PORT})); s.close()" 2>/dev/null; do
    if ! kill -0 "$CARLA_PID" 2>/dev/null; then
        echo "ERROR: CARLA died. Log:"
        tail -40 "$CARLA_LOG"
        exit 1
    fi
    if [ $elapsed -ge $_EXP_TIMEOUT ]; then
        echo "ERROR: Timeout after ${_EXP_TIMEOUT}s. Log:"
        tail -40 "$CARLA_LOG"
        exit 1
    fi
    sleep 3
    elapsed=$((elapsed+3))
    echo "  ${elapsed}s..."
done
echo "CARLA ready (${elapsed}s)"

if [ "${_EXP_VIDEO}" = "1" ]; then
    echo "  Running GPU warmup..."
    python3 "${EXAMPLE_DIR}/warmup_carla.py" </dev/null 2>&1 || echo "  (warmup skipped)"
fi

# ═══════════════════════════════════════════════════════════════════════
# [4/5] Run SHARC experiment
#   SHARC must write the real experiment_list_dir to RUN_DIR_FILE
#   via the SHARC_RUN_DIR_FILE environment variable.
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [4/5] Running SHARC: ${_EXP_CONFIG} ==="
cd "$EXAMPLE_DIR"

rm -f "$RUN_DIR_FILE"

SHARC_RUN_DIR_FILE="$RUN_DIR_FILE" \
sharc --config_filename "${_EXP_CONFIG}" </dev/null

if [ ! -f "$RUN_DIR_FILE" ]; then
    echo "ERROR: SHARC did not write run directory file: $RUN_DIR_FILE"
    exit 1
fi

RESULT_DIR="$(cat "$RUN_DIR_FILE")"
if [ -z "$RESULT_DIR" ] || [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: Invalid RESULT_DIR from $RUN_DIR_FILE: '$RESULT_DIR'"
    exit 1
fi

echo "Experiment result dir: $RESULT_DIR"
echo "result_dir=${RESULT_DIR}" >> "$RUN_INFO_FILE"

echo ""
echo "=== Saving dashboard image ==="
python3 -m sharc.dashboard --save "$EXAMPLE_DIR" </dev/null

# ═══════════════════════════════════════════════════════════════════════
# [4b] Compute experiment metrics
#   Uses RESULT_DIR directly, never EXAMPLE_DIR/latest
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== Computing experiment metrics ==="
METRICS_RESULT_DIR="$RESULT_DIR"
METRICS_SIM_DIR="$(find "$METRICS_RESULT_DIR" -name 'experiment_data.json' -printf '%h\n' 2>/dev/null | head -1)"

if [ -n "$METRICS_SIM_DIR" ]; then
    echo "Metrics simulation dir: $METRICS_SIM_DIR"
    python3 "$EXAMPLE_DIR/compute_metrics.py" "$METRICS_SIM_DIR" </dev/null
else
    echo "WARNING: No experiment_data.json found for metrics"
fi

# ═══════════════════════════════════════════════════════════════════════
# [5/5] Record video (if --video enabled)
#   Uses RESULT_DIR directly, never EXAMPLE_DIR/latest
# ═══════════════════════════════════════════════════════════════════════
if [ "${_EXP_VIDEO}" = "1" ]; then
    echo ""
    echo "=== [5/5] Recording experiment video ==="
    if [ -d "$RESULT_DIR" ]; then
        VIDEO_ARGS="--fps 20"
        if [ "${_EXP_HIGHRES}" = "1" ]; then
            VIDEO_ARGS="$VIDEO_ARGS --highres"
        fi
        python3 "${EXAMPLE_DIR}/record_experiment_video.py" "$RESULT_DIR" \
            $VIDEO_ARGS </dev/null 2>&1 || echo "WARNING: Video recording failed"
    else
        echo "WARNING: Result dir not found — skipping video"
    fi
else
    echo ""
    echo "=== [5/5] Video recording skipped (use --video to enable) ==="
fi

echo ""
echo "=== DONE ==="
echo "Results: $RESULT_DIR"
echo "Run metadata dir: $RUN_META_DIR"
CONTAINER_SCRIPT