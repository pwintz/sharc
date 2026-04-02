#!/bin/bash
# ============================================================================
# CARLA Leaderboard Scenario Runner with SHARC MPC Agent
# ============================================================================
#
# Runs official CARLA leaderboard scenarios using scenario_runner with the
# SHARC MPC controller agent. Supports single scenarios, route-based
# scenarios, video recording, parallel instances, and batch execution.
#
# This script is run from the HOST machine. It manages a Docker container
# running CARLA + scenario_runner + SHARC MPC agent.
#
# ---- COMMANDS ----
#
#   Single scenario:
#     ./run_leaderboard_scenarios.sh --scenario FollowLeadingVehicle_1 --video
#
#   Route-based (leaderboard style):
#     ./run_leaderboard_scenarios.sh --route devtest --route-id 0 --video
#
#   Run all built-in scenarios:
#     ./run_leaderboard_scenarios.sh --run-all --video
#
#   Run batch of scenarios in parallel (3 CARLA instances):
#     ./run_leaderboard_scenarios.sh --run-all --parallel 3 --video
#
#   List available scenarios:
#     ./run_leaderboard_scenarios.sh --list
#
# ---- OPTIONS ----
#
#   --scenario NAME   Run a specific scenario (e.g., FollowLeadingVehicle_1)
#   --route FILE      Route XML (devtest|training|validation|town10 or full path)
#   --route-id ID     Route ID within the route file (default: 0)
#   --run-all         Run all predefined leaderboard scenarios sequentially
#   --parallel N      Use N parallel CARLA instances (default: 1)
#   --video           Record video for each scenario
#   --highres         Use high-resolution video (1280x720)
#   --port PORT       Base CARLA port (default: 2010, instances use +10 spacing)
#   --tm-port PORT    Base Traffic Manager port (default: 8100)
#   --container NAME  Docker container (default: carla-sharc-yasin5)
#   --user NAME       Container user (default: admin)
#   --timeout SECS    CARLA startup timeout (default: 120)
#   --agent-config F  Path to agent config JSON (default: sharc_agent_config.json)
#   --output-dir DIR  Output directory for results
#   --list            List all available scenarios and exit
#   --repetitions N   Run each scenario N times (default: 1)
#   --debug           Enable debug output in scenario_runner
#   -h, --help        Show this help
#
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONTAINER="carla-sharc-yasin5"
CONTAINER_USER="admin"
BASE_PORT=2010
BASE_TM_PORT=8100
TIMEOUT=120
RECORD_VIDEO=0
HIGHRES=0
PARALLEL=1
REPETITIONS=1
DEBUG_MODE=""
SCENARIO_NAME=""
ROUTE_NAME=""
ROUTE_ID="0"
RUN_ALL=0
LIST_SCENARIOS=0
AGENT_CONFIG=""
OUTPUT_DIR=""

CARLA_ROOT="/home/workspace/carla_0.9.16"
SHARC_ROOT="/home/workspace/sharc"
SR_ROOT="/home/workspace/scenario_runner"

# Leaderboard scenarios we test (mapped to CARLA LeaderBoard categories)
ALL_SCENARIOS=(
    # Braking and lane changing (Leaderboard #16)
    "FollowLeadingVehicle_1"
    "FollowLeadingVehicleWithObstacle_1"
    # Obstacle avoidance (Leaderboard #12, #17)
    "StationaryObjectCrossing_1"
    "DynamicObjectCrossing_1"
    # Traffic negotiation (Leaderboard #02-#05)
    "SignalizedJunctionLeftTurn_1"
    "SignalizedJunctionRightTurn_1"
    "NoSignalJunctionCrossing"
    "OppositeVehicleRunningRedLight_1"
    # Control loss (Leaderboard #01)
    "ControlLoss_1"
    # Lane changes
    "OtherLeadingVehicle_1"
    # Turns with cyclists/pedestrians
    "VehicleTurningRight_1"
    "VehicleTurningLeft_1"
    # Maneuvers
    "ManeuverOppositeDirection_1"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "===> $*"; }

show_help() {
    sed -n '2,55p' "$0" | sed 's/^# \?//'
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)      SCENARIO_NAME="$2";     shift 2 ;;
        --route)         ROUTE_NAME="$2";        shift 2 ;;
        --route-id)      ROUTE_ID="$2";          shift 2 ;;
        --run-all)       RUN_ALL=1;              shift   ;;
        --parallel)      PARALLEL="$2";          shift 2 ;;
        --video)         RECORD_VIDEO=1;         shift   ;;
        --highres)       HIGHRES=1;              shift   ;;
        --port)          BASE_PORT="$2";         shift 2 ;;
        --tm-port)       BASE_TM_PORT="$2";      shift 2 ;;
        --container)     CONTAINER="$2";         shift 2 ;;
        --user)          CONTAINER_USER="$2";    shift 2 ;;
        --timeout)       TIMEOUT="$2";           shift 2 ;;
        --agent-config)  AGENT_CONFIG="$2";      shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";        shift 2 ;;
        --repetitions)   REPETITIONS="$2";       shift 2 ;;
        --debug)         DEBUG_MODE="--debug";   shift   ;;
        --list)          LIST_SCENARIOS=1;       shift   ;;
        -h|--help)       show_help; exit 0       ;;
        *) die "Unknown option: $1. Use --help for usage." ;;
    esac
done

# ---------------------------------------------------------------------------
# List scenarios
# ---------------------------------------------------------------------------
if [[ "$LIST_SCENARIOS" -eq 1 ]]; then
    info "Querying scenario_runner for available scenarios..."
    docker exec -u "$CONTAINER_USER" "$CONTAINER" \
        python3 "$SR_ROOT/scenario_runner.py" --list 2>/dev/null
    exit 0
fi

# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------
if [[ -z "$SCENARIO_NAME" && -z "$ROUTE_NAME" && "$RUN_ALL" -eq 0 ]]; then
    echo "No mode specified. Use one of:"
    echo "  --scenario NAME     (single scenario)"
    echo "  --route NAME        (route-based leaderboard)"
    echo "  --run-all           (all predefined scenarios)"
    echo "  --list              (list available scenarios)"
    echo ""
    show_help
    exit 1
fi

# Verify container is running
docker inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null | grep -q true \
    || die "Container '$CONTAINER' is not running. Start it first."

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
AGENT_PY="${SHARC_ROOT}/examples/MPC_example/sharc_mpc_agent.py"
if [[ -z "$AGENT_CONFIG" ]]; then
    AGENT_CONFIG="${SHARC_ROOT}/examples/MPC_example/sharc_agent_config.json"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${SHARC_ROOT}/examples/MPC_example/scenario_runner_results"
fi

# Resolve route file shortnames
resolve_route() {
    local route="$1"
    case "$route" in
        devtest)    echo "${SR_ROOT}/srunner/data/routes_devtest.xml" ;;
        training)   echo "${SR_ROOT}/srunner/data/routes_training.xml" ;;
        validation) echo "${SR_ROOT}/srunner/data/routes_validation.xml" ;;
        town10)     echo "${SR_ROOT}/srunner/data/routes_town10.xml" ;;
        /*) echo "$route" ;;  # absolute path
        *)  echo "${SR_ROOT}/srunner/data/${route}" ;;
    esac
}

# ============================================================================
# Core: run a single scenario on a specific CARLA instance
# ============================================================================
run_single_scenario() {
    local scenario="$1"
    local carla_port="$2"
    local tm_port="$3"
    local run_id="$4"
    local is_route="${5:-0}"
    local route_file="${6:-}"
    local route_id="${7:-0}"

    local result_dir="${OUTPUT_DIR}/${run_id}"

    info "[${run_id}] Starting scenario: ${scenario:-route} on port ${carla_port}"

    # Check that 3 consecutive CARLA ports are free
    for _p in ${carla_port} $((carla_port+1)) $((carla_port+2)); do
        if docker exec -u "$CONTAINER_USER" "$CONTAINER" \
            python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${_p})); s.close()" 2>/dev/null; then
            die "Port ${_p} already in use. Use --port with >=10 spacing."
        fi
    done

    # Run everything inside the container
    docker exec -i \
        -u "$CONTAINER_USER" \
        -e CARLA_PORT="$carla_port" \
        -e TM_PORT="$tm_port" \
        -e SCENARIO_NAME="${scenario}" \
        -e IS_ROUTE="${is_route}" \
        -e ROUTE_FILE="${route_file}" \
        -e ROUTE_ID_VAL="${route_id}" \
        -e RECORD_VIDEO="$RECORD_VIDEO" \
        -e HIGHRES="$HIGHRES" \
        -e TIMEOUT="$TIMEOUT" \
        -e DEBUG_MODE="${DEBUG_MODE}" \
        -e RUN_ID="${run_id}" \
        -e RESULT_DIR="${result_dir}" \
        -e AGENT_PY="${AGENT_PY}" \
        -e AGENT_CONFIG_PATH="${AGENT_CONFIG}" \
        -e CARLA_ROOT="${CARLA_ROOT}" \
        -e SHARC_ROOT="${SHARC_ROOT}" \
        -e SR_ROOT="${SR_ROOT}" \
        -e REPETITIONS="${REPETITIONS}" \
        "$CONTAINER" bash -s <<'INNER_SCRIPT'

set -euo pipefail

# --- Setup environment ---
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true

export SCENARIO_RUNNER_ROOT="${SR_ROOT}"
export PYTHONPATH="${SR_ROOT}:${CARLA_ROOT}/PythonAPI/carla:${SHARC_ROOT}/resources:${PYTHONPATH:-}"
export SHARC_AGENT_OUTPUT_DIR="${RESULT_DIR}"

mkdir -p "${RESULT_DIR}"
CARLA_LOG="${RESULT_DIR}/carla_server.log"

# --- [1/4] Start CARLA ---
echo "===> [1/4] Starting CARLA on port ${CARLA_PORT}..."
if [ "${RECORD_VIDEO}" = "1" ]; then
    xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24 +extension GLX" \
        "${CARLA_ROOT}/CarlaUE4.sh" -RenderOffScreen -nosound \
        -carla-rpc-port="${CARLA_PORT}" > "${CARLA_LOG}" 2>&1 &
else
    DISPLAY= "${CARLA_ROOT}/CarlaUE4.sh" -nullrhi -RenderOffScreen -nosound \
        -carla-rpc-port="${CARLA_PORT}" > "${CARLA_LOG}" 2>&1 &
fi
CARLA_PID=$!
trap 'kill "$CARLA_PID" 2>/dev/null || true; wait "$CARLA_PID" 2>/dev/null || true' EXIT

# Wait for CARLA
elapsed=0
until python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',${CARLA_PORT})); s.close()" 2>/dev/null; do
    if [ $elapsed -ge $TIMEOUT ]; then
        echo "ERROR: CARLA failed to start within ${TIMEOUT}s"
        tail -40 "${CARLA_LOG}"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed+2))
done
echo "===> CARLA ready on port ${CARLA_PORT} (${elapsed}s)"

# --- [2/4] Build scenario_runner command ---
echo "===> [2/4] Building scenario_runner command..."
SR_CMD="python3 ${SR_ROOT}/scenario_runner.py"
SR_CMD="${SR_CMD} --port ${CARLA_PORT}"
SR_CMD="${SR_CMD} --trafficManagerPort ${TM_PORT}"
SR_CMD="${SR_CMD} --sync"
SR_CMD="${SR_CMD} --output --json"
SR_CMD="${SR_CMD} --outputDir ${RESULT_DIR}"
SR_CMD="${SR_CMD} --reloadWorld"
SR_CMD="${SR_CMD} --repetitions ${REPETITIONS}"

if [ -n "${DEBUG_MODE}" ]; then
    SR_CMD="${SR_CMD} --debug"
fi

if [ "${IS_ROUTE}" = "1" ]; then
    SR_CMD="${SR_CMD} --route ${ROUTE_FILE} --route-id ${ROUTE_ID_VAL}"
    SR_CMD="${SR_CMD} --agent ${AGENT_PY}"
    SR_CMD="${SR_CMD} --agentConfig ${AGENT_CONFIG_PATH}"
elif [ -n "${SCENARIO_NAME}" ]; then
    SR_CMD="${SR_CMD} --scenario ${SCENARIO_NAME}"
    SR_CMD="${SR_CMD} --agent ${AGENT_PY}"
    SR_CMD="${SR_CMD} --agentConfig ${AGENT_CONFIG_PATH}"
fi

# --- [3/4] Run scenario ---
echo "===> [3/4] Running: ${SR_CMD}"
echo "     Output dir: ${RESULT_DIR}"

eval "${SR_CMD}" 2>&1 | tee "${RESULT_DIR}/scenario_runner.log"
SR_EXIT=${PIPESTATUS[0]}

echo "===> scenario_runner exited with code ${SR_EXIT}"

# --- [4/4] Summary ---
echo "===> [4/4] Results in: ${RESULT_DIR}"
if [ -f "${RESULT_DIR}/agent_results.json" ]; then
    python3 -c "
import json, sys
with open('${RESULT_DIR}/agent_results.json') as f:
    data = json.load(f)
s = data.get('summary', {})
print(f\"  Steps:           {data['total_steps']}\")
print(f\"  Mean comp time:  {s.get('mean_computation_ms', 0):.1f} ms\")
print(f\"  Max comp time:   {s.get('max_computation_ms', 0):.1f} ms\")
print(f\"  Infeasible:      {s.get('total_infeasible', 0)}\")
print(f\"  Deadline misses: {s.get('deadline_misses', 0)}\")
"
fi

ls -la "${RESULT_DIR}/"
echo "===> Done: ${RUN_ID}"

INNER_SCRIPT

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        info "[${run_id}] SUCCESS"
    else
        info "[${run_id}] FAILED (exit code $exit_code)"
    fi
    return $exit_code
}

# ============================================================================
# Execution modes
# ============================================================================

if [[ -n "$SCENARIO_NAME" ]]; then
    # --- Single scenario mode ---
    run_id="${SCENARIO_NAME}_$(date +%Y%m%d_%H%M%S)"
    run_single_scenario "$SCENARIO_NAME" "$BASE_PORT" "$BASE_TM_PORT" "$run_id"

elif [[ -n "$ROUTE_NAME" ]]; then
    # --- Route-based mode ---
    route_file=$(resolve_route "$ROUTE_NAME")
    run_id="route_${ROUTE_NAME}_${ROUTE_ID}_$(date +%Y%m%d_%H%M%S)"
    run_single_scenario "" "$BASE_PORT" "$BASE_TM_PORT" "$run_id" \
        1 "$route_file" "$ROUTE_ID"

elif [[ "$RUN_ALL" -eq 1 ]]; then
    # --- Run all scenarios ---
    info "Running ${#ALL_SCENARIOS[@]} leaderboard scenarios"
    info "Parallel instances: ${PARALLEL}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PASS=0
    FAIL=0
    TOTAL=${#ALL_SCENARIOS[@]}

    if [[ "$PARALLEL" -le 1 ]]; then
        # Sequential execution
        for scenario in "${ALL_SCENARIOS[@]}"; do
            run_id="${scenario}_${TIMESTAMP}"
            if run_single_scenario "$scenario" "$BASE_PORT" "$BASE_TM_PORT" "$run_id"; then
                ((PASS++))
            else
                ((FAIL++))
            fi
        done
    else
        # Parallel execution with N CARLA instances
        # Create a job queue
        JOB_IDX=0
        PIDS=()
        SCENARIOS_RUNNING=()

        for scenario in "${ALL_SCENARIOS[@]}"; do
            # Wait for a slot if all instances are busy
            while [[ ${#PIDS[@]} -ge $PARALLEL ]]; do
                # Wait for any child to finish
                DONE_PID=""
                for i in "${!PIDS[@]}"; do
                    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                        wait "${PIDS[$i]}" && ((PASS++)) || ((FAIL++))
                        info "Finished: ${SCENARIOS_RUNNING[$i]}"
                        unset 'PIDS[i]'
                        unset 'SCENARIOS_RUNNING[i]'
                        DONE_PID="yes"
                        break
                    fi
                done
                if [[ -z "$DONE_PID" ]]; then
                    sleep 2
                fi
                # Re-index arrays
                PIDS=("${PIDS[@]}")
                SCENARIOS_RUNNING=("${SCENARIOS_RUNNING[@]}")
            done

            # Launch the scenario on the next available port
            slot=${#PIDS[@]}
            port=$((BASE_PORT + slot * 10))
            tm_port=$((BASE_TM_PORT + slot * 10))
            run_id="${scenario}_${TIMESTAMP}"

            info "Launching ${scenario} on port ${port} (slot ${slot})"
            run_single_scenario "$scenario" "$port" "$tm_port" "$run_id" &
            PIDS+=($!)
            SCENARIOS_RUNNING+=("$scenario")
            ((JOB_IDX++))

            # Small delay between launches to avoid port conflicts
            sleep 5
        done

        # Wait for remaining jobs
        for i in "${!PIDS[@]}"; do
            wait "${PIDS[$i]}" && ((PASS++)) || ((FAIL++))
            info "Finished: ${SCENARIOS_RUNNING[$i]}"
        done
    fi

    # Summary
    echo ""
    echo "============================================"
    echo "  LEADERBOARD SCENARIO RESULTS"
    echo "============================================"
    echo "  Total:  ${TOTAL}"
    echo "  Passed: ${PASS}"
    echo "  Failed: ${FAIL}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "============================================"
fi
