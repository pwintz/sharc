#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Run ALL SHARC MPC leaderboard scenarios (parallel + serial) with
# video recording, then compare trajectories.
#
# Uses run_offscreen_experiment.sh for each config, which handles the
# full CARLA lifecycle (start → warmup → experiment → video → stop).
#
# Usage (from repo root):
#   ./run_all_leaderboard.sh                    # All 5 scenarios, both modes
#   ./run_all_leaderboard.sh --scenario dense_traffic   # One scenario only
#   ./run_all_leaderboard.sh --no-video         # Skip video recording
#   ./run_all_leaderboard.sh --parallel-only    # Skip serial runs
#   ./run_all_leaderboard.sh --serial-only      # Skip parallel runs
#   ./run_all_leaderboard.sh --timeout 300      # Custom CARLA timeout
#
# Results are saved under examples/MPC_example/experiments/ with
# timestamped directories.  A summary table is printed at the end.
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_offscreen_experiment.sh"
EXAMPLE_DIR="${SCRIPT_DIR}/examples/MPC_example"
COMPARE_SCRIPT="${EXAMPLE_DIR}/compare_trajectories.py"
RESULTS_LOG="${EXAMPLE_DIR}/experiments/leaderboard_run_$(date +%Y%m%d_%H%M%S).log"

# ── Scenarios (base name → parallel config, serial config) ─────────
SCENARIOS=(
    dense_traffic
    high_speed_avoidance
    lead_vehicle_braking
    pedestrian_crossing
    vehicle_cutin
)

# ── Default options ────────────────────────────────────────────────
VIDEO_FLAG="--video"
RUN_PARALLEL=1
RUN_SERIAL=1
TIMEOUT=300
FILTER_SCENARIO=""

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --scenario)       FILTER_SCENARIO="$2"; shift 2 ;;
        --no-video)       VIDEO_FLAG="";         shift ;;
        --video)          VIDEO_FLAG="--video";  shift ;;
        --parallel-only)  RUN_SERIAL=0;          shift ;;
        --serial-only)    RUN_PARALLEL=0;        shift ;;
        --timeout)        TIMEOUT="$2";          shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[ -x "$RUN_SCRIPT" ] || die "Cannot find run_offscreen_experiment.sh at $RUN_SCRIPT"

mkdir -p "${EXAMPLE_DIR}/experiments"

# ── Logging ────────────────────────────────────────────────────────
log() { echo "$(date '+%H:%M:%S') | $*" | tee -a "$RESULTS_LOG"; }

log "═══════════════════════════════════════════════════════════════"
log " SHARC Leaderboard — Full Run"
log " Video: $([ -n "$VIDEO_FLAG" ] && echo ENABLED || echo disabled)"
log " Parallel: $([ $RUN_PARALLEL -eq 1 ] && echo YES || echo no)"
log " Serial:   $([ $RUN_SERIAL -eq 1 ] && echo YES || echo no)"
log " Timeout:  ${TIMEOUT}s"
log " Log:      ${RESULTS_LOG}"
log "═══════════════════════════════════════════════════════════════"
echo ""

# Track experiment directories for comparison
declare -A PARALLEL_DIRS
declare -A SERIAL_DIRS
FAIL_COUNT=0
PASS_COUNT=0

find_latest_experiment() {
    # Find the most recent experiment directory matching a label slug
    local label_slug="$1"
    ls -dt "${EXAMPLE_DIR}/experiments/"*"--${label_slug}" 2>/dev/null | head -1
}

get_config_slug() {
    # Get the experiment dir slug for a config file: config filename minus .json, spaces/_/caps → dashes
    local config="$1"
    local base="${config%.json}"
    echo "$base" | tr ' _' '-' | tr '[:upper:]' '[:lower:]'
}

run_one() {
    local config="$1"
    local mode="$2"
    local scenario="$3"
    log ""
    log "─── Running: ${config} (${mode}) ───"
    local start_time=$SECONDS

    # Run experiment; ignore exit code (CARLA cleanup returns 143)
    "$RUN_SCRIPT" --config "$config" $VIDEO_FLAG --timeout "$TIMEOUT" 2>&1 | tee -a "$RESULTS_LOG" || true

    local elapsed=$(( SECONDS - start_time ))

    # Check success by finding the experiment directory with experiment_data.json
    local slug
    slug=$(get_config_slug "$config")
    local exp_dir
    exp_dir=$(find_latest_experiment "$slug")

    if [ -n "$exp_dir" ] && find "$exp_dir" -name "experiment_data.json" -print -quit 2>/dev/null | grep -q .; then
        log "✓ ${config} completed in ${elapsed}s"
        log "  Results: $exp_dir"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
        if [ "$mode" = "parallel" ]; then
            PARALLEL_DIRS["$scenario"]="$exp_dir"
        else
            SERIAL_DIRS["$scenario"]="$exp_dir"
        fi
    else
        log "✗ ${config} FAILED after ${elapsed}s (no experiment_data.json found)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
}

# ── Run all scenarios ────────────────────────────────────────────
for scenario in "${SCENARIOS[@]}"; do
    # Filter if requested
    if [ -n "$FILTER_SCENARIO" ] && [ "$scenario" != "$FILTER_SCENARIO" ]; then
        continue
    fi

    log ""
    log "══════════════════════════════════════════════════════════"
    log " Scenario: ${scenario}"
    log "══════════════════════════════════════════════════════════"

    parallel_config="leaderboard_${scenario}.json"
    serial_config="leaderboard_${scenario}_serial.json"

    if [ $RUN_PARALLEL -eq 1 ]; then
        if [ -f "${EXAMPLE_DIR}/simulation_configs/${parallel_config}" ]; then
            run_one "$parallel_config" "parallel" "$scenario"
        else
            log "WARNING: ${parallel_config} not found, skipping"
        fi
    fi

    if [ $RUN_SERIAL -eq 1 ]; then
        if [ -f "${EXAMPLE_DIR}/simulation_configs/${serial_config}" ]; then
            run_one "$serial_config" "serial" "$scenario"
        else
            log "WARNING: ${serial_config} not found, skipping"
        fi
    fi
done

# ── Compare parallel vs serial ──────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════════"
log " Trajectory Comparisons (parallel vs serial)"
log "═══════════════════════════════════════════════════════════════"

if [ $RUN_PARALLEL -eq 1 ] && [ $RUN_SERIAL -eq 1 ]; then
    # Map host paths to container paths for comparison
    HOST_EXAMPLES="/home/meng111/Projects/Yasin/sharc/examples"
    CONTAINER_EXAMPLES="/home/workspace/sharc/examples"

    for scenario in "${SCENARIOS[@]}"; do
        if [ -n "$FILTER_SCENARIO" ] && [ "$scenario" != "$FILTER_SCENARIO" ]; then
            continue
        fi

        p_dir="${PARALLEL_DIRS[$scenario]:-}"
        s_dir="${SERIAL_DIRS[$scenario]:-}"

        if [ -n "$p_dir" ] && [ -n "$s_dir" ]; then
            log ""
            log "── ${scenario}: parallel vs serial ──"
            # Convert host paths to container paths
            p_container="${p_dir/${HOST_EXAMPLES}/${CONTAINER_EXAMPLES}}"
            s_container="${s_dir/${HOST_EXAMPLES}/${CONTAINER_EXAMPLES}}"
            docker exec -u admin carla-sharc-yasin5 bash -c \
                "cd /home/workspace/sharc/examples/MPC_example && \
                 python3 compare_trajectories.py '${p_container}' '${s_container}'" \
                2>&1 | tee -a "$RESULTS_LOG" || log "  (comparison failed)"
        else
            log ""
            log "── ${scenario}: SKIPPED (missing parallel or serial result)"
            [ -z "$p_dir" ] && log "  Missing: parallel"
            [ -z "$s_dir" ] && log "  Missing: serial"
        fi
    done
else
    log " (Skipped — need both parallel and serial runs for comparison)"
fi

# ── Summary ──────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════════"
log " SUMMARY"
log "═══════════════════════════════════════════════════════════════"
log " Passed: ${PASS_COUNT}"
log " Failed: ${FAIL_COUNT}"
log " Full log: ${RESULTS_LOG}"
log ""

for scenario in "${SCENARIOS[@]}"; do
    if [ -n "$FILTER_SCENARIO" ] && [ "$scenario" != "$FILTER_SCENARIO" ]; then
        continue
    fi
    p="${PARALLEL_DIRS[$scenario]:-MISSING}"
    s="${SERIAL_DIRS[$scenario]:-MISSING}"
    log " ${scenario}:"
    log "   parallel: $(basename "$p" 2>/dev/null || echo MISSING)"
    log "   serial:   $(basename "$s" 2>/dev/null || echo MISSING)"
done

log ""
log "═══════════════════════════════════════════════════════════════"
log " DONE"
log "═══════════════════════════════════════════════════════════════"

[ $FAIL_COUNT -eq 0 ] || exit 1
