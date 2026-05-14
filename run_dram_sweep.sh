#!/bin/bash
set -euo pipefail

EXAMPLE="MPC_example"
CONFIG="ped_dist_2500ps.json"

EXAMPLE_DIR="examples/${EXAMPLE}"
CHIP_DIR="${EXAMPLE_DIR}/chip_configs"
LOG_DIR="dram_sweep_logs"

mkdir -p "$LOG_DIR"

TESTS=(
#   HBM
#   high_bw
#   high_latency
#   large_queue
#   low_latency
#   multi_channel
#   multi_rank
#   small_queue
#   LPDDR4
    low_bw
    even_higher_latency
)

PARAMS=(
#   PARAMS.base_HBM   
#   PARAMS.base_high_bw
#   PARAMS.base_high_latency
#   PARAMS.base_large_queue
#   PARAMS.base_low_latency
#   PARAMS.base_multi_channel
#   PARAMS.base_multi_rank
#   PARAMS.base_small_queue
#   PARAMS.base_LPDDR4
    PARAMS.base_low_bw
    PARAMS.base_even_higher_latency
)

if [ "${#TESTS[@]}" -ne "${#PARAMS[@]}" ]; then
    echo "ERROR: TESTS/PARAMS length mismatch"
    exit 1
fi

cp "${CHIP_DIR}/PARAMS.base" "${CHIP_DIR}/PARAMS.base.backup"

restore_params() {
    cp "${CHIP_DIR}/PARAMS.base.backup" "${CHIP_DIR}/PARAMS.base"
}
trap restore_params EXIT

for i in "${!TESTS[@]}"; do
    name="${TESTS[$i]}"
    param="${PARAMS[$i]}"

    echo "========================================"
    echo "Running $((i+1))/${#TESTS[@]}: $name"
    echo "Using PARAMS file: $param"
    echo "========================================"

    cp "${CHIP_DIR}/${param}" "${CHIP_DIR}/PARAMS.base"

    if ./run_offscreen_experiment.sh \
        --example "$EXAMPLE" \
        --config "$CONFIG" \
        --log "${LOG_DIR}/${name}.log" \
        --video; then

        echo "PASS: $name" | tee -a "${LOG_DIR}/summary.log"
    else
        echo "FAIL: $name" | tee -a "${LOG_DIR}/summary.log"
        echo "Continuing to next test..."
    fi

    sleep 5
done

echo "All DRAM sweep tests finished."