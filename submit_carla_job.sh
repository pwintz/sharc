#!/bin/bash
#SBATCH --job-name=sharc_carla
#SBATCH --account=fc_control
#SBATCH --partition=savio4_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/sharc_carla_%A_%a.out

# CARLA + SHARC Job Submission Script for SLURM with Apptainer
#
# Usage:
#   sbatch submit_carla_job.sh                          # Run default config
#   sbatch --array=0-4 submit_carla_job.sh              # Run all 5 test scenarios
#   CONFIG=my_config.json sbatch submit_carla_job.sh    # Custom config
#   VIDEO=1 sbatch submit_carla_job.sh                  # Enable video recording
#
# Prerequisites:
#   1. Build SIF:  apptainer build carla-sharc.sif docker-daemon://carla-sharc
#      or pull:    apptainer pull carla-sharc.sif docker://yourregistry/carla-sharc:latest
#   2. Ensure logs/ directory exists:  
mkdir -p logs
set -euo pipefail

# Parse KEY=VALUE arguments (e.g., ./submit_carla_job.sh VIDEO=1 CONFIG=foo.json)
for arg in "$@"; do
    if [[ "$arg" =~ ^([A-Z_]+)=(.*)$ ]]; then
        export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
    fi
done

# ── Configuration ─────────────────────────────────────────────────────────────
SIF_FILE="${SIF_FILE:-carla-sharc.sif}"
EXAMPLE="${EXAMPLE:-MPC_example}"

# Config files for array jobs (edit as needed)
CONFIG_FILES=(
    "lead_vehicle_safe.json"
    "test_lead_vehicle_serial.json"
    "test_lead_vehicle_parallel.json"
    "test_obstacle_serial.json"
    "test_obstacle_parallel.json"
)

# Select config: array job uses SLURM_ARRAY_TASK_ID, single job uses CONFIG or first entry
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CONFIG="${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}"
    echo "Array task $SLURM_ARRAY_TASK_ID: $CONFIG"
elif [[ -n "${CONFIG:-}" ]]; then
    echo "Using CONFIG=$CONFIG"
else
    CONFIG="${CONFIG_FILES[0]}"
    echo "Using default config: $CONFIG"
fi

# ── Environment ───────────────────────────────────────────────────────────────
module load apptainer 2>/dev/null || true

mkdir -p logs

echo "======================================"
echo "CARLA + SHARC Experiment"
echo "======================================"
echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-$(hostname)}"
echo "SIF:     $SIF_FILE"
echo "Example: $EXAMPLE"
echo "Config:  $CONFIG"
echo "======================================"
echo ""

if [[ ! -f "$SIF_FILE" ]]; then
    echo "ERROR: SIF file '$SIF_FILE' not found."
    echo "Build it first:  apptainer build $SIF_FILE docker-daemon://carla-sharc"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
VIDEO_FLAG=""
if [[ "${VIDEO:-0}" == "1" ]]; then
    VIDEO_FLAG="--video"
fi

./run_experiment_apptainer.sh \
    --sif "$SIF_FILE" \
    --example "$EXAMPLE" \
    --config "$CONFIG" \
    --timeout 600 \
    $VIDEO_FLAG

EXIT_CODE=$?

echo ""
echo "======================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS  $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "FAILED (exit $EXIT_CODE)  $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "======================================"

exit $EXIT_CODE
