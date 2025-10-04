#!/bin/bash
#SBATCH --job-name=sharc_sim
#SBATCH --account=fc_control
#SBATCH --partition=savio4_htc
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/sharc_%A_%a_%x.out

# SHARC Job Submission Script for Savio
# Usage: sbatch submit_job.sh
# Usage with array: sbatch --array=0-4 submit_job.sh
# 
# Logs are stored in logs/ directory:
#   %A = job ID
#   %a = array task ID (if array job)
#   %x = job name
#
# Timestamp is added to log filename and printed in output

# Create logs directory if it doesn't exist
mkdir -p logs

# Create timestamp for this job
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Rename log file to include timestamp (move from generic name to timestamped name)
# This will be executed after SLURM creates the initial log file
if [ -n "$SLURM_JOB_ID" ]; then
    ORIGINAL_LOG="logs/sharc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_NAME}.out"
    TIMESTAMPED_LOG="logs/${TIMESTAMP}_sharc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    # Move will happen at the end of the script
fi

# ============ Configuration ============
# Apptainer image (must exist in current directory)
IMAGE="sharc_latest.sif"

# Example to run (acc_example or cartpole)
EXAMPLE="acc_example"

# Configuration files array (edit as needed)
CONFIG_FILES=(
    "default.json"
)

# For array jobs, select config based on task ID
# For single jobs, use first config
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
    echo "Array job $SLURM_ARRAY_TASK_ID: Using config $CONFIG_FILE"
else
    CONFIG_FILE=${CONFIG_FILES[0]}
    echo "Single job: Using config $CONFIG_FILE"
fi

# ============ Job Execution ============
echo "======================================"
echo "SHARC Simulation Job"
echo "======================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Example: $EXAMPLE"
echo "Config: $CONFIG_FILE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "======================================"
echo ""

# Run simulation with Apptainer
apptainer exec \
    --bind "$(pwd)/resources:/home/dcuser/resources" \
    --bind "$(pwd)/examples:/examples" \
    "$IMAGE" \
    bash -c "cd /examples/$EXAMPLE && sharc --config_filename $CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Simulation completed successfully"
else
    echo "✗ Simulation failed with exit code $EXIT_CODE"
fi
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

# Rename log file to include timestamp
if [ -n "$SLURM_JOB_ID" ] && [ -f "$ORIGINAL_LOG" ]; then
    mv "$ORIGINAL_LOG" "$TIMESTAMPED_LOG"
    echo "Log saved to: $TIMESTAMPED_LOG"
fi

exit $EXIT_CODE

