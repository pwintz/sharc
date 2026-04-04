# Running CARLA + SHARC on SLURM via Apptainer

## Prerequisites

Set cache/tmp dirs to scratch (required on Savio — home quota is too small):

```bash
export APPTAINER_CACHEDIR=/global/scratch/users/$USER/apptainer/cache
export APPTAINER_TMPDIR=/global/scratch/users/$USER/apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
```

## Setup

```bash
git clone https://github.com/pwintz/sharc.git -b batch-replay-determinism
cd sharc
apptainer pull carla-sharc.sif docker://ausar/carla-sharc:latest
```

## Run

```bash
# Default experiment (lead_vehicle_safe.json, no video)
./submit_carla_job.sh

# With video recording (uses GPU rendering via Xvfb)
./submit_carla_job.sh VIDEO=1

# Custom config
./submit_carla_job.sh CONFIG=test_obstacle_serial.json

# Submit via SLURM
sbatch submit_carla_job.sh
VIDEO=1 sbatch submit_carla_job.sh

# Array job (runs CONFIG_FILES[0..4] in parallel)
sbatch --array=0-4 submit_carla_job.sh
```

## Notes

- **No video** (default): CARLA runs with `-nullrhi` — CPU physics only, no GPU, faster startup.
- **With video** (`VIDEO=1`): CARLA uses GPU rendering via Xvfb. Requires `--gres=gpu:1` (already set in `#SBATCH` headers).
- Results are saved to `examples/MPC_example/experiments/`.
- Logs go to `logs/sharc_carla_<jobid>_<arrayid>.out`.
- The SIF file (~70GB) is stored locally; only needs to be pulled once per cluster.
