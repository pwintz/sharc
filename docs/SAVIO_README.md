# Running SHARC on HPC with Apptainer

This guide explains how to run SHARC simulations on an HPC cluster using Apptainer (formerly Singularity).

## Quick Start

```bash
# 1. Clone repository and navigate to it
git clone git@github.com:pwintz/sharc.git && cd sharc

# 2. Load Apptainer module on Savio
module load apptainer

# 3. Setup (pull Docker image and convert to Apptainer)
chmod +x setup_sharc_apptainer.sh
./setup_sharc_apptainer.sh

# 4. Submit job to Savio
sbatch submit_job.sh
```

## Detailed Instructions

### 1. Initial Setup on Savio

First, connect to Savio and prepare the environment:

```bash
# SSH to Savio
ssh username@hpc.brc.berkeley.edu

# Navigate to your project directory
cd /global/scratch/users/$USER/

# Clone SHARC
git clone git@github.com:pwintz/sharc.git
cd sharc

# Load Apptainer module
module load apptainer
```

### 2. Create Apptainer Image

The `setup_sharc_apptainer.sh` script pulls the SHARC Docker image and converts it to Apptainer format:

```bash
./setup_sharc_apptainer.sh
```

This creates `sharc_latest.sif` (~5GB) in your current directory.

### 3. Test Locally (Optional)

Before submitting jobs, test that the image works:

```bash
# Run ACC example with default config
./run_example_apptainer.sh acc_example default.json

# Run cartpole example
./run_example_apptainer.sh cartpole parallel_08.json
```

### 4. Submit Jobs

#### Single Job

Edit `submit_job.sh` to set your desired configuration, then submit:

```bash
sbatch submit_job.sh
```

#### Array Jobs

Submit multiple configurations simultaneously:

```bash
# Submit array job with 4 different configs (0-3)
sbatch --array=0-3 submit_job.sh
```

### 5. Monitor Jobs and View Logs

```bash
# Check job status
squeue -u $USER

# View latest log files (sorted by time)
ls -lt logs/

# View most recent log
ls -t logs/ | head -1 | xargs -I {} cat logs/{}

# View specific job output (use tab completion)
cat logs/2024-10-04_15-30-45_sharc_12345_0.out

```

### 6. Retrieve Results

Results are saved to `examples/[example_name]/experiments/`:

```bash
# List experiment results
ls -lh examples/cartpole/experiments/

# Copy results to your local machine (from your local terminal)
scp -r username@dtn.brc.berkeley.edu:/path/to/sharc/examples/ ./
```

## Configuration Files

### `submit_job.sh`

Key parameters to customize:

```bash
#SBATCH --job-name=sharc_sim      # Job name
#SBATCH --account=fc_control      # Your Savio account
#SBATCH --partition=savio4_htc    # Partition (adjust as needed)
#SBATCH --cpus-per-task=16        # Number of CPUs
#SBATCH --time=72:00:00           # Max runtime

EXAMPLE="cartpole"                # Example to run
CONFIG_FILES=(...)                # Array of config files
```

### Directory Bindings

The scripts bind two directories into the container:
- `resources/` → `/home/dcuser/resources` (SHARC code)
- `examples/` → `/examples` (example projects)

Changes to files in these directories persist after the job completes.

## Example Configurations

### Run ACC Example in Serial Mode

```bash
# In submit_job.sh, set:
EXAMPLE="acc_example"
CONFIG_FILES=("serial.json")

# Submit single job
sbatch submit_job.sh
```

## Differences from Docker

| Docker | Apptainer |
|--------|-----------|
| `docker pull` | `apptainer pull docker://` |
| `docker run` | `apptainer exec` |
| `-v host:container` | `--bind host:container` |
| Root by default | Non-root by default |
