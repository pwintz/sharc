# CARLA + SHARC Docker Environment

A combined Docker environment with CARLA 0.9.16 simulator and SHARC tools.

## Directory Structure

```
/home/workspace/
├── carla_0.9.16/          # CARLA simulator
│   ├── CarlaUE4.sh        # CARLA server
│   └── PythonAPI/         # Python API & examples
├── sharc/                 # SHARC tools
│   ├── resources/
│   └── examples/
└── my_files/              # Your mounted files (edit from VS Code)
```

---

## Quick Start

```bash
# 0. Activate docker group (to avoid sudo)
newgrp docker

# 1. Build the image (one time)
docker build \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -f Dockerfile -t carla-sharc .

# 2. Start the container
./run_carla_sharc.sh

# 3. Inside container: start CARLA server
/home/workspace/carla_0.9.16/CarlaUE4.sh -prefernvidia &

# 4. Run examples
python /home/workspace/carla_0.9.16/PythonAPI/examples/automatic_control.py
```

---

## Headless Experiment Runner (no display required)

Run a full CARLA + SHARC experiment from the **host machine** without a monitor.
The script starts CARLA in headless mode (`-nullrhi`), waits for it to become
ready, runs the SHARC experiment, and saves a dashboard PNG to the results folder.

```bash
# Default: MPC_example with obstacle_constraint.json
./run_offscreen_experiment.sh

# Specific example / config
./run_offscreen_experiment.sh --example MPC_example --config obstacle_constraint.json

# All options
./run_offscreen_experiment.sh --help
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--example NAME` | `MPC_example` | Example directory name |
| `--config FILE` | `obstacle_constraint.json` | Simulation config filename |
| `--container NAME` | `carla-sharc-yasin5` | Docker container name |
| `--user NAME` | `admin` | Container user to run as |
| `--timeout SECS` | `180` | CARLA startup timeout |
| `--log FILE` | _(none)_ | Tee all output to a file on the host |

**Output:** results are written to
`examples/<EXAMPLE>/experiments/<timestamp>--<label>/`,
including `experiment_data.json` and `dashboard_final.png`.

---

## Running the Container

**First, activate docker group (to avoid sudo):**
```bash
newgrp docker
```

Then run:
```bash
./run_carla_sharc.sh
```

Or manually:

```bash
xhost +local:docker

docker run -it --rm \
    --gpus all \
    --privileged \
    --network host \
    --name carla-sharc \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/my_workspace:/home/workspace/my_files \
    carla-sharc
```

---

## Editing Files from VS Code

1. **Run container** (creates `./workspace` folder):
   ```bash
   ./run_carla_sharc.sh
   ```

2. **Open VS Code** on your host:
   ```bash
   code ./workspace
   ```

3. **Edit files** → changes appear instantly in container at `/home/workspace/my_files/`

---

## Running CARLA

### Start CARLA Server

```bash
# Default (with GPU support)
/home/workspace/carla_0.9.16/CarlaUE4.sh -prefernvidia

# Or with options:
/home/workspace/carla_0.9.16/CarlaUE4.sh -prefernvidia -quality-level=Low
/home/workspace/carla_0.9.16/CarlaUE4.sh -prefernvidia -RenderOffScreen
```

### Run Python Examples

Open a **second terminal**:

```bash
docker exec -it carla-sharc bash

python /home/workspace/carla_0.9.16/PythonAPI/examples/automatic_control.py
python /home/workspace/carla_0.9.16/PythonAPI/examples/manual_control.py
python /home/workspace/carla_0.9.16/PythonAPI/examples/generate_traffic.py
```

### Your Own Scripts

```python
# Save to ./workspace/my_script.py on host
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()
print(f"Map: {world.get_map().name}")
```

Run it:
```bash
python /home/workspace/my_files/my_script.py
```

---

## Running SHARC

```bash
cd /home/workspace/sharc/examples/acc_example
python controller_delegator.py
```

---

## Key Paths

| Path | Description |
|------|-------------|
| `/home/workspace/carla_0.9.16/` | CARLA simulator |
| `/home/workspace/carla_0.9.16/PythonAPI/examples/` | CARLA examples |
| `/home/workspace/sharc/` | SHARC tools & examples |
| `/home/workspace/my_files/` | Your mounted files |
| `/opt/tools/` | Scarab, PIN, DynamoRIO |

---

## Troubleshooting

### GUI not working
```bash
xhost +local:docker
```

### GPU not being used (slow performance, nvidia-smi shows 0%)

**Problem:** CARLA is using software rendering instead of GPU.

**Solution 1: Use prefernvidia flag (recommended)**
```bash
/home/workspace/carla_0.9.16/CarlaUE4.sh -prefernvidia
```

**Solution 2: Headless mode (no rendering, fastest)**
```bash
/home/workspace/carla_0.9.16/CarlaUE4.sh -RenderOffScreen -nullrhi
```

**Verify GPU usage:**
```bash
# In another terminal
docker exec -it carla-sharc nvidia-smi
# Should show GPU usage when CARLA is running
```

### CARLA server crashes
```bash
/home/workspace/carla_0.9.16/CarlaUE4.sh -quality-level=Low -ResX=800 -ResY=600 -prefernvidia
```

### Cannot connect to CARLA
```bash
sleep 10  # Wait for server to start
python your_script.py
```
