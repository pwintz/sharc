#!/bin/bash
# CARLA + SHARC Experiment Runner via Apptainer
# Functionally equivalent to run_offscreen_experiment.sh but uses Apptainer
# instead of Docker.  Designed for SLURM/HPC systems.
#
# The script converts a Docker image to an Apptainer SIF (if needed), then
# runs the same experiment logic inside the Apptainer container.
#
# Usage:
#   ./run_experiment_apptainer.sh [--config FILE] [--sif FILE] [--image DOCKER_IMAGE]
#                                 [--example NAME] [--timeout SECS] [--video]
#
# SLURM usage:
#   sbatch --gres=gpu:1 --wrap="./run_experiment_apptainer.sh --config obstacle_constraint.json"
#
# Examples:
#   ./run_experiment_apptainer.sh --config test_lead_vehicle_serial.json
#   ./run_experiment_apptainer.sh --config obstacle_constraint.json --video
#   ./run_experiment_apptainer.sh --sif ./carla-sharc.sif --config test_obstacle_parallel.json

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SIF_FILE="carla-sharc.sif"
DOCKER_IMAGE="carla-sharc"
EXAMPLE_NAME="MPC_example"
CONFIG_NAME="obstacle_constraint.json"
CARLA_PORT=2000
TIMEOUT=300
RECORD_VIDEO=0
HIGHRES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sif)       SIF_FILE="$2";     shift 2 ;;
        --image)     DOCKER_IMAGE="$2"; shift 2 ;;
        --example)   EXAMPLE_NAME="$2"; shift 2 ;;
        --config)    CONFIG_NAME="$2";  shift 2 ;;
        --port)      CARLA_PORT="$2";   shift 2 ;;
        --timeout)   TIMEOUT="$2";      shift 2 ;;
        --video)     RECORD_VIDEO=1;    shift ;;
        --highres)   HIGHRES=1;         shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ── Check prerequisites ──────────────────────────────────────────────────────
command -v apptainer &>/dev/null || die "apptainer not found.  On SLURM: module load apptainer"

# ── Build SIF from Docker if needed ──────────────────────────────────────────
if [[ ! -f "$SIF_FILE" ]]; then
    echo "==> SIF file '$SIF_FILE' not found.  Building from Docker image '$DOCKER_IMAGE' ..."
    apptainer build "$SIF_FILE" "docker-daemon://$DOCKER_IMAGE"
    echo "==> SIF built: $SIF_FILE"
fi

echo "==> SIF:      $SIF_FILE"
echo "==> Example:  $EXAMPLE_NAME / $CONFIG_NAME"
echo "==> Video:    $([ $RECORD_VIDEO -eq 1 ] && echo 'ENABLED' || echo 'disabled')"
echo ""

# ── Detect host Vulkan ICD files ─────────────────────────────────────────────
# CARLA uses Vulkan for GPU rendering. Apptainer --nv handles CUDA/GLX but NOT
# the Vulkan ICD loader config. We must bind them from the host.
VULKAN_BINDS=""
VULKAN_ICD=""
for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d; do
    if [[ -d "$icd_dir" ]]; then
        VULKAN_BINDS="$VULKAN_BINDS --bind $icd_dir:$icd_dir"
    fi
done
# Find the NVIDIA Vulkan ICD file
for icd_file in /usr/share/vulkan/icd.d/nvidia_icd.json \
                /etc/vulkan/icd.d/nvidia_icd.json \
                /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json; do
    if [[ -f "$icd_file" ]]; then
        VULKAN_ICD="$icd_file"
        break
    fi
done

# Also bind EGL vendor files if present (for headless rendering)
EGL_BINDS=""
for egl_dir in /usr/share/glvnd/egl_vendor.d /usr/share/egl/egl_external_platform.d; do
    if [[ -d "$egl_dir" ]]; then
        EGL_BINDS="$EGL_BINDS --bind $egl_dir:$egl_dir"
    fi
done

# ── Run inside Apptainer ─────────────────────────────────────────────────────
# --nv               : NVIDIA GPU pass-through (CUDA + GLX libs)
# --writable-tmpfs   : allow writes to /tmp etc. without needing a writable overlay
# --bind             : mount local resources/examples so edits persist on the host
# --env              : set NVIDIA env vars for full GPU rendering (graphics + video)
#
# The inner script is identical to the CONTAINER_SCRIPT heredoc in
# run_offscreen_experiment.sh, ensuring functional equivalence.
# shellcheck disable=SC2086
apptainer exec \
    --nv \
    --writable-tmpfs \
    --env "NVIDIA_DRIVER_CAPABILITIES=all" \
    --env "NVIDIA_VISIBLE_DEVICES=all" \
    --env "__NV_PRIME_RENDER_OFFLOAD=1" \
    --env "__GLX_VENDOR_LIBRARY_NAME=nvidia" \
    ${VULKAN_ICD:+--env "VK_ICD_FILENAMES=$VULKAN_ICD"} \
    $VULKAN_BINDS \
    $EGL_BINDS \
    --bind "$SCRIPT_DIR/resources:/home/workspace/sharc/resources" \
    --bind "$SCRIPT_DIR/examples:/home/workspace/sharc/examples" \
    "$SIF_FILE" \
    bash -s <<CONTAINER_SCRIPT
set -euo pipefail

export _EXP_EXAMPLE="$EXAMPLE_NAME"
export _EXP_CONFIG="$CONFIG_NAME"
export _EXP_PORT="$CARLA_PORT"
export _EXP_TIMEOUT="$TIMEOUT"
export _EXP_VIDEO="$RECORD_VIDEO"
export _EXP_HIGHRES="$HIGHRES"

CARLA_ROOT=/home/workspace/carla_0.9.16
if [ ! -f "\$CARLA_ROOT/CarlaUE4.sh" ]; then
    CARLA_ROOT=/workspace
fi
SHARC_ROOT=/home/workspace/sharc
EXAMPLE_DIR="\${SHARC_ROOT}/examples/\${_EXP_EXAMPLE}"
CARLA_LOG="\${EXAMPLE_DIR}/carla_server.log"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true

echo "Running as: \$(whoami)"

# ═══════════════════════════════════════════════════════════════════════
# [0/5] GPU diagnostics
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== GPU Diagnostics ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        || echo "  nvidia-smi failed (GPU may not be accessible)"
else
    echo "  WARNING: nvidia-smi not found"
fi
# Check Vulkan availability
if command -v vulkaninfo &>/dev/null; then
    echo "  Vulkan: \$(vulkaninfo --summary 2>/dev/null | grep -i 'gpu' | head -1 || echo 'not available')"
else
    echo "  vulkaninfo not found (Vulkan may still work via ICD)"
fi
# Check if NVIDIA libraries are loadable
if ldconfig -p 2>/dev/null | grep -q libEGL_nvidia; then
    echo "  libEGL_nvidia: found"
else
    echo "  WARNING: libEGL_nvidia not in ldconfig"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════
# [1/5] Kill stale CARLA
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [1/5] Stopping any existing CARLA instance ==="
pkill -9 -f CarlaUE4 2>/dev/null || true
pkill -9 -f Xvfb 2>/dev/null || true
sleep 3

# ═══════════════════════════════════════════════════════════════════════
# [2/5] Start CARLA
# ═══════════════════════════════════════════════════════════════════════
echo ""
if [ "\${_EXP_VIDEO}" = "1" ]; then
    echo "=== [2/5] Starting CARLA (GPU via Xvfb — video recording enabled) ==="
    # Ensure Vulkan ICD is set for the CARLA process
    export VK_ICD_FILENAMES="\${VK_ICD_FILENAMES:-}"
    xvfb-run --auto-servernum --server-args="-screen 0 1920x1080x24 +extension GLX" \\
        "\$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound \\
        -carla-rpc-port="\${_EXP_PORT}" > "\$CARLA_LOG" 2>&1 &
else
    echo "=== [2/5] Starting CARLA (headless, -nullrhi) ==="
    DISPLAY= "\$CARLA_ROOT/CarlaUE4.sh" -nullrhi -RenderOffScreen -nosound \\
        -carla-rpc-port="\${_EXP_PORT}" > "\$CARLA_LOG" 2>&1 &
fi
CARLA_PID=\$!
echo "CARLA PID: \$CARLA_PID | log: \$CARLA_LOG"

trap 'echo "Stopping CARLA..."; kill "\$CARLA_PID" 2>/dev/null; pkill -f CarlaUE4 2>/dev/null; pkill -f Xvfb 2>/dev/null; wait "\$CARLA_PID" 2>/dev/null' EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════
# [3/5] Wait for CARLA RPC (port open + client connection)
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [3/5] Waiting for CARLA on port \${_EXP_PORT} (timeout \${_EXP_TIMEOUT}s) ==="
elapsed=0
until python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',\${_EXP_PORT})); s.close()" 2>/dev/null; do
    if ! kill -0 "\$CARLA_PID" 2>/dev/null && ! pgrep -f CarlaUE4 >/dev/null 2>&1; then
        echo "ERROR: CARLA process died. Last 40 lines of log:"
        tail -40 "\$CARLA_LOG"
        exit 1
    fi
    if [ \$elapsed -ge \$_EXP_TIMEOUT ]; then
        echo "ERROR: Timeout waiting for port after \${_EXP_TIMEOUT}s. Log:"
        tail -40 "\$CARLA_LOG"
        exit 1
    fi
    sleep 3; elapsed=\$((elapsed+3)); echo "  \${elapsed}s..."
done
echo "Port open (\${elapsed}s). Waiting for CARLA RPC to be fully ready..."

# Port open ≠ RPC ready. Wait for the CARLA Python client to connect.
# This is critical: CARLA opens the TCP port before the world is loaded.
rpc_ready=0
for i in \$(seq 1 60); do
    if python3 -c "
import carla, os
port = int(os.getenv('_EXP_PORT', 2000))
c = carla.Client('localhost', port)
c.set_timeout(5.0)
w = c.get_world()
print('  RPC ready: ' + w.get_map().name)
" </dev/null 2>/dev/null; then
        rpc_ready=1
        break
    fi
    sleep 3
    echo "  RPC not ready yet (\$((elapsed + i*3))s)..."
done

if [ \$rpc_ready -eq 0 ]; then
    echo "ERROR: CARLA port is open but RPC never became ready. Log:"
    tail -40 "\$CARLA_LOG"
    exit 1
fi

if [ "\${_EXP_VIDEO}" = "1" ]; then
    echo "  Running GPU warmup..."
    python3 "\${EXAMPLE_DIR}/warmup_carla.py" --wait </dev/null 2>&1 || echo "  (warmup skipped)"
fi

# ═══════════════════════════════════════════════════════════════════════
# [4/5] Run SHARC experiment
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "=== [4/5] Running SHARC: \${_EXP_CONFIG} ==="
cd "\$EXAMPLE_DIR"
sharc --config_filename "\${_EXP_CONFIG}" </dev/null

echo ""
echo "=== Saving dashboard image ==="
python3 -m sharc.dashboard --save "\$EXAMPLE_DIR" </dev/null

echo ""
echo "=== Computing experiment metrics ==="
LATEST_DIR="\$EXAMPLE_DIR/latest"
if [ -L "\$LATEST_DIR" ]; then
    METRICS_RESULT_DIR="\$(readlink -f "\$LATEST_DIR")"
    METRICS_SIM_DIR="\$(find "\$METRICS_RESULT_DIR" -name 'experiment_data.json' -printf '%h\n' 2>/dev/null | head -1)"
    if [ -n "\$METRICS_SIM_DIR" ]; then
        python3 "\$EXAMPLE_DIR/compute_metrics.py" "\$METRICS_SIM_DIR" </dev/null
    else
        echo "WARNING: No experiment_data.json found for metrics"
    fi
else
    echo "WARNING: No 'latest' symlink — skipping metrics"
fi

# ═══════════════════════════════════════════════════════════════════════
# [5/5] Record video (if --video enabled)
# ═══════════════════════════════════════════════════════════════════════
if [ "\${_EXP_VIDEO}" = "1" ]; then
    echo ""
    echo "=== [5/5] Recording experiment video ==="
    LATEST_DIR="\$EXAMPLE_DIR/latest"
    if [ -L "\$LATEST_DIR" ]; then
        RESULT_DIR="\$(readlink -f "\$LATEST_DIR")"
        VIDEO_ARGS="--fps 20"
        if [ "\${_EXP_HIGHRES}" = "1" ]; then
            VIDEO_ARGS="\$VIDEO_ARGS --highres"
        fi
        python3 "\${EXAMPLE_DIR}/record_experiment_video.py" "\$RESULT_DIR" \\
            \$VIDEO_ARGS </dev/null 2>&1 || echo "WARNING: Video recording failed"
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
    echo "Results: \$(readlink -f latest)"
fi
CONTAINER_SCRIPT
