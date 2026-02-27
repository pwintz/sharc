# CARLA Deterministic Replay

Deterministic recording and replay for CARLA scenarios. Uses PID lane following for realistic, collision-free driving with sub-30cm accuracy over 60 seconds.

## Quick Start

```bash
# 1. Start CARLA server (default port 2000)
./CarlaUE4.sh -prefernvidia -RenderOffScreen

# Or with custom port (e.g., 2010) if default port is in use:
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2010

# 2. Activate environment
conda activate carla

# 3. Run (records then replays automatically)
python carla_replay.py --config config.yaml

# Or with custom port:
python carla_replay.py --config config.yaml --port 2010
```

## How We Almost Achieved Determinism

### 1. Physics Engine Precision
- **Max Substeps**: Increased `max_substeps` to 10. This provides the physics engine with finer temporal resolution, significantly reducing numerical drift compared to the default 1 substep.
- **Fixed Timestep**: Strictly enforced `fixed_delta_seconds = 0.05` (20Hz) in synchronous mode.
- **Deterministic Ragdolls**: Enabled `settings.deterministic_ragdolls = True` to remove randomness from physics interactions.

### 2. State Synchronization
- **Settling Phase**: Implemented a configurable settling phase (e.g., 1-10 ticks) after spawning to allow vehicles to reach a stable suspension state before recording begins. This ensures both record and replay start from a physically relaxed state.
- **Consistent Timing**: Carefully aligned the recording and replay loops. We record state *after* a tick (capturing the result of the previous control) and compare it *after* a tick in replay. This avoids off-by-one frame errors that cause massive drift.

### 3. Traffic & Actor Management
- **Deterministic Blueprints**: Instead of random vehicle selection (`filter('vehicle.*')`), we use a fixed, sorted pool of specific blueprints (e.g., Model3, Prius, Charger).
- **Blueprint Recording**: The exact blueprint ID for each NPC is saved in the log (`npc_spawn_indices` now stores `{'index': 60, 'blueprint': 'vehicle.audi.a2'}`). Replay spawns the *exact* same vehicle types, ensuring identical mass/dimensions/physics.
- **Traffic Manager Strategy**:
  - **Recording**: NPCs are controlled by Traffic Manager to generate realistic, reactive behavior (lane changes, yielding).
  - **Replay**: Traffic Manager is **disabled**. NPCs are spawned as **kinematic objects** (`simulate_physics=False`) and teleported frame-by-frame to their recorded positions. This ensures they are non-reactive and follow their recorded paths exactly, eliminating any TM-induced divergence.
- **Clean Slate**: We destroy existing actors before starting to prevent interference from previous runs.

### 4. Control & Logic
- **Ego Vehicle Strategy**:
  - **No Traffic Manager**: We **never** use Traffic Manager for the ego vehicle. TM is reactive to the micro-state of the world; even tiny differences in replay would cause it to make different decisions (e.g., brake slightly harder), leading to massive divergence.
  - **PID Controller**: Instead, we use a deterministic PID lane-following controller. It is stateless and robust, calculating steering/throttle based purely on the current lane deviation.
  - **Physics Replay**: Unlike NPCs, the ego vehicle is replayed with **physics enabled**. We apply the exact recorded control inputs (throttle, steer, brake) to reproduce the physics trajectory. This allows for future "intervention" testing where we might want to change the ego's controls mid-simulation.

### 5. Why Replay from T=0?
- **No Mid-Sim Resets**: We intentionally replay the entire scenario from $t=0$ rather than jumping to a specific timestamp.
- **Physics Continuity**: CARLA's `set_target_velocity()` does not instantaneously force the vehicle's physics state (inertia, suspension, tire friction). Using it to jump to $t=10s$ results in an immediate "jerk" and trajectory divergence.
- **Solution**: By starting from the exact same spawn point and replaying all control inputs from the beginning, we let the physics engine naturally evolve to the desired state, preserving perfect continuity.

## Output

Results saved to `results/YYYYMMDD_HHMMSS/`:
- `config.yaml` - Configuration used
- `scenario.json` - Recorded data
- `comparison.png` - Trajectory visualization
