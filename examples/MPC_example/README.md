# CARLA MPC Example

Obstacle-aware Model Predictive Control (MPC) for autonomous driving in CARLA.
The ego vehicle uses a kinematic bicycle model with hard collision-avoidance and
lane-keeping constraints, solved at each time step via nonlinear MPC.

## Scenarios

| Config | Description |
|--------|-------------|
| `lead_vehicle_stop.json` | Single NPC drives slowly ahead, then stops. Ego must brake via MPC constraints. |
| `obstacle_constraint_parallel.json` | Three NPC vehicles on the same road at 25 m spacing, parallel mode. |
| `leaderboard_dense_traffic.json` | 20 vehicles + 5 walkers, heavy traffic. |
| `leaderboard_lead_vehicle_braking.json` | Lead vehicle hard braking with a long prediction horizon. |
| `leaderboard_pedestrian_crossing.json` | 10 walkers, slow cautious driving. |
| `leaderboard_vehicle_cutin.json` | 15 vehicles with lane-change cut-ins. |
| `leaderboard_high_speed_avoidance.json` | High-speed obstacle avoidance at 25 m/s target. |

## Running an Experiment

Experiments run inside a Docker container (`carla-sharc-yasin5`) with CARLA
0.9.16.  The `run_offscreen_experiment.sh` script handles the full lifecycle:
starting CARLA, running the experiment, saving a dashboard, and optionally
recording a video — all without a monitor.

```bash
# From the repo root:
./run_offscreen_experiment.sh --config lead_vehicle_stop.json --video
./run_offscreen_experiment.sh --config obstacle_constraint_parallel.json --video
```

**Flags:**
- `--config FILE` — Simulation config from `simulation_configs/` (default: `obstacle_constraint.json`)
- `--video` — Record an MP4 replay after the experiment
- `--highres` — 1280×720 video (default: 640×360)
- `--container NAME` — Docker container name (default: `carla-sharc-yasin5`)

**Outputs** are saved under `experiments/<timestamp>--<label>/`:
- `experiment_data.json` — Full trajectory and control history
- `metrics.json` — Collision, feasibility, tracking error, min obstacle distance
- `dashboard_final.png` — Summary dashboard plot
- `<label>_video.mp4` — Replay video (if `--video` was used)

## MPC Controller

The `CarlaConstraintMPCController` formulates driving as a constrained
optimization at each step:

- **Objective:** Track waypoints + maintain target speed, with control effort and jerk penalties.
- **Hard constraints:**
  - Collision avoidance: $\|p_\text{ego}(j) - \hat{p}_\text{obs}(j)\|^2 \geq r_\text{safe}^2$ for each prediction step $j$ and obstacle.
  - Lane keeping: $|\text{lat\_dev}(j)| \leq w_\text{lane}/2$.
- **On infeasibility:** Emergency brake (safety fallback only — no heuristic speed matching or sticky hold).

All braking decisions are made by the MPC solver through its constraints and
cost function.

## NPC Configuration

NPC behavior is configured in the `carla.npcs` section of each scenario JSON:

| Parameter | Description |
|-----------|-------------|
| `road_vehicles` | Number of NPCs spawned ahead on ego's road |
| `road_spacing` | Distance (m) between consecutive road NPCs |
| `road_speed_pct` | CARLA TM speed parameter (0 = speed limit, 100 = stationary) |
| `road_stop_after_s` | Seconds after which road NPCs are commanded to stop (`null` = never) |
