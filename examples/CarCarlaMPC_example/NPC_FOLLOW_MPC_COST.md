# NPC Follow MPC Cost

This note documents what the `CarlaNPCFollowMPCController` is doing today,
not just the original intent. It covers:

- the vehicle model used by the optimizer
- exactly what enters the MPC cost
- how lead vehicles are detected
- how hard constraints are enforced
- what happens when the nonlinear solve becomes infeasible
- the dynamics-side waypoint behavior that matters for this controller

The main implementation lives in:

- `resources/controllers/src/CarlaNPCFollowMPCController.cpp`
- `resources/controllers/include/CarlaNPCFollowMPCController.h`
- `resources/dynamics/carla_mpc_dynamics.py`

## High-Level Intent

This controller is a route-tracking nonlinear MPC for CARLA that can do two
different things depending on traffic:

- Free road:
  track the route centerline and regulate to the nominal `target_speed`.
- Lead vehicle ahead:
  still track the route centerline, but reduce the effective target speed and
  add a following-gap penalty while keeping collision avoidance as a hard
  constraint.

It is not a pure "follow the lead car" controller and it is not a pure
"obstacle avoidance only" controller. It is a hybrid:

- route tracking comes from the waypoint path
- speed regulation comes from a speed cost
- following behavior comes from lead detection plus a follow-gap cost
- safety comes from hard inequality constraints

## State, Input, and Exogenous Input

The controller state is:

`x = [px, py, psi, v]`

Where:

- `px`, `py`: global position
- `psi`: yaw angle
- `v`: longitudinal speed

The control input is:

`u = [a, delta]`

Where:

- `a`: longitudinal acceleration command used by the optimizer
- `delta`: steering angle command

The exogenous input vector `w` is:

`w = [wp_x_1, wp_y_1, ..., wp_x_40, wp_y_40, obs1_x, obs1_y, obs1_vx, obs1_vy, obs1_r, ...]`

So the controller receives:

- `n_waypoints` future route points
- `n_obstacles` obstacle slots

Each obstacle slot contains:

- obstacle position `(x, y)`
- obstacle velocity `(vx, vy)`
- obstacle radius `r`

Unused obstacle slots are filled with sentinels:

- `x = 1e6`
- `y = 1e6`
- `vx = 0`
- `vy = 0`
- `r = 0`

## Vehicle Model Used Inside MPC

The optimizer uses a kinematic bicycle model:

`px_dot = v cos(psi)`

`py_dot = v sin(psi)`

`psi_dot = (v / L) tan(delta)`

`v_dot = a`

Where:

- `L = wheelbase`

This is the prediction model used over the horizon. It is simpler than full
CARLA physics, which is normal for an MPC controller.

## Objective Function

At a high level, the optimizer minimizes:

`J = J_path + J_speed + J_follow_gap + J_effort + J_jerk`

over the prediction horizon.

More explicitly, for each prediction step `j = 1 .. Np`:

`J_stage(j) = gamma^j * [`

`  q_path * d_path(x_j)^2`

`  + q_speed * (v_j - v_eff)^2`

`  + q_follow_gap * max(0, d_des - d_lead(x_j))^2`

`]`

and for each control step `j = 0 .. Nc-1`:

`J_input(j) = gamma^j * [`

`  r_accel * a_j^2`

`  + r_steer * delta_j^2`

`]`

plus jerk penalties:

- first input relative to the previous applied command
- later inputs relative to the previous optimized input

### 1. Path Cost

The path term is:

`J_path = sum_j gamma^j * q_path * d_path(x_j)^2`

Where `d_path(x_j)^2` is the squared distance from the predicted position
`(px_j, py_j)` to the nearest waypoint in the local waypoint set.

What this means in practice:

- larger `q_path` makes the controller hug the route more aggressively
- too large `q_path` can produce twitchy steering or centerline hunting
- too small `q_path` can let the car drift more before correcting

### 2. Speed Cost

The speed term is:

`J_speed = sum_j gamma^j * q_speed * (v_j - v_eff)^2`

Where:

- `v_j` is predicted speed
- `v_eff` is the controller's current effective target speed

Important detail:

- the controller does not always optimize toward raw `target_speed`
- it optimizes toward `effective_target_speed`, which may be reduced by a lead
  vehicle

### 3. Follow-Gap Cost

If a lead obstacle is found, the controller adds:

`J_follow_gap = sum_j gamma^j * q_follow_gap * max(0, d_des - d_lead(x_j))^2`

Where:

- `d_lead(x_j)` is the forward route distance from the predicted ego state to
  the lead obstacle
- `d_des` is currently implemented as:
  `min_follow_distance`

Note the subtlety here:

- `follow_time_gap` is used in target-speed shaping
- the cost gap term itself currently uses only `min_follow_distance`

So the controller has two separate "follow" mechanisms:

- target-speed shaping:
  slow down earlier when a lead vehicle is ahead
- follow-gap cost:
  penalize getting too close

### 4. Control-Effort Cost

The effort term is:

`J_effort = sum_j gamma^j * [r_accel * a_j^2 + r_steer * delta_j^2]`

Interpretation:

- `r_accel` discourages large throttle/brake demands
- `r_steer` discourages large steering magnitudes

If the car is wobbly, `r_steer` is one of the first weights to increase.

### 5. Jerk Cost

The jerk term penalizes changes in control:

- first move:
  `(a_0 - prev_accel)^2` and `(delta_0 - prev_steer)^2`
- later moves:
  `(a_j - a_{j-1})^2` and `(delta_j - delta_{j-1})^2`

Weighted by:

- `r_jerk_v`
- `r_jerk_yaw`

Interpretation:

- `r_jerk_v` smooths longitudinal command changes
- `r_jerk_yaw` smooths steering changes

For wobble reduction, `r_jerk_yaw` is usually the most important smoothing
term.

### 6. Discount Factor

All stage and effort terms are geometrically discounted by `gamma`.

Interpretation:

- `gamma < 1` slightly emphasizes near-term behavior more than far-term
  behavior
- `gamma = 1` would weight all horizon stages equally

## Effective Target Speed Logic

This controller does not blindly chase the nominal `target_speed`.

### No lead vehicle

If no lead obstacle is found:

- `desired_target_speed = target_speed`

### Lead vehicle found

If a lead vehicle is found within `lead_engage_distance`, the controller first
computes a desired gap:

`d_desired = min_follow_distance + follow_time_gap * max(v_ego, v_lead)`

Then it computes a normalized gap ratio over the active engagement region:

`gap_ratio = clamp((d_lead - d_desired) / (lead_engage_distance - d_desired), 0, 1)`

Then:

`desired_target_speed = v_lead + gap_ratio * (target_speed - v_lead)`

Interpretation:

- if the lead is far away but still within engagement range:
  the controller keeps a speed closer to nominal cruise
- if the lead is near the desired gap:
  the controller moves closer to the lead speed
- if the lead is stopped but far away:
  the controller no longer collapses immediately to zero target speed

This was an important fix. Earlier behavior matched `lead_speed` too directly
and caused the ego to stall even with a large open gap.

### Low-pass filtering

The desired target is then smoothed:

`v_eff <- alpha * v_desired + (1 - alpha) * v_eff_prev`

Where:

- `alpha = target_speed_alpha`

Interpretation:

- larger `alpha`:
  faster reaction, less smoothing
- smaller `alpha`:
  slower reaction, smoother target-speed evolution

## Lead Vehicle Detection

Lead detection is route-based, not heading-only.

Both ego and each obstacle are projected onto the current waypoint path.

For each obstacle:

- project obstacle position onto the waypoint polyline
- compute route progress `s`
- compute lateral offset from the path
- reject obstacles that are:
  - behind the ego on path progress
  - laterally outside the lane corridor

The lane corridor used for lead gating is:

`lane_half_width + obstacle_radius + safe_margin`

The selected lead is the closest obstacle ahead in route-progress coordinates.

Lead speed is estimated by projecting obstacle velocity onto the local path
tangent, not just using raw world-frame speed magnitude.

This is why the controller can treat a stopped car ahead as a lead vehicle even
if the ego heading is noisy or the road is curved.

## Hard Constraints

Safety is enforced with hard inequality constraints at every prediction step.

### 1. Collision Avoidance

For obstacle `i` at prediction step `j`:

`c_obs(i, j) = r_safe^2 - ||p_ego(j) - p_obs(j)||^2 <= 0`

Where:

- `p_obs(j)` is obstacle position propagated as:
  `obs_x + j * dt * obs_vx`,
  `obs_y + j * dt * obs_vy`
- `r_safe = ego_radius + obstacle_radius + safe_margin`

Interpretation:

- if the predicted ego comes inside the safety circle, the constraint is
  violated
- this is a hard feasibility condition, not just a soft penalty

### 2. Lane-Keeping Constraint

For each prediction step:

`c_lane(j) = lateral_deviation(j)^2 - lane_half_width^2 <= 0`

Interpretation:

- the predicted position must remain within the lane-width envelope around the
  waypoint path

### Important consequence

Because both collision avoidance and lane keeping are hard constraints:

- the optimizer can become infeasible even on open road if the local
  nonlinear solve struggles numerically
- when that happens, fallback logic matters a lot

## Infeasibility Handling

The controller now uses two different fallback modes.

### If obstacles are active

If the solver is infeasible and at least one obstacle slot is active:

- if speed is already near zero:
  hold
- otherwise:
  brake with `min_accel`
- steering is set to zero

This is the conservative safety behavior.

### If no obstacles are active

If the solver is infeasible and there are no active obstacles:

- use a route-tracking fallback instead of hold/brake
- acceleration is proportional to free-road speed error
- steering is generated from:
  - path-heading error
  - lateral deviation

This was added because repeated infeasible solves on an empty road were causing
the car to lose speed for no good reason.

So:

- obstacle case:
  safety-first fallback
- free-road case:
  continue tracking and regulating speed

## Dynamics-Side Behavior That Matters

The controller itself does not generate waypoints or obstacle lists. Those come
from `carla_mpc_dynamics.py`.

### Waypoint packing

The dynamics code builds a local route window of `n_waypoints` future points and
packs them into `w`.

### Obstacle packing

Nearby in-lane actors are packed into the obstacle slots, sorted by route
progress.

### Important special case for this controller

The generic obstacle-aware dynamics logic can truncate waypoints when an
obstacle blocks the lane ahead. That is useful for some obstacle-avoidance
controllers, but it was wrong for this follow-MPC.

Why it was wrong:

- the truncated path made the lead vehicle project "off path"
- then lead detection failed
- then the ego kept acting like the road was clear

The dynamics code now skips waypoint truncation when
`controller_type == "CarlaNPCFollowMPCController"`.

For this controller, the correct setup is:

- keep the full route centerline in `w`
- provide obstacles separately in obstacle slots

That lets the controller:

- detect the lead vehicle correctly
- track the lane correctly
- keep collision avoidance as a hard constraint

## Current Tuned Weights in `npc_follow_cost_seed42_config.json`

For the current seed-42 config, the cost weights are:

- `q_path = 1.0`
- `q_speed = 4.0`
- `q_follow_gap = 2.0`
- `r_accel = 0.35`
- `r_steer = 3.0`
- `r_jerk_v = 2.5`
- `r_jerk_yaw = 80.0`
- `gamma = 0.95`

Interpretation of this tuning:

- lower `q_path`:
  less aggressive centerline chasing
- higher `q_speed`:
  stronger cruise-speed regulation
- higher `r_steer`:
  less large steering amplitude
- much higher `r_jerk_yaw`:
  much smoother steering evolution
- moderately higher `r_accel` and `r_jerk_v`:
  less harsh longitudinal oscillation

## How to Tune This Cost Function

If the car is too wobbly:

- increase `r_steer`
- increase `r_jerk_yaw`
- optionally reduce `q_path`

If the car is too sluggish on free road:

- increase `q_speed`
- reduce `r_accel`
- reduce `r_jerk_v`

If the car approaches a lead too aggressively:

- increase `q_follow_gap`
- increase `min_follow_distance`
- increase `follow_time_gap`
- reduce `target_speed_alpha` if target-speed changes are too abrupt

If the controller brakes too early for distant slow traffic:

- increase `lead_engage_distance` carefully
- or make target-speed shaping less conservative

If it still becomes numerically infeasible too often:

- smoothing the cost can help
- but some remaining issues are solver/initialization related, not only
  weight-related

## Summary

The current controller is best understood as:

- a nonlinear route-tracking MPC
- with hard lane and collision constraints
- with route-based lead detection
- with lead-aware target-speed shaping
- with a soft following-gap penalty
- and different infeasibility fallbacks for free-road vs obstacle cases

The most important implementation details are:

- speed cost uses `effective_target_speed`, not raw `target_speed`
- lead detection is route-progress based
- collision avoidance is hard-constrained
- the follow-MPC must receive the untruncated route centerline
- no-obstacle infeasibility now uses a tracking fallback instead of stalling
