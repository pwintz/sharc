# Obstacle-Aware MPC for CARLA–SHARC: Formulation Proposals

## 1. Paper Summary

**SHARC-Drive** couples two simulators:
- **SHARC** executes a controller binary on Scarab (cycle-accurate CPU simulator), producing computation time τ_k per sample step.
- **CARLA** provides high-fidelity vehicle dynamics via its physics engine.

The paper's core insight is the **performance cliff**: MPC computation time is state-dependent (harder optimization → more iterations), so a single deadline miss can cascade—stale control drives the vehicle further from reference, which makes the *next* optimization harder, which causes another miss, etc. PID is immune because τ_k ≈ const.

**Current MPC formulation** (kinematic bicycle model, `CarlaKinematicMPCController`):
- State: x = (p_x, p_y, ψ, v) ∈ ℝ⁴
- Control: u = (a, δ) ∈ ℝ² (acceleration, steering angle)
- Exogenous: w = N_w dense waypoints from CARLA's HD map ∈ ℝ^{2·N_w}
- Cost: waypoint tracking (closest-waypoint distance²) + speed tracking + control effort + jerk + γ-discounting
- **No obstacle awareness whatsoever**

---

## 2. Design Requirements

1. **Collision avoidance** — the ego vehicle must not collide with other vehicles, pedestrians, or static obstacles
2. **Waypoint following** — maintain path tracking along the route planner's reference
3. **Forward progress** — make steady headway; don't freeze, oscillate, or reverse when obstacles appear
4. **Meaningful trajectories** — smooth, natural driving behavior (lane-keeping, reasonable speed)
5. **No explicit traffic rules encoded** — behavior should emerge from cost structure, not hard-coded logic
6. **Simplicity** — minimal changes to the existing SHARC architecture; clean CARLA interface

---

## 3. CARLA Interface for Obstacle Information

Before discussing MPC formulations, we need to decide *how* the dynamics bridge obtains obstacle data and passes it to the controller. The exogenous input vector `w` is the interface between the Python dynamics and the C++ controller.

### Interface Option A: World-Query Polling (Recommended)

At each `get_exogenous_input(t)` call, query `world.get_actors()`, filter for vehicles/pedestrians near the ego, extract their (x, y, v_x, v_y) in global frame, and pack them into fixed-size slots in `w`.

```python
# In CarlaMPCDynamics.get_exogenous_input():
actors = self.world.get_actors()
ego_loc = self.vehicle.get_transform().location
obstacles = []
for a in actors:
    if a.id == self.vehicle.id:
        continue
    if not a.type_id.startswith(('vehicle.', 'walker.')):
        continue
    loc = a.get_transform().location
    dist = ego_loc.distance(loc)
    if dist < detection_radius:
        vel = a.get_velocity()
        obstacles.append((loc.x, loc.y, vel.x, vel.y, dist))

# Sort by distance, take closest N_obs
obstacles.sort(key=lambda o: o[4])
obstacles = obstacles[:N_obs]

# Pack into w: [wp_x1, wp_y1, ..., wp_xNw, wp_yNw, ox1, oy1, ovx1, ovy1, ..., oxNo, oyNo, ovxNo, ovyNo]
```

**Pro:** Simple, synchronous, no extra sensors needed, works within SHARC's existing `w` vector interface, gives global-frame positions directly (no coordinate transforms), and is fully deterministic in synchronous mode.

**Con:** `world.get_actors()` returns *all* actors—O(N_total) per call. For typical CARLA scenes (<200 actors) this is negligible. Does not capture static obstacles unless they are CARLA actors (hedges, buildings are not actors—only vehicles, walkers, props).

### Interface Option B: CARLA Radar Sensor

Attach a `sensor.other.radar` to the ego vehicle. It provides detections with (azimuth, altitude, depth, velocity) per target.

**Pro:** Physically realistic sensing model with range/azimuth/velocity; natural fit for sensor-limited studies.

**Con:** Asynchronous sensor callback complicates synchronous SHARC flow; returns data in polar coordinates requiring transform; detection is noisy by design; doesn't naturally fit the `get_exogenous_input()` synchronous call pattern; more complex to integrate; overkill for our goal.

### Interface Option C: CARLA Obstacle Sensor

`sensor.other.obstacle` triggers an event when an obstacle enters a configurable detection range ahead.

**Pro:** Simple event-driven API.

**Con:** Binary (obstacle/no obstacle), lacks position/velocity detail, event-driven rather than polled, insufficient for MPC trajectory planning.

### Recommendation

**Option A (World-Query Polling)** is the right choice. It is:
- Synchronous and deterministic (matches SHARC's tick-based flow)
- Gives complete obstacle state (position + velocity) in global frame
- Zero extra CARLA actors/sensors to manage
- Simplest to pack into the exogenous input vector `w`

The exogenous input vector becomes:

```
w = [wp_x_1, wp_y_1, ..., wp_x_Nw, wp_y_Nw,  |  ox_1, oy_1, ovx_1, ovy_1, ..., ox_No, oy_No, ovx_No, ovy_No]
     \_______________ 2·N_w ________________/     \_________________________ 4·N_obs _________________________/
```

Total exogenous dimension: `2·N_w + 4·N_obs`

Unused obstacle slots (fewer than N_obs obstacles detected) are filled with a sentinel value (e.g., position = (1e6, 1e6), velocity = (0, 0)) so the controller can ignore them.

---

## 4. MPC Formulation Proposals

All proposals share the kinematic bicycle prediction model (eq. 12 in the paper):

```
x̂_{j+1|k} = x̂_{j|k} + T·[v·cos(ψ), v·sin(ψ), (v/L)·tan(δ), a]ᵀ
```

The difference is **how obstacles enter the cost function** and **whether hard constraints are used**.

---

### Proposal A: Exponential Repulsive Barrier in Cost Function

**Idea:** Add a smooth, exponential repulsive penalty for each obstacle. As the predicted trajectory approaches an obstacle, the cost rises sharply but remains differentiable everywhere.

**Cost function:**

```
J = Σ_{j=1}^{Np} γ^j · [
      q_ℓ · d_{j|k}²                          (waypoint tracking)
    + q_v · (v_{j|k} - v*)²                    (speed tracking)
    + Σ_{i=1}^{N_obs} q_obs · exp(-||p̂_{j|k} - o_i||² / (2σ²))   (obstacle repulsion)
  ]
  + Σ_{j=0}^{Nc-1} γ^j · [r_a·a²_{j|k} + r_δ·δ²_{j|k}]         (control effort)
  + jerk terms (unchanged)
```

Where:
- `o_i = (o_{x,i}, o_{y,i})` is the position of obstacle `i`
- `σ` controls the "repulsion radius" — how far away the barrier activates
- `q_obs` controls the strength of the repulsion
- When `||p̂ - o_i||` is large, `exp(...)` ≈ 0 (no effect)
- When `||p̂ - o_i||` → 0, `exp(...)` → `q_obs` (strong penalty)

**Optionally, use predicted obstacle positions** for dynamic obstacles:

```
ô_{i,j} = o_i + j·T·(ov_{x,i}, ov_{y,i})   (constant-velocity motion model)
```

**Pros:**
- Smooth and differentiable everywhere → well-behaved gradients for NLopt
- No inequality constraints needed → `ineq_c = 0` (no template change)
- Always feasible — repulsion is a soft cost, so the solver always finds a solution
- Naturally handles multiple obstacles
- Obstacle "softness" tunable: σ controls warning distance, q_obs controls urgency
- Forward progress emerges from speed-tracking + waypoint-tracking combination
- No traffic rules encoded — behavior emerges from the cost landscape

**Cons:**
- **No hard safety guarantee** — with finite q_obs, the optimizer *can* drive through an obstacle if the waypoint attraction is strong enough
- **Tuning burden** — three new parameters (q_obs, σ, N_obs) interact with existing weights
- **Computational cost** — N_obs extra exponentials per prediction step per obstacle (but exp() is fast)
- **Local minima** — if obstacles block the waypoint path, the optimizer may get trapped between competing gradients (attracted to waypoint, repelled from obstacle)

**Mitigation for safety:** Use a large q_obs (e.g., 100×q_ℓ) and a moderate σ (e.g., 3–5m for vehicles). The exponential barrier rises so steeply that violating it would require an unreasonably large waypoint-tracking benefit.

---

### Proposal B: Inverse-Distance Barrier in Cost Function

**Idea:** Use a 1/d² (or 1/d) penalty that goes to infinity as the predicted trajectory touches an obstacle. This provides stronger repulsion close to obstacles than the exponential.

**Cost function:**

```
J = Σ_{j=1}^{Np} γ^j · [
      q_ℓ · d_{j|k}²
    + q_v · (v_{j|k} - v*)²
    + Σ_{i=1}^{N_obs} q_obs / (||p̂_{j|k} - o_i||² + ε)          (inverse-distance repulsion)
  ]
  + control effort + jerk terms
```

Where `ε > 0` is a small constant (e.g., 0.1) to prevent division by zero / numerical infinity.

**Pros:**
- Stronger repulsion near obstacles than exponential — approaches infinity as d → 0
- Classic artificial potential field formulation, well-studied
- Simple to implement

**Cons:**
- **Gradient explosion** — 1/d² has very large gradients near obstacles, which can cause NLopt convergence issues (step-size oscillation, slow convergence)
- **Strong local minima** — 1/d² creates deep wells that attract the solver to solutions *near* the obstacle surface; combined with waypoint attraction, this produces saddle points and oscillation more readily than the exponential
- **Need ε tuning** — too small → numerical instability; too large → weak repulsion
- **No hard guarantee** — same as Proposal A

**Verdict:** Strictly dominated by Proposal A for gradient-based solvers. The exponential barrier has better numerical conditioning (bounded gradient) and comparable repulsive strength when σ is tuned properly. **Not recommended.**

---

### Proposal C: Hard Inequality Constraints (Keep-Out Circles)

**Idea:** Each obstacle defines a circular keep-out zone as a hard inequality constraint on the optimization.

**Constraints added to the NLMPC problem:**

```
||p̂_{j|k} - o_i||² ≥ r_safe²   for all j = 1,...,Np, i = 1,...,N_obs
```

Equivalently:  `r_safe² - ||p̂_{j|k} - o_i||²  ≤  0`

The cost function remains unchanged (waypoint + speed + effort + jerk).

**Implementation with libmpc++:**
```cpp
// Template: NLMPC<Nx, Nu, Ny, Np, Nc, ineq_c, eq_c>
// ineq_c = Np * N_obs (one constraint per prediction step per obstacle)
nlmpc.setIneqConFunction([this](
    cvec<ineq_c>& constraint,
    const mat<Np + 1, Nx>& X,
    const mat<Np + 1, Nu>& U,
    const double&) {
    int idx = 0;
    for (int j = 1; j <= Np; ++j) {
        double px = X(j, 0), py = X(j, 1);
        for (int i = 0; i < N_obs; ++i) {
            double dx = px - obs_x[i];
            double dy = py - obs_y[i];
            constraint(idx++) = r_safe * r_safe - (dx*dx + dy*dy);
        }
    }
});
```

**Pros:**
- **Hard safety guarantee** — if the solver finds a feasible solution, it provably avoids all obstacles
- Clean mathematical formulation — standard constrained NLP
- No potential-field tuning needed

**Cons:**
- **Infeasibility** — if obstacles surround the vehicle or the safe radius is too large, the problem has no feasible solution. NLopt returns failure, and the ZOH applies the stale input (potentially driving *into* the obstacle)
- **Template parameter change** — `ineq_c` must be `Np * N_obs`, a compile-time constant. This means every possible obstacle count must be set at compile time, wasting template computation when fewer obstacles are present
- **Nonconvex constraints** — keep-out circles are nonconvex. NLopt may fail to find a feasible point, or may take many iterations to satisfy all constraints, *increasing τ_k* — exactly the cascade effect we're studying
- **Dramatically higher computational cost** — each constraint adds Jacobian evaluations. With Np=10 and N_obs=5, that's 50 inequality constraints on top of the NLP
- **Interaction with SHARC's performance-cliff analysis** — hard constraints that increase solver difficulty as obstacles appear *directly amplifies* the cascading-miss phenomenon. This could be scientifically interesting but makes the controller much less robust

**Verdict:** Mathematically clean but **impractical for this project**. The compile-time `ineq_c`, infeasibility risk, and massive computation cost increase make this a poor fit for both the SHARC framework and the paper's goals. The infeasibility problem is particularly dangerous: when the solver fails, ZOH applies stale control, which is the *opposite* of safe behavior.

---

### Proposal D: Soft Constraint via Log-Barrier in Cost (Interior-Point Style)

**Idea:** Add a logarithmic barrier `−log(d² − r²)` that goes to +∞ as the predicted position approaches the keep-out circle boundary, which prevents crossing.

**Cost addition:**

```
J_obs = Σ_{j=1}^{Np} Σ_{i=1}^{N_obs} −μ · log(||p̂_{j|k} - o_i||² − r_safe²)
```

Where μ > 0 is the barrier weight. Undefined (→ +∞) when `||p̂ - o_i||² ≤ r_safe²`.

**Pros:**
- Interior-point barriers are well-studied in convex optimization
- Provides strong repulsion near the boundary with weaker effect far away
- Theoretically converges to the hard-constraint solution as μ → 0

**Cons:**
- **Domain issue** — the log barrier is *undefined* outside the feasible region. If the predicted trajectory starts infeasible (e.g., obstacle suddenly appears very close), the cost is NaN/Inf and the solver fails immediately
- **Not suitable for derivative-free / gradient-free solvers** — NLopt's algorithms don't handle domain boundaries well
- **Worse than exponential** — the exponential barrier (Proposal A) achieves similar repulsion with a function that is defined everywhere, avoiding the domain issue entirely
- **Initialization sensitivity** — requires the initial trajectory to be feasible (all predicted points outside all keep-out circles)

**Verdict:** **Not recommended.** The exponential barrier (Proposal A) is strictly superior for our NLopt-based setting — it provides similar repulsive behavior while being defined and smooth everywhere.

---

### Proposal E: Exponential Barrier + Forward-Progress Heading Reward (Recommended)

**Idea:** Extend Proposal A with an explicit forward-progress term that uses the *heading alignment* between the vehicle and the waypoint path. This prevents the vehicle from freezing, reversing, or oscillating when obstacles partially block the path.

**Key insight:** Speed tracking alone (v → v*) doesn't prevent the vehicle from steering sideways or backward to avoid an obstacle. Adding a heading-alignment reward ensures the vehicle makes progress *along the route* even when detouring.

**Full cost function:**

```
J = Σ_{j=1}^{Np} γ^j · [
      q_ℓ · d_{j|k}²                                                      (1: waypoint tracking)
    + q_v · (v_{j|k} − v*)²                                                (2: speed tracking)
    + Σ_{i=1}^{N_obs} q_obs · exp(−||p̂_{j|k} − ô_{i,j}||² / (2σ²))      (3: obstacle repulsion)
    + q_prog · max(0, −v_{j|k} · cos(ψ_{j|k} − θ_{wp,j}))                (4: forward-progress)
  ]
  + Σ_{j=0}^{Nc-1} γ^j · [r_a · a²_{j|k} + r_δ · δ²_{j|k}]              (5: control effort)
  + Σ_{j=0}^{Nc-1} [r_jerk_v · Δa² + r_jerk_δ · Δδ²]                     (6: jerk smoothness)
```

Where:
- **Term 1** (path tracking): unchanged from current — distance to closest waypoint
- **Term 2** (speed tracking): unchanged — penalize deviation from reference speed
- **Term 3** (obstacle repulsion): exponential barrier with constant-velocity prediction for dynamic obstacles:
  `ô_{i,j} = o_i + j·T·v_obs_i`
- **Term 4** (forward progress): penalizes *negative* projection of velocity onto waypoint direction. `θ_{wp,j}` is the bearing from predicted position to the closest waypoint. The penalty activates only when the vehicle moves *away from* the path heading (cos < 0), i.e., reversing or drifting far off-course. When moving forward along the path, this term is zero.
- **Terms 5–6** (effort + jerk): unchanged from current controller

**Implementation notes:**
- θ_{wp,j} = atan2(wp_y_nearest − p̂_y, wp_x_nearest − p̂_x), the bearing to the closest waypoint from predicted position at step j
- The `max(0, −...)` can be smoothly approximated as `softplus(−v·cos(...))` = `log(1 + exp(−v·cos(...)))` for differentiability
- N_obs is fixed at compile time (e.g., 5 or 8). Unused slots use sentinel positions far away, so their exp() terms vanish automatically
- Predicted obstacle positions `ô_{i,j}` assume constant velocity — simple and adequate for short prediction horizons (1–2 seconds)

**New parameters:**

| Parameter | Description | Suggested range | Role |
|-----------|-------------|-----------------|------|
| `q_obs` | Obstacle repulsion strength | 10–100 × q_ℓ | Higher → stronger avoidance |
| `σ` | Repulsion activation radius [m] | 2–6 | Larger → earlier reaction |
| `q_prog` | Forward-progress weight | 0.1–1.0 | Prevents freezing/reversal |
| `N_obs` | Max obstacle slots | 3–8 | Compile-time constant |
| `r_detect` | Detection radius [m] | 30–50 | In dynamics, not controller |

**Pros:**
- **Differentiable everywhere** — exponential barrier + softplus progress term → clean gradients
- **Always feasible** — no hard constraints, solver always returns a solution
- **Forward progress guaranteed** — heading-alignment term prevents degenerate behaviors
- **Dynamic obstacles handled** — constant-velocity prediction in cost function
- **Naturally produces meaningful behavior:**
  - Vehicle slows down when approaching obstacles (speed tracking + obstacle repulsion balance)
  - Vehicle steers around obstacles (waypoint tracking + repulsion gradient)
  - Vehicle speeds up after passing (speed tracking reasserts)
  - Vehicle doesn't reverse or freeze (forward-progress term)
- **No traffic rules encoded** — all behavior emerges from cost landscape
- **Minimal template changes** — still `ineq_c = 0`, `eq_c = 0`
- **Scientifically interesting** — the exponential terms increase cost landscape complexity, which may increase solver iterations (τ_k grows) → directly connects to the paper's performance-cliff analysis

**Cons:**
- **No hard safety guarantee** — with sufficiently strong waypoint attraction and weak q_obs, the optimizer could choose to clip an obstacle. Mitigation: use q_obs ≫ q_ℓ
- **More parameters to tune** than the current controller (q_obs, σ, q_prog added)
- **N_obs exponentials per step** — minor computational overhead but negligible compared to the NLP solver cost
- **Constant-velocity obstacle prediction is naive** — real vehicles brake, turn, etc. For short horizons (< 2s), this is acceptable; for longer horizons, it may produce overly conservative or insufficient avoidance

---

### Proposal F: Simplified Variant — Exponential Barrier Only (No Progress Term)

**Idea:** Same as Proposal E but drop term 4 (forward progress). Rely solely on speed tracking + waypoint tracking to maintain forward motion.

**Pros:**
- Simplest extension — only add the obstacle repulsion exponential to the existing cost
- Fewer parameters to tune (no q_prog)
- Adequate for scenarios where obstacles don't block the full path ahead

**Cons:**
- **Can produce freezing behavior** — when a large obstacle blocks the path directly ahead, The waypoint tracking pulls forward, obstacle repulsion pushes back, speed tracking pulls to v* but direction is ambiguous → vehicle may oscillate or stop
- **Less robust** than Proposal E in complex scenarios

**Verdict:** Good as a *first implementation* for simplicity; upgrade to Proposal E if freezing behavior is observed.

---

## 5. Comparison Summary

| Criterion | A: Exponential | B: 1/d² | C: Hard Constraints | D: Log-Barrier | **E: Exp + Progress** | F: Exp Only |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Safety guarantee | Soft | Soft | **Hard** | Soft | Soft | Soft |
| Always feasible | **Yes** | **Yes** | No | No | **Yes** | **Yes** |
| Gradient quality | **Good** | Poor | N/A | Poor domain | **Good** | **Good** |
| Template changes | None | None | `ineq_c` change | None | None | None |
| Forward progress | Implicit | Implicit | Implicit | Implicit | **Explicit** | Implicit |
| Dynamic obstacles | ✓ | ✓ | ✓ | ✓ | **✓** | ✓ |
| Params to tune | +2 | +2 | +1 | +2 | +3 | +2 |
| Computation overhead | Low | Low | **High** | Low | Low | Low |
| Solver convergence | **Stable** | Unstable | Hard | Domain issues | **Stable** | **Stable** |
| SHARC compatibility | **Full** | Full | Partial | Full | **Full** | **Full** |
| Cascade-miss interaction | Moderate | Moderate | **Extreme** | Moderate | Moderate | Moderate |

---

## 6. Recommendation

**Primary recommendation: Proposal E** (Exponential Barrier + Forward-Progress Heading Reward)

This is the cleanest formulation that satisfies all five requirements:
1. Collision avoidance via exponential repulsion → naturally emerging behavior
2. Waypoint following → unchanged from current formulation
3. Forward progress → explicit heading-alignment term
4. Meaningful trajectories → smoothness terms + γ-discounting + natural cost landscape
5. No traffic rules → all behavior emerges from quadratic/exponential cost terms

**Implementation plan** if selected:
1. **Python dynamics** (`CarlaMPCObstacleDynamics`): extend `CarlaMPCDynamics` to pack obstacle data into `w`
2. **C++ controller** (`CarlaObstacleMPCController`): extend `CarlaKinematicMPCController` with new cost terms
3. **Config** (`base_config.json`): add obstacle parameters under `mpc_options`
4. **CMake**: add the new source file to the build

**Fallback: Proposal F** if we want minimal complexity first.

---

## 7. Open Questions for Discussion

See clarification questions in the conversation.
