// controller/CarlaConstraintMPCController.cpp
// Hard-constraint obstacle MPC:  no collisions + stay in lane.
//
//   Exogenous input layout (same as CarlaObstacleMPCController):
//     w[0 .. 2*N_w-1]           = waypoints (wx1, wy1, …)
//     w[2*N_w + 5*i + 0..4]    = obstacle i (x, y, vx, vy, radius)
//   Sentinel: x ≥ 5e5 → empty slot.
//
//   Inequality constraints (all must be ≤ 0):
//     For each prediction step j = 1..Np:
//       - Per obstacle i:  r_safe² - ||p(j) - ô_i(j)||²   (collision)
//       - Lane keeping:    lat_dev(j)² - half_w²            (lane)
//     Total: Np * (N_obs + 1) = TNIEQ
//
//   On solver infeasibility → emergency brake (a = min_accel, δ = 0).

#include "CarlaConstraintMPCController.h"

#if TNX == 4 && defined(TNIEQ) && TNIEQ > 0
#include "sharc/utils.hpp"
#include "debug_levels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

static constexpr double SENTINEL_THRESHOLD = 5e5;

// ------------------------------------------------------------------ //
//  Setup                                                              //
// ------------------------------------------------------------------ //

void CarlaConstraintMPCController::setup(const nlohmann::json& json_data) {
    const auto& sp = json_data.at("system_parameters");

    sample_time  = sp.at("sample_time").get<double>();
    wheelbase    = sp.at("mpc_options").at("wheelbase").get<double>();
    target_speed = sp.at("target_speed").get<double>();

    // Cost weights (tracking only)
    const auto& cost = sp.at("mpc_options").at("cost_weights");
    q_path     = cost.at("q_path").get<double>();
    q_speed    = cost.at("q_speed").get<double>();
    r_accel    = cost.at("r_accel").get<double>();
    r_steer    = cost.at("r_steer").get<double>();
    r_jerk_v   = cost.at("r_jerk_v").get<double>();
    r_jerk_yaw = cost.at("r_jerk_yaw").get<double>();
    gamma      = cost.at("gamma").get<double>();

    // Constraint parameters
    ego_radius     = cost.value("ego_radius", 2.5);
    safe_margin    = cost.value("safe_margin", 0.5);
    lane_half_width = cost.value("lane_half_width", 1.75);

    // Input limits
    const auto& limits = sp.at("mpc_options").at("input_limits");
    max_accel = limits.at("max_accel").get<double>();
    min_accel = limits.at("min_accel").get<double>();
    max_steer = limits.at("max_steer").get<double>();
    min_steer = limits.at("min_steer").get<double>();

    // Waypoints & obstacles
    n_waypoints = sp.at("mpc_options").at("n_waypoints").get<int>();
    n_obstacles = sp.at("mpc_options").at("n_obstacles").get<int>();
    assert(Ndu == 2 * n_waypoints + 5 * n_obstacles
           && "TNDU must equal 2*n_waypoints + 5*n_obstacles");
    assert(ineq_c == Np * (n_obstacles + 1)
           && "TNIEQ must equal Np * (n_obstacles + 1)");

    wp_x.resize(n_waypoints);
    wp_y.resize(n_waypoints);
    obstacles.resize(n_obstacles);

    // ---- NLMPC ---------------------------------------------------- //
    nlmpc.setLoggerLevel(mpc::Logger::NORMAL);
    nlmpc.setDiscretizationSamplingTime(sample_time);

    // Kinematic bicycle model
    nlmpc.setStateSpaceFunction(
        [this](xVec& dx, const xVec& x, const uVec& u, const unsigned int&) {
            double psi = x(2), v = x(3);
            double a = u(0), delta = u(1);
            dx(0) = v * std::cos(psi);
            dx(1) = v * std::sin(psi);
            dx(2) = (v / wheelbase) * std::tan(delta);
            dx(3) = a;
        });

    // Objective: pure tracking (no obstacle terms — safety via constraints)
    nlmpc.setObjectiveFunction(
        [this](
            const mpc::mat<Np + 1, Nx>& X,
            const mpc::mat<Np + 1, Ny>&,
            const mpc::mat<Np + 1, Nu>& U,
            const double&) -> double
        {
            double J = 0.0;
            double gj = gamma;
            for (int j = 1; j <= Np; ++j) {
                double px = X(j, 0), py = X(j, 1), v = X(j, 3);
                J += gj * q_path * closestWaypointDistSq(px, py);
                double ev = v - effective_target_speed;
                J += gj * q_speed * ev * ev;
                gj *= gamma;
            }
            // Control effort
            double gk = 1.0;
            for (int j = 0; j < Nc; ++j) {
                J += gk * r_accel * U(j, 0) * U(j, 0);
                J += gk * r_steer * U(j, 1) * U(j, 1);
                gk *= gamma;
            }
            // Jerk penalty
            {
                double da = U(0, 0) - prev_accel;
                double dd = U(0, 1) - prev_steer;
                J += r_jerk_v * da * da;
                J += r_jerk_yaw * dd * dd;
            }
            for (int j = 1; j < Nc; ++j) {
                double da = U(j, 0) - U(j - 1, 0);
                double dd = U(j, 1) - U(j - 1, 1);
                J += r_jerk_v * da * da;
                J += r_jerk_yaw * dd * dd;
            }
            return J;
        });

    // ---- Inequality constraints ----------------------------------- //
    // Layout (all ≤ 0 for feasibility):
    //   idx = (j-1)*(n_obstacles+1) + i    → collision with obstacle i at step j
    //   idx = (j-1)*(n_obstacles+1) + n_obs → lane keeping at step j
    nlmpc.setIneqConFunction(
        [this](
            cvec<ineq_c>& c,
            const mpc::mat<Np + 1, Nx>& X,
            const mpc::mat<Np + 1, Ny>&,
            const mpc::mat<Np + 1, Nu>&,
            const double&)
        {
            int idx = 0;
            for (int j = 1; j <= Np; ++j) {
                double px = X(j, 0);
                double py = X(j, 1);
                double dt_pred = j * sample_time;

                // Collision avoidance
                for (int i = 0; i < n_obstacles; ++i) {
                    if (obstacles[i].x > SENTINEL_THRESHOLD) {
                        c(idx++) = -1e6;  // trivially satisfied
                        continue;
                    }
                    double ox = obstacles[i].x + dt_pred * obstacles[i].vx;
                    double oy = obstacles[i].y + dt_pred * obstacles[i].vy;
                    double r_safe = ego_radius + obstacles[i].radius + safe_margin;
                    double dx = px - ox;
                    double dy = py - oy;
                    c(idx++) = r_safe * r_safe - (dx * dx + dy * dy);
                }

                // Lane keeping: lat_dev² ≤ half_w²
                double ld = lateralDeviation(px, py);
                c(idx++) = ld * ld - lane_half_width * lane_half_width;
            }
        });

    // Input bounds
    uVec umin, umax;
    umin(0) = min_accel; umin(1) = min_steer;
    umax(0) = max_accel; umax(1) = max_steer;
    nlmpc.setInputBounds(umin, umax, {0, Nc});

    control.setZero();

    // State persistence
    experiment_dir = json_data.value("experiment_dir", "");
    state_file = experiment_dir.empty() ? "" : experiment_dir + "/mpc_state.json";
    if (!state_file.empty()) load_state();
}

// ------------------------------------------------------------------ //
//  Control step                                                       //
// ------------------------------------------------------------------ //

void CarlaConstraintMPCController::calculateControl(int k, double t,
                                                     const xVec& x, const wVec& w) {
    // Unpack waypoints
    for (int i = 0; i < n_waypoints; ++i) {
        wp_x[i] = w(2 * i);
        wp_y[i] = w(2 * i + 1);
    }

    // Unpack obstacle slots
    const int obs_offset = 2 * n_waypoints;
    for (int i = 0; i < n_obstacles; ++i) {
        obstacles[i].x      = w(obs_offset + 5 * i + 0);
        obstacles[i].y      = w(obs_offset + 5 * i + 1);
        obstacles[i].vx     = w(obs_offset + 5 * i + 2);
        obstacles[i].vy     = w(obs_offset + 5 * i + 3);
        obstacles[i].radius = w(obs_offset + 5 * i + 4);
    }

    state = x;

    // --- Adapt reference speed to closest leading vehicle ----------------
    effective_target_speed = target_speed;
    {
        double px = x(0), py = x(1), psi = x(2);
        double cos_psi = std::cos(psi), sin_psi = std::sin(psi);
        for (int i = 0; i < n_obstacles; ++i) {
            if (obstacles[i].x > SENTINEL_THRESHOLD) continue;
            double dx = obstacles[i].x - px;
            double dy = obstacles[i].y - py;
            double lon = dx * cos_psi + dy * sin_psi;  // ahead if > 0
            if (lon > 0) {
                // Project obstacle velocity onto ego heading
                double obs_fwd_speed = obstacles[i].vx * cos_psi
                                     + obstacles[i].vy * sin_psi;
                effective_target_speed = std::min(effective_target_speed,
                                                  std::max(obs_fwd_speed, 0.0));
            }
        }
    }

    // --- Sticky hold: once stopped near obstacles, stay stopped -----------
    if (sticky_hold) {
        bool all_clear = true;
        double px = x(0), py = x(1);
        for (int i = 0; i < n_obstacles; ++i) {
            if (obstacles[i].x > SENTINEL_THRESHOLD) continue;
            double dx = px - obstacles[i].x;
            double dy = py - obstacles[i].y;
            double dist = std::sqrt(dx * dx + dy * dy);
            double r_safe = ego_radius + obstacles[i].radius + safe_margin;
            if (dist < 2.0 * r_safe) { all_clear = false; break; }
        }
        if (all_clear) {
            sticky_hold = false;
        } else {
            control(0) = 0.0;
            control(1) = 0.0;
            int active_obs = 0;
            for (int i = 0; i < n_obstacles; ++i)
                if (obstacles[i].x < SENTINEL_THRESHOLD) ++active_obs;
            std::cout << "[CarlaConstraintMPC] k=" << k
                      << " STICKY_HOLD v=" << x(3)
                      << " obs=" << active_obs << "/" << n_obstacles
                      << std::endl;
            latest_metadata.clear();
            latest_metadata["k"] = k;
            latest_metadata["controller"] = "CarlaConstraintMPCController";
            latest_metadata["is_feasible"] = false;
            latest_metadata["sticky_hold"] = true;
            prev_accel = 0.0;
            prev_steer = 0.0;
            if (!state_file.empty()) save_state();
            return;
        }
    }

    mpc_result = nlmpc.optimize(state, control);

    // Count active obstacles
    int active_obs = 0;
    for (int i = 0; i < n_obstacles; ++i)
        if (obstacles[i].x < SENTINEL_THRESHOLD) ++active_obs;

    // On infeasibility → emergency brake (or hold if already stopped)
    if (!mpc_result.is_feasible) {
        double v = x(3);
        if (std::abs(v) < 0.1) {
            // Already (nearly) stopped — hold position, engage sticky hold
            control(0) = 0.0;
            control(1) = 0.0;
            if (active_obs > 0) sticky_hold = true;
        } else {
            // Still moving — hard brake
            control(0) = min_accel;
            control(1) = 0.0;
        }
        std::cout << "[CarlaConstraintMPC] k=" << k
                  << " INFEASIBLE → " << (std::abs(v) < 0.1 ? "HOLD" : "BRAKE")
                  << " a=" << control(0) << " v=" << v
                  << " obs=" << active_obs << "/" << n_obstacles
                  << std::endl;
    } else {
        control = mpc_result.cmd;
        std::cout << "[CarlaConstraintMPC] k=" << k
                  << " a=" << control(0) << " delta=" << control(1)
                  << " cost=" << mpc_result.cost
                  << " obs=" << active_obs << "/" << n_obstacles
                  << std::endl;
    }

    // Metadata for logging / trajectory drawing
    latest_metadata.clear();
    latest_metadata["k"]             = k;
    latest_metadata["t"]             = t;
    latest_metadata["controller"]    = "CarlaConstraintMPCController";
    latest_metadata["solver_status"] = mpc_result.solver_status;
    latest_metadata["is_feasible"]   = mpc_result.is_feasible;
    latest_metadata["cost"]          = mpc_result.cost;

    auto opt_seq = nlmpc.getOptimalSequence();
    std::vector<double> traj_x, traj_y;
    for (int j = 0; j <= Np; ++j) {
        traj_x.push_back(opt_seq.state(j, 0));
        traj_y.push_back(opt_seq.state(j, 1));
    }
    latest_metadata["traj_x"] = traj_x;
    latest_metadata["traj_y"] = traj_y;

    prev_accel = control(0);
    prev_steer = control(1);

    if (!state_file.empty()) save_state();
}

// ------------------------------------------------------------------ //
//  State persistence                                                  //
// ------------------------------------------------------------------ //

void CarlaConstraintMPCController::save_state() const {
    nlohmann::json s;
    s["prev_accel"]  = prev_accel;
    s["prev_steer"]  = prev_steer;
    s["sticky_hold"] = sticky_hold;
    std::ofstream f(state_file);
    if (f.is_open()) f << s.dump(2);
}

void CarlaConstraintMPCController::load_state() {
    std::ifstream f(state_file);
    if (!f.is_open()) return;
    try {
        nlohmann::json s;
        f >> s;
        prev_accel   = s.value("prev_accel", 0.0);
        prev_steer   = s.value("prev_steer", 0.0);
        sticky_hold  = s.value("sticky_hold", false);
    } catch (...) {}
}

// ------------------------------------------------------------------ //
//  Helpers                                                            //
// ------------------------------------------------------------------ //

double CarlaConstraintMPCController::closestWaypointDistSq(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < n_waypoints; ++i) {
        double dx = px - wp_x[i];
        double dy = py - wp_y[i];
        double d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return best;
}

double CarlaConstraintMPCController::lateralDeviation(double px, double py) const {
    // Perpendicular distance from (px,py) to the closest waypoint segment.
    // Sign comes from the cross-product: positive = left of path.
    double min_proj_dist_sq = std::numeric_limits<double>::max();
    double best_lat = 0.0;

    for (int i = 0; i < n_waypoints - 1; ++i) {
        double ax = wp_x[i],     ay = wp_y[i];
        double bx = wp_x[i + 1], by = wp_y[i + 1];

        double abx = bx - ax, aby = by - ay;
        double apx = px - ax, apy = py - ay;
        double ab_len_sq = abx * abx + aby * aby;
        if (ab_len_sq < 1e-12) continue;

        // Project onto segment, clamp to [0, 1]
        double t = std::max(0.0, std::min(1.0,
                   (apx * abx + apy * aby) / ab_len_sq));

        double cx = ax + t * abx;
        double cy = ay + t * aby;
        double dx = px - cx, dy = py - cy;
        double dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_proj_dist_sq) {
            min_proj_dist_sq = dist_sq;
            // Signed lateral deviation via cross product
            best_lat = (abx * apy - aby * apx) / std::sqrt(ab_len_sq);
        }
    }
    return best_lat;
}

REGISTER_CONTROLLER("CarlaConstraintMPCController", CarlaConstraintMPCController)
#endif
