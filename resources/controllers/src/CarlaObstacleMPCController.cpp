// controller/CarlaObstacleMPCController.cpp
// Obstacle-aware kinematic bicycle-model MPC.
//   Exogenous input layout:
//     w[0 .. 2*N_w-1]                   = waypoints  (wx1, wy1, …)
//     w[2*N_w + 5*i + 0..4]             = obstacle i (x, y, vx, vy, radius)
//   Sentinel: obstacle slot with x ≥ 5e5 is treated as empty.

#include "CarlaObstacleMPCController.h"

#if TNX == 4
#include "sharc/utils.hpp"
#include "debug_levels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

static constexpr double SENTINEL_THRESHOLD = 5e5;

// Numerically stable softplus: log(1 + exp(x))
static inline double softplus(double x) {
    if (x > 20.0) return x;           // avoid exp overflow
    if (x < -20.0) return std::exp(x); // negligible
    return std::log(1.0 + std::exp(x));
}

// ------------------------------------------------------------------ //
//  Setup                                                              //
// ------------------------------------------------------------------ //

void CarlaObstacleMPCController::setup(const nlohmann::json& json_data) {
    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("Start of CarlaObstacleMPCController::setup()")
    }

    const auto& sp = json_data.at("system_parameters");

    // --- Vehicle / timing ------------------------------------------------ //
    sample_time  = sp.at("sample_time").get<double>();
    wheelbase    = sp.at("mpc_options").at("wheelbase").get<double>();
    target_speed = sp.at("target_speed").get<double>();

    // --- Cost weights — tracking ----------------------------------------- //
    const auto& cost = sp.at("mpc_options").at("cost_weights");
    q_path     = cost.at("q_path").get<double>();
    q_speed    = cost.at("q_speed").get<double>();
    r_accel    = cost.at("r_accel").get<double>();
    r_steer    = cost.at("r_steer").get<double>();
    r_jerk_v   = cost.at("r_jerk_v").get<double>();
    r_jerk_yaw = cost.at("r_jerk_yaw").get<double>();
    gamma      = cost.at("gamma").get<double>();

    // --- Cost weights — obstacle avoidance ------------------------------- //
    q_obs      = cost.at("q_obs").get<double>();
    sigma_obs  = cost.at("sigma_obs").get<double>();
    q_prog     = cost.value("q_prog", 0.0);        // 0 → Proposal F
    ego_radius = cost.value("ego_radius", 2.5);

    // --- Input limits ---------------------------------------------------- //
    const auto& limits = sp.at("mpc_options").at("input_limits");
    max_accel = limits.at("max_accel").get<double>();
    min_accel = limits.at("min_accel").get<double>();
    max_steer = limits.at("max_steer").get<double>();
    min_steer = limits.at("min_steer").get<double>();

    // --- Waypoints & obstacles ------------------------------------------- //
    n_waypoints = sp.at("mpc_options").at("n_waypoints").get<int>();
    n_obstacles = sp.at("mpc_options").at("n_obstacles").get<int>();
    assert(Ndu == 2 * n_waypoints + 5 * n_obstacles
           && "TNDU must equal 2*n_waypoints + 5*n_obstacles");

    wp_x.resize(n_waypoints);
    wp_y.resize(n_waypoints);
    obstacles.resize(n_obstacles);

    // --- NLMPC configuration --------------------------------------------- //
    nlmpc.setLoggerLevel(mpc::Logger::NORMAL);
    nlmpc.setDiscretizationSamplingTime(sample_time);

    // Kinematic bicycle model: x = [px, py, psi, v]
    nlmpc.setStateSpaceFunction(
        [this](xVec& dx, const xVec& x, const uVec& u, const unsigned int&) {
            const double psi   = x(2);
            const double v     = x(3);
            const double a     = u(0);
            const double delta = u(1);

            dx(0) = v * std::cos(psi);
            dx(1) = v * std::sin(psi);
            dx(2) = (v / wheelbase) * std::tan(delta);
            dx(3) = a;
        });

    // Objective function — waypoint + speed + obstacle barrier + forward progress
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
                const double px  = X(j, 0);
                const double py  = X(j, 1);
                const double psi = X(j, 2);
                const double v   = X(j, 3);

                // ---- Waypoint tracking ---- //
                double dmin_sq = closestWaypointDistSq(px, py);
                J += gj * q_path * dmin_sq;

                // ---- Speed tracking ---- //
                double ev = v - target_speed;
                J += gj * q_speed * ev * ev;

                // ---- Obstacle repulsive barrier ---- //
                const double dt_pred = j * sample_time;
                for (int i = 0; i < n_obstacles; ++i) {
                    if (obstacles[i].x > SENTINEL_THRESHOLD) continue;

                    // Constant-velocity prediction
                    double ox = obstacles[i].x  + dt_pred * obstacles[i].vx;
                    double oy = obstacles[i].y  + dt_pred * obstacles[i].vy;
                    double r  = obstacles[i].radius;

                    double dx_o = px - ox;
                    double dy_o = py - oy;
                    double center_dist_sq = dx_o * dx_o + dy_o * dy_o;

                    // Effective distance: subtract combined bounding radii
                    double combined_r  = ego_radius + r;
                    double d_eff_sq    = center_dist_sq - combined_r * combined_r;
                    if (d_eff_sq < 0.0) d_eff_sq = 0.0;  // overlapping → max penalty

                    J += gj * q_obs * std::exp(-d_eff_sq / (2.0 * sigma_obs * sigma_obs));
                }

                // ---- Forward-progress heading penalty (Proposal E) ---- //
                if (q_prog > 0.0) {
                    int wp_idx     = closestWaypointIndex(px, py);
                    int target_idx = std::min(wp_idx + 1, n_waypoints - 1);
                    double dx_wp   = wp_x[target_idx] - px;
                    double dy_wp   = wp_y[target_idx] - py;
                    double theta_wp = std::atan2(dy_wp, dx_wp);

                    // Penalise negative forward velocity along waypoint heading
                    double forward = v * std::cos(psi - theta_wp);
                    J += gj * q_prog * softplus(-forward);
                }

                gj *= gamma;
            }

            // ---- Control effort ---- //
            double gk = 1.0;
            for (int j = 0; j < Nc; ++j) {
                J += gk * r_accel * U(j, 0) * U(j, 0);
                J += gk * r_steer * U(j, 1) * U(j, 1);
                gk *= gamma;
            }

            // ---- Jerk penalty ---- //
            {
                double da = U(0, 0) - prev_accel;
                double dd = U(0, 1) - prev_steer;
                J += r_jerk_v   * da * da;
                J += r_jerk_yaw * dd * dd;
            }
            for (int j = 1; j < Nc; ++j) {
                double da = U(j, 0) - U(j - 1, 0);
                double dd = U(j, 1) - U(j - 1, 1);
                J += r_jerk_v   * da * da;
                J += r_jerk_yaw * dd * dd;
            }

            return J;
        });

    // Input bounds
    uVec umin, umax;
    umin(0) = min_accel; umin(1) = min_steer;
    umax(0) = max_accel; umax(1) = max_steer;
    nlmpc.setInputBounds(umin, umax, {0, Nc});

    control.setZero();

    // State persistence
    experiment_dir = json_data.value("experiment_dir", "");
    state_file     = experiment_dir.empty() ? "" : experiment_dir + "/mpc_state.json";
    if (!state_file.empty())
        load_state();

    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("End of CarlaObstacleMPCController::setup()")
    }
}

// ------------------------------------------------------------------ //
//  Control step                                                       //
// ------------------------------------------------------------------ //

void CarlaObstacleMPCController::calculateControl(int k, double t,
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

    mpc_result = nlmpc.optimize(state, control);
    if (mpc_result.is_feasible)
        control = mpc_result.cmd;

    prev_accel = control(0);
    prev_steer = control(1);
}

void CarlaObstacleMPCController::postControl(int k, double t,
                                              const xVec& x, const wVec& w) {
    // Count active obstacles for console log
    int active_obs = 0;
    for (int i = 0; i < n_obstacles; ++i)
        if (obstacles[i].x < SENTINEL_THRESHOLD) ++active_obs;

    latest_metadata.clear();
    latest_metadata["k"]             = k;
    latest_metadata["t"]             = t;
    latest_metadata["controller"]    = "CarlaObstacleMPCController";
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

    std::cout << "[CarlaObstacleMPC] k=" << k
              << " a=" << control(0) << " delta=" << control(1)
              << " cost=" << mpc_result.cost
              << " obs=" << active_obs << "/" << n_obstacles
              << (mpc_result.is_feasible ? "" : " FALLBACK")
              << std::endl;

    if (!state_file.empty())
        save_state();
}

// ------------------------------------------------------------------ //
//  State persistence                                                  //
// ------------------------------------------------------------------ //

void CarlaObstacleMPCController::save_state() const {
    nlohmann::json s;
    s["prev_accel"] = prev_accel;
    s["prev_steer"] = prev_steer;
    std::ofstream f(state_file);
    if (f.is_open()) f << s.dump(2);
}

void CarlaObstacleMPCController::load_state() {
    std::ifstream f(state_file);
    if (!f.is_open()) return;
    try {
        nlohmann::json s;
        f >> s;
        prev_accel = s.value("prev_accel", 0.0);
        prev_steer = s.value("prev_steer", 0.0);
    } catch (...) {}
}

// ------------------------------------------------------------------ //
//  Helpers                                                            //
// ------------------------------------------------------------------ //

double CarlaObstacleMPCController::closestWaypointDistSq(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < n_waypoints; ++i) {
        double dx = px - wp_x[i];
        double dy = py - wp_y[i];
        double d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return best;
}

int CarlaObstacleMPCController::closestWaypointIndex(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    int idx = 0;
    for (int i = 0; i < n_waypoints; ++i) {
        double dx = px - wp_x[i];
        double dy = py - wp_y[i];
        double d2 = dx * dx + dy * dy;
        if (d2 < best) { best = d2; idx = i; }
    }
    return idx;
}

REGISTER_CONTROLLER("CarlaObstacleMPCController", CarlaObstacleMPCController)
#endif
