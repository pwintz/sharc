// controller/CarlaKinematicMPCController.cpp
#include "CarlaKinematicMPCController.h"

#if TNX == 4
#include "sharc/utils.hpp"
#include "debug_levels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

void CarlaKinematicMPCController::setup(const nlohmann::json& json_data) {
    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("Start of CarlaKinematicMPCController::setup()")
    }

    const auto& sp = json_data.at("system_parameters");

    // --- Vehicle / timing ------------------------------------------------ //
    sample_time  = sp.at("sample_time").get<double>();
    wheelbase    = sp.at("mpc_options").at("wheelbase").get<double>();
    target_speed = sp.at("target_speed").get<double>();

    // --- Cost weights ---------------------------------------------------- //
    const auto& cost = sp.at("mpc_options").at("cost_weights");
    q_path     = cost.at("q_path").get<double>();
    q_speed    = cost.at("q_speed").get<double>();
    r_accel    = cost.at("r_accel").get<double>();
    r_steer    = cost.at("r_steer").get<double>();
    r_jerk_v   = cost.at("r_jerk_v").get<double>();
    r_jerk_yaw = cost.at("r_jerk_yaw").get<double>();
    gamma      = cost.at("gamma").get<double>();

    // --- Input limits ---------------------------------------------------- //
    const auto& limits = sp.at("mpc_options").at("input_limits");
    max_accel = limits.at("max_accel").get<double>();
    min_accel = limits.at("min_accel").get<double>();
    max_steer = limits.at("max_steer").get<double>();
    min_steer = limits.at("min_steer").get<double>();

    // --- Waypoints ------------------------------------------------------- //
    n_waypoints = sp.at("mpc_options").at("n_waypoints").get<int>();
    assert(Ndu == 2 * n_waypoints && "TNDU must equal 2 * n_waypoints");

    wp_x.resize(n_waypoints);
    wp_y.resize(n_waypoints);

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
                double px = X(j, 0);
                double py = X(j, 1);
                double v  = X(j, 3);

                double dmin_sq = closestWaypointDistSq(px, py);
                J += gj * q_path * dmin_sq;

                double ev = v - target_speed;
                J += gj * q_speed * ev * ev;

                gj *= gamma;
            }

            double gk = 1.0;
            for (int j = 0; j < Nc; ++j) {
                J += gk * r_accel * U(j, 0) * U(j, 0);
                J += gk * r_steer * U(j, 1) * U(j, 1);
                gk *= gamma;
            }

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

    uVec umin, umax;
    umin(0) = min_accel; umin(1) = min_steer;
    umax(0) = max_accel; umax(1) = max_steer;
    nlmpc.setInputBounds(umin, umax, {0, Nc});

    control.setZero();

    experiment_dir = json_data.value("experiment_dir", "");
    state_file     = experiment_dir.empty() ? "" : experiment_dir + "/mpc_state.json";
    if (!state_file.empty())
        load_state();

    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("End of CarlaKinematicMPCController::setup()")
    }
}

void CarlaKinematicMPCController::calculateControl(int k, double t,
                                           const xVec& x, const wVec& w) {
    for (int i = 0; i < n_waypoints; ++i) {
        wp_x[i] = w(2 * i);
        wp_y[i] = w(2 * i + 1);
    }
    state = x;

    mpc_result = nlmpc.optimize(state, control);
    control    = mpc_result.cmd;

    latest_metadata.clear();
    latest_metadata["k"]                = k;
    latest_metadata["t"]                = t;
    latest_metadata["controller"]       = "CarlaKinematicMPCController";
    latest_metadata["solver_status"]    = mpc_result.solver_status;
    latest_metadata["is_feasible"]      = mpc_result.is_feasible;
    latest_metadata["cost"]             = mpc_result.cost;

    auto opt_seq = nlmpc.getOptimalSequence();
    std::vector<double> traj_x, traj_y;
    for (int j = 0; j <= Np; ++j) {
        traj_x.push_back(opt_seq.state(j, 0));
        traj_y.push_back(opt_seq.state(j, 1));
    }
    latest_metadata["traj_x"] = traj_x;
    latest_metadata["traj_y"] = traj_y;

    std::cout << "[CarlaKinematicMPC] k=" << k
              << " a=" << control(0) << " delta=" << control(1)
              << " cost=" << mpc_result.cost << std::endl;

    prev_accel = control(0);
    prev_steer = control(1);

    if (!state_file.empty())
        save_state();
}

void CarlaKinematicMPCController::save_state() const {
    nlohmann::json s;
    s["prev_accel"] = prev_accel;
    s["prev_steer"] = prev_steer;
    std::ofstream f(state_file);
    if (f.is_open()) f << s.dump(2);
}

void CarlaKinematicMPCController::load_state() {
    std::ifstream f(state_file);
    if (!f.is_open()) return;
    try {
        nlohmann::json s;
        f >> s;
        prev_accel  = s.value("prev_accel", 0.0);
        prev_steer  = s.value("prev_steer", 0.0);
    } catch (...) {}
}

double CarlaKinematicMPCController::closestWaypointDistSq(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < n_waypoints; ++i) {
        double dx = px - wp_x[i];
        double dy = py - wp_y[i];
        double d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return best;
}

void CarlaKinematicMPCController::accelToThrottleBrake(double a,
                                               double& throttle,
                                               double& brake) const {
    if (a >= 0.0) {
        throttle = std::min(a / max_accel, 1.0);
        brake    = 0.0;
    } else {
        throttle = 0.0;
        brake    = std::min(-a / (-min_accel), 1.0);
    }
}

REGISTER_CONTROLLER("CarlaKinematicMPCController", CarlaKinematicMPCController)
#endif

