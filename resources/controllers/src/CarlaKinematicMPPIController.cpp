// controller/CarlaKinematicMPPIController.cpp
#include "CarlaKinematicMPPIController.h"

#if TNX == 4
#include "sharc/utils.hpp"
#include "debug_levels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

void CarlaKinematicMPPIController::setup(const nlohmann::json& json_data) {
    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("Start of CarlaKinematicMPPIController::setup()")
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

    // --- MPPI Options ---------------------------------------------------- //
    if (sp.at("mpc_options").contains("mppi_options")) {
        const auto& mppi = sp.at("mpc_options").at("mppi_options");
        num_rollouts = mppi.at("num_rollouts").get<int>();
        lambda       = mppi.at("lambda").get<double>();
        sigma_accel  = mppi.at("sigma_accel").get<double>();
        sigma_steer  = mppi.at("sigma_steer").get<double>();
        rho          = mppi.at("rho").get<double>();
        beta         = mppi.at("beta").get<double>();
    }

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

    // --- MPPI configuration --------------------------------------------- //
    // Using the explicitly confirmed types and enums from libmpc 1.0.0
    nlmpc = std::make_unique<mpc::NLMPC<>>(Nx, Nu, Ny, Np, Nc, ineq_c, eq_c, mpc::MPPI);
    nlmpc->setLoggerLevel(mpc::Logger::NORMAL);

    // Sample time for discrete simulation in MPPI
    nlmpc->setDiscretizationSamplingTime(sample_time);

    // Kinematic model: px, py, psi, v
    nlmpc->setStateSpaceFunction([this](cvec<>& dx, const cvec<>& x, const cvec<>& u, const unsigned int&) {
        const double psi   = x(2);
        const double v     = x(3);
        const double a     = u(0);
        const double delta = u(1);
        dx(0) = v * std::cos(psi);
        dx(1) = v * std::sin(psi);
        dx(2) = (v / wheelbase) * std::tan(delta);
        dx(3) = a;
    });

    nlmpc->setOutputFunction([](cvec<>& y, const cvec<>& x, const cvec<>&, const unsigned int&) {
        y << x(0), x(1);
    });

    // MPPI objective
    nlmpc->setObjectiveFunction([this](const mat<>& X, const mat<>& Y, const mat<>& U, const double&) {
        double total_cost = 0.0;
        for (int i = 0; i <= Np; ++i) {
            double px = X(i, 0);
            double py = X(i, 1);
            double d2 = closestWaypointDistSq(px, py);
            
            double v = X(i, 3);
            double ev = v - target_speed;
            
            double stage_cost = q_path * d2 + q_speed * ev * ev;
            total_cost += stage_cost;
            
            if (i < Nc) {
                total_cost += r_accel * U(i, 0) * U(i, 0) + r_steer * U(i, 1) * U(i, 1);
            }
        }
        return total_cost;
    });

    // MPPI Parameters
    mpc::MPPIParameters p;
    p.num_rollouts = (size_t)num_rollouts;
    p.maximum_iteration = 10; // Standard for MPPI control
    p.lambda = lambda;
    p.sigma.resize(Nu);
    p.sigma << sigma_accel, sigma_steer;
    p.rho = rho;
    p.beta = beta;
    p.integration_substeps = 5;
    
    nlmpc->setOptimizerParameters(p);

    // Input bounds
    cvec<Nu> umin, umax;
    umin(0) = min_accel; umin(1) = min_steer;
    umax(0) = max_accel; umax(1) = max_steer;
    nlmpc->setInputBounds(umin, umax, {0, Nc});

    control.setZero();

    experiment_dir = json_data.value("experiment_dir", "");
    state_file     = experiment_dir.empty() ? "" : experiment_dir + "/mppi_state.json";
    if (!state_file.empty())
        load_state();

    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("End of CarlaKinematicMPPIController::setup()")
    }
}

void CarlaKinematicMPPIController::calculateControl(int k, double t,
                                           const xVec& x, const wVec& w) {
    // Input Validation
    for (int i = 0; i < x.size(); ++i) {
        if (std::isnan(x(i)) || std::isinf(x(i))) {
            std::cerr << "[CarlaKinematicMPPI] CRITICAL: NaN/Inf detected in state x(" << i << ")=" << x(i) << " at k=" << k << std::endl;
            return;
        }
    }
    for (int i = 0; i < w.size(); ++i) {
        if (std::isnan(w(i)) || std::isinf(w(i))) {
            std::cerr << "[CarlaKinematicMPPI] CRITICAL: NaN/Inf detected in waypoints w(" << i << ")=" << w(i) << " at k=" << k << std::endl;
            return;
        }
    }

    for (int i = 0; i < n_waypoints; ++i) {
        wp_x[i] = w(2 * i);
        wp_y[i] = w(2 * i + 1);
    }
    state = x;

    // In libmpc 1.0.0, the command is in the Result struct
    mpc_result = nlmpc->optimize(state, control);
    control    = mpc_result.cmd;

    latest_metadata.clear();
    latest_metadata["k"]                = k;
    latest_metadata["t"]                = t;
    latest_metadata["controller"]       = "CarlaKinematicMPPIController";
    latest_metadata["cost"]             = mpc_result.cost;
    latest_metadata["iterations"]       = 100;
    latest_metadata["status"]           = "Success";
    latest_metadata["constraint_error"] = 0.0;
    latest_metadata["dual_residual"]     = 0.0;

    auto opt_seq = nlmpc->getOptimalSequence();
    std::vector<double> traj_x, traj_y;
    for (int j = 0; j <= Np; ++j) {
        traj_x.push_back(opt_seq.state(j, 0));
        traj_y.push_back(opt_seq.state(j, 1));
    }
    latest_metadata["traj_x"] = traj_x;
    latest_metadata["traj_y"] = traj_y;

    std::cout << "[CarlaKinematicMPPI] k=" << k
              << " a=" << control(0) << " delta=" << control(1)
              << " cost=" << mpc_result.cost << std::endl;

    prev_accel = control(0);
    prev_steer = control(1);

    if (!state_file.empty())
        save_state();
}

void CarlaKinematicMPPIController::save_state() const {
    nlohmann::json s;
    s["prev_accel"] = prev_accel;
    s["prev_steer"] = prev_steer;
    std::ofstream f(state_file);
    if (f.is_open()) f << s.dump(2);
}

void CarlaKinematicMPPIController::load_state() {
    std::ifstream f(state_file);
    if (!f.is_open()) return;
    try {
        nlohmann::json s;
        f >> s;
        prev_accel  = s.value("prev_accel", 0.0);
        prev_steer  = s.value("prev_steer", 0.0);
    } catch (...) {}
}

double CarlaKinematicMPPIController::closestWaypointDistSq(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < n_waypoints; ++i) {
        double dx = px - wp_x[i];
        double dy = py - wp_y[i];
        double d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return best;
}

void CarlaKinematicMPPIController::accelToThrottleBrake(double a,
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

REGISTER_CONTROLLER("CarlaKinematicMPPIController", CarlaKinematicMPPIController)
#endif
