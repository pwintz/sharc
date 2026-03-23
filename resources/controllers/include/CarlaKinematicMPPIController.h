// controller/CarlaKinematicMPPIController.h
#pragma once

#ifndef CARLA_KINEMATIC_MPPI_CONTROLLER_H
#define CARLA_KINEMATIC_MPPI_CONTROLLER_H

#include "controller.h"
#if TNX == 4

#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

using namespace mpc;

class CarlaKinematicMPPIController : public Controller {
private:
    static constexpr int Nx  = TNX;
    static constexpr int Nu  = TNU;
    static constexpr int Ndu = TNDU;
    static constexpr int Ny  = TNY;
    static constexpr int Np  = PREDICTION_HORIZON;
    static constexpr int Nc  = CONTROL_HORIZON;
    static constexpr int ineq_c = 0;
    static constexpr int eq_c   = 0;

    double wheelbase = 2.87;
    double sample_time   = 0.1;    // [s]
    double target_speed  = 10.0;   // [m/s] reference speed

    double q_path     = 1.0;   
    double q_speed    = 0.5;   
    double r_accel    = 0.0;   
    double r_steer    = 0.0;   
    double r_jerk_v   = 0.1;   
    double r_jerk_yaw = 0.1;   
    double gamma      = 1.0;   

    // MPPI Parameters
    int num_rollouts = 512;
    double lambda = 1.0;
    double sigma_accel = 0.1;
    double sigma_steer = 0.1;
    double rho = 0.2;
    double beta = 0.9;

    double max_accel =  3.0;  // [m/s^2]
    double min_accel = -5.0;  // [m/s^2]
    double max_steer =  0.7;  // [rad]
    double min_steer = -0.7;  // [rad]

    int n_waypoints = 0;

    // Use dynamic allocation for NLMPC to pass OptimizerType at runtime
    std::unique_ptr<mpc::NLMPC<>> nlmpc;
    Result<> mpc_result;

    std::vector<double> wp_x;
    std::vector<double> wp_y;

    double prev_accel = 0.0;
    double prev_steer = 0.0;

    std::string experiment_dir;
    std::string state_file;
    void save_state() const;
    void load_state();

    double closestWaypointDistSq(double px, double py) const;
    void accelToThrottleBrake(double a, double& throttle, double& brake) const;

public:
    CarlaKinematicMPPIController(const nlohmann::json& json_data) : Controller(json_data) {
        setup(json_data);
    }

    void calculateControl(int k, double t, const xVec& x, const wVec& w) override;

protected:
    void setup(const nlohmann::json& json_data) override;
};

#endif // TNX == 4
#endif // CARLA_KINEMATIC_MPPI_CONTROLLER_H
