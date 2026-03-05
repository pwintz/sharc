// controller/CarlaDynamicMPCController.h
#pragma once

#ifndef CARLA_DYNAMIC_MPC_CONTROLLER_H
#define CARLA_DYNAMIC_MPC_CONTROLLER_H

#include "controller.h"
#if TNX == 6

#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

using namespace mpc;

class CarlaDynamicMPCController : public Controller {
private:
    static constexpr int Nx  = TNX;
    static constexpr int Nu  = TNU;
    static constexpr int Ndu = TNDU;
    static constexpr int Ny  = TNY;
    static constexpr int Np  = PREDICTION_HORIZON;
    static constexpr int Nc  = CONTROL_HORIZON;
    static constexpr int ineq_c = 0;
    static constexpr int eq_c   = 0;

    double mass = 1845.0;
    double Iz = 2500.0;
    double l_f = 1.35;
    double l_r = 1.52;
    double C_f = 80000.0;
    double C_r = 80000.0;
    double sample_time   = 0.1;    // [s]
    double target_speed  = 20.0;   // [m/s] reference speed

    double q_path     = 1.0;   
    double q_speed    = 0.5;   
    double r_accel    = 0.0;   
    double r_steer    = 0.0;   
    double r_jerk_v   = 0.1;   
    double r_jerk_yaw = 0.1;   
    double gamma      = 1.0;   

    double max_accel =  3.0;  // [m/s^2]
    double min_accel = -5.0;  // [m/s^2]  (braking)
    double max_steer =  0.7;  // [rad]
    double min_steer = -0.7;  // [rad]

    int n_waypoints = 0;

    NLMPC<Nx, Nu, Ny, Np, Nc, ineq_c, eq_c> nlmpc;
    Result<Nu> mpc_result;

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
    CarlaDynamicMPCController(const nlohmann::json& json_data) : Controller(json_data) {
        setup(json_data);
    }

    void calculateControl(int k, double t, const xVec& x, const wVec& w) override;

protected:
    void setup(const nlohmann::json& json_data) override;
};

#endif // TNX == 6
#endif // CARLA_DYNAMIC_MPC_CONTROLLER_H
