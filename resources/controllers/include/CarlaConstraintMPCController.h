// controller/CarlaConstraintMPCController.h
// Constraint-based obstacle-aware kinematic MPC controller.
//
// Instead of soft exponential barriers (Proposals E/F), uses HARD constraints:
//   1. Collision avoidance: ||p_ego - p_obs||² ≥ r_safe²  at every Np step
//   2. Lane keeping: |lateral_deviation| ≤ lane_half_width  at every Np step
//
// When a stopped car is ahead, the solver can't find a collision-free +
// in-lane trajectory at the current speed → the optimal action is to brake.
#pragma once

#ifndef CARLA_CONSTRAINT_MPC_CONTROLLER_H
#define CARLA_CONSTRAINT_MPC_CONTROLLER_H

#include "controller.h"

// Only compile when we have a 4-state kinematic model AND constraint slots
#if TNX == 4 && defined(TNIEQ) && TNIEQ > 0

#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

using namespace mpc;

class CarlaConstraintMPCController : public Controller {
private:
    static constexpr int Nx  = TNX;
    static constexpr int Nu  = TNU;
    static constexpr int Ndu = TNDU;
    static constexpr int Ny  = TNY;
    static constexpr int Np  = PREDICTION_HORIZON;
    static constexpr int Nc  = CONTROL_HORIZON;
    static constexpr int ineq_c = TNIEQ;  // Np * (n_obstacles + 1)
    static constexpr int eq_c   = 0;

    // Vehicle parameters
    double wheelbase    = 2.87;
    double sample_time  = 0.1;
    double target_speed = 20.0;
    double effective_target_speed = 20.0;  // adapted per step to leading vehicle

    // Cost weights (tracking only — no obstacle cost terms)
    double q_path     = 1.0;
    double q_speed    = 0.5;
    double r_accel    = 0.0;
    double r_steer    = 0.0;
    double r_jerk_v   = 0.1;
    double r_jerk_yaw = 0.1;
    double gamma      = 1.0;

    // Constraint parameters
    double ego_radius      = 2.5;   // ego bounding-circle radius [m]
    double safe_margin      = 0.5;   // extra buffer beyond bounding radii [m]
    double lane_half_width  = 1.75;  // half lane width [m]

    // Input limits
    double max_accel =  3.0;
    double min_accel = -5.0;
    double max_steer =  0.7;
    double min_steer = -0.7;

    // Waypoints & obstacles
    int n_waypoints = 0;
    int n_obstacles = 0;

    NLMPC<Nx, Nu, Ny, Np, Nc, ineq_c, eq_c> nlmpc;
    Result<Nu> mpc_result;

    std::vector<double> wp_x, wp_y;

    struct ObstacleData { double x, y, vx, vy, radius; };
    std::vector<ObstacleData> obstacles;

    double prev_accel = 0.0;
    double prev_steer = 0.0;
    bool   sticky_hold = false;   // Once stopped near obstacles, stay stopped
    double min_clear_dist = 0.0;  // 2 * r_safe — release hold when all obs this far

    std::string experiment_dir;
    std::string state_file;
    void save_state() const;
    void load_state();

    double closestWaypointDistSq(double px, double py) const;
    double lateralDeviation(double px, double py) const;

public:
    CarlaConstraintMPCController(const nlohmann::json& json_data) : Controller(json_data) {
        setup(json_data);
    }

    void calculateControl(int k, double t, const xVec& x, const wVec& w) override;

protected:
    void setup(const nlohmann::json& json_data) override;
};

#endif // TNX == 4 && TNIEQ > 0
#endif // CARLA_CONSTRAINT_MPC_CONTROLLER_H
