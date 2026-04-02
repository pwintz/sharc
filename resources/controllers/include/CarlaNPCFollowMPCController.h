// controller/CarlaNPCFollowMPCController.h
// Hard-constraint, lead-vehicle-aware CARLA MPC controller.
#pragma once

#ifndef CARLA_NPC_FOLLOW_MPC_CONTROLLER_H
#define CARLA_NPC_FOLLOW_MPC_CONTROLLER_H

#include "controller.h"

#if TNX == 4 && defined(TNIEQ) && TNIEQ > 0

#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>
#include <limits>

using namespace mpc;

class CarlaNPCFollowMPCController : public Controller {
private:
    static constexpr int Nx  = TNX;
    static constexpr int Nu  = TNU;
    static constexpr int Ndu = TNDU;
    static constexpr int Ny  = TNY;
    static constexpr int Np  = PREDICTION_HORIZON;
    static constexpr int Nc  = CONTROL_HORIZON;
    static constexpr int ineq_c = TNIEQ;
    static constexpr int eq_c   = 0;

    struct ObstacleData {
        double x;
        double y;
        double vx;
        double vy;
        double radius;
    };

    double wheelbase = 2.87;
    double sample_time = 0.1;
    double target_speed = 20.0;
    double effective_target_speed = 20.0;
    double last_effective_target_speed = 20.0;

    double q_path = 1.0;
    double q_heading = 1.0;
    double q_speed = 0.5;
    double q_follow_gap = 1.0;
    double r_accel = 0.0;
    double r_steer = 0.0;
    double r_jerk_v = 0.1;
    double r_jerk_yaw = 0.1;
    double gamma = 1.0;

    double ego_radius = 2.5;
    double safe_margin = 0.5;
    double lane_half_width = 1.75;
    double follow_time_gap = 1.6;
    double min_follow_distance = 8.0;
    double target_speed_alpha = 0.35;
    double lead_engage_distance = 30.0;

    double max_accel = 3.0;
    double min_accel = -5.0;
    double max_steer = 0.7;
    double min_steer = -0.7;
    double max_steer_step = 0.01;

    int n_waypoints = 0;
    int n_obstacles = 0;

    NLMPC<Nx, Nu, Ny, Np, Nc, ineq_c, eq_c> nlmpc;
    Result<Nu> mpc_result;

    std::vector<double> wp_x;
    std::vector<double> wp_y;
    std::vector<ObstacleData> obstacles;

    double prev_accel = 0.0;
    double prev_steer = 0.0;

    std::string experiment_dir;
    std::string state_file;

    double closestWaypointDistSq(double px, double py) const;
    double lateralDeviation(double px, double py) const;
    double pathHeading(double px, double py) const;
    static double wrapAngle(double angle);
    bool projectOntoWaypointPath(double px,
                                 double py,
                                 double& path_s,
                                 double& lateral_offset,
                                 double* heading = nullptr) const;
    bool findLeadObstacle(const xVec& x,
                          int& lead_index,
                          double& forward_distance,
                          double& lateral_offset,
                          double& lead_speed) const;
    double limitSteeringStep(double desired_steer) const;
    void save_state() const;
    void load_state();

public:
    CarlaNPCFollowMPCController(const nlohmann::json& json_data) : Controller(json_data) {
        setup(json_data);
    }

    void calculateControl(int k, double t, const xVec& x, const wVec& w) override;

protected:
    void setup(const nlohmann::json& json_data) override;
};

#endif
#endif
