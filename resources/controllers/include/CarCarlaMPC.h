// controller/NLMPCController.h
#pragma once

#ifndef NLMPC_CONTROLLER_H
#define NLMPC_CONTROLLER_H
#include "controller.h"
#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

#include "nlohmann/json.hpp"
#include <mutex>
#include <vector>

// This code requires the following preprocessor variables to be defined:
// * PREDICTION_HORIZON
// * CONTROL_HORIZON
// * TNX
// * TNU
// * TNDU
// * TNY

class CarCarlaMPC : public Controller {
private:
    constexpr static int Tnx = TNX;
    constexpr static int Tnu = TNU;
    constexpr static int Tndu = TNDU;
    constexpr static int Tny = TNY;
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon = CONTROL_HORIZON;
    constexpr static int ineq_c = 0;
    constexpr static int eq_c = 0;

    uVec umin_, umax_;
    xVec xmin_, xmax_;

    Eigen::Vector4d x_ref;  

    int n_waypoints = 0;
    std::vector<Eigen::Vector2d> waypoints_;
    mutable std::mutex waypoints_mtx_;

    // constants
    double lr, lf;   
    double sample_time;
    double input_cost_weight;
    int debug_level_ = 0;
    double termVelocity;
    // MPC Computation Result
    Result<Tnu> nlmpc_step_result;

    NLMPC<Tnx, Tnu, Tny, prediction_horizon, control_horizon, ineq_c, eq_c> nlmpc;


    //Var for PID of velocity
    double prev_error = 0.0;
    double integral_error = 0.0;

    double Kp = .5;
    double Ki = .05;
    double Kd = .005;

    double paperObjfunc(
        const mpc::mat<prediction_horizon + 1, Tnx>& X,
        const mpc::mat<prediction_horizon + 1, Tnu>& U,
        const Eigen::VectorXd& Q,
        const Eigen::Vector2d& Rdiag,
        const Eigen::Vector2d& Rd) const;

    double trackingObjfunc(
        const mpc::mat<prediction_horizon + 1, Tnx>& X,
        const mpc::mat<prediction_horizon + 1, Tnu>& U,
        const Eigen::VectorXd& Q,
        const Eigen::Vector2d& Rdiag,
        const Eigen::Vector2d& Rd) const;

    mutable int last_wp_idx_ = 0;

    int closestIdxInWindow(const std::vector<Eigen::Vector2d>& wps,
        double px, double py,
        int last_idx) const;

public:
    // Constructor that initializes dimensions and calls setup
    CarCarlaMPC(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
};
#endif // NLMPC_CONTROLLER_H
