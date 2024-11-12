// controller/ACC2_Controller.h
#pragma once

#include "controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

// This code requires the following preprocessor variables to be defined:
// * PREDICTION_HORIZON
// * CONTROL_HORIZON
// * TNX
// * TNU
// * TNDU
// * TNY

class ACC2_Controller : public Controller {
private:
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon    = CONTROL_HORIZON;

    // double convergence_termination_tol;
    double sample_time;

    // System-specific Parameters
    double mass;  // Mass
    double v_des; // Desired velocity.
    double d_min; // Minimum headway.
    double v_max; // Maximum velocity.
    double F_accel_max; // Maximum acceleration force.
    double F_brake_max; // Maximum braking force.
    double F_accel_time_constant; // Time constant of acceleration force
    double max_braking; // Braking acceleration magnitude lower bound.
    double max_braking_front; // Braking acceleration magnitude lower bound for front vehicle.
    double beta; // From friction force: F = beta + gamma v^2. 
    double gamma;// From friction force: F = beta + gamma v^2. 

    // Cost weights
    double       output_cost_weight;
    double        input_cost_weight; // Input weight in cost function.
    double  delta_input_cost_weight; // Input diff weight in cost funciton.

    double h_prev = 0.0;

    Eigen::Matrix<double, TNDU, PREDICTION_HORIZON> w_series; // Store

    // MPC Computation Result and LMPC object
    Result<Tnu> lmpc_step_result;
    LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;

    // Set the initial constraint values
    void setConstraints();

    // Update the state space matrices based on the linearization of
    // the dynamics around the current velocity.
    void updateStateSpaceMatrices(double v);
    void updateTerminalConstraint(double v_front_underestimate);

    void setOptimizerParameters(const nlohmann::json &json_data);
    void setWeights(const nlohmann::json &json_data);
    void setReferences(const nlohmann::json &json_data);

    // Generated Values Used in MPC Constraints
    mat<Tnx, Tnx>    A;
    mat<Tnu, Tnu>    B;
    mat<Tndu, Tndu> Bd;
    double worst_case_front_velocity; // E.g., "\hat{v}_F"

    // void updateMpcConstraints();

public:
    // Constructor that initializes dimensions and calls setup
    ACC2_Controller(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void calculateControl(const xVec &x, const wVec &w) override;
    
};
