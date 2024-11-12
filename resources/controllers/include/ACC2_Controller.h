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

    // Component indices
    int p_index = 0;
    int h_index = 1; // Higher than mine :`(
    int v_index = 2;
    int F_index = 3;


    // System-specific Parameters
    double mass;  // Mass
    double v_des; // Desired velocity.
    double d_min; // Minimum headway.
    double v_max; // Maximum velocity.
    double F_accel_max; // Maximum acceleration force.
    double F_brake_max; // Maximum braking force.
    double F_accel_time_constant; // Time constant of acceleration force
    double max_brake_acceleration; // Braking acceleration magnitude lower bound.
    double max_brake_acceleration_front; // Braking acceleration magnitude lower bound for front vehicle.
    double beta; // From friction force: F = beta + gamma v^2. 
    double gamma;// From friction force: F = beta + gamma v^2. 

    // Cost weights
    double       output_cost_weight;
    double        input_cost_weight; // Input weight in cost function.
    double  delta_input_cost_weight; // Input diff weight in cost funciton.

    // Create an array to store a series of predictions for w.
    Eigen::Matrix<double, TNDU, PREDICTION_HORIZON> w_series; 

    // MPC Computation Result and LMPC object
    Result<Tnu> lmpc_step_result;
    LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;

    // === State Matrices ===
    // Continuous time
    mat<Tnx, Tnx> Ac;
    mat<Tnx, Tnu> Bc;
    mat<Tnx, Tndu> Bc_disturbance;
    mat<Tny, Tnx> C;

    // Discrete time
    mat<Tnx, Tnx> Ad;
    mat<Tnx, Tnu> Bd;
    mat<Tnx, Tndu> Bd_disturbance;  
    mat<Tny, Tndu> Cd_disturbance;

    // Output Reference Value
    yVec yRef;

    // Box Constraints
    xVec xmin;
    xVec xmax;
    yVec ymin;
    yVec ymax;
    uVec umin;
    uVec umax;

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
    // mat<Tnx, Tnx>    A;
    // mat<Tnu, Tnu>    B;
    // mat<Tndu, Tndu> Bd;
    // double worst_case_front_velocity; // E.g., "\hat{v}_F"

    // void updateMpcConstraints();

public:
    // Constructor that initializes dimensions and calls setup
    ACC2_Controller(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
    
protected:
    void setup(const nlohmann::json &json_data) override;
};
