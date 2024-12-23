// controller/ACC_Controller.h
#pragma once

#include "controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

// These MUST be overwritten
#ifndef PREDICTION_HORIZON
  #define PREDICTION_HORIZON -1
#endif
#ifndef CONTROL_HORIZON
  #define CONTROL_HORIZON -1
#endif

class ACC_Controller : public Controller {
private:
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon    = CONTROL_HORIZON;

    // double convergence_termination_tol;
    double sample_time;

    // Component indices
    int p_index = 0;
    int h_index = 1; // Higher than mine :`(
    int v_index = 2;

    // System-specific Parameters
    double mass;  // Mass
    double v_des; // Desired velocity.
    double d_min; // Minimum headway.
    double v_max; // Maximum velocity.
    double F_accel_max; // Maximum acceleration force.
    double F_brake_max; // Maximum braking force.
    double max_brake_acceleration; // Braking acceleration magnitude lower bound.
    double max_brake_acceleration_front; // Braking acceleration magnitude lower bound for front vehicle.
    double beta; // From friction force: F = beta + gamma v^2. 
    double gamma;// From friction force: F = beta + gamma v^2. 

    // Cost weights
    double       output_cost_weight;
    double        input_cost_weight; // Input weight in cost function.
    double  delta_input_cost_weight; // Input diff weight in cost function.

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

public:
    // Constructor that initializes dimensions and calls setup
    ACC_Controller(const nlohmann::json &json_data) : Controller(json_data) {
        assert(prediction_horizon > 0);
        assert(control_horizon > 0);
        setup(json_data);  // Call setup in the derived class constructor
    }

    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
    
protected:
    void setup(const nlohmann::json &json_data) override;
};
