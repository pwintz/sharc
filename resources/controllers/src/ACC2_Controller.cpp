// controller/ACC2_Controller.cpp
#include "scarabintheloop_utils.hpp"
#include "ACC2_Controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include <cmath>
#include <algorithm>
#include <debug_levels.hpp>

void ACC2_Controller::setup(const nlohmann::json &json_data){
    PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::setup()")

    // Load generic system parameters
    sample_time              = json_data.at("system_parameters").at("sample_time");
    
    // Load system-specific parameters.
    mass                     = json_data.at("system_parameters").at("mass");
    v_des                    = json_data.at("system_parameters").at("v_des");
    d_min                    = json_data.at("system_parameters").at("d_min");
    v_max                    = json_data.at("system_parameters").at("v_max");
    F_accel_max              = json_data.at("system_parameters").at("F_accel_max");
    F_brake_max              = json_data.at("system_parameters").at("F_brake_max");
    F_accel_time_constant    = json_data.at("system_parameters").at("F_accel_time_constant");
    max_braking              = json_data.at("system_parameters").at("max_braking");
    max_braking_front        = json_data.at("system_parameters").at("max_braking_front");
    beta                     = json_data.at("system_parameters").at("beta");
    gamma                    = json_data.at("system_parameters").at("gamma");

    // Cost weights
    output_cost_weight       = 1;
    input_cost_weight        = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");
    delta_input_cost_weight  = json_data.at("system_parameters").at("mpc_options").at("delta_input_cost_weight");

    // bool use_state_after_delay_prediction = json_data.at("system_parameters").at("mpc_options").at("use_state_after_delay_prediction");

    w_series.resize(Tndu, prediction_horizon);

    // PRINT_WITH_FILE_LOCATION("Creating Matrices");
    updateStateSpaceMatrices(0);
    // PRINT_WITH_FILE_LOCATION("Finished creating Matrices");

    setOptimizerParameters(json_data);
    setWeights(json_data);
    setConstraints();
    setReferences(json_data);
}

void ACC2_Controller::calculateControl(const xVec &x, const wVec &w){
    // Calculate the control value "u" based on the current state (or state estimate) "x" and the 
    // current exogenuous input (or its estimate) "w".

    state            = x;
    exogeneous_input = w;

      // Update the state matrices to linearize around the current velocity.
    double v = x[2];
    updateStateSpaceMatrices(v);

    // Update terminal constraint based on the measurement of the front vehicle's velocity.
    double v_front_underestimate = w[0];
    updateTerminalConstraint(v_front_underestimate);

    PRINT_WITH_FILE_LOCATION("w_series.resize(...)")
    w_series.resize(Tndu, prediction_horizon);
    w_series(0, 0) = w[0];
    w_series(0, 1) = w[1];
    for (int col = 1; col < prediction_horizon; col++){
      double prev_worst_case_velocity = w_series(0, col-1);
        w_series(0, col) = std::max(0.0, prev_worst_case_velocity - max_braking_front * sample_time);
        w_series(1, col) = 1.0; 
    }
    lmpc.setExogenuosInputs(w_series);

    // Call LMPC control calculation here
    lmpc_step_result = lmpc.step(state, control);
    control = lmpc_step_result.cmd;

    nlohmann::json metadata_json;

    // if (debug_optimizer_stats_level >= 1) 
    // {
    //   PRINT_WITH_FILE_LOCATION("Optimizer Info")
    //   PRINT_WITH_FILE_LOCATION("         Return code: " << lmpc_step_result.retcode)
    //   PRINT_WITH_FILE_LOCATION("       Result status: " << lmpc_step_result.status)
    //   PRINT_WITH_FILE_LOCATION("Number of iterations: " << lmpc_step_result.num_iterations)
    //   PRINT_WITH_FILE_LOCATION("                Cost: " << lmpc_step_result.cost)
    //   PRINT_WITH_FILE_LOCATION("    Constraint error: " << lmpc_step_result.primal_residual)
    //   PRINT_WITH_FILE_LOCATION("          Dual error: " << lmpc_step_result.dual_residual)
    // }
}

void ACC2_Controller::updateStateSpaceMatrices(double v0) {
    /* 
    Update the state space matrices based on the linearization of the dynamics centered at 
    the current velocity "v0".
    */
    if (global_debug_levels.debug_program_flow_level >= 2){ 
      PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::updateStateSpaceMatrices")
    }

    // Define continuous-time state matrix A_c
    mat<Tnx, Tnx> Ac;
    double tau = F_accel_time_constant;
    Ac << 0, 0,          1,         0, // p   (position)
          0, 0,         -1,         0, // h   (headway)
          0, 0, 2*gamma*v0,    1/mass, // v   (velocity)
          0, 0,          0,    -1/tau; // F^a (acceleration force)

    // Define continuous-time input matrix B_c
    mat<Tnx, Tnu> Bc;
    Bc <<      0,          0, // p   (position)
               0,          0, // h   (headway)
               0,    -1/mass, // v   (velocity)
           1/tau,          0; // F^a (acceleration force)
    //       ^             ^   
    // u:  T^a_ref         T^b    

    // State to output matrix
    mat<Tny, Tnx> C;
    C <<   0, 0, 1, 0;
    //     ^  ^  ^  ^
    // x:  p  h  v  F^a.

    // Constant force from friction: k = β + γv²
    double k = beta + gamma * std::pow(v0, 2); 

    // Set input disturbance matrix.
    mat<Tnx, Tndu> Bc_disturbance;
    Bc_disturbance << 0, 0, 
                      1, 0, 
                      0, k, 
                      0, 0;
    
    // ======= Discrete-time Matrices ========
    mat<Tnx, Tnx> Ad;
    mat<Tnx, Tnu> Bd;
    mat<Tnx, Tndu> Bd_disturbance;                           // State disturbance matrix
    mat<Tny, Tndu> Cd_disturbance = mat<Tny, Tndu>::Zero() ; // Output disturbance matrix

    discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, sample_time, Ad, Bd, Bd_disturbance);
    
    // Set the state-space model in LMPC
    lmpc.setStateSpaceModel(Ad, Bd, C);
    lmpc.setDisturbances(Bd_disturbance, Cd_disturbance);
}

void ACC2_Controller::setOptimizerParameters(const nlohmann::json &json_data) {
    PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::setOptimizerParameters")
    LParameters params;
    params.alpha = 1.6;
    params.rho   = 1e-6;
    params.adaptive_rho      = true;
    params.eps_rel           = json_data.at("system_parameters").at("osqp_options").at("rel_tolerance");
    params.eps_abs           = json_data.at("system_parameters").at("osqp_options").at("abs_tolerance");
    params.eps_prim_inf      = json_data.at("system_parameters").at("osqp_options").at("primal_infeasibility_tolerance");
    params.eps_dual_inf      = json_data.at("system_parameters").at("osqp_options").at("dual_infeasibility_tolerance");
    params.time_limit        = json_data.at("system_parameters").at("osqp_options").at("time_limit");
    params.maximum_iteration = json_data.at("system_parameters").at("osqp_options").at("maximum_iteration");
    params.verbose           = json_data.at("system_parameters").at("osqp_options").at("verbose");
    params.enable_warm_start = json_data.at("system_parameters").at("mpc_options").at("enable_mpc_warm_start");
    params.polish            = true;

    lmpc.setOptimizerParameters(params);
}

void ACC2_Controller::setWeights(const nlohmann::json &json_data) {
    PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::setWeights()")
    if (output_cost_weight < 0) {
        throw std::invalid_argument("The output weight was negative.");
    }

    // mat<Tny, Tny> OutputWMat     = mat<Tnx, Tnx>::Identity();
    // mat<Tnu, Tnu> InputWMat;     // = mat<Tnu>::Ones() * inputWeight;
    // mat<Tnu> DeltaInputWMat = mat<Tnu, Tnu>::Identity() * delta_input_cost_weight;

    yVec outputWeight     = yVec::Ones() * output_cost_weight;
    uVec inputWeight      = uVec::Ones() * input_cost_weight;
    uVec deltaInputWeight = uVec::Ones() * delta_input_cost_weight;

    printVector("outputWeight",     outputWeight);
    printVector("inputWeight",      inputWeight);
    printVector("deltaInputWeight", deltaInputWeight);

    lmpc.setObjectiveWeights(outputWeight, inputWeight, deltaInputWeight, {0, prediction_horizon});
}

void ACC2_Controller::setConstraints() {
  PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::setConstraints()")
  const double inf = std::numeric_limits<double>::infinity();
  xVec xmin;
  xVec xmax;
  yVec ymin;
  yVec ymax;
  uVec umin;
  uVec umax;

  xmin <<    -inf, // p   (position)
            d_min, // h   (headway)
                0, // v   (velocity)
                0; // F^a (acceleration force)

  xmax <<     inf, // p   (position)
              inf, // h   (headway)
            v_max, // v   (velocity)
      F_accel_max; // F^a (acceleration force)

  ymin <<     0; // velocity

  ymax << v_max; // velocity

  umin << 0, // acceleration force reference
          0; // braking force

  umax << F_accel_max, // acceleration force reference
          F_brake_max; // braking force

  lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});

  // Initially, set a terminal constraint that assume the front vehicle is not moving, 
  // until we have a sensor measurement to the contrary.
  updateTerminalConstraint(0);
}

void ACC2_Controller::updateTerminalConstraint(double v_front_underestimate) {
  // ==== Set the terminal constraint ====
  //    h(end) - v(end)*v_max/2a > d_min - v_f(end)^2 / 2a_F
  if (global_debug_levels.debug_program_flow_level >= 2) {
    PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::updateTerminalConstraint()")
  }

  // Create a vector 'c' for a constraint in the form "cᵀ x ≥ s_min."
  xVec xc;
  xc   << 0, 1, -v_max/(2*max_braking), 0;
  //     ^  ^         ^                ^
  // x:  p  h         v               F^a.

  // Compute the velocity at the prediction horizon if it brakes as fast a possible.
  double worst_case_front_velocity_at_ph 
            = std::max(0.0, v_front_underestimate  - prediction_horizon * max_braking_front * sample_time);

  // Calculate "d_min - v_f(end)^2 / 2a_F"
  double constraint_min = d_min - pow(worst_case_front_velocity_at_ph, 2) / (2*max_braking_front);
  
  uVec uc = uVec::Zero();
  // Set the scalar constraint as a terminal constraint (using the slice "{prediction_horizon, prediction_horizon}"
  // means this contraint is only applied at the last step in the prediction horizon.
  lmpc.setScalarConstraint(constraint_min, inf, xc, uc, {prediction_horizon, prediction_horizon});
}

void ACC2_Controller::setReferences(const nlohmann::json &json_data) {
    yVec yRef;
    yRef << v_des;
    lmpc.setReferences(yRef, uVec::Zero(), uVec::Zero(), {0, prediction_horizon});
}

// Register the controller
REGISTER_CONTROLLER("ACC2_Controller", ACC2_Controller)