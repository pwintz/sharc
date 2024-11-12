// controller/ACC2_Controller.cpp
#include "scarabintheloop_utils.hpp"
#include "ACC2_Controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include <cmath>
#include <algorithm>
#include <debug_levels.hpp>
#include <assert.h> 
#include "nlohmann/json.hpp"

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
    max_brake_acceleration       = json_data.at("system_parameters").at("max_brake_acceleration");
    max_brake_acceleration_front = json_data.at("system_parameters").at("max_brake_acceleration_front");
    beta                     = json_data.at("system_parameters").at("beta");
    gamma                    = json_data.at("system_parameters").at("gamma");

    // Check the values are reasonable.
    assert(mass > 0);
    assert(v_des > 0);
    assert(d_min > 0);
    assert(v_max > 0);
    assert(F_accel_max > 0);
    assert(F_brake_max > 0);
    assert(F_accel_time_constant > 0);
    assert(max_brake_acceleration > 0);
    assert(max_brake_acceleration_front > 0);
    assert(beta  > 0);
    assert(gamma > 0) ;

    // Cost weights
    output_cost_weight       = json_data.at("system_parameters").at("mpc_options").at("output_cost_weight");
    input_cost_weight        = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");
    delta_input_cost_weight  = json_data.at("system_parameters").at("mpc_options").at("delta_input_cost_weight");

    // bool use_state_after_delay_prediction = json_data.at("system_parameters").at("mpc_options").at("use_state_after_delay_prediction");

    updateStateSpaceMatrices(0);
    setOptimizerParameters(json_data);
    setWeights(json_data);
    setConstraints();
    setReferences(json_data);

    if (global_debug_levels.debug_dynamics_level >= 1) {
      printMat("Ad", Ad);
      printMat("Bd", Bd);
      printMat("Bd_disturbance", Bd_disturbance);
      printVector("xmin", xmin);
      printVector("xmax", xmax);
      printVector("ymin", ymin);
      printVector("ymax", ymax);
      printVector("umin", umin);
      printVector("umax", umax);
    }

    // Check that the initial condition does not violate the constraints.
    xVec x0;
    uVec u0;
    yVec y0;
    loadColumnValuesFromJson(x0, json_data, "x0");
    loadColumnValuesFromJson(u0, json_data, "u0");
    control = u0;
    y0 = C * x0;

    PRINT_WITH_FILE_LOCATION("Checking initial conditions satisfy constraints.")
    // assertVectorAlmostLessThan("xmin", xmin, "  x0",   x0, 1e-1);
    // assertVectorAlmostLessThan("  x0",   x0, "xmax", xmax, 1e-1);
    // assertVectorAlmostLessThan("umin", umin, "  u0",   u0, 1e-1);
    // assertVectorAlmostLessThan("  u0",   u0, "umax", umax, 1e-1);
    // assertVectorAlmostLessThan("ymin", ymin, "  y0",   y0, 1e-1);
    // assertVectorAlmostLessThan("  y0",   y0, "ymax", ymax, 1e-1);
    
}

void ACC2_Controller::calculateControl(int k, double t, const xVec &x, const wVec &w){
    // Calculate the control value "u" based on the current state (or state estimate) "x" and the 
    // current exogenuous input (or its estimate) "w".

    PRINT_WITH_FILE_LOCATION("Checking constraints for 'x' at start of calculateControl(k=" << k << ").")
    // assertVectorAlmostLessThan("xmin", xmin, "   x",    x, 1e-1);
    // assertVectorAlmostLessThan("   x",    x, "xmax", xmax, 1e-1);

    state            = x;
    exogeneous_input = w;

    // Update the state matrices to linearize around the current velocity.
    double v = x[v_index];
    updateStateSpaceMatrices(v);

    // Update terminal constraint based on the measurement of the front vehicle's velocity.
    double v_front_underestimate = w[0];
    updateTerminalConstraint(v_front_underestimate);

    for (int col = 0; col < prediction_horizon; col++){
      w_series(0, col) = std::max(0.0, v_front_underestimate - col * sample_time * max_brake_acceleration_front);
      // w_series(0, col) = v_front_underestimate;
      w_series(1, col) = 0.0; 
    }
    PRINT("w (predicted):\n" << w_series)
    lmpc.setExogenuosInputs(w_series);

    // Call LMPC control calculation here
    lmpc_step_result = lmpc.step(state, control);
    control = lmpc_step_result.cmd;
    PRINT("Result of calculateControl(): " << control)

    latest_metadata.clear();
    latest_metadata["iterations"]      = lmpc_step_result.num_iterations;
    latest_metadata["retcode"]         = lmpc_step_result.retcode;
    latest_metadata["cost"]            = lmpc_step_result.cost;
    latest_metadata["constraint_error"] = lmpc_step_result.primal_residual;
    latest_metadata["dual_residual"]   = lmpc_step_result.dual_residual;
    latest_metadata["status"]          = mpc::SolutionStats::resultStatusToString(lmpc_step_result.status);

    mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();
    mat<prediction_horizon, Tnx> opt_state_seq  = optimal_sequence.state;
    mat<prediction_horizon, Tnu> opt_input_seq  = optimal_sequence.input;
    mat<prediction_horizon, Tny> opt_output_seq = optimal_sequence.output ;

    xVec x_opt_next = opt_state_seq.row(0);
    PRINT("Checking if x satisfies xmin <= x <= xmax for k=" << k)
    // assertVectorAlmostLessThan(  "xmin",       xmin, "x^*(0)", x_opt_next, 1e-1);
    // assertVectorAlmostLessThan("x^*(0)", x_opt_next,   "xmax",       xmax, 1e-1);
    
    uVec u_opt_next = opt_input_seq.row(0);
    PRINT("Checking if u satisfies umin <= u <= umax for k=" << k)
    // assertVectorAlmostLessThan(  "umin",       umin, "u^*(0)", u_opt_next, 1e-1);
    // assertVectorAlmostLessThan("u^*(0)", u_opt_next,   "umax",       umax, 1e-1);

    PRINT("control - u_opt_next: " << control - u_opt_next)

    if (global_debug_levels.debug_optimizer_stats_level >= 1) 
    {
      PRINT_WITH_FILE_LOCATION("Optimizer Info")
      PRINT("         Return code: " << lmpc_step_result.retcode)
      PRINT("       Result status: " << lmpc_step_result.status)
      PRINT("Number of iterations: " << lmpc_step_result.num_iterations)
      PRINT("                Cost: " << lmpc_step_result.cost)
      PRINT("    Constraint error: " << lmpc_step_result.primal_residual)
      PRINT("          Dual error: " << lmpc_step_result.dual_residual)
      PRINT("  Optimal x Sequence:\n" << opt_state_seq)
      PRINT("  Optimal u Sequence:\n" << opt_input_seq)
      PRINT("  Optimal y Sequence:\n" << opt_output_seq)
    }

    yVec y = C * x;
    yVec y_err = y - yRef;
    printVector("y", y);
    printVector("y_err", y_err);

    // double v = x[2];
    // double friction_linearized = beta - gamma * std::pow(v0, 2); 
    // double drag_linearized = -2*gamma*v0/mass * v;
    // double accelerator_force = 
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
    double tau = F_accel_time_constant;
    Ac << 0, 0,               1,         0, // p   (position)
          0, 0,              -1,         0, // h   (headway)
          0, 0, -2*gamma*v0/mass,         1, // v   (velocity)
          0, 0,               0,    -1/tau; // F^a (acceleration force)

    // Define continuous-time input matrix B_c
    Bc <<      0,          0, // p   (position)
               0,          0, // h   (headway)
               0,         -1, // v   (velocity)
           1/tau,          0; // F^a (acceleration force)
    //       ^             ^   
    // u:  T^a_ref         T^b    

    printMat("Ac (continuous)", Ac);
    printMat("Bc (continuous)", Bc);

    // State to output matrix
    C <<   0, 0, 1, 0;
    //     ^  ^  ^  ^
    // x:  p  h  v  F^a.

    // Constant force from friction: k = β + γv²
    double k = beta - gamma * std::pow(v0, 2); 
    assert(k > 0);
    PRINT("-2*gamma*v0/mass: " << -2*gamma*v0/mass)
    PRINT("               k: " << k)
    PRINT("         -k/mass: " << -k/mass)

    // Set input disturbance matrix.
    Bc_disturbance << 0, 0, 
                      1, 0, 
                      0, -k/mass, 
                      // 0, 0, 
                      0, 0;
    
    // ======= Discrete-time Matrices ========                         
    // State disturbance matrix
    Cd_disturbance = mat<Tny, Tndu>::Zero() ; // Output disturbance matrix

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

    if (global_debug_levels.debug_dynamics_level >=1){
      printVector(    "outputWeight",     outputWeight);
      printVector(     "inputWeight",      inputWeight);
      printVector("deltaInputWeight", deltaInputWeight);
    }

    lmpc.setObjectiveWeights(outputWeight, inputWeight, deltaInputWeight, {-1, -1});
}

void ACC2_Controller::setConstraints() {
  if (global_debug_levels.debug_program_flow_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Start of ACC2_Controller::setConstraints()")
  }
  const double inf = std::numeric_limits<double>::infinity();

  umin << 0, // acceleration force reference
          0; // braking force

  umax << 1200*F_accel_max/mass, // acceleration force reference
          1200*F_brake_max/mass; // braking force

  xmin <<       0, // p     (position)
            d_min, // h     (headway)
                0, // v     (velocity)
                0; // F^a/M (normalized acceleration force)

  xmax <<          inf, // p     (position)
                   inf, // h     (headway)
                 v_max, // v     (velocity)
               umax[0]; // F^a/M (normalized acceleration force)

  ymin <<     0; // velocity

  ymax << v_max; // velocity

  printVector("xmax: ", xmax);
  printVector("umax: ", umax);
  printVector("ymax: ", ymax);

  lmpc.setConstraints(xmin, umin, ymin,  
                      xmax, umax, ymax, {0, prediction_horizon});

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
  xc  << 0, 1, -v_max/(2*max_brake_acceleration), 0;
  //     ^  ^         ^                ^
  // x:  p  h         v               F^a.

  // Compute the velocity at the prediction horizon if it brakes as fast a possible.
  double worst_case_front_velocity_at_ph 
            = std::max(0.0, v_front_underestimate  - prediction_horizon * max_brake_acceleration_front * sample_time);

  // Calculate "d_min - v_f(end)^2 / 2a_F"
  double constraint_min = d_min - pow(worst_case_front_velocity_at_ph, 2) / (2*max_brake_acceleration_front);

  PRINT("worst_case_front_velocity_at_ph=" << worst_case_front_velocity_at_ph)
  PRINT("                 constraint_min=" << constraint_min)
  PRINT("        needed headway at v_des=" << constraint_min + v_des * v_max/(2*max_brake_acceleration))
  
  uVec uc = uVec::Zero();
  // Set the scalar constraint as a terminal constraint (using the slice "{prediction_horizon-1, prediction_horizon}")
  // means this contraint is only applied at the last step in the prediction horizon.
  // lmpc.setScalarConstraint(constraint_min, inf, xc, uc, {prediction_horizon-1, prediction_horizon});
}

void ACC2_Controller::setReferences(const nlohmann::json &json_data) {
    yRef << v_des;
    assert(v_des > 0);
    PRINT("v_des: " << v_des)
    printVector("yRef", yRef);
    lmpc.setReferences(yRef, uVec::Zero(), uVec::Zero(), {-1, -1});
}

// Register the controller
REGISTER_CONTROLLER("ACC2_Controller", ACC2_Controller)