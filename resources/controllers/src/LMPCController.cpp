// controller/LMPCController.cpp
#include "sharc/utils.hpp"
#include "debug_levels.hpp"
#include "LMPCController.h"

void LMPCController::setup(const nlohmann::json &json_data){
    // Load system parameters
    convergence_termination_tol = json_data.at("system_parameters").at("convergence_termination_tol");
    sample_time                 = json_data.at("system_parameters").at("sample_time");
    lead_car_input              = json_data.at("system_parameters").at("lead_car_input");
    tau                         = json_data.at("system_parameters").at("tau");
    
    // bool use_state_after_delay_prediction = json_data.at("system_parameters").at("mpc_options").at("use_state_after_delay_prediction");

    // PRINT_WITH_FILE_LOCATION("Creating Matrices");
    createStateSpaceMatrices(json_data);
    // PRINT_WITH_FILE_LOCATION("Finished creating Matrices");

    setOptimizerParameters(json_data);
    setWeights(json_data);
    setConstraints(json_data);
    setReferences(json_data);
}

void LMPCController::calculateControl(int k, double t, const xVec &x, const wVec &w){
    state = x;

    // Call LMPC control calculation here
    lmpc_step_result = lmpc.optimize(state, control);
    control = lmpc_step_result.cmd;

    latest_metadata.clear();
    latest_metadata["iterations"]       = lmpc_step_result.num_iterations;
    latest_metadata["solver_status"]    = lmpc_step_result.solver_status;
    latest_metadata["solver_status_msg"]= lmpc_step_result.solver_status_msg;
    latest_metadata["is_feasible"]      = lmpc_step_result.is_feasible;
    latest_metadata["cost"]             = lmpc_step_result.cost;
    latest_metadata["constraint_error"] = lmpc_step_result.primal_residual;
    latest_metadata["dual_residual"]    = lmpc_step_result.dual_residual;
    latest_metadata["status"]           = mpc::SolutionStats::resultStatusToString(lmpc_step_result.status);

    mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();
    auto opt_state_seq  = optimal_sequence.state;
    auto opt_output_seq = optimal_sequence.output ;
    auto opt_input_seq  = optimal_sequence.input;

    if (global_debug_levels.debug_optimizer_stats_level >= 1) 
    {
      PRINT_WITH_FILE_LOCATION("Optimizer Info")
      PRINT("       solver_status: " << lmpc_step_result.solver_status)
      PRINT("   solver_status_msg: " << lmpc_step_result.solver_status_msg)
      PRINT("         is_feasible: " << lmpc_step_result.is_feasible)
      PRINT("       Result status: " << lmpc_step_result.status)
      PRINT("Number of iterations: " << lmpc_step_result.num_iterations)
      PRINT("                Cost: " << lmpc_step_result.cost)
      PRINT("    Constraint error: " << lmpc_step_result.primal_residual)
      PRINT("          Dual error: " << lmpc_step_result.dual_residual)
      PRINT("  Optimal x Sequence:\n" << opt_state_seq)
      PRINT("  Optimal u Sequence:\n" << opt_input_seq)
      PRINT("  Optimal y Sequence:\n" << opt_output_seq)
      
    }
}

void LMPCController::createStateSpaceMatrices(const nlohmann::json &json_data) {
    // Define continuous-time state matrix A_c
    mat<Tnx, Tnx> Ac;
    Ac << 0,      1, 0,  0,      0,
          0, -1/tau, 0,  0,      0,
          1,      0, 0, -1,      0,
          0,      0, 0,  0,      1,
          0,      0, 0,  0, -1/tau;

    // Define continuous-time input matrix B_c
    mat<Tnx, Tnu> Bc;
    Bc << 0,
          0,
          0, 
          0, 
          -1/tau;

    // State to output matrix
    mat<Tny, Tnx> C;
    C << 0, 0, 1, 0, 0, 
         1, 0, 0,-1, 0;

    // Set input disturbance matrix.
    mat<Tnx, Tndu> Bc_disturbance;
    Bc_disturbance << 0, 1/tau, 0, 0, 0;
    
    mat<Tnx, Tnx> Ad;
    mat<Tnx, Tnu> Bd;
    mat<Tnx, Tndu> Bd_disturbance;

    discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, sample_time, Ad, Bd, Bd_disturbance);
    
    // Set the state-space model in LMPC
    lmpc.setStateSpaceModel(Ad, Bd, C);
    lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());
}

void LMPCController::setOptimizerParameters(const nlohmann::json &json_data) {
    LParameters params;
    params.alpha = 1.6;
    params.rho = 1e-6;
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

void LMPCController::setWeights(const nlohmann::json &json_data) {
    double outputWeight = json_data.at("system_parameters").at("mpc_options").at("output_cost_weight");
    double inputWeight = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");

    if (outputWeight < 0) {
        throw std::invalid_argument("The output weight was negative.");
    }

    yVec OutputW     = yVec::Ones() * outputWeight;
    uVec InputW      = uVec::Ones() * inputWeight;
    uVec DeltaInputW = uVec::Zero();

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});
}

void LMPCController::setConstraints(const nlohmann::json &json_data) {
    auto constraint_data = json_data.at("system_parameters").at("constraints");
    xVec xmin, xmax;
    loadColumnValuesFromJson(xmin, constraint_data, "xmin");
    loadColumnValuesFromJson(xmax, constraint_data, "xmax");

    yVec ymin, ymax;
    loadColumnValuesFromJson(ymin, constraint_data, "ymin");
    loadColumnValuesFromJson(ymax, constraint_data, "ymax");

    uVec umin, umax;
    loadColumnValuesFromJson(umin, constraint_data, "umin");
    loadColumnValuesFromJson(umax, constraint_data, "umax");

    bool are_x_bounds_okay = lmpc.setStateBounds(xmin, xmax, {-1, -1});
    bool are_u_bounds_okay = lmpc.setInputBounds(umin, umax, {-1, -1});
    bool are_y_bounds_okay = lmpc.setOutputBounds(ymin, ymax, {-1, -1});
    assert(are_x_bounds_okay);
    assert(are_u_bounds_okay);
    assert(are_y_bounds_okay);

    // lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});
}

void LMPCController::setReferences(const nlohmann::json &json_data) {
    yVec yRef;
    loadColumnValuesFromJson(yRef, json_data.at("system_parameters"), "yref");
    lmpc.setReferences(yRef, uVec::Zero(), uVec::Zero(), {0, prediction_horizon});
}

// Register the controller
REGISTER_CONTROLLER("LMPCController", LMPCController)