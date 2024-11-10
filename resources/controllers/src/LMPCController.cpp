// controller/LMPCController.cpp
#include "scarabintheloop_utils.hpp"
#include "LMPCController.h"

void LMPCController::setup(const nlohmann::json &json_data){
    // Load system parameters
    convergence_termination_tol = json_data.at("system_parameters").at("convergence_termination_tol");
    sample_time                 = json_data.at("system_parameters").at("sample_time");
    lead_car_input              = json_data.at("system_parameters").at("lead_car_input");
    tau                         = json_data.at("system_parameters").at("tau");
    
    bool use_state_after_delay_prediction = json_data.at("system_parameters").at("mpc_options").at("use_state_after_delay_prediction");


    // PRINT_WITH_FILE_LOCATION("Creating Matrices");
    createStateSpaceMatrices(json_data);
    // PRINT_WITH_FILE_LOCATION("Finished creating Matrices");

    setOptimizerParameters(json_data);
    setWeights(json_data);
    setConstraints(json_data);
    setReferences(json_data);
}

void LMPCController::update_internal_state(const Eigen::VectorXd &x){
    state = x;
}

void LMPCController::calculateControl(){
    // Call LMPC control calculation here
    lmpc_step_result = lmpc.step(state, control);
    control = lmpc_step_result.cmd;
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

    cvec<Tny> OutputW     = cvec<Tny>::Ones() * outputWeight;
    cvec<Tnu> InputW      = cvec<Tnu>::Ones() * inputWeight;
    cvec<Tnu> DeltaInputW = cvec<Tnu>::Zero();

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});
}

void LMPCController::setConstraints(const nlohmann::json &json_data) {
    auto constraint_data = json_data.at("system_parameters").at("constraints");
    cvec<Tnx> xmin, xmax;
    loadColumnValuesFromJson(xmin, constraint_data, "xmin");
    loadColumnValuesFromJson(xmax, constraint_data, "xmax");

    cvec<Tny> ymin, ymax;
    loadColumnValuesFromJson(ymin, constraint_data, "ymin");
    loadColumnValuesFromJson(ymax, constraint_data, "ymax");

    cvec<Tnu> umin, umax;
    loadColumnValuesFromJson(umin, constraint_data, "umin");
    loadColumnValuesFromJson(umax, constraint_data, "umax");

    lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});
}

void LMPCController::setReferences(const nlohmann::json &json_data) {
    cvec<Tny> yRef;
    loadColumnValuesFromJson(yRef, json_data.at("system_parameters"), "yref");
    lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});
}

// Register the controller
REGISTER_CONTROLLER("LMPCController", LMPCController)