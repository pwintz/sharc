// RLC_Controller.cpp — RLC LMPC Controller (ACC-style structure)

#include "RLC_Controller.h"
#include "sharc/utils.hpp"

#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include "nlohmann/json.hpp"
#include <debug_levels.hpp>


// ======= Setup =======
void RLC_Controller::setup(const nlohmann::json &json_data) {
  if (global_debug_levels.debug_program_flow_level >= 2) {
    PRINT_WITH_FILE_LOCATION("Start of RLC_Controller::setup()")
  }

  // ---- Load required parameters ----
  sample_time = json_data.at("system_parameters").at("sample_time").get<double>();
  assert(sample_time > 0.0);

  const auto &sp = json_data.at("system_parameters");
  R = sp.at("R").get<double>();
  L = sp.at("L").get<double>();
  Cval = sp.at("C").get<double>();
  assert(R > 0.0 && L > 0.0 && Cval > 0.0);

  // Reference voltage for vC
  v_ref = sp.at("v_ref").get<double>();

  // ---- MPC/OSQP options (ACC-style) ----
  output_cost_weight      = sp.at("mpc_options").at("output_cost_weight").get<double>();
  input_cost_weight       = sp.at("mpc_options").at("input_cost_weight").get<double>();
  delta_input_cost_weight = sp.at("mpc_options").at("delta_input_cost_weight").get<double>();

  // Optional: robustly read bounds (defaults if missing)
  const double inf = std::numeric_limits<double>::infinity();

  vC_min = sp.contains("vC_min") ? sp.at("vC_min").get<double>() : -inf;
  vC_max = sp.contains("vC_max") ? sp.at("vC_max").get<double>() :  inf;

  iL_min = sp.contains("iL_min") ? sp.at("iL_min").get<double>() : -inf;
  iL_max = sp.contains("iL_max") ? sp.at("iL_max").get<double>() :  inf;

  u_min  = sp.contains("u_min")  ? sp.at("u_min").get<double>()  : -inf;  // source voltage lower bound
  u_max  = sp.contains("u_max")  ? sp.at("u_max").get<double>()  :  inf;  // source voltage upper bound
  assert(u_min < u_max);

  updateStateSpaceMatrices_RLC(R, L, Cval);
  setOptimizerParameters(json_data);
  setWeights(json_data);
  setConstraints();
  setReferences(json_data);

  // Initialize control for warm start
  control.setZero(); // 1×1
  if (global_debug_levels.debug_program_flow_level >= 2) {
    PRINT_WITH_FILE_LOCATION("End of RLC_Controller::setup()")
  }
}

// ======= Per-step control =======
void RLC_Controller::calculateControl(int k, double t, const xVec &x, const wVec & /*w*/) {
  // No exogenous inputs for this simple RLC LMPC.
  state = x;
  xVec States_X = x;
  printVector("X vals", States_X);

  // Solve LMPC with warm start (previous 'control')
  lmpc_step_result = lmpc.optimize(state, control);
  control = lmpc_step_result.cmd; // keep for next warm start

  latest_metadata.clear();
  latest_metadata["iterations"]       = lmpc_step_result.num_iterations;
  latest_metadata["solver_status"]    = lmpc_step_result.solver_status;
  latest_metadata["solver_status_msg"]= lmpc_step_result.solver_status_msg;
  latest_metadata["is_feasible"]      = lmpc_step_result.is_feasible;
  latest_metadata["cost"]             = lmpc_step_result.cost;
  latest_metadata["constraint_error"] = lmpc_step_result.primal_residual;
  latest_metadata["dual_residual"]    = lmpc_step_result.dual_residual;
  latest_metadata["status"]           = mpc::SolutionStats::resultStatusToString(lmpc_step_result.status);

  // Diagnostics
  if (global_debug_levels.debug_optimizer_stats_level >= 1) {
    PRINT_WITH_FILE_LOCATION("RLC LMPC Step")
    PRINT("k=" << k << " t=" << t)
    PRINT("solver_status: " << lmpc_step_result.solver_status
      << "  feasible: " << lmpc_step_result.is_feasible
      << "  iters: " << lmpc_step_result.num_iterations
      << "  cost: " << lmpc_step_result.cost)
    PRINT("primal_res: " << lmpc_step_result.primal_residual
      << "  dual_res: " << lmpc_step_result.dual_residual)
    PRINT("u*: " << control.transpose())
  }

  // (Optional) inspect optimal sequences
  if (global_debug_levels.debug_optimizer_stats_level >= 2) {
    mpc::OptSequence seq = lmpc.getOptimalSequence();
    PRINT("x*:\n" << seq.state)
    PRINT("u*:\n" << seq.input)
    PRINT("y*:\n" << seq.output)
  }

  // (Optional) track y, error
  yVec y = Cmat * state;                   // y = vC
  yVec y_err = y - yRef;                // tracking error
  printVector("y", y);
  printVector("y_err", y_err);
}

// ======= Model (series RLC) =======
void RLC_Controller::updateStateSpaceMatrices_RLC(double R, double L, double C) {
  if (global_debug_levels.debug_program_flow_level >= 2) {
    PRINT_WITH_FILE_LOCATION("Start of updateStateSpaceMatrices_RLC()")
  }

  // Continuous-time model (ZOH discretized)
  // x = [vC; iL], u = v_src
  // vC' = (1/C)*iL
  // iL' = -(1/L)*vC - (R/L)*iL + (1/L)*u
  Ac <<  0.0,     1.0/C,
        -1.0/L,  -R/L;

  Bc <<  0.0,
         1.0/L;

  // Output: track capacitor voltage only (ny = 1)
  Cmat.setZero();
  Cmat(0, 0) = 1.0; // y = vC
  // Cmat(1, 1) = 1.0; // y1 = iL

  // Discretize (your utils provide this overload)
  discretization<Tnx, Tnu>(Ac, Bc, sample_time, Ad, Bd);

  // Register in LMPC
  lmpc.setStateSpaceModel(Ad, Bd, Cmat);

  if (global_debug_levels.debug_dynamics_level >= 1) {
    printMat("Ac", Ac);
    printMat("Bc", Bc);
    printMat("Ad", Ad);
    printMat("Bd", Bd);
    printMat("C",  Cmat);
  }
}

// ======= Optimizer params (OSQP via LMPC) =======
void RLC_Controller::setOptimizerParameters(const nlohmann::json &json_data) {
  if (global_debug_levels.debug_program_flow_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Start of RLC_Controller::setOptimizerParameters()")
  }

  const auto &oq = json_data.at("system_parameters").at("osqp_options");

  LParameters params;
  params.alpha              = 1.6;
  params.rho                = 1e-2; // small but not too small; adaptive_rho will retune
  params.adaptive_rho       = true;
  params.eps_rel            = oq.at("rel_tolerance").get<double>();
  params.eps_abs            = oq.at("abs_tolerance").get<double>();
  params.eps_prim_inf       = oq.at("primal_infeasibility_tolerance").get<double>();
  params.eps_dual_inf       = oq.at("dual_infeasibility_tolerance").get<double>();
  params.time_limit         = 0; // wall-clock disabled (match your ACC choice)
  params.maximum_iteration  = oq.at("maximum_iteration").get<int>();
  params.verbose            = oq.at("verbose").get<bool>();
  params.enable_warm_start  = json_data.at("system_parameters").at("mpc_options").at("enable_mpc_warm_start").get<bool>();
  params.polish             = true;

  lmpc.setOptimizerParameters(params);
}

// ======= Weights =======
void RLC_Controller::setWeights(const nlohmann::json & /*json_data*/) {
  if (global_debug_levels.debug_program_flow_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Start of RLC_Controller::setWeights()")
  }
  if (output_cost_weight < 0.0) {
    throw std::invalid_argument("The output weight was negative.");
  }

  yVec yW;
  yW.setZero();                 // [0, 0]
  yW(0) = output_cost_weight;   // [Q_vC, 0]
  uVec uW = uVec::Ones() * input_cost_weight;
  uVec dW = uVec::Ones() * delta_input_cost_weight;

  if (global_debug_levels.debug_dynamics_level >= 1) {
    printVector("outputWeight(y)", yW);
    printVector("inputWeight(u)",  uW);
    printVector("deltaInputWeight(du)", dW);
  }

  // {-1,-1} → apply over entire horizon (ACC-style)
  lmpc.setObjectiveWeights(yW, uW, dW, {-1, -1});
}

void RLC_Controller::setConstraints() {
  if (global_debug_levels.debug_program_flow_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Start of RLC_Controller::setConstraints()")
  }
  const double inf = std::numeric_limits<double>::infinity();

  // // State bounds (x = [vC; iL])  — OK to use << because Tnx==2
  // xmin << vC_min, iL_min;
  // xmax << vC_max, iL_max;
  xmin.setConstant(-inf);
  xmax.setConstant( inf);

  // // Input bounds (u is 1×1) — OK to use << because Tnu==1
  // umin << u_min;
  // umax << u_max;

  umin.setConstant(-inf);
  umax.setConstant( inf);

  // Output bounds (y = [vC; iL]) — DO NOT use << when TNY=2
  ymin.setConstant(-inf);
  ymax.setConstant( inf);
  //ymin(0) = vC_min;  ymax(0) = vC_max;     // voltage bounds

  if (global_debug_levels.debug_dynamics_level >= 1) {
    printVector("xmin", xmin);
    printVector("xmax", xmax);
    printVector("umin", umin);
    printVector("umax", umax);
    printVector("ymin", ymin);
    printVector("ymax", ymax);
  }

  bool okx = lmpc.setStateBounds( xmin, xmax, {0, prediction_horizon});
  bool oku = lmpc.setInputBounds( umin, umax, {0, control_horizon});
  bool oky = lmpc.setOutputBounds(ymin, ymax, {0, prediction_horizon});
  assert(okx && oku && oky);
}


// ======= References =======
void RLC_Controller::setReferences(const nlohmann::json & /*json_data*/) {
  if (global_debug_levels.debug_program_flow_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Start of RLC_Controller::setReferences()")
  }

  // Track vC to v_ref; iL reference is irrelevant if weight = 0
  yRef << v_ref;

  if (global_debug_levels.debug_dynamics_level >= 1) {
    PRINT("v_ref: " << v_ref)
    printVector("yRef", yRef);
  }

  // No input references; apply across entire horizon
  lmpc.setReferences(yRef, uVec::Zero(), uVec::Zero(), {0, prediction_horizon});
}

// Register the controller
REGISTER_CONTROLLER("RLC_Controller", RLC_Controller)
