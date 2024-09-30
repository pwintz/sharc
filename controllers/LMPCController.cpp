// controller/LMPCController.cpp
#include "LMPCController.h"

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    // From https://stackoverflow.com/a/26221725/6651650
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ 
      throw std::runtime_error( "string_format(): Error during formatting." ); 
    }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template<int rows, int cols>
void loadMatrixValuesFromJson(mat<rows, cols>& mat_out, json json_data, std::string key) {
      if (json_data[key].size() != rows*cols)
      {
        throw std::invalid_argument(string_format(
              "loadMatrixValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in mat_out (%dx%d).", 
              json_data[key].size(),  key.c_str(), rows, cols));
      }

      int i = 0;
      for (auto& element : json_data[key]) {
        mat_out[i] = element;
        i++;
      }
}

void LMPCController::setup(const nlohmann::json &json_data){
    // Load system parameters
    convergence_termination_tol = json_data["system_parameters"]["convergence_termination_tol"];
    sample_time = json_data["system_parameters"]["sample_time"];
    lead_car_input = json_data["system_parameters"]["lead_car_input"];
    tau = json_data["system_parameters"]["tau"];

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
    control = lmpc.step(state, control).cmd;
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
    params.adaptive_rho = true;
    params.eps_rel = json_data["system_parameters"]["osqp_options"]["rel_tolerance"];
    params.eps_abs = json_data["system_parameters"]["osqp_options"]["abs_tolerance"];
    params.eps_prim_inf = json_data["system_parameters"]["osqp_options"]["primal_infeasibility_tolerance"];
    params.eps_dual_inf = json_data["system_parameters"]["osqp_options"]["dual_infeasibility_tolerance"];
    params.time_limit = json_data["system_parameters"]["osqp_options"]["time_limit"];
    params.maximum_iteration = json_data["system_parameters"]["osqp_options"]["maximum_iteration"];
    params.verbose = json_data["system_parameters"]["osqp_options"]["verbose"];
    params.enable_warm_start = json_data["system_parameters"]["mpc_options"]["enable_mpc_warm_start"];
    params.polish = true;

    lmpc.setOptimizerParameters(params);
}

void LMPCController::setWeights(const nlohmann::json &json_data) {
    double outputWeight = json_data["system_parameters"]["mpc_options"]["output_cost_weight"];
    double inputWeight = json_data["system_parameters"]["mpc_options"]["input_cost_weight"];

    if (outputWeight < 0) {
        throw std::invalid_argument("The output weight was negative.");
    }

    cvec<Tny> OutputW = cvec<Tny>::Ones() * outputWeight;
    cvec<Tnu> InputW = cvec<Tnu>::Ones() * inputWeight;
    cvec<Tnu> DeltaInputW = cvec<Tnu>::Zero();

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});
}

void LMPCController::setConstraints(const nlohmann::json &json_data) {
    auto constraint_data = json_data["system_parameters"]["constraints"];
    cvec<Tnx> xmin, xmax;
    loadMatrixValuesFromJson(xmin, constraint_data, "xmin");
    loadMatrixValuesFromJson(xmax, constraint_data, "xmax");

    cvec<Tny> ymin, ymax;
    loadMatrixValuesFromJson(ymin, constraint_data, "ymin");
    loadMatrixValuesFromJson(ymax, constraint_data, "ymax");

    cvec<Tnu> umin, umax;
    loadMatrixValuesFromJson(umin, constraint_data, "umin");
    loadMatrixValuesFromJson(umax, constraint_data, "umax");

    lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});
}

void LMPCController::setReferences(const nlohmann::json &json_data) {
    cvec<Tny> yRef;
    loadMatrixValuesFromJson(yRef, json_data["system_parameters"], "yref");
    lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});
}

// Register the controller
REGISTER_CONTROLLER("LMPCController", LMPCController)