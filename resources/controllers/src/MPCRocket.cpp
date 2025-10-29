// controller/LMPCController.cpp  (adapted for MPCRocket)
#include "MPCRocket.h"
#include "sharc/utils.hpp"
#include <mpc/LMPC.hpp> // ensure correct include if needed

void MPCRocket::setup(const nlohmann::json &json_data){
    // Load the parameters
    g = json_data.at("system_parameters").at("g");
    mass = json_data.at("system_parameters").at("mass");
    auto jx0 = json_data.at("x0");
    for(int i = 0; i < Tnx; ++i){
        x0(i) = jx0.at(i).get<double>();
    }
    auto ju0 = json_data.at("u0");
    for(int i = 0; i < Tnu; ++i){
        u0(i) = ju0.at(i).get<double>();
    }
    // prediction horizon - ensure this is read or a class default
    pred_hor = json_data.at("system_parameters").at("mpc_options").at("prediction_horizon");

    // Build matrices and set them on the LMPC object
    setUpMatricies(json_data);

    // Zero disturbances initially (shape: Nx x Ndu, Ny x Ndu)
    lmpc.setDisturbances(mpc::mat<Tnx, Tndu>::Zero(),
                         mpc::mat<Tny, Tndu>::Zero());

    // Weights
    mpc::cvec<Tnu> InputW;
    mpc::cvec<Tny> OutputW;
    double outputWeight = json_data.at("system_parameters").at("mpc_options").at("output_cost_weight");
    double inputWeight = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");

    if (outputWeight < 0) {
        throw std::invalid_argument("The output weight was negative.");
    }

    OutputW     = yVec::Ones() * outputWeight;
    InputW      = uVec::Ones() * inputWeight;
    uVec DeltaInputW = uVec::Zero();

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

    // Horizon slice (use mpc::HorizonSlice or braced if your helper provides it)
    mpc::HorizonSlice slice(0, pred_hor);


    // Constraints: define xmin/xmax vectors (sizes must match Tnx)
    mpc::cvec<Tnx> xmin, xmax;
    xmin << 0, 0; // <-- resize/initialize correctly for Tnx; adjust if Tnx > 2
    xmax <<  2000,  10000;

    // if you need infinities, use mpc::inf or set large numbers consistent with library
    // Example (if more states exist, initialize accordingly)
    // xmin(2) = -mpc::inf; xmax(2) = mpc::inf; etc.

    mpc::cvec<Tny> ymin = mpc::cvec<Tny>::Zero();
    mpc::cvec<Tny> ymax = mpc::cvec<Tny>::Constant(mpc::inf);

    mpc::cvec<Tnu> umin = mpc::cvec<Tnu>::Zero();
    mpc::cvec<Tnu> umax = mpc::cvec<Tnu>::Ones() * 30.0;

    lmpc.setStateBounds(xmin, xmax, slice);
    lmpc.setInputBounds(umin, umax, slice);
    lmpc.setOutputBounds(ymin, ymax, slice);

    // References
    mpc::cvec<Tny> yRef = mpc::cvec<Tny>::Zero();
    // set desired output(s)
    // if Tny==1:
    double offset = mass * g;
    mpc::cvec<Tnu> uRef = mpc::cvec<Tnu>::Constant(offset);
    
    
    yRef(0) = 150.0;

    lmpc.setReferences(yRef, uRef, uRef, slice);

    // initialize prev_u for warm-starts
    prev_u = u0;
}

// Build A, B, Bd, C and set them on lmpc (not a separate controller object)
void MPCRocket::setUpMatricies(const nlohmann::json &json_data){
    // Continuous/discrete model (adapt to your system dimensions)
    mpc::mat<Tnx, Tnx> Ad;
    Ad << 0, 1,
          0, 0;

    mpc::mat<Tnx, Tnu> Bd;
    Bd << 0,
          1.0 / mass;

    Bd_disturbance.setZero();        // makes all columns zero
    Bd_disturbance(0, 0) = 0.0;      // first disturbance channel contribution to state 0
    Bd_disturbance(1, 0) = -g;
    
    mpc::mat<Tny, Tnx> Cd;
    Cd.setIdentity();

    Cd_disturbance = mat<Tny, Tndu>::Zero() ; // Output disturbance matrix

    
    // Set the state-space model in LMPC
    lmpc.setStateSpaceModel(Ad, Bd, Cd);
    lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());

    LParameters params;

    params.alpha = 1.6;
    params.rho = 1e-6;
    params.eps_rel = 1e-4;
    params.eps_abs = 1e-4;
    params.eps_prim_inf = 1e-3;
    params.eps_dual_inf = 1e-3;
    params.time_limit = 0;
    params.enable_warm_start = false;
    params.verbose = false;
    params.adaptive_rho = true;
    params.polish = true;

    lmpc.setOptimizerParameters(params);

}

void MPCRocket::calculateControl(int k, double t, const xVec &x, const wVec &w){
    prev_u = control;

    // Convert x to mpc state vector
    state = x;

    // Call LMPC optimizer exactly as in your working example
    lmpc_step_result = lmpc.optimize(state, control);
    control = lmpc_step_result.cmd;

    // store metadata if you want similar to working example
    // mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();

    // warm start next time
    mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();
    auto opt_state_seq  = optimal_sequence.state;
    auto opt_output_seq = optimal_sequence.output ;
    auto opt_input_seq  = optimal_sequence.input;
    
}


// Register
REGISTER_CONTROLLER("MPCRocket", MPCRocket)
