// controller/LMPCController.cpp  (adapted for MPCRocket)
#include "MPCRocket.h"
#include "sharc/utils.hpp"
#include <mpc/LMPC.hpp> // ensure correct include if needed

void MPCRocket::setup(const nlohmann::json &json_data){
    // Load the parameters
    g = json_data.at("system_parameters").at("g");
    mass = json_data.at("system_parameters").at("mass");
    double refheight = json_data.at("controller_parameters").at("target_height");
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
    lmpc.setDisturbances(Bd_disturbance,Cd_disturbance);

    // Weights

    double outputWeight = json_data.at("system_parameters").at("mpc_options").at("output_cost_weight");
    double inputWeight = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");

    if (outputWeight < 0) {
        throw std::invalid_argument("The output weight was negative.");
    }

    mpc::cvec<Tny> OutputW = mpc::cvec<Tny>::Ones() * outputWeight;
    mpc::cvec<Tnu> InputW  = mpc::cvec<Tnu>::Ones() * inputWeight;
    mpc::cvec<Tnu> DeltaUWeight = mpc::cvec<Tnu>::Constant(0.1);

    mpc::HorizonSlice slice(0, pred_hor);

    lmpc.setObjectiveWeights(OutputW, InputW, DeltaUWeight, slice);

    // Horizon slice (use mpc::HorizonSlice or braced if your helper provides it)


    mpc::cvec<Tnx> xmin, xmax;

    // Constraints: define xmin/xmax vectors (sizes must match Tnx)
    xmin = mpc::cvec<Tnx>::Constant(-mpc::inf);
    xmax = mpc::cvec<Tnx>::Constant( mpc::inf);
    // example override for physical bounds (change to your real bounds)
    xmin(0) = 0.0;            // altitude >= 0
    xmin(1) = -mpc::inf;      // velocity unbounded below (example)
    xmax(0) = 2000.0;         // altitude upper bound
    xmax(1) = 10000.0;        // velocity upper bound (example)

    // if you need infinities, use mpc::inf or set large numbers consistent with library
    // Example (if more states exist, initialize accordingly)
    // xmin(2) = -mpc::inf; xmax(2) = mpc::inf; etc.

    mpc::cvec<Tny> ymin = mpc::cvec<Tny>::Constant(-mpc::inf);
    mpc::cvec<Tny> ymax = mpc::cvec<Tny>::Constant(mpc::inf);

    mpc::cvec<Tnu> umin = mpc::cvec<Tnu>::Zero();
    mpc::cvec<Tnu> umax = mpc::cvec<Tnu>::Ones() * 30.0;

    lmpc.setStateBounds(xmin, xmax, slice);
    lmpc.setInputBounds(umin, umax, slice);
    lmpc.setOutputBounds(ymin, ymax, slice);


    // References
    double offset = mass * g;                  // ~19.62 N
    mpc::cvec<Tnu> uRef = mpc::cvec<Tnu>::Constant(offset);
    mpc::cvec<Tny> yRef = mpc::cvec<Tny>::Zero();
    yRef(0) = refheight;                           // meters
    lmpc.setReferences(yRef, uRef, uRef, slice);

    // Warm start with absolute last control
    prev_u = u0;                      

    mpc::LParameters params;

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


// Build A, B, Bd, C and set them on lmpc (not a separate controller object)
void MPCRocket::setUpMatricies(const nlohmann::json &json_data){
    double dt = json_data.at("system_parameters").at("sample_time");
    
    mpc::mat<Tnx, Tnx> Ad;
    Ad << 1.0, dt,
    0.0, 1.0;

    mpc::mat<Tnx, Tnu> Bd;
    double b0 = (dt*dt) / (2.0 * mass);
    double b1 = dt / mass;
    Bd << b0,
        b1;

    // Output: height only
    mpc::mat<Tny, Tnx> Cd;
    Cd.setZero();
    Cd(0,0) = 1.0;

    // No explicit disturbance model (simplest, consistent with uRef = m*g)
    lmpc.setStateSpaceModel(Ad, Bd, Cd);
    //lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());

    

}

void MPCRocket::calculateControl(int k, double t, const xVec &x, const wVec &w){

    // Convert x to mpc state vector
    state = x;

    // Call LMPC optimizer exactly as in your working example
    lmpc_step_result = lmpc.optimize(state, prev_u);
    control = lmpc_step_result.cmd;

    // store metadata if you want similar to working example
    // mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();

    prev_u = control;

    // warm start next time
    mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();
    auto opt_state_seq  = optimal_sequence.state;
    auto opt_output_seq = optimal_sequence.output ;
    auto opt_input_seq  = optimal_sequence.input;

    double u_aug_arr[Tnu + Tndu];
    for(int i=0;i<Tnu;++i) u_aug_arr[i] = static_cast<double>(control(i));
    for(int j=0;j<Tndu;++j) u_aug_arr[Tnu + j] = 0.0; // no commanded disturbance

    
}


// Register
REGISTER_CONTROLLER("MPCRocket", MPCRocket)
