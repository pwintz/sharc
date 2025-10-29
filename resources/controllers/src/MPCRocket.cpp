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
    input_cost_weight = json_data.at("system_parameters").at("mpc_options").at("input_cost_weight");
    output_cost_weight = json_data.at("system_parameters").at("mpc_options").at("output_cost_weight");

    OutputW = mpc::cvec<Tny>::Ones() * output_cost_weight;
    InputW  = mpc::cvec<Tnu>::Ones() * input_cost_weight;
    mpc::cvec<Tnu> DeltaUWeight = mpc::cvec<Tnu>::Zero();

    // Horizon slice (use mpc::HorizonSlice or braced if your helper provides it)
    mpc::HorizonSlice slice(0, pred_hor);

    // Set objective weights on the LMPC
    bool ok = lmpc.setObjectiveWeights(OutputW, InputW, DeltaUWeight, slice);
    if(!ok){
        throw std::runtime_error("Failed to set LMPC objective weights");
    }

    // Constraints: define xmin/xmax vectors (sizes must match Tnx)
    mpc::cvec<Tnx> xmin, xmax;
    xmin << -M_PI/6, -M_PI/6; // <-- resize/initialize correctly for Tnx; adjust if Tnx > 2
    xmax <<  M_PI/6,  M_PI/6;

    // if you need infinities, use mpc::inf or set large numbers consistent with library
    // Example (if more states exist, initialize accordingly)
    // xmin(2) = -mpc::inf; xmax(2) = mpc::inf; etc.

    mpc::cvec<Tny> ymin = mpc::cvec<Tny>::Constant(-mpc::inf);
    mpc::cvec<Tny> ymax = mpc::cvec<Tny>::Constant(mpc::inf);

    mpc::cvec<Tnu> umin = mpc::cvec<Tnu>::Zero();
    mpc::cvec<Tnu> umax = mpc::cvec<Tnu>::Ones() * 30.0;
    umin -= u0;
    umax  -= u0;

    lmpc.setStateBounds(xmin, xmax, slice);
    lmpc.setInputBounds(umin, umax, slice);
    lmpc.setOutputBounds(ymin, ymax, slice);

    // References
    mpc::cvec<Tny> yRef = mpc::cvec<Tny>::Zero();
    // set desired output(s)
    // if Tny==1:
    yRef(0) = 150.0;

    lmpc.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), slice);

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

    // Set the state-space model on the LMPC object (A, B, C)
    lmpc.setStateSpaceModel(Ad, Bd, Cd);

    // Provide disturbance mapping separately
    lmpc.setDisturbances(Bd_disturbance, mpc::mat<Tny, Tndu>::Zero());
}

void MPCRocket::calculateControl(int k, double t, const xVec &x, const wVec &w){
    // Convert x to mpc state vector
    mpc::cvec<Tnx> state;
    for(int i=0;i<Tnx;++i) state(i) = x[i];

    // Call LMPC optimizer exactly as in your working example
    auto lmpc_step_result = lmpc.optimize(state, prev_u);
    mpc::cvec<Tnu> control = lmpc_step_result.cmd;

    // store metadata if you want similar to working example
    // mpc::OptSequence optimal_sequence = lmpc.getOptimalSequence();

    // warm start next time
    prev_u = control;

    // assemble augmented control if required by lower level:
    // if lower-level expects [u, disturbance], make an array and call base method
    double u_aug_arr[Tnu + Tndu];
    for(int i=0;i<Tnu;++i) u_aug_arr[i] = static_cast<double>(control(i));
    for(int j=0;j<Tndu;++j) u_aug_arr[Tnu + j] = 0.0; // no commanded disturbance

    // pass to the actuator interface. If your controller base provides setControlInput,
    // call that (or maybe it's this->setControlInput):
    // If it's a base-class function:
    control = Eigen::Map<mpc::cvec<Tnu>>(u_aug_arr);;
    // else, if you have a different interface, call it accordingly.
}

// Register
REGISTER_CONTROLLER("MPCRocket", MPCRocket)
