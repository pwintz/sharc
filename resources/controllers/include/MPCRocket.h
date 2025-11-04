#pragma once

#include "controller.h"
#include "sharc/utils.hpp"
#include <mpc/LMPC.hpp>
using namespace mpc;



class MPCRocket : public Controller {
private:

    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon    = CONTROL_HORIZON;


    // Discrete time
    mat<Tnx, Tnx> Ad;
    mat<Tnx, Tnu> Bd;
    mat<Tnx, Tndu> Bd_disturbance; 
    mat<Tny, Tndu> Cd_disturbance;

    double g;
    double mass;
    mpc::cvec<Tnx> x0;
    mpc::cvec<Tnu> u0;
    double input_cost_weight;
    double output_cost_weight;
    int pred_hor;

    Result<Tnu> lmpc_step_result;

    mpc::cvec<Tnu> prev_u;

    

    LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;

    
public:
    // Constructor that initializes dimensions and calls setup
    MPCRocket(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
    void setUpMatricies(const nlohmann::json &json_data);

};