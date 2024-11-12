// controller/NLMPCController.h
#pragma once

#ifndef NLMPC_CONTROLLER_H
#define NLMPC_CONTROLLER_H
#include "controller.h"
#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

#include "nlohmann/json.hpp"

// This code requires the following preprocessor variables to be defined:
// * PREDICTION_HORIZON
// * CONTROL_HORIZON
// * TNX
// * TNU
// * TNDU
// * TNY

class NLMPCController : public Controller {
private:
    constexpr static int Tnx = TNX;
    constexpr static int Tnu = TNU;
    constexpr static int Tndu = TNDU;
    constexpr static int Tny = TNY;
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon = CONTROL_HORIZON;
    constexpr static int ineq_c = 0;
    constexpr static int eq_c = 0;

    // constants
    // double M, m, J, l, c, gamma, g;    
    // double sample_time;
    // double input_cost_weight;

    NLMPC<Tnx, Tnu, Tny, prediction_horizon, control_horizon, ineq_c, eq_c> nlmpc;

public:
    // Constructor that initializes dimensions and calls setup
    NLMPCController(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
};
#endif // NLMPC_CONTROLLER_H
