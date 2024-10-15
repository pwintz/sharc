// controller/NLMPCController.h
#pragma once

#ifndef NLMPC_CONTROLLER_H
#define NLMPC_CONTROLLER_H
#include "controller.h"
#include <mpc/NLMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

// These values are used in compile time so we need to define them inside the code
#ifndef PREDICTION_HORIZON
#define PREDICTION_HORIZON 5
#endif

#ifndef CONTROL_HORIZON
#define CONTROL_HORIZON 2
#endif

#ifndef TNX
#define TNX 4
#endif

#ifndef TNU
#define TNU 1
#endif

#ifndef TNDU
#define TNDU 0
#endif

#ifndef TNY
#define TNY 0
#endif

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
    void update_internal_state(const Eigen::VectorXd &x) override;
    void calculateControl() override;
};
#endif // NLMPC_CONTROLLER_H
