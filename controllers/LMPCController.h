// controller/LMPCController.h
#pragma once

#ifndef LMPC_CONTROLLER_H
#define LMPC_CONTROLLER_H
#include "controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

// These values are used in compile time so we need to define them inside the code
#ifndef PREDICTION_HORIZON
#define PREDICTION_HORIZON 5
#endif

#ifndef CONTROL_HORIZON
#define CONTROL_HORIZON 3
#endif

#ifndef TNX
#define TNX 5
#endif

#ifndef TNU
#define TNU 1
#endif

#ifndef TNDU
#define TNDU 1
#endif

#ifndef TNY
#define TNY 2
#endif

class LMPCController : public Controller {
private:
    constexpr static int Tnx = TNX;
    constexpr static int Tnu = TNU;
    constexpr static int Tndu = TNDU;
    constexpr static int Tny = TNY;
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon = CONTROL_HORIZON;

    double convergence_termination_tol;
    double sample_time;
    double lead_car_input;
    double tau;

    LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;

    void createStateSpaceMatrices(const nlohmann::json &json_data);
    void setOptimizerParameters(const nlohmann::json &json_data);
    void setWeights(const nlohmann::json &json_data);
    void setConstraints(const nlohmann::json &json_data);
    void setReferences(const nlohmann::json &json_data);

public:
    // Constructor that initializes dimensions and calls setup
    LMPCController(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void update_internal_state(const Eigen::VectorXd &x) override;
    void calculateControl() override;
};
#endif // LMPC_CONTROLLER_H
