// controller/LMPCController.h
#pragma once

#include "controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

// This code requires the following preprocessor variables to be defined:
// * PREDICTION_HORIZON
// * CONTROL_HORIZON
// * TNX
// * TNU
// * TNDU
// * TNY

class LMPCController : public Controller {
private:
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon    = CONTROL_HORIZON;

    double convergence_termination_tol;
    double sample_time;
    double lead_car_input;
    double tau;
    bool use_state_after_delay_prediction;

    double max_brake_acceleration;       // Braking acceleration magnitude lower bound.
    double max_brake_acceleration_front; // Braking acceleration magnitude lower bound for front vehicle.

    // Create an array to store a series of predictions for w.
    Eigen::Matrix<double, TNDU, PREDICTION_HORIZON> w_series;

    // MPC Computation Result
    Result<Tnu> lmpc_step_result;

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
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
    // Result<Tnu> getLastLmpcStepResult() {
    //   return lmpc_step_result;
    // }
    // Eigen::VectorXd getLatestControl() {
    //   return lmpc_step_result.cmd;
    // }
    // Result<Tnu> getLastLmpcIterations() {
    //   return lmpc_step_result.iterations;
    // }
};
