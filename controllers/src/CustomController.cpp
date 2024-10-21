#include "CustomController.h"

void CustomController::setup(const nlohmann::json &json_data){
    // Load parameters from json as you want like below and setup the controller
    // tau = json_data.at("system_parameters").at("tau");
}

void CustomController::update_internal_state(const Eigen::VectorXd &x){
    // Use this to update the internal state that is used for control calculations
    state = x;
}

void CustomController::calculateControl(){
    // Calculate the control input, feel free to use internal state and last control as below
    // control = lmpc.step(state, control).cmd;
}

// Register the controller using a name of your choice that will be used in json to call
REGISTER_CONTROLLER("CustomController", CustomController)