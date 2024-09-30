#pragma once

#ifndef CUSTOM_CONTROLLER_H
#define CUSTOM_CONTROLLER_H
#include "controller.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

class CustomController : public Controller {
public:
    // Constructor that initializes dimensions and calls setup
    CustomController(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void update_internal_state(const Eigen::VectorXd &x) override;
    void calculateControl() override;
};
#endif // CUSTOM_CONTROLLER_H
