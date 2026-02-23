#pragma once

#include "controller.h"

class CustomController : public Controller {
public:
    // Constructor that initializes dimensions and calls setup
    CustomController(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }

    void setup(const nlohmann::json &json_data) override;
    void calculateControl(int k, double t, const xVec &x,  const wVec &w) override;
    nlohmann::json getLatestMetadata() const;  // ← add this line

private:
    double Kp, Ki, Kd;
    double dt;
    double target_height;  

    double integral_error = 0.0;
    double prev_error = 0.0;

    double m, g;
};


