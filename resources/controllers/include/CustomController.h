// #pragma once
// 
// #include "controller.h"
// 
// class CustomController : public Controller {
// public:
//     // Constructor that initializes dimensions and calls setup
//     CustomController(const nlohmann::json &json_data) : Controller(json_data) {
//         setup(json_data);  // Call setup in the derived class constructor
//     }
// 
//     void setup(const nlohmann::json &json_data) override;
//     void calculateControl(int k, double t, const xVec &x, const wVec &w) override;
// };
