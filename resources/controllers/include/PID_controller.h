#pragma once

#include "controller.h"
#include "nlohmann/json.hpp"
#include <deque>
#include <algorithm>


class PID_controller : public Controller {
private:
    double  P_lat      = 0.3; 
    double  I_lat      = 0.0;  
    double  D_lat      = 0.05;

    double  P_lon      = 0.4; 
    double  I_lon      = 0.05;  
    double  D_lon      = 0.0;

    double sample_time  = 0.1;
    double target_speed = 60;

    double dt           = 0.03;

    // PID controller 
    double last_steer = 0.0;

    int idx_x, idx_y, idx_yaw, idx_speed;
    int idx_wp_x, idx_wp_y, idx_target_speed;
    // // ---------- Longitudinal PID ----------
    // double kp_lon, ki_lon, kd_lon;
    // double dt;
    std::deque<double> lon_error_buf;

    // // ---------- Lateral PID ----------
    // double kp_lat, ki_lat, kd_lat;
    std::deque<double> lat_error_buf;

    // ---------- Limits ----------
    double max_throttle;
    double max_brake;
    double max_steer;

    // ---------- Helpers ----------
    double pidLongitudinal(double target_speed, double current_speed);
    double pidLateral(double heading_error);

public:
    // Constructor that initializes dimensions and calls setup
    PID_controller(const nlohmann::json &json_data) : Controller(json_data) {
        setup(json_data);  // Call setup in the derived class constructor
    }
   
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;

protected:
    void setup(const nlohmann::json &json_data) override;
};


