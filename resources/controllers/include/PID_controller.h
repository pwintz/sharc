#pragma once

#include "controller.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <fstream>
#include <string>


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

    // PID controller 
    double last_steer = 0.0;

    int idx_x, idx_y, idx_yaw, idx_speed;
    int idx_wp_x, idx_wp_y, idx_target_speed;

    // ---------- Longitudinal PID ----------
    double lon_integral  = 0.0;
    double lon_prev_error = 0.0;
    bool   lon_has_prev   = false;

    // ---------- Lateral PID ----------
    double lat_integral  = 0.0;
    double lat_prev_error = 0.0;
    bool   lat_has_prev   = false;

    // ---------- Limits ----------
    double max_throttle;
    double max_brake;
    double max_steer;

    // ---------- Cross-batch state persistence ----------
    std::string experiment_dir;  // path to the experiment dir (parent of all batch dirs)
    std::string pid_state_file;  // experiment_dir/pid_state.json
    void save_pid_state() const;
    void load_pid_state();

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


