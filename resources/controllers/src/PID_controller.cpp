#include "PID_controller.h"
#include "sharc/utils.hpp"

#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include <debug_levels.hpp>
#include <Eigen/Dense>


void PID_controller::setup(const nlohmann::json &json_data){
    // Load parameters from json as you want like below and setup the controller
    // tau = json_data.at("system_parameters").at("tau");

    // why global debug levels cannot be accessed?
    if (global_debug_levels.debug_program_flow_level >= 2) {
    PRINT_WITH_FILE_LOCATION("Start of PID_controller::setup()")
    }

    // ---- Load required parameters ----
    sample_time = json_data.at("system_parameters").at("sample_time").get<double>();
    assert(sample_time > 0.0);

    const auto &sp = json_data.at("system_parameters");
    // add system parameters required for PID_controller

    // Reference speed for vehicle
    target_speed = sp.at("target_speed").get<double>();

    // ---- PID_controller parameters ----
    P_lat = sp.at("PID_options").at("lat").at("P").get<double>();
    I_lat = sp.at("PID_options").at("lat").at("I").get<double>();
    D_lat = sp.at("PID_options").at("lat").at("D").get<double>();

    P_lon = sp.at("PID_options").at("lon").at("P").get<double>();
    I_lon = sp.at("PID_options").at("lon").at("I").get<double>();
    D_lon = sp.at("PID_options").at("lon").at("D").get<double>();

    max_throttle = sp.value("max_throttle", 0.75);
    max_brake    = sp.value("max_brake", 0.3);
    max_steer    = sp.value("max_steer", 0.8);

    lon_integral   = 0.0;
    lon_prev_error = 0.0;
    lon_has_prev   = false;
    lat_integral   = 0.0;
    lat_prev_error = 0.0;
    lat_has_prev   = false;
    last_steer     = 0.0;

    // ---- Cross-batch state persistence ------------------------------- //
    // experiment_dir is the parent directory shared across all batches.
    // If a pid_state.json file exists there (written by the previous batch),
    // we restore the integrator state so the PID behaves as one continuous run.
    experiment_dir  = json_data.value("experiment_dir", "");
    pid_state_file  = experiment_dir.empty() ? "" : experiment_dir + "/pid_state.json";
    if (!pid_state_file.empty())
        load_pid_state();
    // ------------------------------------------------------------------ //
    
    // Initialize control for warm start
    control.setZero(); // 1×1
    if (global_debug_levels.debug_program_flow_level >= 2) {
        PRINT_WITH_FILE_LOCATION("End of PID_controller::setup()")
    }
    std::cout << "setup done" << std::endl;
}

double PID_controller::pidLongitudinal(double target_speed,
                                         double current_speed)
{
    double error = target_speed - current_speed;

    // Running integral (trapezoidal rule)
    lon_integral += error * sample_time;

    // Derivative (backward difference)
    double de = 0.0;
    if (lon_has_prev)
        de = (error - lon_prev_error) / sample_time;
    lon_prev_error = error;
    lon_has_prev   = true;

    double u = P_lon * error
             + I_lon * lon_integral
             + D_lon * de;

    return std::clamp(u, -1.0, 1.0);
}

double PID_controller::pidLateral(double heading_error)
{
    // Running integral (trapezoidal rule)
    lat_integral += heading_error * sample_time;

    // Derivative (backward difference)
    double de = 0.0;
    if (lat_has_prev)
        de = (heading_error - lat_prev_error) / sample_time;
    lat_prev_error = heading_error;
    lat_has_prev   = true;

    double u = P_lat * heading_error
             + I_lat * lat_integral
             + D_lat * de;

    return std::clamp(u, -1.0, 1.0);
}


double computeHeadingError(
    double ego_x, double ego_y,
    double yaw,
    double wp_x, double wp_y)
{
    // Vehicle forward vector
    Eigen::Vector3d v_vec(std::cos(yaw), std::sin(yaw), 0.0);

    // Vector to waypoint
    Eigen::Vector3d w_vec(wp_x - ego_x, wp_y - ego_y, 0.0);

    double wv_norm = v_vec.norm() * w_vec.norm();

    double dot;
    if (wv_norm == 0.0) {
        dot = 1.0;
    } else {
        dot = std::acos(
            std::clamp(v_vec.dot(w_vec) / wv_norm, -1.0, 1.0)
        );
    }

    Eigen::Vector3d cross = v_vec.cross(w_vec);
    if (cross.z() < 0)
        dot *= -1.0;

    return dot; // radians
}

double limitSteerRate(double steer, double last)
{
    if (steer > last + 0.1)
        steer = last + 0.1;
    else if (steer < last - 0.1)
        steer = last - 0.1;

    return steer;
}

// ---------------------------------------------------------------------------
// Cross-batch PID state persistence
// ---------------------------------------------------------------------------

void PID_controller::save_pid_state() const
{
    nlohmann::json state;
    state["lon_integral"]   = lon_integral;
    state["lon_prev_error"] = lon_prev_error;
    state["lon_has_prev"]   = lon_has_prev;
    state["lat_integral"]   = lat_integral;
    state["lat_prev_error"] = lat_prev_error;
    state["lat_has_prev"]   = lat_has_prev;
    state["last_steer"]     = last_steer;

    std::ofstream f(pid_state_file);
    if (f.is_open()) {
        f << state.dump(2);
        std::cout << "[PID] Saved integrator state to " << pid_state_file << std::endl;
    } else {
        std::cerr << "[PID] WARNING: could not write pid_state.json to " << pid_state_file << std::endl;
    }
}

void PID_controller::load_pid_state()
{
    std::ifstream f(pid_state_file);
    if (!f.is_open()) {
        std::cout << "[PID] No prior integrator state found — starting fresh." << std::endl;
        return;
    }

    try {
        nlohmann::json state;
        f >> state;
        lon_integral   = state.value("lon_integral",   0.0);
        lon_prev_error = state.value("lon_prev_error", 0.0);
        lon_has_prev   = state.value("lon_has_prev",   false);
        lat_integral   = state.value("lat_integral",   0.0);
        lat_prev_error = state.value("lat_prev_error", 0.0);
        lat_has_prev   = state.value("lat_has_prev",   false);
        last_steer     = state.value("last_steer",     0.0);
        std::cout << "[PID] Restored integrator state from " << pid_state_file
                  << " (lon_integral=" << lon_integral << ")" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "[PID] WARNING: failed to parse pid_state.json: " << e.what()
                  << " — starting fresh." << std::endl;
    }
}

void PID_controller::calculateControl(int k, double t, const xVec &x, const wVec &w){
    // Calculate the control input, feel free to use internal state and last control as below
    // control = lmpc.step(state, control).cmd;
    std::cout << "start calculation" << std::endl;
    state = x;
    xVec States_X = x;
    printVector("X vals", States_X);

    double ego_x = x(0);
    double ego_y = x(1);
    double yaw   = x(2);
    double speed = x(3);

    double wp_x  = x(4); 
    double wp_y  = x(5);

    double accel = pidLongitudinal(target_speed, speed);

    std::cout << "target_speed loaded = " << target_speed << std::endl;

    std::cout << "speed=" << speed
          << " target=" << target_speed
          << " accel=" << accel << std::endl;

    double heading_error = computeHeadingError(ego_x, ego_y, yaw, wp_x, wp_y);

    double steer = pidLateral(heading_error);
    steer = limitSteerRate(steer, last_steer);
    last_steer = steer;

    //assert(idx_x < x.size());
    //assert(idx_wp_x < w.size());    


    // Same throttle / brake logic
    control(0) = accel >= 0 ? std::min(accel, max_throttle) : 0.0;
    control(1) = accel <  0 ? std::min(-accel, max_brake)  : 0.0;
    control(2) = std::clamp(steer, -max_steer, max_steer);

    // for testing purpose for now, set control to constant values
    //control.setConstant(1.0);

    // set latest metadata
    //latest_metadata = nlohmann::json::object();
    latest_metadata.clear();
    latest_metadata["k"] = k;
    latest_metadata["t"] = t;
    latest_metadata["controller"] = "PID_Controller";

    // Persist integrator + derivative state so the next batch starts warm.
    if (!pid_state_file.empty())
        save_pid_state();

    std::cout << "end calculation" << std::endl;
}
// PID functions


// Register the controller using a name of your choice that will be used in json to call
REGISTER_CONTROLLER("PID_controller", PID_controller)