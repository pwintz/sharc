#include "CustomController.h" 
void CustomController::setup(const nlohmann::json &json_data){
    m = json_data.at("system_parameters").at("mass"); 
    g = json_data.at("system_parameters").at("g"); 
    dt = json_data.at("system_parameters").at("sample_time"); 
    // Load PID parameters 
    Kp = json_data.at("controller_parameters").at("Kp"); 
    Ki = json_data.at("controller_parameters").at("Ki"); 
    Kd = json_data.at("controller_parameters").at("Kd"); 
    // Desired height 
    target_height = json_data.at("controller_parameters").at("target_height"); } 
    
void CustomController::calculateControl(int k, double t, const xVec &x, const wVec &w){ 
    // State vector: [h, v] 
    double velocity = x(1); 
    double height = x(0); 
    // Compute tracking error 
    double error = target_height - height; 
    // PID terms 
    integral_error += error * dt; 
    double derivative_error = (error - prev_error) / dt; 
    // Basic PID control (with gravity compensation) 
    double F = m * g + Kp * error + Ki * integral_error + Kd * derivative_error; 
    // no neg thrusts 
    if (F < 0.0) F = 0.0;
    if (F > 50) F = 50;
     // Set control output
    control(0) = F; 
    // Save error for next iteration 
    prev_error = error; } 

nlohmann::json CustomController::getLatestMetadata() const {
    nlohmann::json metadata = nlohmann::json::object(); 
    return metadata; }
    
// Register the controller using a name of your choice that will be used in json to call 
REGISTER_CONTROLLER("CustomController", CustomController)