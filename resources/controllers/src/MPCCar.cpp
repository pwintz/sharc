// controller/NLMPCController.cpp
#include "MPCCar.h"

// constants need to be declared globally or segmentation fault happens (why?)

void MPCCar::setup(const nlohmann::json &json_data){
    // Load system parameters
    this->lr = json_data.at("system_parameters").at("lr");
    this->lf = json_data.at("system_parameters").at("lf");
    this->sample_time = json_data.at("system_parameters").at("sample_time");
    this->input_cost_weight = json_data.at("system_parameters")
                                  .at("mpc_options")
                                  .at("input_cost_weight");
    nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setDiscretizationSamplingTime(this->sample_time);

    // Dynamics differential equation
    auto stateEq = [&](xVec &x_dot, const xVec &x, const uVec &u) {
        assert(u.size() == Tnu && "Control vector u has wrong dimension");

        // States
        double X     = x(0);   // global x position
        double Y     = x(1);   // global y position
        double v     = x(2);   // speed
        double psi   = x(3);   // heading angle

        // Inputs
        double a     = u(0);   // acceleration
        double beta  = u(1);   // slip angle

        // Compute derivatives (continuous-time)
        x_dot(0) = v * std::cos(psi + beta);       // x_dot
        x_dot(1) = v * std::sin(psi + beta);       // y_dot
        x_dot(2) = a;                              // v_dot
        x_dot(3) = (v / this->lr) * std::sin(beta);      // psi_dot
    };

    nlmpc.setStateSpaceFunction(
        [&](xVec &dx, const xVec &x, const uVec &u, const unsigned int &) { 
            stateEq(dx, x, u); 
        }
    );
    nlmpc.setObjectiveFunction([&](
        const mpc::mat<prediction_horizon + 1, TNX> &x,
        const mpc::mat<prediction_horizon + 1, TNY> &,
        const mpc::mat<prediction_horizon + 1, TNU> &u,
        double)
    { 
        // State cost weights from JSON
        std::vector<double> w = json_data.at("system_parameters").at("mpc_options").at("state_cost_weights").get<std::vector<double>>();
        Eigen::VectorXd state_cost_weights = 
            Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(w.data(), w.size());

        // Quadratic cost: sum(Qx + Ru)
        double cost = 0.0;
        for (int k = 0; k < prediction_horizon + 1; ++k) {
            Eigen::VectorXd xk = x.row(k);
            Eigen::VectorXd uk = u.row(k);
            cost += (xk.transpose() * state_cost_weights.asDiagonal() * xk)(0,0);
            cost += this->input_cost_weight * uk.squaredNorm();
        }
        return cost;
    });
    nlmpc.setOutputFunction([&](yVec &y, const xVec &x, const uVec &, const unsigned int &) {
        y(0) = x(0);  // x-position
        y(1) = x(1);  // y-position
    });
    
}
void MPCCar::calculateControl(int k, double t, const xVec &x, const wVec &w){
    state = x;
    if (control.size() != Tnu) {
        control.resize(Tnu);
        control.setZero();
    }
    
    // Call NLMPC control calculation here
    nlmpc_step_result = nlmpc.optimize(state, control);
    control = nlmpc_step_result.cmd;

    latest_metadata.clear();
    latest_metadata["iterations"]       = nlmpc_step_result.num_iterations;
    latest_metadata["solver_status"]    = nlmpc_step_result.solver_status;
    latest_metadata["solver_status_msg"]= nlmpc_step_result.solver_status_msg;
    latest_metadata["is_feasible"]      = nlmpc_step_result.is_feasible;
    latest_metadata["cost"]             = nlmpc_step_result.cost;
    latest_metadata["constraint_error"] = nlmpc_step_result.primal_residual;
    latest_metadata["dual_residual"]    = nlmpc_step_result.dual_residual;
    latest_metadata["status"]           = mpc::SolutionStats::resultStatusToString(nlmpc_step_result.status);
}

// Register the controller
REGISTER_CONTROLLER("MPCCar", MPCCar)
