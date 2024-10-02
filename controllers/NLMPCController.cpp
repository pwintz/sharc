// controller/NLMPCController.cpp
#include "NLMPCController.h"

// constants need to be declared globally or segmentation fault happens (why?)
double M, m, J, l, c, my_gamma, g, sample_time, input_cost_weight;

void NLMPCController::setup(const nlohmann::json &json_data){
    // Load system parameters
    M = json_data["system_parameters"]["M"];
    m = json_data["system_parameters"]["m"];
    J = json_data["system_parameters"]["J"];
    l = json_data["system_parameters"]["l"];
    c = json_data["system_parameters"]["c"];
    my_gamma = json_data["system_parameters"]["gamma"];
    g = json_data["system_parameters"]["g"];
    sample_time = json_data["system_parameters"]["sample_time"];
    input_cost_weight = json_data["system_parameters"]["mpc_options"]["input_cost_weight"];

    nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setContinuosTimeModel(sample_time);

    // Dynamics differential equation
    auto stateEq = [&](
                       mpc::cvec<TNX> &x_dot_,
                       const mpc::cvec<TNX> &x_,
                       const mpc::cvec<TNU> &u)
    {
        // Constants
        double M_t = M + m;
        double J_t = J + m * std::pow(l, 2);
        // Recover state parameters
        double x = x_(0);     // position of the base
        double theta = x_(1); // angle of the pendulum
        double vx = x_(2);    // velocity of the base
        double omega = x_(3); // angular rate of the pendulum

        // Compute common terms
        double s_t = std::sin(theta);
        double c_t = std::cos(theta);
        double o_2 = std::pow(omega, 2);
        double l_2 = std::pow(l, 2);

        // Calculate derivatives
        x_dot_(0) = vx;
        x_dot_(1) = omega;
        x_dot_(2) = (-m * l * s_t * o_2 + m * g * (m * l_2 / J_t) * s_t * c_t -
                     c * vx - (my_gamma / J_t) * m * l * c_t * omega + u(0)) /
                    (M_t - m * (m * l_2 / J_t) * c_t * c_t);

        x_dot_(3) = (-m * l_2 * s_t * c_t * o_2 + M_t * g * l * s_t - c * l * c_t * vx -
                    my_gamma * (M_t / m) * omega + l * c_t * u(0)) /
                    (J_t * (M_t / m) - m * (l * c_t) * (l * c_t));
    };

    nlmpc.setStateSpaceFunction([&](
                                        mpc::cvec<TNX> &dx,
                                        const mpc::cvec<TNX> &x,
                                        const mpc::cvec<TNU> &u,
                                        const unsigned int &)
                                    { stateEq(dx, x, u); });
    nlmpc.setObjectiveFunction([&](
                                       const mpc::mat<prediction_horizon + 1, TNX> &x,
                                       const mpc::mat<prediction_horizon + 1, TNY> &,
                                       const mpc::mat<prediction_horizon + 1, TNU> &u,
                                       double)
                                   { 
                                    std::vector<double> w = json_data["system_parameters"]["mpc_options"]["state_cost_weights"].get<std::vector<double>>();
                                    Eigen::VectorXd state_cost_weights = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(w.data(), w.size()); // create a weight vector
                                    return (x * state_cost_weights.asDiagonal()).array().square().sum() + u.array().square().sum() * input_cost_weight; });
}

void NLMPCController::update_internal_state(const Eigen::VectorXd &x){
    state = x;
}

void NLMPCController::calculateControl(){
    // Call NLMPC control calculation here
    control = nlmpc.step(state, control).cmd;
}

// Register the controller
REGISTER_CONTROLLER("NLMPCController", NLMPCController)