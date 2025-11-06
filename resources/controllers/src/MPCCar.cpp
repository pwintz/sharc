// controller/NLMPCController.cpp
#include "MPCCar.h"


void MPCCar::setup(const nlohmann::json &json_data){
    // Load system parameters
    this->lr = json_data.at("system_parameters").at("lr");
    this->lf = json_data.at("system_parameters").at("lf");
    this->sample_time = json_data.at("system_parameters").at("sample_time");
    this->input_cost_weight = json_data.at("system_parameters")
                                  .at("mpc_options")
                                  .at("input_cost_weight");
                                  



    // nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setLoggerLevel(mpc::Logger::log_level::ALERT);

    // nlmpc.setLoggerLevel(mpc::Logger::log_level::DEEP);
    nlmpc.setDiscretizationSamplingTime(this->sample_time);


    // Dynamics differential equation
    // nlmpc.setStateSpaceFunction(
    //     [this](mpc::cvec<Tnx>& x_dot,
    //            const mpc::cvec<Tnx>& x,
    //            const mpc::cvec<Tnu>& u,
    //            const unsigned int&) {
    //         assert(u.size() == Tnu && "Control vector u has wrong dimension");
    
    //         if (!x.allFinite() || !u.allFinite()) {
    //             std::cerr << "[ERROR] NaN or Inf in dynamics input!" << std::endl;
    //             x_dot.setZero();
    //             return;
    //         }
    
    //         x_dot.setZero();
    //         double X = x(0), Y = x(1), v = x(2), psi = x(3);
    //         double a = u(0), beta = u(1);
    
    //         x_dot(0) = v * std::cos(psi + beta);
    //         x_dot(1) = v * std::sin(psi + beta);
    //         x_dot(2) = a;
    //         x_dot(3) = (v / this->lr) * std::sin(beta);
    //     });
    nlmpc.setStateSpaceFunction(
        [this](mpc::cvec<Tnx>& x_next,
                   const mpc::cvec<Tnx>& x,
                   const mpc::cvec<Tnu>& u,
                   const unsigned int&) {
            assert(u.size() == Tnu && "Control vector u has wrong dimension");
    
            if (!x.allFinite() || !u.allFinite()) {
                std::cerr << "[ERROR] NaN or Inf in dynamics input!" << std::endl;
                x_next.setZero();
                return;
            }
            double Ts = this-> sample_time;
    
            // Extract states
            double X   = x(0);
            double Y   = x(1);
            double v   = x(2);
            double psi = x(3);
    
            // Inputs
            double a    = u(0);
            double beta = u(1);
    
            // Continuous dynamics (x_dot)
            double x_dot_0 = v * std::cos(psi + beta);
            double x_dot_1 = v * std::sin(psi + beta);
            double x_dot_2 = a;
            double x_dot_3 = (v / this->lr) * std::sin(beta);
    
            // Euler discretization: x_{k+1} = x_k + Ts * x_dot
            x_next(0) = X   + Ts * x_dot_0;
            x_next(1) = Y   + Ts * x_dot_1;
            x_next(2) = v   + Ts * x_dot_2;
            x_next(3) = psi + Ts * x_dot_3;
        });
    
    
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
            Eigen::VectorXd xk = x.row(k).transpose().eval();
            Eigen::VectorXd uk = u.row(k).transpose().eval();
            
            cost += (xk.transpose() * state_cost_weights.asDiagonal() * xk)(0,0);
            cost += this->input_cost_weight * uk.squaredNorm();
        }
        Eigen::VectorXd x_ref(4);
        x_ref << 10.0, 0.0, 0.0, 0.0; // for example, target x=10m ahead

        for (int k = 0; k < prediction_horizon + 1; ++k) {
            Eigen::VectorXd xk = x.row(k).transpose();
            Eigen::VectorXd uk = u.row(k).transpose();
            
            Eigen::VectorXd err = xk - x_ref;  // deviation from reference
            cost += (err.transpose() * state_cost_weights.asDiagonal() * err)(0,0);
            cost += this->input_cost_weight * uk.squaredNorm();
        }
        return cost;
    });
    nlmpc.setOutputFunction([&](yVec &y, const xVec &x, const uVec &, const unsigned int &) {
        y(0) = x(0);  // x-position
        y(1) = x(1);  // y-position
    });
    
    
    // --- pull MPC options (OPTIONAL: use your compile-time constants if available) ---
    const auto& mpc_opts = json_data.at("system_parameters").at("mpc_options");
    const int PRED_H = mpc_opts.at("prediction_horizon").get<int>();  // 20
    const int CTRL_H = mpc_opts.at("control_horizon").get<int>();     // 4

    // --- constraints object ---
    const auto& c = json_data.at("system_parameters").at("constraints");

    // 1) Persist vectors from JSON, then Map (const!) to fixed-size Eigen types
    const std::vector<double> v_umin = c.at("umin").get<std::vector<double>>();  // size = TNU
    const std::vector<double> v_umax = c.at("umax").get<std::vector<double>>();
    const std::vector<double> v_xmin = c.at("xmin").get<std::vector<double>>();  // size = TNX
    const std::vector<double> v_xmax = c.at("xmax").get<std::vector<double>>();

    umin_ = Eigen::Map<const uVec>(v_umin.data());
    umax_ = Eigen::Map<const uVec>(v_umax.data());
    xmin_ = Eigen::Map<const xVec>(v_xmin.data());
    xmax_ = Eigen::Map<const xVec>(v_xmax.data());

    nlmpc.setInputBounds(umin_, umax_, mpc::HorizonSlice{0, CTRL_H});    // length = Tch
    nlmpc.setStateBounds(xmin_, xmax_, mpc::HorizonSlice{0, PRED_H});    // length = Tph


    NLParameters params;

    params.relative_ftol = 1e-10;
    params.relative_xtol = 1e-10;
    params.absolute_ftol = 1e-10;
    params.absolute_xtol = 1e-10;
    params.time_limit = 0;

    params.hard_constraints = true;
    params.enable_warm_start = false;

    nlmpc.setOptimizerParameters(params);
    }
void MPCCar::calculateControl(int k, double t, const xVec &x, const wVec &w){
    state = x;
    // if (control.size() != Tnu) {
    //     control.resize(Tnu);
    //     control.setZero();
    // }
    
    // // Call NLMPC control calculation here
    // nlmpc_step_result = nlmpc.optimize(state, control);
    // control = nlmpc_step_result.cmd;

    // sanity init
    if (control.size()!=Tnu) { control.resize(Tnu); control.setZero(); }

    // simple guards
    auto finite = [](const auto& v){ for (int i=0;i<v.size();++i) if(!std::isfinite(v(i))) return false; return true; };
    if (!finite(state))  throw std::runtime_error("state has NaN/Inf");
    if (!finite(control)) std::cerr << "[warn] control had NaN/Inf, zeroed.\n";

    // one-line summaries
    auto v2s = [](const auto& v){
        std::ostringstream oss; oss.setf(std::ios::fixed); oss<<std::setprecision(6);
        for (int i=0;i<v.size();++i){ if(i) oss<<','; oss<<v(i); } return oss.str();
    };
    std::cerr << "[NLMPC] state=" << v2s(state) << " control=" << v2s(control) << "\n";

    try {
        nlmpc_step_result = nlmpc.optimize(state, control);
    } catch (const std::exception& e) {
        std::cerr << "[NLMPC] optimize() threw: " << e.what() << "\n";
        throw;
    }
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
