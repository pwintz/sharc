// controller/NLMPCController.cpp
#include "MPCCar.h"


void MPCCar::setup(const nlohmann::json &json_data){
    // Load system parameters
    this->debug_level_ = json_data.at("==== Debgugging Levels ====")
    .at("debug_program_flow_level")
    .get<int>();
    this->lr = json_data.at("system_parameters").at("lr").get<double>();
    this->lf = json_data.at("system_parameters").at("lf").get<double>();
    this->sample_time = json_data.at("system_parameters").at("sample_time").get<double>();
                                
    // Weights for the cost function                              
    auto jx = json_data.at("controller_parameters").at("terminal_state");
    for(int i = 0; i < Tnx; ++i){
        x_ref(i) = jx.at(i).get<double>();
    }
    //Initiallizations
    Eigen::Vector4d Q;
    Eigen::Vector4d Qf;
    Eigen::Vector2d Rdiag;     // a, beta
    Eigen::Vector2d Rd; // Δa, Δbeta   
    Eigen::Vector2d u_ref; u_ref << 0.0, 0.0;        
    auto jc = json_data.at("controller_parameters");
    auto jQ  = jc.at("Q");
    auto jQf = jc.at("Qf");
    auto jR  = jc.at("Rdiag");
    auto jRd = jc.at("Rd");

    for (int i=0; i<4; ++i) {
        Q(i)  = jQ.at(i).get<double>();
        Qf(i) = jQf.at(i).get<double>();
    }
    for (int i=0; i<2; ++i) {
        Rdiag(i) = jR.at(i).get<double>();
        Rd(i)    = jRd.at(i).get<double>();
    }                    



    // nlmpc.setLoggerLevel(mpc::Logger::log_level::NORMAL);
    nlmpc.setLoggerLevel(mpc::Logger::log_level::ALERT);

    // nlmpc.setLoggerLevel(mpc::Logger::log_level::DEEP);
    nlmpc.setDiscretizationSamplingTime(this->sample_time);

    // Dynamics differential equation
    nlmpc.setStateSpaceFunction(
        [this](mpc::cvec<Tnx>& dx,
               const mpc::cvec<Tnx>& x,
               const mpc::cvec<Tnu>& u,
               const unsigned int&) {
            assert(u.size() == Tnu && "Control vector u has wrong dimension");
    
            if (!x.allFinite() || !u.allFinite()) {
                std::cerr << "[ERROR] NaN or Inf in dynamics input!\n";
                dx.setZero();
                return;
            }
    
            const double v   = x(2);
            const double psi = x(3);
            const double a    = u(0);
            const double beta = u(1);
    
            dx(0) = v * std::cos(psi + beta);        // Ẋ
            dx(1) = v * std::sin(psi + beta);        // Ẏ
            dx(2) = a;                                // v̇
            dx(3) = (v / this->lr) * std::sin(beta); // ψ̇
        });
    
    nlmpc.setObjectiveFunction(
        [this, Q, Qf, Rdiag, Rd, u_ref](
            const mpc::mat<prediction_horizon + 1, Tnx>& X,
            const mpc::mat<prediction_horizon + 1, Tny>& /*Y*/,
            const mpc::mat<prediction_horizon + 1, Tnu>& U,
            const double& /*t*/)
        {
            constexpr double BIG = 1e12;
    
            if (!X.allFinite() || !U.allFinite()) return BIG;
    
            double J = 0.0;
    
            for (int k = 0; k < X.rows() - 1; ++k) {
                Eigen::Vector4d e = X.row(k).transpose() - x_ref;
                J += (Q.array() * e.array().square()).sum();
    
                if (k < U.rows()) {
                    Eigen::Vector2d uk = U.row(k).transpose();
                    J += (Rdiag.array() * uk.array().square()).sum();
                }
            }

            Eigen::Vector4d ef = X.row(X.rows() - 1).transpose() - x_ref;

            J += (Qf.array() * ef.array().square()).sum();

            if (!std::isfinite(J)) return BIG;
            return J;
        });
        
    
    
    nlmpc.setOutputFunction([&](yVec &y, const xVec &x, const uVec &, const unsigned int &) {
        y(0) = x(0);  // x-position
        y(1) = x(1);  // y-position
    });
    
    
    // --- pull MPC options -
    const auto& mpc_opts = json_data.at("system_parameters").at("mpc_options");
    const int PRED_H = mpc_opts.at("prediction_horizon").get<int>();
    const int CTRL_H = mpc_opts.at("control_horizon").get<int>(); 
    
    // Enforce compile-time horizons match JSON
    if ( PRED_H != PREDICTION_HORIZON) {
        std::ostringstream oss;
        oss << "prediction_horizon mismatch: JSON has " << PRED_H
            << ", but controller is compiled with " << PREDICTION_HORIZON
            << ". Update JSON or recompile with matching horizon.";
        throw std::runtime_error(oss.str());
    }

    if (CTRL_H != CONTROL_HORIZON) {
        std::ostringstream oss;
        oss << "control_horizon mismatch: JSON has " << CTRL_H
            << ", but controller is compiled with " << CONTROL_HORIZON
            << ". Update JSON or recompile with matching horizon.";
        throw std::runtime_error(oss.str());
    }

    // --- constraints object ---
    const auto& c = json_data.at("system_parameters").at("constraints");

    // 1) Persist vectors from JSON, then Map (const!) to fixed-size Eigen types
    const std::vector<double> v_umin = c.at("umin").get<std::vector<double>>();  // size = TNU
    const std::vector<double> v_umax = c.at("umax").get<std::vector<double>>();
    const std::vector<double> v_xmin = c.at("xmin").get<std::vector<double>>();  // size = TNX
    const std::vector<double> v_xmax = c.at("xmax").get<std::vector<double>>();

    //Ensure size of maped objects
    auto require_size = [](const std::vector<double>& v,
            std::size_t expected,
            const std::string& name)
        {
        if (v.size() != expected) {
        std::ostringstream oss;
        oss << "JSON vector '" << name << "' has size "
        << v.size() << ", expected " << expected;
        throw std::runtime_error(oss.str());
        }
    };
    require_size(v_umin, Tnu, "constraints.umin");
    require_size(v_umax, Tnu, "constraints.umax");
    require_size(v_xmin, Tnx, "constraints.xmin");
    require_size(v_xmax, Tnx, "constraints.xmax");

    umin_ = Eigen::Map<const uVec>(v_umin.data());
    umax_ = Eigen::Map<const uVec>(v_umax.data());
    xmin_ = Eigen::Map<const xVec>(v_xmin.data());
    xmax_ = Eigen::Map<const xVec>(v_xmax.data());

    nlmpc.setInputBounds(umin_, umax_, mpc::HorizonSlice{0, CTRL_H});    // length = control_horizon
    nlmpc.setStateBounds(xmin_, xmax_, mpc::HorizonSlice{0, PRED_H});    // length = prediction_horizon


    NLParameters params;
    params.maximum_iteration = mpc_opts.at("params.maximum_iteration").get<double>();  
    params.relative_ftol   =   mpc_opts.at("params.relative_ftol").get<double>();
    params.relative_xtol   =   mpc_opts.at("params.relative_xtol").get<double>();
    params.absolute_ftol   =   mpc_opts.at("params.absolute_ftol").get<double>();
    params.absolute_xtol   =   mpc_opts.at("params.absolute_xtol").get<double>(); 
    params.hard_constraints =  mpc_opts.value("params.hard_constraints", true);
    params.enable_warm_start = mpc_opts.value("warm_start", true);

    nlmpc.setOptimizerParameters(params);

    }
void MPCCar::calculateControl(int k, double t, const xVec &x, const wVec &w){
    state = x;
    if (control.size()!=Tnu) { control.resize(Tnu); control.setZero(); }

    // simple guards
    auto finite = [](const auto& v){ for (int i=0;i<v.size();++i) if(!std::isfinite(v(i))) return false; return true; };
    if (!finite(state))  throw std::runtime_error("state has NaN/Inf");
    if (!finite(control)) std::cerr << "[warn] control had NaN/Inf, zeroed.\n";

    // one-line summaries
    if (debug_level_ >= 1) {
        auto v2s = [](const auto& v){
            std::ostringstream oss; oss.setf(std::ios::fixed); oss<<std::setprecision(6);
            for (int i=0;i<v.size();++i){ if(i) oss<<','; oss<<v(i); } return oss.str();
        };
    
        std::cerr << "[NLMPC] state=" << v2s(state) << " control=" << v2s(control) << "\n";
    }

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
