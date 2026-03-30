// controller/NLMPCController.cpp
#include "CarCarlaMPC.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include <Eigen/Dense>



static inline double wrapAngle(double a) {
    // wrap to [-pi, pi]
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

static inline double parseTargetSpeedMps(
    const nlohmann::json& controller_parameters)
{
    const double value = controller_parameters.at("TerminalVelocity").get<double>();
    const std::string units = controller_parameters.value(
        "TerminalVelocityUnits", std::string("mps"));

    if (units == "mps" || units == "m/s") {
        return value;
    }
    if (units == "kmph" || units == "km/h" || units == "kph") {
        return value / 3.6;
    }

    throw std::runtime_error(
        "controller_parameters.TerminalVelocityUnits must be 'mps' or 'kmph'.");
}

double CarCarlaMPC::paperObjfunc(
    const mpc::mat<prediction_horizon + 1, Tnx>& X,
    const mpc::mat<prediction_horizon + 1, Tnu>& U,
    const Eigen::VectorXd& Q,
    const Eigen::Vector2d& Rdiag,
    const Eigen::Vector2d& Rd) const
{
    constexpr double BIG = 1e12;
    if (!X.allFinite() || !U.allFinite()) return BIG;

    const double q_e   = Q(0);
    const double q_v   = Q(3);
    const double v_ref = this->termVelocity;

    // copy waypoints thread-safely
    std::vector<Eigen::Vector2d> wps;
    {
        std::lock_guard<std::mutex> lk(waypoints_mtx_);
        wps = waypoints_;
    }
    if (wps.empty()) return BIG;

    auto closestDistSq = [&](double px, double py) -> double {
        double best = BIG;
        for (const auto& w : wps) {
            const double dx = px - w.x();
            const double dy = py - w.y();
            const double d2 = dx*dx + dy*dy;
            if (d2 < best) best = d2;
        }
        return best;
    };

    const int Np = X.rows() - 1;
    const int Nc = std::min<int>(control_horizon, Np);

    double J = 0.0;

    // tracking + speed
    for (int j = 1; j <= Np; ++j) {
        const double d2 = closestDistSq(X(j,0), X(j,1));
        const double ev = X(j,3) - v_ref;
        J += q_e * d2 + q_v * (ev * ev);
    }

    // control magnitude
    for (int j = 0; j < Nc; ++j) {
        Eigen::Vector2d uj = U.row(j).head<2>().transpose();
        J += (Rdiag.array() * uj.array().square()).sum();
    }

    // control smoothness
    for (int j = 1; j < Nc; ++j) {
        Eigen::Vector2d du = (U.row(j).head<2>() - U.row(j-1).head<2>()).transpose();
        J += (Rd.array() * du.array().square()).sum();
    }

    if (!std::isfinite(J)) return BIG;
    return J;
}

static inline double distPointToSegSq(
    double px, double py,
    double ax, double ay,
    double bx, double by)
{
    const double abx = bx - ax, aby = by - ay;
    const double apx = px - ax, apy = py - ay;
    const double ab2 = abx*abx + aby*aby;

    if (ab2 < 1e-12) {
        const double dx = px - ax, dy = py - ay;
        return dx*dx + dy*dy;
    }

    double t = (apx*abx + apy*aby) / ab2;
    t = std::max(0.0, std::min(1.0, t));

    const double cx = ax + t*abx;
    const double cy = ay + t*aby;

    const double dx = px - cx, dy = py - cy;
    return dx*dx + dy*dy;
}

int CarCarlaMPC::closestIdxInWindow(const std::vector<Eigen::Vector2d>& wps,
        double px, double py,
        int last_idx) const
{
    constexpr int W = 20; // search window (+/- W indices)
    const int n = static_cast<int>(wps.size());
    if (n == 0) return 0;

    last_idx = std::max(0, std::min(last_idx, n-1));
    const int i0 = std::max(0, last_idx - W);
    const int i1 = std::min(n - 1, last_idx + W);

    double best = 1e18;
    int best_i = last_idx;
    for (int i = i0; i <= i1; ++i) {
        const double dx = px - wps[i].x();
        const double dy = py - wps[i].y();
        const double d2 = dx*dx + dy*dy;
        if (d2 < best) {
            best = d2;
            best_i = i;
        }
    }

    return best_i;
}

double CarCarlaMPC::trackingObjfunc(
    const mpc::mat<prediction_horizon + 1, Tnx>& X,
    const mpc::mat<prediction_horizon + 1, Tnu>& U,
    const Eigen::VectorXd& Q,
    const Eigen::Vector2d& Rdiag,
    const Eigen::Vector2d& Rd) const
{
    constexpr double BIG = 1e12;
    if (!X.allFinite() || !U.allFinite()) return BIG;

    const int Np = (int)X.rows() - 1;
    const int Nc = std::min<int>(CONTROL_HORIZON, Np); // use your compile-time constant

    const double q_e   = Q(0);
    const double q_v   = Q(3);  // velocity weight is Q[3] for [px, py, psi, v]
    const double v_ref = this->termVelocity;

    // copy waypoints thread-safely
    std::vector<Eigen::Vector2d> wps;
    {
        std::lock_guard<std::mutex> lk(waypoints_mtx_);
        wps = waypoints_;
    }
    if ((int)wps.size() < 2) return BIG;

    // choose a local path segment using windowed closest-index
    int i0 = closestIdxInWindow(wps, X(0,0), X(0,1), last_wp_idx_);
    // enforce monotonic-ish progress (prevents jumping backwards)
    i0 = std::max(i0, last_wp_idx_);
    i0 = std::min(i0, (int)wps.size() - 2);

    double J = 0.0;

    // tracking + speed
    for (int j = 1; j <= Np; ++j) {
        const int seg_idx = std::min(i0 + (j - 1), (int)wps.size() - 2);
        const auto& A = wps[seg_idx];
        const auto& B = wps[seg_idx + 1];
        const double ey2 = distPointToSegSq(
            X(j,0), X(j,1),
            A.x(), A.y(),
            B.x(), B.y());

        const double ev = X(j,3) - v_ref;  // use velocity component
        J += q_e * ey2 + q_v * (ev * ev);
    }

    // control magnitude
    for (int j = 0; j < Nc; ++j) {
        Eigen::Vector2d uj = U.row(j).head<2>().transpose();
        J += (Rdiag.array() * uj.array().square()).sum();
    }

    // control smoothness
    for (int j = 1; j < Nc; ++j) {
        Eigen::Vector2d du = (U.row(j).head<2>() - U.row(j-1).head<2>()).transpose();
        J += (Rd.array() * du.array().square()).sum();
    }

    if (!std::isfinite(J)) return BIG;
    return J;
}

void CarCarlaMPC::setup(const nlohmann::json &json_data){
    // Load system parameters
    this->debug_level_ = json_data.at("==== Debgugging Levels ====")
    .at("debug_program_flow_level")
    .get<int>();

    this->lr = json_data.at("system_parameters").at("lr").get<double>();
    this->lf = json_data.at("system_parameters").at("lf").get<double>();
    this->sample_time = json_data.at("system_parameters").at("sample_time").get<double>();

    // n_waypoints is provided by mpc_options; fallback to exogenous_input_dimension/2 when missing.
    const auto &mpc_opts = json_data.at("system_parameters").at("mpc_options");
    if (mpc_opts.contains("n_waypoints")) {
        n_waypoints = mpc_opts.at("n_waypoints").get<int>();
    } else if (json_data.at("system_parameters").contains("exogenous_input_dimension")) {
        const int exo_dim = json_data.at("system_parameters").at("exogenous_input_dimension").get<int>();
        n_waypoints = exo_dim / 2;
        if (n_waypoints * 2 != exo_dim) {
            throw std::runtime_error("exogenous_input_dimension is not even; cannot infer n_waypoints.");
        }
    } else {
        throw std::runtime_error("n_waypoints missing in mpc_options and exogenous_input_dimension missing in system_parameters.");
    }


    if (Tndu != 2 * n_waypoints) {
        std::ostringstream oss;
        oss << "TNDU must equal 2 * n_waypoints for CarCarlaMPC, got TNDU=" << Tndu
            << " and n_waypoints=" << n_waypoints;
        throw std::runtime_error(oss.str());
    }

    {
        std::lock_guard<std::mutex> lk(waypoints_mtx_);
        waypoints_.assign(n_waypoints, Eigen::Vector2d::Zero());
    }

    // Initiallizations
    Eigen::VectorXd Q(Tnx);
    Eigen::VectorXd Qf(Tnx);

    // Control penalties:
    // - Rdiag: penalize u = [a, delta]
    // - Rd: penalize Δu = [Δa, Δdelta]
    Eigen::Vector2d Rdiag;     // a, delta
    Eigen::Vector2d Rd;        // Δa, Δdelta
    Eigen::Vector2d u_ref; u_ref << 0.0, 0.0;

    auto jc  = json_data.at("controller_parameters");
    auto jQ  = jc.at("Q");
    auto jR  = jc.at("Rdiag");
    auto jRd = jc.at("Rd");
    this->termVelocity = parseTargetSpeedMps(jc);
    objective_mode_ = jc.value("cost_function", std::string("paper"));
    if (objective_mode_ != "paper" && objective_mode_ != "tracking") {
        throw std::runtime_error(
            "controller_parameters.cost_function must be either 'paper' or 'tracking'.");
    }

    if (jQ.size() != Tnx) {
        std::ostringstream oss;
        oss << "controller_parameters.Q must have exactly " << Tnx << " elements, got " << jQ.size();
        throw std::runtime_error(oss.str());
    }

    for (int i=0; i<Tnx; ++i) {
        Q(i)  = jQ.at(i).get<double>();
    }
    for (int i=0; i<2; ++i) {
        Rdiag(i) = jR.at(i).get<double>();
        Rd(i)    = jRd.at(i).get<double>();
    }

    // nlmpc.setLoggerLevel(mpc::Logger::LogLevel::NORMAL);
    nlmpc.setLoggerLevel(mpc::Logger::LogLevel::ALERT);
    // nlmpc.setLoggerLevel(mpc::Logger::LogLevel::DEEP);
    nlmpc.setDiscretizationSamplingTime(this->sample_time);

    // Dynamics differential equation
    // State vector: [x, y, psi, v]
    //   x(0)=x, x(1)=y, x(2)=psi, x(3)=v
    // The last entry is heading, the second to last is speed.
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

            dx.setZero();

            const double psi   = x(2);
            const double v     = x(3);
            const double a     = u(0);
            const double delta = u(1);
            const double wheelbase = this->lf + this->lr;

            dx(0) = v * std::cos(psi);                      // x_dot
            dx(1) = v * std::sin(psi);                      // y_dot
            dx(2) = (v / wheelbase) * std::tan(delta);      // psi_dot
            dx(3) = a;                                      // v_dot
        });

    nlmpc.setObjectiveFunction(
        [this, Q, Rdiag, Rd, u_ref](
            const mpc::mat<prediction_horizon + 1, Tnx>& X,
            const mpc::mat<prediction_horizon + 1, Tny>& /*Y*/,
            const mpc::mat<prediction_horizon + 1, Tnu>& U,
            const double& /*t*/)
        {
            if (this->objective_mode_ == "tracking") {
                return this->trackingObjfunc(X, U, Q, Rdiag, Rd);
            }
            return this->paperObjfunc(X, U, Q, Rdiag, Rd);
            
        });

    nlmpc.setOutputFunction([&](yVec &y, const xVec &x, const uVec &, const unsigned int &) {
        y(0) = x(0);  // x-position
        y(1) = x(1);  // y-position
    });

    // --- pull MPC options -
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

    const std::vector<double> v_umin = c.at("umin").get<std::vector<double>>();
    const std::vector<double> v_umax = c.at("umax").get<std::vector<double>>();
    const std::vector<double> v_xmin = c.at("xmin").get<std::vector<double>>();
    const std::vector<double> v_xmax = c.at("xmax").get<std::vector<double>>();

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

    nlmpc.setInputBounds(umin_, umax_, mpc::HorizonSlice{0, CTRL_H});
    nlmpc.setStateBounds(xmin_, xmax_, mpc::HorizonSlice{0, PRED_H});

    NLParameters params;
    params.maximum_iteration  = mpc_opts.at("params.maximum_iteration").get<double>();
    params.relative_ftol      = mpc_opts.at("params.relative_ftol").get<double>();
    params.relative_xtol      = mpc_opts.at("params.relative_xtol").get<double>();
    params.absolute_ftol      = mpc_opts.at("params.absolute_ftol").get<double>();
    params.absolute_xtol      = mpc_opts.at("params.absolute_xtol").get<double>();
    params.hard_constraints   = mpc_opts.value("params.hard_constraints", true);
    params.enable_warm_start  = mpc_opts.value("warm_start", true);

    nlmpc.setOptimizerParameters(params);
}



void CarCarlaMPC::calculateControl(int k, double t, const xVec &x, const wVec &w) {
    // Use the spawned vehicle state, NOT any X0 from config file
    state = x;

    // Initialize control vector if needed
    if (control.size() != Tnu) {
        control.resize(Tnu);
        control.setZero();
    }

    // simple guards
    auto finite = [](const auto& v) {
        for (int i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v(i))) return false;
        }
        return true;
    };

    if (!finite(state)) {
        throw std::runtime_error("state has NaN/Inf");
    }

    if (!finite(control)) {
        std::cerr << "[warn] control had NaN/Inf, zeroed.\n";
    }

    if (w.size() != Tndu) {
        std::ostringstream oss;
        oss << "waypoint vector has size " << w.size() << ", expected " << Tndu;
        throw std::runtime_error(oss.str());
    }

    {
        std::lock_guard<std::mutex> lk(waypoints_mtx_);
        if ((int)waypoints_.size() != n_waypoints) {
            waypoints_.assign(n_waypoints, Eigen::Vector2d::Zero());
        }
        for (int i = 0; i < n_waypoints; ++i) {
            const double wx = w(2 * i);
            const double wy = w(2 * i + 1);
            if (!std::isfinite(wx) || !std::isfinite(wy)) {
                std::ostringstream oss;
                oss << "waypoint " << i << " contains NaN/Inf";
                throw std::runtime_error(oss.str());
            }
            waypoints_[i] = Eigen::Vector2d(wx, wy);
        }
    }


    // one-line summaries
    if (debug_level_ >= 1) {
        auto v2s = [](const auto& v) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);
            for (int i = 0; i < v.size(); ++i) {
                if (i) oss << ',';
                oss << v(i);
            }
            return oss.str();
        };

        std::cerr << "[NLMPC] state=" << v2s(state) << " control=" << v2s(control) << "\n";
    }

    const bool near_standstill = std::abs(state(3)) < 0.5;
    if (near_standstill && std::abs(control(0)) < 1e-6 && std::abs(control(1)) < 1e-6) {
        control(0) = std::min(umax_(0), 1.0);
        control(1) = 0.0;
    }

    bool used_launch_fallback = false;
    bool used_hold_fallback = false;
    try {
        nlmpc_step_result = nlmpc.optimize(state, control);
    } catch (const std::exception& e) {
        std::cerr << "[NLMPC] optimize() threw: " << e.what() << "\n";
        throw;
    }

    const bool result_control_finite = finite(nlmpc_step_result.cmd);
    const bool bad_solve =
        !result_control_finite ||
        nlmpc_step_result.status == mpc::ResultStatus::ERROR ||
        !nlmpc_step_result.is_feasible;

    if (!bad_solve) {
        control = nlmpc_step_result.cmd;
        last_feasible_control_ = control;
        has_last_feasible_control_ = true;
    } else if (near_standstill) {
        control(0) = std::min(umax_(0), 1.0);
        control(1) = 0.0;
        used_launch_fallback = true;
    } else if (has_last_feasible_control_) {
        control = last_feasible_control_;
        used_hold_fallback = true;
    } else {
        control(0) = std::min(umax_(0), 0.3);
        control(1) = 0.0;
        used_hold_fallback = true;
    }

    {
        std::vector<Eigen::Vector2d> wps;
        { std::lock_guard<std::mutex> lk(waypoints_mtx_); wps = waypoints_; }
        if (wps.size() >= 2) {
            int idx = closestIdxInWindow(wps, state(0), state(1), last_wp_idx_);
            last_wp_idx_ = std::max(last_wp_idx_, idx);
            last_wp_idx_ = std::min(last_wp_idx_, (int)wps.size()-2);
        }
    }

    // control(0) = PIDVelocityControl(x(2));
    latest_metadata.clear();
    auto resultStatusToString = [](mpc::ResultStatus s) -> std::string {
        switch (s) {
            case mpc::ResultStatus::SUCCESS:       return "SUCCESS";
            case mpc::ResultStatus::MAX_ITERATION: return "MAX_ITERATION";
            case mpc::ResultStatus::INFEASIBLE:    return "INFEASIBLE";
            case mpc::ResultStatus::ERROR:         return "ERROR";
            default:                               return "UNKNOWN";
        }
    };
    latest_metadata["k"]                 = k;
    latest_metadata["t"]                 = t;
    latest_metadata["controller"]        = "CarCarlaMPC";
    latest_metadata["solver_status"]     = nlmpc_step_result.solver_status;
    latest_metadata["solver_status_msg"] = nlmpc_step_result.solver_status_msg;
    latest_metadata["is_feasible"]       = nlmpc_step_result.is_feasible;
    latest_metadata["cost"]              = nlmpc_step_result.cost;
    latest_metadata["status"]            = resultStatusToString(nlmpc_step_result.status);
    latest_metadata["used_launch_fallback"] = used_launch_fallback;
    latest_metadata["used_hold_fallback"] = used_hold_fallback;
    latest_metadata["cost_function"]     = objective_mode_;

    auto opt_seq = nlmpc.getOptimalSequence();
    std::vector<double> traj_x, traj_y;
    traj_x.reserve(prediction_horizon + 1);
    traj_y.reserve(prediction_horizon + 1);
    for (int j = 0; j <= prediction_horizon; ++j) {
        traj_x.push_back(opt_seq.state(j, 0));
        traj_y.push_back(opt_seq.state(j, 1));
    }
    latest_metadata["traj_x"] = traj_x;
    latest_metadata["traj_y"] = traj_y;
}

// void CarCarlaMPC::logDebugInfo() const {
//     std::cerr << "[DEBUG] Control inputs: a=" << control(0) << ", delta=" << control(1) << "\n";
//     std::cerr << "[DEBUG] Optimization status: " << resultStatusToString(nlmpc_step_result.status) << "\n";
//     std::cerr << "[DEBUG] Solver feasible: " << nlmpc_step_result.is_feasible << "\n";
//     std::cerr << "[DEBUG] Cost: " << nlmpc_step_result.cost << "\n";
// }

// Register the controller
REGISTER_CONTROLLER("CarCarlaMPC", CarCarlaMPC)
