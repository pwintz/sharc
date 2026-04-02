// controller/CarlaNPCFollowMPCController.cpp
// Follows lead NPC speed while keeping hard collision constraints.

#include "CarlaNPCFollowMPCController.h"

#if TNX == 4 && defined(TNIEQ) && TNIEQ > 0
#include "sharc/utils.hpp"
#include "debug_levels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

static constexpr double SENTINEL_THRESHOLD = 5e5;

void CarlaNPCFollowMPCController::setup(const nlohmann::json& json_data) {
    const auto& sp = json_data.at("system_parameters");
    const auto& cost = sp.at("mpc_options").at("cost_weights");

    sample_time = sp.at("sample_time").get<double>();
    wheelbase = sp.at("mpc_options").at("wheelbase").get<double>();
    target_speed = sp.at("target_speed").get<double>();
    effective_target_speed = target_speed;
    last_effective_target_speed = target_speed;

    q_path = cost.at("q_path").get<double>();
    q_heading = cost.value("q_heading", q_path);
    q_speed = cost.at("q_speed").get<double>();
    q_follow_gap = cost.value("q_follow_gap", 0.0);
    r_accel = cost.at("r_accel").get<double>();
    r_steer = cost.at("r_steer").get<double>();
    r_jerk_v = cost.at("r_jerk_v").get<double>();
    r_jerk_yaw = cost.at("r_jerk_yaw").get<double>();
    gamma = cost.at("gamma").get<double>();

    ego_radius = cost.value("ego_radius", 2.5);
    safe_margin = cost.value("safe_margin", 0.5);
    lane_half_width = cost.value("lane_half_width", 1.75);
    follow_time_gap = cost.value("follow_time_gap", 1.6);
    min_follow_distance = cost.value("min_follow_distance", 8.0);
    target_speed_alpha = cost.value("target_speed_alpha", 0.35);
    lead_engage_distance = cost.value("lead_engage_distance", 30.0);

    const auto& limits = sp.at("mpc_options").at("input_limits");
    max_accel = limits.at("max_accel").get<double>();
    min_accel = limits.at("min_accel").get<double>();
    max_steer = limits.at("max_steer").get<double>();
    min_steer = limits.at("min_steer").get<double>();

    n_waypoints = sp.at("mpc_options").at("n_waypoints").get<int>();
    n_obstacles = sp.at("mpc_options").at("n_obstacles").get<int>();
    assert(Ndu == 2 * n_waypoints + 5 * n_obstacles
           && "TNDU must equal 2*n_waypoints + 5*n_obstacles");
    assert(ineq_c == Np * (n_obstacles + 1)
           && "TNIEQ must equal Np * (n_obstacles + 1)");

    wp_x.resize(n_waypoints);
    wp_y.resize(n_waypoints);
    obstacles.resize(n_obstacles);

    nlmpc.setLoggerLevel(mpc::Logger::NORMAL);
    nlmpc.setDiscretizationSamplingTime(sample_time);

    nlmpc.setStateSpaceFunction(
        [this](xVec& dx, const xVec& x, const uVec& u, const unsigned int&) {
            const double psi = x(2);
            const double v = x(3);
            const double a = u(0);
            const double delta = u(1);
            dx(0) = v * std::cos(psi);
            dx(1) = v * std::sin(psi);
            dx(2) = (v / wheelbase) * std::tan(delta);
            dx(3) = a;
        });

    nlmpc.setObjectiveFunction(
        [this](
            const mpc::mat<Np + 1, Nx>& X,
            const mpc::mat<Np + 1, Ny>&,
            const mpc::mat<Np + 1, Nu>& U,
            const double&) -> double
        {
            double J = 0.0;
            double gj = gamma;
            for (int j = 1; j <= Np; ++j) {
                const double px = X(j, 0);
                const double py = X(j, 1);
                const double v = X(j, 3);

                double path_s = 0.0;
                double lat_error = 0.0;
                double path_heading = 0.0;
                if (projectOntoWaypointPath(px, py, path_s, lat_error, &path_heading)) {
                    J += gj * q_path * lat_error * lat_error;
                    const double heading_error = wrapAngle(X(j, 2) - path_heading);
                    J += gj * q_heading * heading_error * heading_error;
                } else {
                    J += gj * q_path * closestWaypointDistSq(px, py);
                }
                double speed_reference = effective_target_speed;

                int lead_index = -1;
                double forward_distance = std::numeric_limits<double>::infinity();
                double lateral_offset = 0.0;
                double lead_speed = target_speed;
                xVec x_pred;
                x_pred << X(j, 0), X(j, 1), X(j, 2), X(j, 3);
                const bool lead_found = findLeadObstacle(
                    x_pred, lead_index, forward_distance, lateral_offset, lead_speed);
                if (lead_found && forward_distance <= lead_engage_distance) {
                    speed_reference = std::min(target_speed, lead_speed);
                }

                const double ev = v - speed_reference;
                J += gj * q_speed * ev * ev;

                if (q_follow_gap > 0.0) {
                    if (lead_found) {
                        const double desired_gap = min_follow_distance;
                        const double gap_error = std::max(0.0, desired_gap - forward_distance);
                        J += gj * q_follow_gap * gap_error * gap_error;
                    }
                }

                gj *= gamma;
            }

            double gk = 1.0;
            for (int j = 0; j < Nc; ++j) {
                J += gk * r_accel * U(j, 0) * U(j, 0);
                J += gk * r_steer * U(j, 1) * U(j, 1);
                gk *= gamma;
            }

            {
                const double da = U(0, 0) - prev_accel;
                const double dd = U(0, 1) - prev_steer;
                J += r_jerk_v * da * da;
                J += r_jerk_yaw * dd * dd;
            }
            for (int j = 1; j < Nc; ++j) {
                const double da = U(j, 0) - U(j - 1, 0);
                const double dd = U(j, 1) - U(j - 1, 1);
                J += r_jerk_v * da * da;
                J += r_jerk_yaw * dd * dd;
            }

            return J;
        });

    nlmpc.setIneqConFunction(
        [this](
            cvec<ineq_c>& c,
            const mpc::mat<Np + 1, Nx>& X,
            const mpc::mat<Np + 1, Ny>&,
            const mpc::mat<Np + 1, Nu>&,
            const double&)
        {
            int idx = 0;
            for (int j = 1; j <= Np; ++j) {
                const double px = X(j, 0);
                const double py = X(j, 1);
                const double dt_pred = j * sample_time;

                for (int i = 0; i < n_obstacles; ++i) {
                    if (obstacles[i].x > SENTINEL_THRESHOLD) {
                        c(idx++) = -1e6;
                        continue;
                    }
                    const double ox = obstacles[i].x + dt_pred * obstacles[i].vx;
                    const double oy = obstacles[i].y + dt_pred * obstacles[i].vy;
                    const double r_safe = ego_radius + obstacles[i].radius + safe_margin;
                    const double dx = px - ox;
                    const double dy = py - oy;
                    c(idx++) = r_safe * r_safe - (dx * dx + dy * dy);
                }

                const double ld = lateralDeviation(px, py);
                c(idx++) = ld * ld - lane_half_width * lane_half_width;
            }
        });

    uVec umin, umax;
    umin(0) = min_accel;
    umin(1) = min_steer;
    umax(0) = max_accel;
    umax(1) = max_steer;
    nlmpc.setInputBounds(umin, umax, {0, Nc});

    control.setZero();

    experiment_dir = json_data.value("experiment_dir", "");
    state_file = experiment_dir.empty() ? "" : experiment_dir + "/mpc_state.json";
    if (!state_file.empty()) {
        load_state();
    }
}

double CarlaNPCFollowMPCController::limitSteeringStep(double desired_steer) const {
    const double bounded = std::clamp(desired_steer, min_steer, max_steer);
    const double lower = std::max(min_steer, prev_steer - max_steer_step);
    const double upper = std::min(max_steer, prev_steer + max_steer_step);
    return std::clamp(bounded, lower, upper);
}

void CarlaNPCFollowMPCController::calculateControl(int k, double t,
                                                   const xVec& x, const wVec& w) {
    for (int i = 0; i < n_waypoints; ++i) {
        wp_x[i] = w(2 * i);
        wp_y[i] = w(2 * i + 1);
    }

    const int obs_offset = 2 * n_waypoints;
    for (int i = 0; i < n_obstacles; ++i) {
        obstacles[i].x = w(obs_offset + 5 * i + 0);
        obstacles[i].y = w(obs_offset + 5 * i + 1);
        obstacles[i].vx = w(obs_offset + 5 * i + 2);
        obstacles[i].vy = w(obs_offset + 5 * i + 3);
        obstacles[i].radius = w(obs_offset + 5 * i + 4);
    }

    state = x;

    int lead_index = -1;
    double lead_forward_distance = std::numeric_limits<double>::infinity();
    double lead_lateral_offset = 0.0;
    double lead_speed = target_speed;
    const bool lead_found = findLeadObstacle(
        x, lead_index, lead_forward_distance, lead_lateral_offset, lead_speed);

    double desired_target_speed = target_speed;
    if (lead_found && lead_forward_distance <= lead_engage_distance) {
        const double desired_gap =
            min_follow_distance + follow_time_gap * std::max(x(3), lead_speed);
        const double speed_blend_span =
            std::max(1e-3, lead_engage_distance - desired_gap);
        const double gap_ratio = std::clamp(
            (lead_forward_distance - desired_gap) / speed_blend_span, 0.0, 1.0);
        desired_target_speed = lead_speed +
                               gap_ratio * (target_speed - lead_speed);
        desired_target_speed = std::clamp(desired_target_speed, 0.0, target_speed);
    }
    const double alpha = std::clamp(target_speed_alpha, 0.0, 1.0);
    effective_target_speed =
        alpha * desired_target_speed + (1.0 - alpha) * last_effective_target_speed;
    last_effective_target_speed = effective_target_speed;

    mpc_result = nlmpc.optimize(state, control);

    int active_obs = 0;
    for (int i = 0; i < n_obstacles; ++i) {
        if (obstacles[i].x < SENTINEL_THRESHOLD) {
            ++active_obs;
        }
    }

    if (!mpc_result.is_feasible) {
        const double v = x(3);
        if (active_obs == 0) {
            const double speed_error = target_speed - v;
            control(0) = std::clamp(0.8 * speed_error, min_accel, max_accel);
            // When the optimizer hits a precision limit on easy, obstacle-free
            // lane-following, keep the lateral command smooth instead of
            // recomputing an aggressive tracking correction that can oscillate.
            control(1) = limitSteeringStep(0.5 * prev_steer);
            std::cout << "[CarlaNPCFollowMPC] k=" << k
                      << " INFEASIBLE -> STRAIGHT_FALLBACK"
                      << " a=" << control(0)
                      << " delta=" << control(1)
                      << " v=" << v
                      << " obs=" << active_obs << "/" << n_obstacles
                      << std::endl;
        } else {
            control(0) = (std::abs(v) < 0.1) ? 0.0 : min_accel;
            control(1) = 0.0;
            std::cout << "[CarlaNPCFollowMPC] k=" << k
                      << " INFEASIBLE -> " << (std::abs(v) < 0.1 ? "HOLD" : "BRAKE")
                      << " a=" << control(0)
                      << " v=" << v
                      << " obs=" << active_obs << "/" << n_obstacles
                      << std::endl;
        }
    } else {
        control = mpc_result.cmd;
        control(1) = limitSteeringStep(control(1));
        std::cout << "[CarlaNPCFollowMPC] k=" << k
                  << " a=" << control(0)
                  << " delta=" << control(1)
                  << " target_v=" << effective_target_speed
                  << " obs=" << active_obs << "/" << n_obstacles
                  << std::endl;
    }

    latest_metadata.clear();
    latest_metadata["k"] = k;
    latest_metadata["t"] = t;
    latest_metadata["controller"] = "CarlaNPCFollowMPCController";
    latest_metadata["solver_status"] = mpc_result.solver_status;
    latest_metadata["is_feasible"] = mpc_result.is_feasible;
    latest_metadata["cost"] = mpc_result.cost;
    latest_metadata["cost_function"] = "npc_follow_constraint";
    latest_metadata["effective_target_speed"] = effective_target_speed;
    latest_metadata["speed_cost_reference"] =
        (lead_found && lead_forward_distance <= lead_engage_distance)
            ? std::min(target_speed, lead_speed)
            : effective_target_speed;
    latest_metadata["lead_obstacle_found"] = lead_found;
    latest_metadata["lead_obstacle_index"] = lead_index;
    latest_metadata["lead_obstacle_distance"] = lead_found ? lead_forward_distance : -1.0;
    latest_metadata["lead_obstacle_speed"] = lead_found ? lead_speed : target_speed;
    latest_metadata["lead_obstacle_lateral_offset"] = lead_found ? lead_lateral_offset : 0.0;

    auto opt_seq = nlmpc.getOptimalSequence();
    std::vector<double> traj_x, traj_y;
    for (int j = 0; j <= Np; ++j) {
        traj_x.push_back(opt_seq.state(j, 0));
        traj_y.push_back(opt_seq.state(j, 1));
    }
    latest_metadata["traj_x"] = traj_x;
    latest_metadata["traj_y"] = traj_y;

    prev_accel = control(0);
    prev_steer = control(1);

    if (!state_file.empty()) {
        save_state();
    }
}

void CarlaNPCFollowMPCController::save_state() const {
    nlohmann::json s;
    s["prev_accel"] = prev_accel;
    s["prev_steer"] = prev_steer;
    s["last_effective_target_speed"] = last_effective_target_speed;
    std::ofstream f(state_file);
    if (f.is_open()) {
        f << s.dump(2);
    }
}

void CarlaNPCFollowMPCController::load_state() {
    std::ifstream f(state_file);
    if (!f.is_open()) {
        return;
    }
    try {
        nlohmann::json s;
        f >> s;
        prev_accel = s.value("prev_accel", 0.0);
        prev_steer = s.value("prev_steer", 0.0);
        last_effective_target_speed = s.value("last_effective_target_speed", target_speed);
    } catch (...) {}
}

double CarlaNPCFollowMPCController::closestWaypointDistSq(double px, double py) const {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < n_waypoints; ++i) {
        const double dx = px - wp_x[i];
        const double dy = py - wp_y[i];
        const double d2 = dx * dx + dy * dy;
        if (d2 < best) {
            best = d2;
        }
    }
    return best;
}

double CarlaNPCFollowMPCController::lateralDeviation(double px, double py) const {
    double min_proj_dist_sq = std::numeric_limits<double>::max();
    double best_lat = 0.0;

    for (int i = 0; i < n_waypoints - 1; ++i) {
        const double ax = wp_x[i];
        const double ay = wp_y[i];
        const double bx = wp_x[i + 1];
        const double by = wp_y[i + 1];

        const double abx = bx - ax;
        const double aby = by - ay;
        const double apx = px - ax;
        const double apy = py - ay;
        const double ab2 = abx * abx + aby * aby;
        if (ab2 < 1e-9) {
            continue;
        }

        const double tau = std::clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0);
        const double proj_x = ax + tau * abx;
        const double proj_y = ay + tau * aby;
        const double dx = px - proj_x;
        const double dy = py - proj_y;
        const double dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_proj_dist_sq) {
            min_proj_dist_sq = dist_sq;
            const double cross = abx * (py - ay) - aby * (px - ax);
            best_lat = (cross >= 0.0 ? 1.0 : -1.0) * std::sqrt(dist_sq);
        }
    }

    return best_lat;
}

double CarlaNPCFollowMPCController::pathHeading(double px, double py) const {
    double best_dist_sq = std::numeric_limits<double>::max();
    double best_heading = 0.0;

    for (int i = 0; i < n_waypoints - 1; ++i) {
        const double ax = wp_x[i];
        const double ay = wp_y[i];
        const double bx = wp_x[i + 1];
        const double by = wp_y[i + 1];
        const double abx = bx - ax;
        const double aby = by - ay;
        const double ab2 = abx * abx + aby * aby;
        if (ab2 < 1e-9) {
            continue;
        }

        const double apx = px - ax;
        const double apy = py - ay;
        const double tau = std::clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0);
        const double proj_x = ax + tau * abx;
        const double proj_y = ay + tau * aby;
        const double dx = px - proj_x;
        const double dy = py - proj_y;
        const double dist_sq = dx * dx + dy * dy;

        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_heading = std::atan2(aby, abx);
        }
    }

    return best_heading;
}

double CarlaNPCFollowMPCController::wrapAngle(double angle) {
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

bool CarlaNPCFollowMPCController::findLeadObstacle(const xVec& x,
                                                   int& lead_index,
                                                   double& forward_distance,
                                                   double& lateral_offset,
                                                   double& lead_speed) const {
    lead_index = -1;
    forward_distance = std::numeric_limits<double>::infinity();
    lateral_offset = 0.0;
    lead_speed = target_speed;

    double ego_s = 0.0;
    double ego_lat = 0.0;
    if (!projectOntoWaypointPath(x(0), x(1), ego_s, ego_lat)) {
        return false;
    }

    for (int i = 0; i < n_obstacles; ++i) {
        if (obstacles[i].x > SENTINEL_THRESHOLD) {
            continue;
        }

        double obs_s = 0.0;
        double obs_lat = 0.0;
        if (!projectOntoWaypointPath(obstacles[i].x, obstacles[i].y, obs_s, obs_lat)) {
            continue;
        }

        const double forward = obs_s - ego_s;
        const double lateral = obs_lat;
        const double lane_window = lane_half_width + obstacles[i].radius + safe_margin;
        if (forward <= 0.0 || std::abs(lateral) > lane_window) {
            continue;
        }

        if (forward < forward_distance) {
            forward_distance = forward;
            lateral_offset = lateral;
            double seg_dx = 1.0;
            double seg_dy = 0.0;
            double best_dist_sq = std::numeric_limits<double>::max();
            for (int j = 0; j < n_waypoints - 1; ++j) {
                const double ax = wp_x[j];
                const double ay = wp_y[j];
                const double bx = wp_x[j + 1];
                const double by = wp_y[j + 1];
                const double abx = bx - ax;
                const double aby = by - ay;
                const double ab2 = abx * abx + aby * aby;
                if (ab2 < 1e-9) {
                    continue;
                }
                const double apx = obstacles[i].x - ax;
                const double apy = obstacles[i].y - ay;
                const double tau = std::clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0);
                const double proj_x = ax + tau * abx;
                const double proj_y = ay + tau * aby;
                const double dx = obstacles[i].x - proj_x;
                const double dy = obstacles[i].y - proj_y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    seg_dx = abx;
                    seg_dy = aby;
                }
            }
            const double seg_norm = std::hypot(seg_dx, seg_dy);
            const double tx = seg_norm > 1e-9 ? seg_dx / seg_norm : 1.0;
            const double ty = seg_norm > 1e-9 ? seg_dy / seg_norm : 0.0;
            lead_speed = std::max(0.0, obstacles[i].vx * tx + obstacles[i].vy * ty);
            lead_index = i;
        }
    }

    return lead_index >= 0;
}

bool CarlaNPCFollowMPCController::projectOntoWaypointPath(double px,
                                                          double py,
                                                          double& path_s,
                                                          double& lateral_offset,
                                                          double* heading) const {
    if (n_waypoints < 2) {
        return false;
    }

    double accumulated_s = 0.0;
    double best_s = 0.0;
    double best_lat = 0.0;
    double best_heading = 0.0;
    double best_dist_sq = std::numeric_limits<double>::max();
    bool found = false;

    for (int i = 0; i < n_waypoints - 1; ++i) {
        const double ax = wp_x[i];
        const double ay = wp_y[i];
        const double bx = wp_x[i + 1];
        const double by = wp_y[i + 1];
        const double abx = bx - ax;
        const double aby = by - ay;
        const double seg_len = std::hypot(abx, aby);
        if (seg_len < 1e-9) {
            continue;
        }

        const double apx = px - ax;
        const double apy = py - ay;
        const double tau = std::clamp((apx * abx + apy * aby) / (seg_len * seg_len), 0.0, 1.0);
        const double proj_x = ax + tau * abx;
        const double proj_y = ay + tau * aby;
        const double dx = px - proj_x;
        const double dy = py - proj_y;
        const double dist_sq = dx * dx + dy * dy;

        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_s = accumulated_s + tau * seg_len;
            const double cross = abx * (py - ay) - aby * (px - ax);
            best_lat = (cross >= 0.0 ? 1.0 : -1.0) * std::sqrt(dist_sq);
            best_heading = std::atan2(aby, abx);
            found = true;
        }

        accumulated_s += seg_len;
    }

    if (!found) {
        return false;
    }

    path_s = best_s;
    lateral_offset = best_lat;
    if (heading != nullptr) {
        *heading = best_heading;
    }
    return true;
}

REGISTER_CONTROLLER("CarlaNPCFollowMPCController", CarlaNPCFollowMPCController)
#endif
