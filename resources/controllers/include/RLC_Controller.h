// controller/RLC_Controller.h
#pragma once

#include "controller.h"
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include <limits>
#include <cassert>

using namespace mpc;

// ===== Compile-time horizons (must be provided at build time) =====
#ifndef PREDICTION_HORIZON
  #define PREDICTION_HORIZON -1
#endif
#ifndef CONTROL_HORIZON
  #define CONTROL_HORIZON -1
#endif

class RLC_Controller : public Controller {
private:
    // Mirror ACC style: compile-time constants
    constexpr static int prediction_horizon = PREDICTION_HORIZON;
    constexpr static int control_horizon    = CONTROL_HORIZON;

    // ===== System / sampling =====
    double sample_time = 0.0;

    // ===== Series RLC parameters =====
    // State x = [ vC ; iL ], input u = v_src, output y = vC
    double R     = 1.0;  // Ohm
    double L     = 1.0;  // Henry
    double Cval  = 1.0;  // Farad

    // ===== Objective weights =====
    // Applied uniformly across the horizon (unless you override per-slice)
    double       output_cost_weight      = 10.0; // on y (vC)
    double        input_cost_weight      = 0.1;  // on u
    double  delta_input_cost_weight      = 0.0;  // on Δu

    // ===== State-space (continuous & discrete) =====
    mat<Tnx, Tnx> Ac;        // continuous A
    mat<Tnx, Tnu> Bc;        // continuous B

    mat<Tnx, Tnx> Ad;        // discrete A
    mat<Tnx, Tnu> Bd;        // discrete B
    mat<Tny, Tnx> Cmat;      // output (track vC)

    // ===== References =====
    double v_ref = 5.0;      // desired capacitor voltage
    yVec   yRef;             // reference vector for LMPC (size Tny)

    // ===== Box constraints =====
    // Defaults to ±∞ (no constraint) unless overridden by JSON
    double vC_min = -std::numeric_limits<double>::infinity();
    double vC_max =  std::numeric_limits<double>::infinity();
    double iL_min = -std::numeric_limits<double>::infinity();
    double iL_max =  std::numeric_limits<double>::infinity();
    double u_min  = -std::numeric_limits<double>::infinity(); // source voltage min
    double u_max  =  std::numeric_limits<double>::infinity(); // source voltage max

    // Vectors passed to LMPC
    xVec xmin, xmax;
    yVec ymin, ymax;
    uVec umin, umax;

    // ===== LMPC objects (compile-time horizon) =====
    LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;
    Result<Tnu>                                                    lmpc_step_result;

    // ===== Internal working state =====
    //xVec state;    // latest state used for optimize()
    //uVec control;  // last applied control (for warm start)

    // ===== Helpers (ACC-style structure) =====
    void updateStateSpaceMatrices_RLC(double R, double L, double C);
    void setOptimizerParameters(const nlohmann::json &json_data);
    void setWeights(const nlohmann::json &json_data);
    void setConstraints();
    void setReferences(const nlohmann::json &json_data);

public:
    // Constructor asserts horizons and calls setup (same as ACC)
    explicit RLC_Controller(const nlohmann::json &json_data)
    : Controller(json_data) {
        assert(prediction_horizon > 0);
        assert(control_horizon    > 0);
        setup(json_data);
    }

    // ACC-style interface
    void calculateControl(int k, double t, const xVec &x, const wVec &w) override;

protected:
    void setup(const nlohmann::json &json_data) override;
};
