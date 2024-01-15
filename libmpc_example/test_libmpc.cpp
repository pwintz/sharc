// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
#include <iostream>
#include <mpc/LMPC.hpp>
 
using namespace mpc;

int main()
{
  std::cout << "Successfully included mpc/LMPC.hpp" << std::endl;
  const int Tnx = 12; // State dimension
  const int Tnu = 4;  // Control dimension
  const int Tndu = 1; // Control disturbance dimension
  const int Tny = 12; // Output dimension
  const int prediction_horizon = 2;
  const int control_horizon = 1;

  LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;
  std::cout << "Created LMPC object" << std::endl;

  LParameters params;

  params.alpha = 1.6;
  params.rho = 1e-6;
  params.eps_rel = 1e-4;
  params.eps_abs = 1e-4;
  params.eps_prim_inf = 1e-3;
  params.eps_dual_inf = 1e-3;
  params.time_limit = 0;
  params.enable_warm_start = true;
  params.verbose = false;
  params.adaptive_rho = true;
  params.polish = true;

  lmpc.setOptimizerParameters(params);
  std::cout << "Set parameters" << std::endl;

  lmpc.setLoggerLevel(Logger::log_level::NORMAL);

  mat<Tnx, Tnx> Ad;
  Ad << 1,       0,   0, 0, 0, 0,    0.1,       0,       0,      0,      0,      0,
        0,       1,   0, 0, 0, 0,      0,     0.1,       0,      0,      0,      0,
        0,       0,   1, 0, 0, 0,      0,       0,     0.1,      0,      0,      0,
   0.0488,       0,   0, 1, 0, 0, 0.0016,       0,       0, 0.0992,      0,      0,
        0, -0.0488,   0, 0, 1, 0,      0, -0.0016,       0,      0, 0.0992,      0,
        0,       0,   0, 0, 0, 1,      0,       0,       0,      0,      0, 0.0992,
        0,       0,   0, 0, 0, 0,      1,       0,       0,      0,      0,      0,
        0,       0,   0, 0, 0, 0,      0,       1,       0,      0,      0,      0,
        0,       0,   0, 0, 0, 0,      0,       0,       1,      0,      0,      0,
   0.9734,       0,   0, 0, 0, 0, 0.0488,       0,       0, 0.9846,      0,      0,
        0, -0.9734,   0, 0, 0, 0,      0, -0.0488,       0,      0, 0.9846,      0,
        0,       0,   0, 0, 0, 0,      0,       0,       0,      0,       0, 0.9846;

  mat<Tnx, Tnu> Bd;
  Bd << 0, -0.0726, 0, 0.0726,
        -0.0726, 0, 0.0726, 0,
        -0.0152, 0.0152, -0.0152, 0.0152,
        0, -0.0006, -0.0000, 0.0006,
        0.0006, 0, -0.0006, 0,
        0.0106, 0.0106, 0.0106, 0.0106,
        0, -1.4512, 0, 1.4512,
        -1.4512, 0, 1.4512, 0,
        -0.3049, 0.3049, -0.3049, 0.3049,
        0, -0.0236, 0, 0.0236,
        0.0236, 0, -0.0236, 0,
        0.2107, 0.2107, 0.2107, 0.2107;

  mat<Tny, Tnx> Cd;
  Cd.setIdentity();

  mat<Tny, Tnu> Dd;
  Dd.setZero();

  lmpc.setStateSpaceModel(Ad, Bd, Cd);

  lmpc.setDisturbances(
      mat<Tnx, Tndu>::Zero(),
      mat<Tny, Tndu>::Zero());

  cvec<Tnu> InputW, DeltaInputW;
  cvec<Tny> OutputW;

  OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
  InputW << 0.1, 0.1, 0.1, 0.1;
  DeltaInputW << 0, 0, 0, 0;

  lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

  cvec<Tnx> xmin, xmax;
  xmin << -M_PI / 6, -M_PI / 6, -inf, -inf, -inf, -1,
      -inf, -inf, -inf, -inf, -inf, -inf;

  xmax << M_PI / 6, M_PI / 6, inf, inf, inf, inf,
      inf, inf, inf, inf, inf, inf;

  cvec<Tny> ymin, ymax;
  ymin.setOnes();
  ymin *= -inf;
  ymax.setOnes();
  ymax *= inf;

  cvec<Tnu> umin, umax;
  double u0 = 10.5916;
  umin << 9.6, 9.6, 9.6, 9.6;
  umin.array() -= u0;
  umax << 13, 13, 13, 13;
  umax.array() -= u0;

  lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});

  cvec<Tny> yRef;
  yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});

  auto res = lmpc.step(cvec<Tnx>::Zero(), cvec<Tnu>::Zero());
  lmpc.getOptimalSequence();

  return 0;
}