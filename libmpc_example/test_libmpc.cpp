// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
#include <iostream>
#include <mpc/LMPC.hpp>
#include "scarab_markers.h"


// fstream provides I/O to file pipes.
#include <fstream>
 
using namespace mpc;

int main()
{

  scarab_begin(); // this is needed for '--pintool_args -fast_forward_to_start_inst 1'

  std::cout << "Successfully included mpc/LMPC.hpp" << std::endl;
  const int Tnx = 4; // State dimension
  const int Tnu = 2;  // Control dimension
  const int Tndu = 1; // Control disturbance dimension
  const int Tny = 4; // Output dimension
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

  // Continuous-time Matrices for \dot{x} = Ax + bu. 
  // Model is the HCW equations for relative satellite motion.
  mat<Tnx, Tnx> Ac;
  mat<Tnx, Tnu> Bc;
  double n = 1e-3; // For low-Earth orbit.

  // Define matrix A_c
  Ac << 0,       0,    1,   0,
        0,       0,    0,   1,
        3 * n*n, 0,    0, 2*n,
        0,       0, -2*n,   0;

  // Define matrix B
  Bc << 0, 0,
        0, 0,
        0, 1,
        1, 0;

  // Define the time variable t
  double t = 2.0;

  // Compute the matrix exponential exp(A*t)
  mat<Tnx, Tnx> Ad;
  Ad << 1.0001, 0, 9.9998, 0.1000,
          -0.0000, 1.0000, -0.1000, 9.9993,
          0.0000, 0, 1.0000, 0.0200,
          -0.0000, 0, -0.0200, 0.9998;

  mat<Tnx, Tnu> Bd;
  Bd << 49.9996, 0.3333,
        -0.3333, 49.9983,
        9.9998, 0.1000,
        -0.1000, 9.9993;
      
    // State to output matrix
    mat<Tny, Tnx> Cd;
    Cd.setIdentity();

    // Input to output matrix
    mat<Tny, Tnu> Dd;
    Dd.setZero();

    lmpc.setStateSpaceModel(Ad, Bd, Cd);

    lmpc.setDisturbances(
        mat<Tnx, Tndu>::Zero(),
        mat<Tny, Tndu>::Zero());

// ======== Weights ========== //

     // Output Weights
     cvec<Tny> OutputW;
     OutputW << 10, 10, 10, 10;

     // Input Weights
     cvec<Tnu> InputW;
     InputW << 0.1, 0.1;

     // Input change weights
     cvec<Tnu> DeltaInputW;
     DeltaInputW << 0, 0;

     lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

// // ======== Constraints ========== //
//     cvec<Tnx> xmin, xmax;
//     xmin << -M_PI / 6, -M_PI / 6, -inf, -inf, -inf, -1,
//         -inf, -inf, -inf, -inf, -inf, -inf;

//     xmax << M_PI / 6, M_PI / 6, inf, inf, inf, inf,
//         inf, inf, inf, inf, inf, inf;

//     cvec<Tny> ymin, ymax;
//     ymin.setOnes();
//     ymin *= -inf;
//     ymax.setOnes();
//     ymax *= inf;

//     cvec<Tnu> umin, umax;
//     double u0 = 10.5916;
//     umin << 9.6, 9.6, 9.6, 9.6;
//     umin.array() -= u0;
//     umax << 13, 13, 13, 13;
//     umax.array() -= u0;

//     lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});

  // Output reference point
  cvec<Tny> yRef;
  yRef << 0, 0, 0, 0;

  lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});

  // I/O Setup
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");

  // File names for the pipes we use for communicating with Python.
  std::string x_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_c++_to_py";
  std::string u_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/u_c++_to_py";
  std::string x_in_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_py_to_c++";

  // Open the pipes to Python. Each time we open one of these streams, this process pauses until 
  // it is connected to the Python process (or another process). The order needs to match the order
  // in the reader process.
  std::cout << "About to open x_outfile output stream. Waiting for reader." << std::endl;
  std::ofstream x_outfile(x_out_filename);
  std::cout << "About to open u_outfile output stream. Waiting for reader." << std::endl;
  std::ofstream u_outfile(u_out_filename);
  std::cout << "About to open x_infile output stream. Waiting for reader." << std::endl;
  std::ifstream x_infile(x_in_filename);
  std::cout << "All files open." << std::endl;

     mpc::cvec<Tnx> modelX, modeldX;
     mpc::cvec<Tnu> last_u;

     modelX << 100, 100, 100, -100;
     last_u = cvec<Tnu>::Zero();

     for (int i = 0; i < 100; i++)
     {
          // Begin a batch of Scarab statistics
          scarab_roi_dump_begin(); 

          // Compute a step of the MPC controller.
          auto res = lmpc.step(modelX, last_u);
          auto optseq = lmpc.getOptimalSequence();

          // Save a batch of Scarab statistics
          scarab_roi_dump_end();

          // std::cout << "About to print to file." << std::endl;
          //     scarab_roi_dump_begin(); 
          // //     scarab_roi_dump_end();std::cout
          // x_outfile << Ad.format(fmt) << std::endl << std::endl;

          // std::cout << Ad.format(fmt) << std::endl << std::endl;
          // std::cout << optseq.input.row(0) << std::endl;
          auto u = optseq.input.row(0);
          auto x = optseq.state.row(0);
          std::cout << "x:" << x << std::endl;

          x_outfile << x.row(0).format(fmt) << std::endl;
          u_outfile << u.row(0).format(fmt) << std::endl;

          // Read and print the contents of the file
          std::string line_from_py;
          std::cout << "Getting line from python: " << line_from_py << std::endl;
          std::getline(x_infile, line_from_py);
          std::cout << "Line from python: " << line_from_py << std::endl;

          // Create a stringstream to parse the string
          std::stringstream ss(line_from_py);

          for (int j_entry = 0; j_entry < modelX.rows(); j_entry++) {
               char comma; // to read and discard the comma
               ss >> modelX(j_entry) >> comma;
               // std::cout << "Value from stringstream:" << modelX(j_entry) << std::endl;
               // modelX(j_entry) = value;

               // std::cout << "Finished printing to file." << std::endl;
          }
          std::cout << "x' from Python: " << modelX.transpose().format(fmt) << std::endl;

          // Set the model state to the next value of x from the optimal sequence. 
          modelX = x;
          last_u = u;
     }    

     x_outfile.close();
     x_infile.close();

     return 0;
}
