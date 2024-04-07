// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
#include <iostream>
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include "scarab_markers.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

// Load a header file for parsing and generating JSON files.
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// fstream provides I/O to file pipes.
#include <fstream>
 
using namespace mpc;

template<int rows, int cols>
void loadMatrixValuesFromJson(mat<rows, cols>& mat_out, json json_data, std::string key) {
      int i = 0;
      for (auto& element : json_data[key]) {
        mat_out[i] = element;
        i++;
      }
}

template<int n>
void printVector(std::string description, cvec<n>& vec_to_print){
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");
  std::cout << description << ":" << std::endl << vec_to_print.transpose().format(fmt) << std::endl;
}

template<int rows, int cols>
void printMat(std::string description, mat<rows, cols>& mat_to_print){
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");
  std::cout << description << ": " << std::endl << mat_to_print.format(fmt) << std::endl;
}

template<int rows, int cols>
std::vector<double> matToStdVector(mat<rows,cols>& matrix){
  std::vector<double> matrix_entries;
  for (int i_row = 0; i_row < matrix.rows() ; i_row++)
  {
    for (int j_col = 0; j_col < matrix.cols() ; j_col++)
    {
      matrix_entries.push_back(matrix(i_row, j_col));
    }
  }
  return matrix_entries;
}

int main()
{

  Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");

  std::ifstream json_file("/workspaces/ros-docker/libmpc_example/config.json");
  json json_data = json::parse(json_file);
  json_file.close();
  
  int n_time_steps = json_data["n_time_steps"];

  // Vector sizes.
  const int Tnx = 5;  // State dimension
  const int Tnu = 1;  // Control dimension
  const int Tndu = 1; // Exogenous control (disturbance) dimension
  const int Tny = 2;  // Output dimension

  // MPC options.
  const int prediction_horizon = 4;
  const int control_horizon = 1;

  // Discretization Options.
  double sample_time = json_data["sample_time"];

  // Set whether the evolution of the dyanmics are computed extenally 
  // (with Python) or are done within this process, using libmpc.
  bool use_external_dyanmics_computation = json_data["use_external_dyanmics_computation"];

  std::cout << "Creating Matrices" << std::endl;
  // Continuous-time Matrices for \dot{x} = Ax + bu. 
  // Model is the HCW equations for relative satellite motion.
  mat<Tnx, Tnx> Ac;
  mat<Tnx, Tnu> Bc;
  
  double tau = 0.5;
  // Define continuous-time state matrix A_c
  Ac << 0,      1, 0,  0,      0, // v1
        0, -1/tau, 0,  0,      0, // a1
        1,      0, 0, -1,      0, // d2
        0,      0, 0,  0,      1, // v2
        0,      0, 0,  0, -1/tau; // a2

  // Define continuous-time input matrix B_c
  Bc <<    0,
           0,
           0, 
           0, 
        -1/tau;

  // State to output matrix
  mat<Tny, Tnx> Cd;
  Cd << 0, 0, 1, 0, 0, 
        1, 0, 0,-1, 0;

  // Set input disturbance matrix.
  mat<Tnx, Tndu> Bc_disturbance;
  mat<Tnx, Tndu> Bd_disturbance;
  Bc_disturbance << 0, 1/tau, 0, 0, 0;

  json json_out_data;
  json_out_data["state_dimension"] = Tnx;
  json_out_data["input_dimension"] = Tnu;
  std::vector<double> Ac_entries = matToStdVector(Ac);
  std::vector<double> Bc_entries = matToStdVector(Bc);
  json_out_data["Ac_entries"] = Ac_entries;
  json_out_data["Bc_entries"] = Bc_entries;
  // std::cout << json_out_data["Ac_entries"] << std::endl;
  // std::cout << json_out_data["Bc_entries"] << std::endl;

  // write prettified JSON to another file
  std::ofstream json_outfile("/workspaces/ros-docker/libmpc_example/system_dynamics.json");
  json_outfile 
    << std::setw(4) // Makes the JSON file better formatted.
    << json_out_data 
    << std::endl;

  mat<Tnx, Tnx> Ad;
  mat<Tnx, Tnu> Bd;
  // Compute the discretization of Ac and Bc, storing them in Ad and Bd.
  discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, sample_time, Ad, Bd, Bd_disturbance);

  std::cout << "Finished creating Matrices" << std::endl;

  std::cout << "Creating LMPC object..." << std::endl;
  LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;
  std::cout << "Finished creating LMPC object." << std::endl;

  LParameters params;
  params.alpha = 1.6;
  params.rho = 1e-6;
  params.eps_rel = 1e-4;
  params.eps_abs = 1e-4;
  params.eps_prim_inf = 1e-3;
  params.eps_dual_inf = 1e-3;
  params.time_limit = json_data["osqp_time_limit"];
  params.enable_warm_start = json_data["enable_mpc_warm_start"];
  params.verbose = json_data["osqp_verbose"];
  params.adaptive_rho = true;
  params.polish = true;

  std::cout << "Set parameters..." << std::endl;
  lmpc.setOptimizerParameters(params);
  std::cout << "Finshed setting parameters" << std::endl;

  lmpc.setStateSpaceModel(Ad, Bd, Cd);
  lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());

  // ======== Weights ========== //

  // Output Weights
  cvec<Tny> OutputW;
  OutputW.setOnes();
  OutputW *= 10;

  // Input Weights
  cvec<Tnu> InputW;
  InputW.setOnes();
  InputW *= 0.1;

  // Input change weights
  cvec<Tnu> DeltaInputW;
  DeltaInputW.setZero();

  lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

  // Set 
  mat<Tndu, prediction_horizon> disturbance;
  disturbance.setZero();
  lmpc.setExogenuosInputs(disturbance);

  // ======== Constraints ========== //

  // State constraints.
  cvec<Tnx> xmin, xmax;
  loadMatrixValuesFromJson(xmin, json_data, "xmin");
  loadMatrixValuesFromJson(xmax, json_data, "xmax");

  // Output constraints
  cvec<Tny> ymin, ymax;
  ymin.setOnes();
  ymin *= -inf;
  ymax.setOnes();
  ymax *= inf;

  // Control constraints.
  cvec<Tnu> umin, umax;
  loadMatrixValuesFromJson(umin, json_data, "umin");
  loadMatrixValuesFromJson(umax, json_data, "umax");

  lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});

  // Output reference point
  cvec<Tny> yRef;
  yRef.setZero();

  lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});

  // I/O Setup

  // File names for the pipes we use for communicating with Python.
  std::string x_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_c++_to_py";
  std::string u_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/u_c++_to_py";
  std::string x_in_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_py_to_c++";
  std::ofstream x_outfile;
  std::ofstream u_outfile;
  std::ifstream x_infile;
  if (use_external_dyanmics_computation) {
    // Open the pipes to Python. Each time we open one of these streams, this process pauses until 
    // it is connected to the Python process (or another process). 
    // The order needs to match the order in the reader process.
    std::cout << "About to open x_outfile output stream. Waiting for reader." << std::endl;
    x_outfile.open(x_out_filename,std::ofstream::out);
    std::cout << "About to open u_outfile output stream. Waiting for reader." << std::endl;
    u_outfile.open(u_out_filename,std::ofstream::out);
    std::cout << "About to open x_infile output stream. Waiting for reader." << std::endl;
    x_infile.open(x_in_filename,std::ofstream::in);
    std::cout << "All files open." << std::endl;
  }

  // State vector state.
  mpc::cvec<Tnx> modelX, modeldX;
  // Set the initial value.
  loadMatrixValuesFromJson(modelX, json_data, "x0");

  // Create a vector for storing the control input from the previous time step.
  mpc::cvec<Tnu> u;
  u = cvec<Tnu>::Zero();

  scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'

  for (int i = 0; i < n_time_steps; i++)
  {
    std::cout << std::endl << "====== Starting loop #" << i << " ======" << std::endl;
    std::cout << "At x=" << modelX.transpose().format(fmt) << std::endl;

    // Begin a batch of Scarab statistics
    scarab_roi_dump_begin(); 

    // Compute a step of the MPC controller.
    printVector("modelX before step", modelX);
    Result res = lmpc.step(modelX, u);
    printVector("modelX after step", modelX);
    auto u = res.cmd;
    printVector("u = res.cmd = ", u );

    // Save a batch of Scarab statistics
    scarab_roi_dump_end();

    // OptSequence has three properties: state, input, and output. 
    // Each predicted time step is stored in one row.
    // There are <prediction_horizon> many rows. 
    OptSequence optseq = lmpc.getOptimalSequence();

    // std::cout << "About to print to file." << std::endl;

    // The output of u and x are row vectors.
    u = optseq.input.row(0);
    // std::cout << "x_out: " << x_out << std::endl;
    // std::cout << "u: " << u << std::endl;

    // std::cout << "optseq.state.row(0):" << optseq.state.row(0).format(fmt) << std::endl;
    // std::cout << "optseq.state.row(1):" << optseq.state.row(1).format(fmt) << std::endl;
    // std::cout << "optseq.state.row(2):" << optseq.state.row(2).format(fmt) << std::endl;

    if (use_external_dyanmics_computation) {
      // Format the numerical arrays into strings (single lines) that we pass to Python.
      auto x_out = modelX.transpose(); // optseq.state.row(0);
      auto x_out_str = x_out.format(fmt);
      auto u_out_str = u.format(fmt);

      // Pass the strings for x_out and u to the json_outfiles, 
      // which are watched by Python.
      std::cout << "x (send to Python): " << x_out_str << std::endl;
      x_outfile << "Loop " << i << ": " << x_out_str << std::endl;

      std::cout << "u (send to Python): " << u_out_str << std::endl;
      u_outfile << "Loop " << i << ": " << u_out_str << std::endl;

      // Read and print the contents of the file from Python.
      std::string line_from_py;
      // std::cout << "Getting line from python: " << std::endl;
      std::getline(x_infile, line_from_py);
      std::cout << "Line from python: " << line_from_py << std::endl;

      // Create a stringstream to parse the string
      std::stringstream ss(line_from_py);

      for (int j_entry = 0; j_entry < modelX.rows(); j_entry++) {
        char comma; // to read and discard the comma
        ss >> modelX(j_entry) >> comma;
        // std::cout << "Value from stringstream:" << modelX(j_entry) << std::endl;
        // modelX(j_entry) = value;
      }
      std::cout << "x (from Python): " << modelX.transpose().format(fmt) << std::endl;
    } else {
      printVector("modelX", modelX);
      printMat("optseq.state", optseq.state);
      // std::cout << "optseq.state.row(0): " << optseq.state.row(0) << std::endl;
      // "modelX" is a column vector, but Eigen automatically transposes the row vector when 
      // assigning the values from optseq.state.row(0).
      modelX = optseq.state.row(0);
      // std::cout << "modelX: " << modelX << std::endl;
      printVector("modelX", modelX);
    }
    
    // Set the model state to the next value of x from the optimal sequence. 
    // modelX = x_out;
    // u = u;
  } 

  if (use_external_dyanmics_computation)
  {
    x_outfile.close();
    x_infile.close();
    u_outfile.close();
  }

  return 0;
}
