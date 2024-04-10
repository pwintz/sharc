// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
#include <iostream>
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include "scarab_markers.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;

#ifdef PREDICTION_HORIZON
#else
  #define PREDICTION_HORIZON 5
#endif
#ifdef CONTROL_HORIZON
#else
  #define CONTROL_HORIZON 3
#endif

// Load a header file for parsing and generating JSON files.
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// fstream provides I/O to file pipes.
#include <fstream>
 
using namespace mpc;

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    // From https://stackoverflow.com/a/26221725/6651650
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ 
      throw std::runtime_error( "string_format(): Error during formatting." ); 
    }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template<int rows, int cols>
void loadMatrixValuesFromJson(mat<rows, cols>& mat_out, json json_data, std::string key) {
      if (json_data[key].size() != rows*cols)
      {
        throw std::invalid_argument(string_format(
              "loadMatrixValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in mat_out (%dx%d).", 
              json_data[key].size(),  key.c_str(), rows, cols));
      }

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

int debug_interfile_communication = 0;
int debug_optimizer_stats = 0;
Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");

// File names for the pipes we use for communicating with Python.
std::string x_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_c++_to_py";
std::string u_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/u_c++_to_py";
std::string x_in_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/x_py_to_c++";
std::string optimizer_info_out_filename = "/workspaces/ros-docker/libmpc_example/sim_dir/optimizer_info.csv";

int main()
{
  std::ifstream config_json_file("/workspaces/ros-docker/libmpc_example/config.json");
  json json_data = json::parse(config_json_file);
  config_json_file.close();

  json out_data;
  std::string out_data_file_name = "/workspaces/ros-docker/libmpc_example/data_out.json";
  std::ifstream data_out_json_file(out_data_file_name);
  if (data_out_json_file.good()) 
  {
    out_data = json::parse(data_out_json_file);
  } 
  data_out_json_file.close();
  
  int n_time_steps = json_data["n_time_steps"];

  // Vector sizes.
  const int Tnx = 5;  // State dimension
  const int Tnu = 1;  // Control dimension
  const int Tndu = 1; // Exogenous control (disturbance) dimension
  const int Tny = 2;  // Output dimension

  // MPC options.
  const int prediction_horizon = PREDICTION_HORIZON;
  const int control_horizon = CONTROL_HORIZON;

  // Discretization Options.
  double sample_time = json_data["sample_time"];

  // Set whether the evolution of the dyanmics are computed extenally 
  // (with Python) or are done within this process, using libmpc.
  bool use_external_dynamics_computation = json_data["use_external_dynamics_computation"];

  PRINT("Creating Matrices");
  // Continuous-time Matrices for \dot{x} = Ax + bu. 
  // Model is the HCW equations for relative satellite motion.
  mat<Tnx, Tnx> Ac;
  mat<Tnx, Tnu> Bc;
  mat<Tny, Tnx> Cd;
  
  double lead_car_input = json_data["lead_car_input"];
  double tau = json_data["tau"];
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
  json_out_data["prediction_horizon"] = prediction_horizon;
  json_out_data["control_horizon"] = control_horizon;
  
  std::ofstream optimizer_info_out_file(optimizer_info_out_filename);
  optimizer_info_out_file << "num_iterations,cost,primal_residual,dual_residual" << std::endl;

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

  PRINT("Finished creating Matrices");

  PRINT("Creating LMPC object...");
  LMPC<Tnx, Tnu, Tndu, Tny, prediction_horizon, control_horizon> lmpc;
  PRINT("Finished creating LMPC object.");

  LParameters params;
  // ADMM relaxation parameter (see https://osqp.org/docs/solver/index.html#algorithm)
  params.alpha = 1.6;
  // ADMM rho step (see https://osqp.org/docs/solver/index.html#algorithm)
  params.rho = 1e-6;
  params.adaptive_rho = true;
  // Relative tolerance
  params.eps_rel = json_data["osqp_rel_tolerance"];
  // Absolute tolerance
  params.eps_abs = json_data["osqp_abs_tolerance"];
  // Primal infeasibility tolerance
  params.eps_prim_inf = json_data["osqp_primal_infeasibility_tolerance"];
  // Dual infeasibility tolerance
  params.eps_dual_inf = json_data["osqp_dual_infeasibility_tolerance"];
  // Runtime limit in seconds
  params.time_limit = json_data["osqp_time_limit"];
  params.maximum_iteration = json_data["osqp_maximum_iteration"];
  params.enable_warm_start = json_data["enable_mpc_warm_start"];
  params.verbose = json_data["osqp_verbose"];
  params.polish = true;

  PRINT("Set parameters...");
  lmpc.setOptimizerParameters(params);
  PRINT("Finshed setting parameters");

  lmpc.setStateSpaceModel(Ad, Bd, Cd);
  lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());

  // ======== Weights ========== //

  // Output Weights
  cvec<Tny> OutputW;
  OutputW.setOnes();
  OutputW *= json_data["output_cost_weight"];

  // Input Weights
  cvec<Tnu> InputW;
  InputW.setOnes();
  InputW *= json_data["input_cost_weight"];

  // Input change weights
  cvec<Tnu> DeltaInputW;
  DeltaInputW.setZero();

  lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

  // Set 
  mat<Tndu, prediction_horizon> disturbance_input;
  disturbance_input.setOnes();
  disturbance_input *= lead_car_input;
  lmpc.setExogenuosInputs(disturbance_input);
  // printMat("Disturbance input", disturbance_input);
  // printMat("Disturbance matirx", Bd_disturbance);
  auto disturbance_vec = Bd_disturbance;
  disturbance_vec *= disturbance_input(0);
  // printMat("Disturbance vector", disturbance_vec);

  // ======== Constraints ========== //

  // State constraints.
  cvec<Tnx> xmin, xmax;
  loadMatrixValuesFromJson(xmin, json_data, "xmin");
  loadMatrixValuesFromJson(xmax, json_data, "xmax");

  // Output constraints
  cvec<Tny> ymin, ymax;
  loadMatrixValuesFromJson(ymin, json_data, "ymin");
  loadMatrixValuesFromJson(ymax, json_data, "ymax");

  // Control constraints.
  cvec<Tnu> umin, umax;
  loadMatrixValuesFromJson(umin, json_data, "umin");
  loadMatrixValuesFromJson(umax, json_data, "umax");

  lmpc.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, prediction_horizon});

  // Output reference point
  cvec<Tny> yRef;
  loadMatrixValuesFromJson(yRef, json_data, "yref");

  lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});

  // I/O Setup

  std::ofstream x_outfile;
  std::ofstream u_outfile;
  std::ifstream x_infile;
  if (use_external_dynamics_computation) {
    // Open the pipes to Python. Each time we open one of these streams, this process pauses until 
    // it is connected to the Python process (or another process). 
    // The order needs to match the order in the reader process.
    PRINT("About to open x_outfile output stream. Waiting for reader.");
    x_outfile.open(x_out_filename,std::ofstream::out);
    PRINT("About to open u_outfile output stream. Waiting for reader.");
    u_outfile.open(u_out_filename,std::ofstream::out);
    PRINT("About to open x_infile output stream. Waiting for reader.");
    x_infile.open(x_in_filename,std::ofstream::in);
    PRINT("All files open.");
  }

  // State vector state.
  mpc::cvec<Tnx> modelX, modeldX;
  // Set the initial value.
  loadMatrixValuesFromJson(modelX, json_data, "x0");

  // Output vector.
  mpc::cvec<Tny> y;

  // Create a vector for storing the control input from the previous time step.
  mpc::cvec<Tnu> u;
  u = cvec<Tnu>::Zero();

  scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'

  for (int i = 0; i < n_time_steps; i++)
  {
    PRINT(std::endl << "====== Starting loop #" << i << " ======");
    PRINT("At x=" << modelX.transpose().format(fmt));

    // Begin a batch of Scarab statistics
    scarab_roi_dump_begin(); 

      // Compute a step of the MPC controller. This does NOT change the value of modelX.
      Result res = lmpc.step(modelX, u);

    // Save a batch of Scarab statistics
    scarab_roi_dump_end();

    auto u = res.cmd;
    // printVector("u  from res.cmd", u );
    
    if (debug_optimizer_stats >= 1) 
    {
      PRINT("Optimizer exit reason: " << res.status_string)
      PRINT("return code: " << res.retcode)
      PRINT("Result status: " << res.status)
      PRINT("Number of iterations: " << res.num_iterations)
      PRINT("Cost: " << res.cost)
      PRINT("Constraint error: " << res.primal_residual)
      PRINT("Dual error: " << res.dual_residual)
    }

    optimizer_info_out_file << res.num_iterations << "," << res.cost << "," << res.primal_residual << "," << res.dual_residual << std::endl;  

    // OptSequence has three properties: state, input, and output. 
    // Each predicted time step is stored in one row.
    // There are <prediction_horizon> many rows. 
    OptSequence optseq = lmpc.getOptimalSequence();

    if (use_external_dynamics_computation) {
      // Format the numerical arrays into strings (single lines) that we pass to Python.
      auto x_out = modelX.transpose();
      auto x_out_str = x_out.format(fmt);
      auto u_out_str = u.format(fmt);

      // Pass the strings for x_out and u to the json_outfiles, 
      // which are watched by Python.
      if (debug_interfile_communication >= 1) 
      {
        PRINT("x (send to Python): " << x_out_str);
      } 
      x_outfile << "Loop " << i << ": " << x_out_str << std::endl;

      if (debug_interfile_communication >= 1) 
      {
        PRINT("u (send to Python): " << u_out_str);
      } 
      u_outfile << "Loop " << i << ": " << u_out_str << std::endl;

      // Read and print the contents of the file from Python.
      std::string line_from_py;
      
      if (debug_interfile_communication >= 2)
      {
        PRINT("Getting line from python: ");
      }
      std::getline(x_infile, line_from_py);
      if (debug_interfile_communication >= 2)
      {
        PRINT("Line from python: " << line_from_py);
      }

      // Create a stringstream to parse the string
      std::stringstream ss(line_from_py);

      for (int j_entry = 0; j_entry < modelX.rows(); j_entry++) {
        char comma; // to read and discard the comma
        ss >> modelX(j_entry) >> comma;
      }
      if (debug_interfile_communication >= 1) 
      {
        PRINT("x (from Python): " << modelX.transpose().format(fmt));
      }
    } else {
      printVector("modelX", modelX);
      printMat("optseq.state", optseq.state);
      modelX = Ad * modelX + Bd * u + Bd_disturbance*lead_car_input;
      printVector("modelX", modelX);
    }
    
    // Set the model state to the next value of x from the optimal sequence. 
    y = Cd * modelX;
    if ((y-yRef).norm() < 1e-2)
    {
      PRINT("Converged to the origin after " << i << " steps.");
      break;
    }
    
  } 
  printVector("y", y);

  if (use_external_dynamics_computation)
  {
    x_outfile << "Done" << std::endl;
    // u_outfile << "Done" << std::endl;
    x_outfile.close();
    x_infile.close();
    u_outfile.close();
  }

  optimizer_info_out_file.close();

  return 0;
}
