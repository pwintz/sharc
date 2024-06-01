// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
#include <iostream>
#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include "scarab_markers.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/algorithm/string/predicate.hpp>
#include "dr_api.h"

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;

// Necseesary for Dynamorio
bool my_setenv(const char *var, const char *value)
{
#ifdef UNIX
    return setenv(var, value, 1 /*override*/) == 0;
#else
    return SetEnvironmentVariable(var, value) == TRUE;
#endif
}

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

Eigen::IOFormat fmt(4, // precision
                    0, // flags
                    ", ", // coefficient separator
                    "\n", // row prefix
                    "[", // matrix prefix
                    "]"); // matrix suffix.
Eigen::IOFormat fmt_high_precision(20, // precision
                    0, // flags
                    ", ", // coefficient separator
                    "\n", // row prefix
                    "[", // matrix prefix
                    "]"); // matrix suffix.

inline void assert_file_exists(const std::string& name) {
  // Check that a file exists.
  std::cout << "Checking if \"" + name + "\" exists." << std::endl;
  // std::ifstream f(name.c_str());
  // if (f.good()) {
  if (std::filesystem::exists(name)){
    return;
  } else {
    throw std::runtime_error("The file \"" + name + "\" does not exist. The current working directory is \"" + std::filesystem::current_path().string() + "\"."); 
  }
}

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
  std::cout << description << ":" << std::endl << vec_to_print.transpose().format(fmt) << std::endl;
}

template<int rows, int cols>
void printMat(std::string description, mat<rows, cols>& mat_to_print){
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

int debug_interfile_communication_level;
int debug_optimizer_stats_level;
int debug_dynamics_level;
// Eigen::IOFormat fmt(4, 0, ", ", "\n", "", "");

std::string example_path;

template<int rows>
void sendCVec(std::string label, int i_loop, mpc::cvec<rows> x, std::ofstream& outfile) 
{
  auto out_str = x.transpose().format(fmt_high_precision);
  if (debug_interfile_communication_level >= 1) 
  {
    PRINT(label << " (send to Python): " << out_str);
  } 
  outfile << "Loop " << i_loop << ": " << out_str << std::endl;
}

void sendDouble(std::string label, int i_loop, double x, std::ofstream& outfile) 
{
  if (debug_interfile_communication_level >= 1) 
  {
    PRINT(label << " (send to Python): " << x);
  } 
  outfile << "Loop " << i_loop << ": " << x << std::endl;
}

int main()
{
  /* We also test -rstats_to_stderr */
  if (!my_setenv("DYNAMORIO_OPTIONS",
              "-stderr_mask 0xc -rstats_to_stderr "
              "-client_lib ';;-offline'"))
   std::cerr << "failed to set env var!\n";
  // PRINT("PREDICTION_HORIZON" << PREDICTION_HORIZON)
  // PRINT("CONTROL_HORIZON" << CONTROL_HORIZON)
  // return 0;
  auto cwd = std::filesystem::current_path();
  if ( boost::algorithm::ends_with(cwd.string(), "sim_dir") )
  {
      example_path = cwd.parent_path().string() + "/";
  } else
  {
    example_path = cwd.string() + "/";
  }
  PRINT("Example path: " << example_path)

  // File names for the pipes we use for communicating with Python.
  // std::string data_out_file_path = "data_out.json";

  std::string sim_dir_path = example_path + "sim_dir/";
  std::string config_file_path = sim_dir_path + "config.json";
  std::string x_out_filename = sim_dir_path + "x_c++_to_py";
  std::string x_predict_out_filename = sim_dir_path + "x_predict_c++_to_py";
  std::string t_predict_out_filename = sim_dir_path + "t_predict_c++_to_py";
  std::string iterations_out_filename = sim_dir_path + "iterations_c++_to_py";
  std::string u_out_filename = sim_dir_path + "u_c++_to_py";
  std::string x_in_filename = sim_dir_path + "x_py_to_c++";
  std::string t_delays_in_filename = sim_dir_path + "t_delay_py_to_c++";
  std::string optimizer_info_out_filename = sim_dir_path + "optimizer_info.csv";

  std::string system_dynamics_filename = sim_dir_path + "system_dynamics.json";

  assert_file_exists(config_file_path);
  std::ifstream config_json_file(config_file_path);
  json json_data = json::parse(config_json_file);
  config_json_file.close();

  int n_time_steps = json_data["n_time_steps"];

  auto debug_config = json_data["==== Debgugging Levels ===="];
  debug_interfile_communication_level = debug_config["debug_interfile_communication_level"];
  debug_optimizer_stats_level = debug_config["debug_optimizer_stats_level"];
  debug_dynamics_level = debug_config["debug_dynamics_level"];

  bool use_state_after_delay_prediction = json_data["use_state_after_delay_prediction"];

  // Vector sizes.
  const int Tnx = 5;  // State dimension
  const int Tnu = 1;  // Control dimension
  const int Tndu = 1; // Exogenous control (disturbance) dimension
  const int Tny = 2;  // Output dimension

  // MPC options.
  const int prediction_horizon = PREDICTION_HORIZON;
  const int control_horizon = CONTROL_HORIZON;
  
  // Convergence termination tolerance: How close to the reference do we need to be stop?
  double convergence_termination_tol = json_data["convergence_termination_tol"];

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
  mat<Tny, Tnx> C;
  mat<Tnx, Tndu> Bc_disturbance;
  
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
  C << 0, 0, 1, 0, 0, 
       1, 0, 0,-1, 0;

  // Set input disturbance matrix.
  Bc_disturbance << 0, 1/tau, 0, 0, 0;

  json json_out_data;
  json_out_data["state_dimension"] = Tnx;
  json_out_data["input_dimension"] = Tnu;
  std::vector<double> Ac_entries = matToStdVector(Ac);
  std::vector<double> Bc_entries = matToStdVector(Bc);
  std::vector<double> Bc_disturbances_entries = matToStdVector(Bc);
  json_out_data["Ac_entries"] = Ac_entries;
  json_out_data["Bc_entries"] = Bc_entries;
  json_out_data["Bc_disturbances_entries"] = Bc_disturbances_entries;
  json_out_data["prediction_horizon"] = prediction_horizon;
  json_out_data["control_horizon"] = control_horizon;
  
  std::ofstream optimizer_info_out_file(optimizer_info_out_filename);
  optimizer_info_out_file << "num_iterations,cost,primal_residual,dual_residual" << std::endl;

  // write prettified JSON to another file
  std::ofstream json_outfile(system_dynamics_filename);
  json_outfile 
    << std::setw(4) // Makes the JSON file better formatted.
    << json_out_data 
    << std::endl;

  mat<Tnx, Tnx> Ad;
  mat<Tnx, Tnu> Bd;
  mat<Tnx, Tndu> Bd_disturbance;
  // Compute the discretization of Ac and Bc, storing them in Ad and Bd.
  discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, sample_time, Ad, Bd, Bd_disturbance);

  if (debug_dynamics_level >= 1)
  {
    PRINT("=== Discretization Output ===")
    PRINT("Ad:")
    PRINT(Ad)
    PRINT("Bd:")
    PRINT(Bd)
    PRINT("Bd_disturbance:")
    PRINT(Bd_disturbance)
  }

  mat<Tnx, Tnx> Ad_predictive;
  mat<Tnx, Tnu> Bd_predictive;
  mat<Tnx, Tndu> Bd_disturbance_predictive;

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
  PRINT("Finished setting optimizer parameters");

  lmpc.setStateSpaceModel(Ad, Bd, C);
  lmpc.setDisturbances(Bd_disturbance, mat<Tny, Tndu>::Zero());

  // ======== Weights ========== //

  double outputWeight = json_data["output_cost_weight"];
  double inputWeight = json_data["input_cost_weight"];

  // Output Weights
  cvec<Tny> OutputW;
  OutputW.setOnes();
  OutputW *= outputWeight;

  // Input Weights
  cvec<Tnu> InputW;
  InputW.setOnes();
  InputW *= inputWeight;

  if ( outputWeight < 0 )
  {
    throw std::invalid_argument( "The output weight was negative." );
  }

  // Input change weights
  cvec<Tnu> DeltaInputW;
  DeltaInputW.setZero();

  lmpc.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, prediction_horizon});

  PRINT("Finished setting weights.");

  // Set disturbances
  mat<Tndu, prediction_horizon> disturbance_input;
  disturbance_input.setOnes();
  disturbance_input *= lead_car_input;
  lmpc.setExogenuosInputs(disturbance_input);
  auto disturbance_vec = Bd_disturbance;
  disturbance_vec *= disturbance_input(0);

  PRINT("Finished setting disturbances.");

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

  PRINT("Finished setting constraints.");

  // Output reference point
  cvec<Tny> yRef;
  loadMatrixValuesFromJson(yRef, json_data, "yref");

  lmpc.setReferences(yRef, cvec<Tnu>::Zero(), cvec<Tnu>::Zero(), {0, prediction_horizon});

  PRINT("Finished setting references.");

  // I/O Setup
  std::ofstream x_outfile;
  std::ofstream x_predict_outfile;
  std::ofstream t_predict_outfile;
  std::ofstream iterations_outfile;
  std::ofstream u_outfile;
  std::ifstream x_infile;
  std::ifstream t_delays_infile;
  if (use_external_dynamics_computation) {
    // Open the pipes to Python. Each time we open one of these streams, 
    // this process pauses until 
    // it is connected to the Python process (or another process). 
    // The order needs to match the order in the reader process.
    
    PRINT("Starting IO setup.");

    assert_file_exists(x_out_filename);
    PRINT("About to open x_outfile output stream. Waiting for reader.");
    x_outfile.open(x_out_filename, std::ofstream::out);

    assert_file_exists(u_out_filename);
    PRINT("About to open u_outfile output stream. Waiting for reader.");
    u_outfile.open(u_out_filename, std::ofstream::out);

    assert_file_exists(x_predict_out_filename);
    PRINT("About to open x_predict_outfile output stream. Waiting for reader.");
    x_predict_outfile.open(x_predict_out_filename, std::ofstream::out);

    assert_file_exists(t_predict_out_filename);
    PRINT("About to open t_predict_outfile output stream. Waiting for reader.");
    t_predict_outfile.open(t_predict_out_filename, std::ofstream::out);

    assert_file_exists(iterations_out_filename);
    PRINT("About to open iterations_outfile output stream. Waiting for reader.");
    iterations_outfile.open(iterations_out_filename, std::ofstream::out);

    // In files.
    assert_file_exists(x_in_filename);
    PRINT("About to open x_infile input stream. Waiting for reader.");
    x_infile.open(x_in_filename, std::ofstream::in);

    assert_file_exists(t_delays_in_filename);
    PRINT("About to open t_delays_infile input stream. Waiting for reader.");
    t_delays_infile.open(t_delays_in_filename, std::ofstream::in);
    PRINT("All files open.");
  }

  // State vector state.
  mpc::cvec<Tnx> modelX, modeldX;
  mpc::cvec<Tnx> x_predict;
  // Set the initial value.
  loadMatrixValuesFromJson(modelX, json_data, "x0");
  
  // Predicted time of finishing computation.
  double t_predict;

  // Output vector.
  mpc::cvec<Tny> y;

  // Create a vector for storing the control input from the previous time step.
  mpc::cvec<Tnu> u;
  u = cvec<Tnu>::Zero();

  // Store the delay time required to compute the previous controller value.
  double t_delay_prev = 0;

  scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'

  for (int i = 0; i < n_time_steps; i++)
  {
    PRINT(std::endl << "====== Starting loop #" << i << " ======");
    PRINT("At x=" << modelX.transpose().format(fmt));


    // Begin a batch of Scarab statistics
    scarab_roi_dump_begin(); 
    dr_app_setup_and_start();

      if (use_state_after_delay_prediction)
      {
        t_predict = t_delay_prev;
        discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, t_predict, Ad_predictive, Bd_predictive, Bd_disturbance_predictive);
        
        x_predict = Ad_predictive * modelX + Bd_predictive * u + Bd_disturbance_predictive*lead_car_input;
        
        if (debug_dynamics_level >= 1)
        {
          PRINT("====== Ad_predictive: ")
          PRINT(Ad_predictive)
          PRINT("====== Bd_predictive: ")
          PRINT(Bd_predictive)
          PRINT("====== Bd_disturbance_predictive: ")
          PRINT(Bd_disturbance_predictive)
        }
        // int n_time_step_division = 1000;
        // double delta_t = t_delay_prev / double(n_time_step_division);
        // x_predict = modelX;
        // for (int i = 0; i < n_time_step_division; i++)
        // {
        //   // Do Euler's forward method for computing predicted state.
        //   x_predict += delta_t * (Ac * modelX + Bc * u + Bc_disturbance*lead_car_input);
        //   t_predict += delta_t;
        //   PRINT("t_predict: " << t_predict << ", after i=" << i << " Euler Forward steps. (delta_t=" << delta_t <<")")
        // }
        // x_predict = modelX + t_predict * (Ac * modelX + Bc * u);
      } else {
        x_predict = modelX;
        t_predict = 0;
      }
      if (debug_dynamics_level >= 1) 
      {
        PRINT("====== t_delay_prev: " << t_delay_prev);
        PRINT("====== t_predict: " << t_predict);
        PRINT("====== x_predict: " << x_predict.transpose());
      }

      // Compute a step of the MPC controller. This does NOT change the value of modelX.
      Result res = lmpc.step(x_predict, u);

    // Save a batch of Scarab statistics
    scarab_roi_dump_end();  
    dr_app_stop_and_cleanup();

    u = res.cmd;
    
    if (debug_optimizer_stats_level >= 1) 
    {
      PRINT("Optimizer Info")
      PRINT("         Return code: " << res.retcode)
      PRINT("       Result status: " << res.status)
      PRINT("Number of iterations: " << res.num_iterations)
      PRINT("                Cost: " << res.cost)
      PRINT("    Constraint error: " << res.primal_residual)
      PRINT("          Dual error: " << res.dual_residual)
    }

    optimizer_info_out_file << res.num_iterations << "," << res.cost << "," << res.primal_residual << "," << res.dual_residual << std::endl;  

    // if ( res.num_iterations == 0 )
    // {
    //   throw std::range_error( "Number of iterations was zero." );
    // }
    // if ( res.cost < 0)
    // {
    //   throw std::range_error( "The cost was negative." );
    // }

    // OptSequence has three properties: state, input, and output. 
    // Each predicted time step is stored in one row.
    // There are <prediction_horizon> many rows. 
    OptSequence optseq = lmpc.getOptimalSequence();

    if (use_external_dynamics_computation) {
      // Format the numerical arrays into strings (single lines) that we pass to Python.     

      sendCVec("x", i, modelX, x_outfile);
      sendCVec("x prediction", i, x_predict, x_predict_outfile);
      sendDouble("t prediction ", i, t_predict, t_predict_outfile);
      sendDouble("iterations ", i, res.num_iterations, iterations_outfile);
      sendCVec("u", i, u, u_outfile);

      // Read and print the contents of the file from Python.
      std::string x_in_line_from_py;
      std::string t_delay_in_line_from_py;
      
      if (debug_interfile_communication_level >= 2)
      {
        PRINT("Getting 'x' line from Python: ");
      }
      std::getline(x_infile, x_in_line_from_py);
      if (debug_interfile_communication_level >= 2)
      {
        PRINT("Getting 't_delays' line from Python: ");
      }
      std::getline(t_delays_infile, t_delay_in_line_from_py);
      if (debug_interfile_communication_level >= 2)
      {
        PRINT("Lines received from Python:")
        PRINT("\t      x: " << x_in_line_from_py);
        PRINT("\tt_delay: " << t_delay_in_line_from_py);
      }

      // Create a stringstream to parse the string
      std::stringstream x_ss(x_in_line_from_py);
      std::stringstream t_delay_ss(t_delay_in_line_from_py);
      
      for (int j_entry = 0; j_entry < modelX.rows(); j_entry++) {
        char comma; // to read and discard the comma
        x_ss >> modelX(j_entry) >> comma;
      }
      t_delay_ss >> t_delay_prev;
      if (debug_interfile_communication_level >= 1) 
      {
        PRINT("x (from Python): " << modelX.transpose().format(fmt_high_precision));
        PRINT("t_delay_prev (from Python): " << t_delay_prev);
      }
    } else {
      printVector("modelX", modelX);
      printMat("optseq.state", optseq.state);
      modelX = Ad * modelX + Bd * u + Bd_disturbance*lead_car_input;
      printVector("modelX", modelX);
    }
    
    // Set the model state to the next value of x from the optimal sequence. 
    y = C * modelX;
    if ((y-yRef).norm() < convergence_termination_tol)
    {
      PRINT("Converged to the origin after " << i << " steps.");
      break;
    }
    
  } 
  // printVector("y", y);

  if (use_external_dynamics_computation)
  {
    x_outfile << "Done" << std::endl;
    x_outfile.close();
    u_outfile.close();
    t_predict_outfile.close();
    x_predict_outfile.close();
    iterations_outfile.close();
    x_infile.close();
    t_delays_infile.close();
  }

  optimizer_info_out_file.close();

  return 0;
}
