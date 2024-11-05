// 
// Compile with g++ -c test_libmpc.cpp -I ~/libmpc-0.4.0/include/ -I /usr/include/eigen3/ -I /usr/local/include -std=c++20
// #include <cstdio>
#include <iostream>

#include "scarab_markers.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/algorithm/string/predicate.hpp>
#include <regex> // Used to find and rename Dynamorio trace directories.
#include "controller.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
using namespace mpc;

// #define USE_DYNAMORIO

// #define USE_DYNAMORIO
#ifdef USE_DYNAMORIO
  #include "dr_api.h"
  bool my_setenv(const char *var, const char *value){
    #ifdef UNIX
        return setenv(var, value, 1 /*override*/) == 0;
    #else
        return SetEnvironmentVariable(var, value) == TRUE;
    #endif
  }
#endif
// #include "scarabintheloop.h"

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_WITH_FILE_LOCATION(x) std::cout << __FILE__  << ":" << __LINE__ << ": " << x << std::endl;

// Load a header file for parsing and generating JSON files.
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// fstream provides I/O to file pipes.
#include <fstream>
 
Eigen::IOFormat fmt(3, // precision
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

inline void assertFileExists(const std::string& name) {
  // Check that a file exists.
  // std::cout << "Checking if \"" + name + "\" exists." << std::endl;
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
    PRINT_WITH_FILE_LOCATION(label << " (send to Python): " << out_str);
  } 
  outfile << "Loop " << i_loop << ": " << out_str << std::endl;
}

void sendDouble(std::string label, int i_loop, double x, std::ofstream& outfile) 
{
  if (debug_interfile_communication_level >= 1) 
  {
    PRINT_WITH_FILE_LOCATION(label << " (send to Python): " << x);
  } 
  outfile << "Loop " << i_loop << ": " << x << std::endl;
}


template<int rows>
void readVec(std::ifstream& x_infile, mpc::cvec<rows>& x) {
    std::string x_in_line_from_py;

    // Check debug level and print the appropriate message
    if (debug_interfile_communication_level >= 2) {
        PRINT_WITH_FILE_LOCATION("Getting 'x' line from Python: ");
    }

    // Get the line from the file
    std::getline(x_infile, x_in_line_from_py);

    if (debug_interfile_communication_level >= 2) {
        PRINT_WITH_FILE_LOCATION("Getting 't_delays' line from Python: ");
    }

    // Parse the line into the x object
    std::stringstream x_ss(x_in_line_from_py);
    for (int j_entry = 0; j_entry < x.rows(); j_entry++) {
        char comma; // to read and discard the comma
        x_ss >> x(j_entry) >> comma;
    }
}


int main()
// int main(int argc, char *argv[])
{
  PRINT("==== Start of acc_controller.main() ====")
  // /* We also test -rstats_to_stderr */
  // if (!my_setenv("DYNAMORIO_OPTIONS",
  //             "-stderr_mask 0xc -rstats_to_stderr "
  //             "-client_lib ';;-offline'")){
  //   std::cerr << "failed to set env var!\n";
  // }
  #ifdef USE_DYNAMORIO
    PRINT("Using DynamoRio.")

    /* We also test -rstats_to_stderr */
    if (!my_setenv("DYNAMORIO_OPTIONS",
                   "-stderr_mask 0xc -rstats_to_stderr "
                   "-client_lib ';;-offline'"))
        std::cerr << "failed to set env var!\n";
  #else 
    PRINT("Not using DynamoRio")
  #endif
  PRINT("PREDICTION_HORIZON: " << PREDICTION_HORIZON)
  PRINT("CONTROL_HORIZON: " << CONTROL_HORIZON)
  // return 0;

  // if (argc == 1){
  //     PRINT_WITH_FILE_LOCATION("Adaptive Cruise Control (ACC) controller. Example usage:")
  //     PRINT_WITH_FILE_LOCATION("\tacc_controller <simulation_directory>")
  //     PRINT_WITH_FILE_LOCATION("Typically, you wouldn't call this directly. Instead, call")
  //     PRINT_WITH_FILE_LOCATION("\t run_scarabintheloop <example_directory>.")
  //     return 0;
  // }

  // std::string sim_dir(argv[1]);
  std::string cwd = std::filesystem::current_path().string();
  std::string sim_dir = cwd + "/";
  PRINT_WITH_FILE_LOCATION("Simulation directory: " << sim_dir)

  // auto cwd = std::filesystem::current_path();
  // if ( boost::algorithm::ends_with(cwd.string(), "sim_dir") )
  // {
  //     example_path = cwd.parent_path().string() + "/";
  // } else
  // {
  //   example_path = cwd.string() + "/";
  // }
  // PRINT_WITH_FILE_LOCATION("Example path: " << example_path)

  // File names for the pipes we use for communicating with Python.
  // std::string data_out_file_path = "data_out.json";

  std::string config_file_path            = sim_dir + "config.json";
  std::string x_out_filename              = sim_dir + "x_c++_to_py";
  std::string x_predict_out_filename      = sim_dir + "x_predict_c++_to_py";
  std::string t_predict_out_filename      = sim_dir + "t_predict_c++_to_py";
  std::string iterations_out_filename     = sim_dir + "iterations_c++_to_py";
  std::string u_out_filename              = sim_dir + "u_c++_to_py";
  std::string x_in_filename               = sim_dir + "x_py_to_c++";
  std::string t_delays_in_filename        = sim_dir + "t_delay_py_to_c++";
  std::string optimizer_info_out_filename = sim_dir + "optimizer_info.csv";

  assertFileExists(config_file_path);
  assertFileExists(x_out_filename);
  assertFileExists(x_predict_out_filename);
  assertFileExists(t_predict_out_filename);
  assertFileExists(iterations_out_filename);
  assertFileExists(u_out_filename);
  assertFileExists(x_in_filename);
  assertFileExists(t_delays_in_filename);
  
  // std::string system_dynamics_filename = sim_dir + "system_dynamics.json";
  // std::string system_dyanmics_lock_filename = system_dynamics_filename + ".lock";
  // PRINT_WITH_FILE_LOCATION("system_dynamics_filename: " << system_dynamics_filename)

  std::ifstream config_json_file(config_file_path);
  json json_data = json::parse(config_json_file);
  config_json_file.close();

  int n_time_steps = json_data.at("n_time_steps");
  auto time_indices = json_data.at("time_indices");
  for (int time_index : time_indices){
    PRINT("time_index" << time_index)
  }

  auto debug_config = json_data["==== Debgugging Levels ===="];
  debug_interfile_communication_level = debug_config["debug_interfile_communication_level"];
  debug_optimizer_stats_level = debug_config["debug_optimizer_stats_level"];
  debug_dynamics_level = debug_config["debug_dynamics_level"];

  bool use_state_after_delay_prediction = json_data.at("system_parameters").at("mpc_options").at("use_state_after_delay_prediction");

  // Vector sizes. How to fix these so that we don't modify this file
  const int Tnx = 4;  // State dimension
  const int Tnu = 1;  // Control dimension
  const int Tndu = 0; // Exogenous control (disturbance) dimension
  const int Tny = 0;  // Output dimension

  // Controller* controller = Controller::createController("LMPCController");
  Controller* controller = Controller::createController(json_data.at("system_parameters").at("controller_type"), json_data);

  // Set whether the evolution of the dyanmics are computed extenally 
  // (with Python) or are done within this process, using libmpc.
  bool use_external_dynamics_computation = json_data["Simulation Options"]["use_external_dynamics_computation"];

  std::ofstream optimizer_info_out_file(optimizer_info_out_filename);
  optimizer_info_out_file << "num_iterations,cost,primal_residual,dual_residual" << std::endl;

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
    // this process pauses until it is connected to the Python process (or another process). 
    // The order needs to match the order in the reader process.
    
    PRINT_WITH_FILE_LOCATION("Starting IO setup.");

    assertFileExists(x_out_filename);
    PRINT_WITH_FILE_LOCATION("Opening x_outfile. Waiting for reader.");
    x_outfile.open(x_out_filename, std::ofstream::out);

    assertFileExists(u_out_filename);
    PRINT_WITH_FILE_LOCATION("Opening u_outfile. Waiting for reader.");
    u_outfile.open(u_out_filename, std::ofstream::out);

    assertFileExists(x_predict_out_filename);
    PRINT_WITH_FILE_LOCATION("Opening x_predict_outfile. Waiting for reader.");
    x_predict_outfile.open(x_predict_out_filename, std::ofstream::out);

    assertFileExists(t_predict_out_filename);
    PRINT_WITH_FILE_LOCATION("Opening t_predict_outfile. Waiting for reader.");
    t_predict_outfile.open(t_predict_out_filename, std::ofstream::out);

    assertFileExists(iterations_out_filename);
    PRINT_WITH_FILE_LOCATION("Opening iterations_outfile. Waiting for reader.");
    iterations_outfile.open(iterations_out_filename, std::ofstream::out);

    // In files.
    assertFileExists(x_in_filename);
    PRINT_WITH_FILE_LOCATION("Opening x_infile. Waiting for reader.");
    x_infile.open(x_in_filename, std::ofstream::in);

    assertFileExists(t_delays_in_filename);
    PRINT_WITH_FILE_LOCATION("Opening t_delays_infile. Waiting for reader.");
    t_delays_infile.open(t_delays_in_filename, std::ofstream::in);
    PRINT_WITH_FILE_LOCATION("All files open.");
  }

  // State vector state.
  mpc::cvec<Tnx> modelX, modeldX;
  mpc::cvec<Tnx> x_predict;
  
  // Predicted time of finishing computation.
  double t_predict;

  // Output vector.
  mpc::cvec<Tny> y;

  // Create a vector for storing the control input from the previous time step.
  mpc::cvec<Tnu> u;

  // Set the initial values from JSON.
  // loadMatrixValuesFromJson(modelX, json_data, "x0");
  readVec(x_infile, modelX);

  loadMatrixValuesFromJson(u, json_data, "u0");

  // Store the delay time required to compute the previous controller value.
  double t_delay_prev = 0;

  if (!json_data["Simulation Options"]["use_fake_scarab_computation_times"] && !json_data["Simulation Options"]["parallel_scarab_simulation"]) {
    scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'
  }

  for (int i = 0; i < n_time_steps; i++)
  {
    PRINT(std::endl << "====== Starting loop #" << i+1 << " of " << n_time_steps << " ======");
    PRINT("x=" << modelX.transpose().format(fmt));
    PRINT("u=" << u.transpose().format(fmt));

    // Begin a batch of Scarab statistics
    #ifdef USE_DYNAMORIO
        PRINT_WITH_FILE_LOCATION("Starting DynamoRIO region of interest.")
        dr_app_setup_and_start();
    #else
      if (!json_data["Simulation Options"]["use_fake_scarab_computation_times"]) {
        PRINT_WITH_FILE_LOCATION("Starting Scarab region of interest.")
        scarab_roi_dump_begin(); 
      }
    #endif

    if (use_state_after_delay_prediction)
    {
      // t_predict = t_delay_prev;
      // discretization<Tnx, Tnu, Tndu>(Ac, Bc, Bc_disturbance, t_predict, Ad_predictive, Bd_predictive, Bd_disturbance_predictive);
      
      // x_predict = Ad_predictive * modelX + Bd_predictive * u + Bd_disturbance_predictive*lead_car_input;
      
      // if (debug_dynamics_level >= 1)
      // {
      //   PRINT("====== Ad_predictive: ")
      //   PRINT(Ad_predictive)
      //   PRINT("====== Bd_predictive: ")
      //   PRINT(Bd_predictive)
      //   PRINT("====== Bd_disturbance_predictive: ")
      //   PRINT(Bd_disturbance_predictive)
      // }
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
    // Result res = lmpc.step(x_predict, u);

    // // Update internal state and calculate control
    controller->update_internal_state(x_predict);
    controller->calculateControl();
    u = controller->control;

    #ifdef USE_DYNAMORIO
      PRINT_WITH_FILE_LOCATION("End of DynamoRIO region of interest.")
      dr_app_stop_and_cleanup();

      std::filesystem::path folder(sim_dir);
      if(!std::filesystem::is_directory(folder))
      {
          throw std::runtime_error(folder.string() + " is not a folder");
      }
      std::vector<std::string> file_list;
      PRINT("DynamoRIO Trace Files in " << folder.string())

      // Regex to match stings like "<path to sim dir>/drmemtrace.acc_controller_5_2_dynamorio.268046.6519.dir"
      std::regex dynamorio_trace_filename_regex("(.*)drmemtrace\\.acc_controller_\\d_\\d_dynamorio\\.\\d+\\.\\d+\\.dir");
      
      bool is_first_dir_found = true;
      for (const auto& folder_entry : std::filesystem::directory_iterator(folder)) {
          const auto full_path = folder_entry.path().string();
          bool is_dynamorio_trace = std::regex_match(full_path, dynamorio_trace_filename_regex);
          if (is_dynamorio_trace) {
            PRINT("- " << folder_entry.path().string())
            std::string new_dir_name("dynamorio_trace_" +  std::to_string(i));
            std::string new_path = std::regex_replace(full_path, dynamorio_trace_filename_regex, new_dir_name);
            PRINT("Renaming")
            PRINT("\t     " << folder_entry.path().string())
            PRINT("\t to: " << new_path)
            std::filesystem::rename(folder_entry.path().string(), new_dir_name);
            // Check thfat we don't have multiple DynamoRIO trace files, which would indicate that we failed to rename them incrementally. 
            if (!is_first_dir_found) {
              throw std::runtime_error("Multiple DynamoRIO trace folders found in " + folder.string());
            }
            is_first_dir_found = false;
          } else {
            // PRINT("- Not a DyanmoRIO trace directory: " << full_path)
          }
      }
      if (is_first_dir_found) {
        throw std::runtime_error("No DynamoRIO trace folder found in " + folder.string());
      }

      PRINT_WITH_FILE_LOCATION("Finished saving trace with DyanmoRIO")
    #else
      if (!json_data["Simulation Options"]["use_fake_scarab_computation_times"]) {
        // Save a batch of Scarab statistics
        PRINT_WITH_FILE_LOCATION("End of Scarab region of interest.")
        scarab_roi_dump_end();  
      }
    #endif

    // u = res.cmd;
    
    // if (debug_optimizer_stats_level >= 1) 
    // {
    //   PRINT_WITH_FILE_LOCATION("Optimizer Info")
    //   PRINT_WITH_FILE_LOCATION("         Return code: " << res.retcode)
    //   PRINT_WITH_FILE_LOCATION("       Result status: " << res.status)
    //   PRINT_WITH_FILE_LOCATION("Number of iterations: " << res.num_iterations)
    //   PRINT_WITH_FILE_LOCATION("                Cost: " << res.cost)
    //   PRINT_WITH_FILE_LOCATION("    Constraint error: " << res.primal_residual)
    //   PRINT_WITH_FILE_LOCATION("          Dual error: " << res.dual_residual)
    // }

    // optimizer_info_out_file << res.num_iterations << "," << res.cost << "," << res.primal_residual << "," << res.dual_residual << std::endl;  

    // OptSequence has three properties: state, input, and output. 
    // Each predicted time step is stored in one row.
    // There are <prediction_horizon> many rows. 
    // OptSequence optseq = lmpc.getOptimalSequence();

    if (use_external_dynamics_computation) {
      // Format the numerical arrays into strings (single lines) that we pass to Python.     
      sendCVec("x", i, modelX, x_outfile);
      sendCVec("x prediction", i, x_predict, x_predict_outfile);
      sendDouble("t prediction ", i, t_predict, t_predict_outfile);
      sendDouble("iterations ", i, 100, iterations_outfile);
      sendCVec("u", i, u, u_outfile);

      // Read and print the contents of the file from Python.
      readVec(x_infile, modelX);

      std::string t_delay_in_line_from_py;

      std::getline(t_delays_infile, t_delay_in_line_from_py);
      if (debug_interfile_communication_level >= 2)
      {
        PRINT_WITH_FILE_LOCATION("Lines received from Python:")
        // PRINT_WITH_FILE_LOCATION("\t      x: " << x_in_line_from_py);
        PRINT_WITH_FILE_LOCATION("\tt_delay: " << t_delay_in_line_from_py);
      }

      // Create a stringstream to parse the string
      std::stringstream t_delay_ss(t_delay_in_line_from_py);
      
      if (debug_interfile_communication_level >= 2)
      {
        PRINT_WITH_FILE_LOCATION("Getting 't_delays' line from Python: ");
      }

      t_delay_ss >> t_delay_prev;
      if (debug_interfile_communication_level >= 1) 
      {
        PRINT_WITH_FILE_LOCATION("x (from Python): " << modelX.transpose().format(fmt));
        PRINT_WITH_FILE_LOCATION("t_delay_prev (from Python): " << t_delay_prev);
      }
    } else { // If use internal dynamics computation
      // printVector("modelX", modelX);
      // printMat("optseq.state", optseq.state);
      // modelX = Ad * modelX + Bd * u + Bd_disturbance*lead_car_input;
      // printVector("modelX", modelX);
    }
    
    // Set the model state to the next value of x from the optimal sequence. 
    // y = C * modelX;
    // if ((y-controller.yRef).norm() < controller.convergence_termination_tol)
    // {
    //   PRINT_WITH_FILE_LOCATION("Converged to the origin after " << i << " steps.");
    //   break;
    // }
    
  } 
  PRINT_WITH_FILE_LOCATION("Finshed looping through " << n_time_steps << " time steps.")
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

  // optimizer_info_out_file.close();

  return 0;
}
