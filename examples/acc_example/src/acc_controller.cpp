#include <iostream>

#include <mpc/LMPC.hpp>
#include <mpc/Utils.hpp>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/algorithm/string/predicate.hpp>
#include <regex> // Used to find and rename Dynamorio trace directories.
#include "controller.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;
using namespace mpc;

#ifdef USE_DYNAMORIO
  #include "dr_api.h"
  bool my_setenv(const char *var, const char *value){
    #ifdef UNIX
        return setenv(var, value, 1 /*override*/) == 0;
    #else
        return SetEnvironmentVariable(var, value) == TRUE;
    #endif
  }
// #elif defined(USE_EXECUTION_DRIVEN_SCARAB)
#else
  #define USE_EXECUTION_DRIVEN_SCARAB
  #include "scarab_markers.h"
#endif

// Define a macro to print a value 
#define PRINT(x) std::cout << x << std::endl;
#define PRINT_WITH_FILE_LOCATION(x) std::cout << __FILE__  << ":" << __LINE__ << ": " << x << std::endl;

// fstream provides I/O to file pipes.
#include <fstream>
 
Eigen::IOFormat fmt(3, // precision
                    0,    // flags
                    ", ", // coefficient separator
                    "\n", // row prefix
                    "[",  // matrix prefix
                    "]"); // matrix suffix.
Eigen::IOFormat fmt_high_precision(20, // precision
                    0,    // flags
                    ", ", // coefficient separator
                    "\n", // row prefix
                    "[",  // matrix prefix
                    "]"); // matrix suffix.

inline void assertFileExists(const std::string& name) {
  // Check that a file exists.
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
void loadMatrixValuesFromJson(mat<rows, cols>& mat_out, nlohmann::json json_data, std::string key) {
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

std::string example_path;

class PipeReader {
  protected:
    std::string filename;
    std::ifstream pipe_file;

  public:
    PipeReader(std::string _filename) {
      filename = _filename;
    }

    void open() {
      assertFileExists(filename);
      PRINT_WITH_FILE_LOCATION("Opening " << filename << ". Waiting for writer...");
      pipe_file.open(filename, std::ofstream::in);
    }

    void close() {
      PRINT_WITH_FILE_LOCATION("Closing " << filename << "...")
      pipe_file.close();
    }
};

class PipeDoubleReader : public PipeReader {
  public:
    // Constructor (call parent constructor).
    PipeDoubleReader (std::string filename) : PipeReader (filename) {}

    void read(std::string label, double& value) {
      // Check debug level and print the appropriate message
      if (debug_interfile_communication_level >= 2) {
          PRINT_WITH_FILE_LOCATION("Getting '" << label << "' line from Python...");
      }

      std::string value_in_line_from_py;

      std::getline(pipe_file, value_in_line_from_py);
      if (debug_interfile_communication_level >= 2)
      {
        PRINT_WITH_FILE_LOCATION("Lines received from Python:")
        PRINT_WITH_FILE_LOCATION("\t" << label << ": " << value_in_line_from_py);
      }

      // Create a stringstream to parse the string
      std::stringstream value_ss(value_in_line_from_py);
      value_ss >> value;
  }
};

class PipeVectorReader : public PipeReader {
  public:
    // Constructor (call parent constructor).
    PipeVectorReader (std::string filename) : PipeReader (filename) {}
    
    template<int rows>
    void read(std::string label, mpc::cvec<rows>& x) {
        std::string x_in_line_from_py;

        // Check debug level and print the appropriate message
        if (debug_interfile_communication_level >= 2) {
            PRINT_WITH_FILE_LOCATION("Getting '" << label << "' line from Python: ");
        }

        // Get the line from the file
        std::getline(pipe_file, x_in_line_from_py);

        // Parse the line into the x object
        std::stringstream x_ss(x_in_line_from_py);
        for (int j_entry = 0; j_entry < x.rows(); j_entry++) {
            char comma; // to read and discard the comma
            x_ss >> x(j_entry) >> comma;
        }

        if (debug_interfile_communication_level >= 2) {
          PRINT("Received " << label << "=" << x.transpose().format(fmt) << " from " << filename);
        }
    }
};


class PipeWriter {
  protected:
    std::string filename;
    std::ofstream pipe_file;

  public:
    PipeWriter(std::string _filename) {
      filename = _filename;
    }

    void open() {
      assertFileExists(filename);
      PRINT_WITH_FILE_LOCATION("Opening " << filename << ". Waiting for reader...");
      pipe_file.open(filename, std::ofstream::out);
    }

    void close() {
      PRINT_WITH_FILE_LOCATION("Closing " << filename << "...")
      pipe_file.close();
    }
};

class PipeDoubleWriter : public PipeWriter {
  public:
    // Constructor (call parent constructor).
    PipeDoubleWriter (std::string filename) : PipeWriter (filename) {}

    void write(std::string label, int i_loop, double x) 
    {
      if (debug_interfile_communication_level >= 1) 
      {
        PRINT_WITH_FILE_LOCATION(label << " (send to Python): " << x);
      } 
      pipe_file << "Loop " << i_loop << ": " << x << std::endl;
    }
};

class PipeVectorWriter : public PipeWriter {
  public:
    // Constructor (call parent constructor).
    PipeVectorWriter (std::string filename) : PipeWriter (filename) {}

    template<int rows>
    void write(std::string label, int i_loop, mpc::cvec<rows> x) 
    {
      auto out_str = x.transpose().format(fmt_high_precision);
      if (debug_interfile_communication_level >= 1) 
      {
        PRINT_WITH_FILE_LOCATION("Send \"" << label << "\" to Python: " << out_str);
      } 
      pipe_file << "Loop " << i_loop << ": " << out_str << std::endl;
      if (debug_interfile_communication_level >= 2) 
      {
        PRINT_WITH_FILE_LOCATION("SENT \"" << label << "\" to Python: " << out_str);
      } 
    }
};

nlohmann::json readJson(std::string file_path) {
  assertFileExists(file_path);
  std::ifstream json_file(file_path);
  nlohmann::json json_data(json::parse(json_file));
  json_file.close();
  return json_data;
}

int main()
// int main(int argc, char *argv[])
{
  PRINT("==== Start of acc_controller.main() ====")
  #ifdef USE_DYNAMORIO
    PRINT_WITH_FILE_LOCATION("Using DynamoRio.")

    /* We also test -rstats_to_stderr */
    if (!my_setenv("DYNAMORIO_OPTIONS",
                   "-stderr_mask 0xc -rstats_to_stderr "
                   "-client_lib ';;-offline'"))
        std::cerr << "failed to set env var!\n";
  #elif defined(USE_EXECUTION_DRIVEN_SCARAB)
    PRINT_WITH_FILE_LOCATION("Running without DyanmoRIO (Execution-driven Scarab or just plain execution)")
  #else
    PRINT_WITH_FILE_LOCATION("Not using Scarab or DyanmoRIO to track statistics.")
  #endif
  PRINT("PREDICTION_HORIZON: " << PREDICTION_HORIZON)
  PRINT("CONTROL_HORIZON: " << CONTROL_HORIZON)

  std::string cwd = std::filesystem::current_path().string();
  std::string sim_dir = cwd + "/";
  PRINT_WITH_FILE_LOCATION("Simulation directory: " << sim_dir)

  // Writers
  PipeDoubleWriter    t_predict_writer(sim_dir + "t_predict_c++_to_py");
  PipeVectorWriter x_prediction_writer(sim_dir + "x_predict_c++_to_py");
  PipeDoubleWriter   iterations_writer(sim_dir + "iterations_c++_to_py");
  PipeVectorWriter            u_writer(sim_dir + "u_c++_to_py");
  // Readers
  PipeVectorReader      x_reader(sim_dir + "x_py_to_c++");
  PipeDoubleReader delays_reader(sim_dir + "t_delay_py_to_c++");
  std::string optimizer_info_out_filename = sim_dir + "optimizer_info.csv";

  // Open and read JSON 
  nlohmann::json json_data = readJson(sim_dir + "config.json");

  int max_time_steps = json_data.at("max_time_steps");

  nlohmann::json debug_config = json_data.at("==== Debgugging Levels ====");
  debug_interfile_communication_level = debug_config.at("debug_interfile_communication_level");
  debug_optimizer_stats_level         = debug_config.at("debug_optimizer_stats_level");
  debug_dynamics_level                = debug_config.at("debug_dynamics_level");

  // Vector sizes.
  const int Tnx  =  TNX; // State dimension
  const int Tnu  =  TNU; // Control dimension
  const int Tndu = TNDU; // Exogenous control (disturbance) dimension
  const int Tny  =  TNY; // Output dimension

  std::string controller_type = json_data.at("system_parameters").at("controller_type");
  Controller* controller = Controller::createController(controller_type, json_data);

  std::ofstream optimizer_info_out_file(optimizer_info_out_filename);
  optimizer_info_out_file << "num_iterations,cost,primal_residual,dual_residual" << std::endl;

  // I/O Setup

  // Open the pipes to Python. Each time we open one of these streams, 
  // this process pauses until it is connected to the Python process (or another process). 
  // The order needs to match the order in the reader process.
  PRINT_WITH_FILE_LOCATION("Starting Pipe Readers/Writers setup.");

  // Open out files.
  u_writer.open();
  x_prediction_writer.open();
  t_predict_writer.open();
  iterations_writer.open();

  // In files.
  x_reader.open();
  delays_reader.open();
  PRINT_WITH_FILE_LOCATION("All files open.");

  // State vector state.
  mpc::cvec<Tnx> modelX, modeldX;
  mpc::cvec<Tnx> x_predict;
  
  // Predicted time of finishing computation.
  double t_predict;

  // Output vector.
  mpc::cvec<Tny> y;

  // Create a vector for storing the control input from the previous time step.
  mpc::cvec<Tnu> u;
  
  loadMatrixValuesFromJson(u, json_data, "u0");

  // Store the delay time required to compute the previous controller value.
  double t_delay_prev = 0;

  #if defined(USE_EXECUTION_DRIVEN_SCARAB)
    if (!json_data.at("Simulation Options").at("use_fake_delays")) {
      scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'
    }
  #endif


  for (int i = 0; i < max_time_steps; i++)
  {
    PRINT(std::endl << "====== Starting loop #" << i+1 << " of " << max_time_steps << " ======");
    
    // Read the value of 'x' for this iteration.
    x_reader.read("x", modelX);

    // Begin a batch of Scarab statistics
    #if defined(USE_DYNAMORIO)
        PRINT_WITH_FILE_LOCATION("Starting DynamoRIO region of interest.")
        dr_app_setup_and_start();
    #elif defined(USE_EXECUTION_DRIVEN_SCARAB)
        PRINT_WITH_FILE_LOCATION("Starting Scarab region of interest.")
        scarab_roi_dump_begin(); 
    #else
        PRINT_WITH_FILE_LOCATION("Starting region of interest without statistics recording.")
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
      PRINT("t_delay_prev: " << t_delay_prev);
      PRINT("t_predict: " << t_predict);
      PRINT("x_predict: " << x_predict.transpose());
    }

    // Compute a step of the MPC controller. This does NOT change the value of modelX.
    // Result res = lmpc.step(x_predict, u);

    // // Update internal state and calculate control
    controller->update_internal_state(x_predict);
    controller->calculateControl();
    u = controller->getLastControl();
    // iterations = controller->getLastLmpcIterations()

    #if defined(USE_DYNAMORIO)
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
    #elif defined(USE_EXECUTION_DRIVEN_SCARAB)
        // Save a batch of Scarab statistics
        PRINT_WITH_FILE_LOCATION("End of Scarab region of interest.")
        scarab_roi_dump_end();  
    #else
        PRINT_WITH_FILE_LOCATION("End of region of interest. Statistics recording is disabled.")
    #endif

    // optimizer_info_out_file << res.num_iterations << "," << res.cost << "," << res.primal_residual << "," << res.dual_residual << std::endl;  

    // if ( res.num_iterations == 0 )
    // {
    //   throw std::range_error( "Number of iterations was zero." );
    // }
    // if ( res.cost < 0)
    // {
    //   PRINT_WITH_FILE_LOCATION("The cost was " << res.cost)
    //   throw std::range_error( "The cost was negative." );
    // }

    // OptSequence has three properties: state, input, and output. 
    // Each predicted time step is stored in one row.
    // There are <prediction_horizon> many rows. 
    // OptSequence optseq = lmpc.getOptimalSequence();

    u_writer.write(                      "u", i, u);
    x_prediction_writer.write("x prediction", i, x_predict);
    t_predict_writer.write( "t prediction", i, t_predict);
    iterations_writer.write( "iterations ", i, -1); // TODO: Fix iterations. Currently sending "-1" because we aren't reading the iterations.

    delays_reader.read("t_delay", t_delay_prev);
    
    if (debug_interfile_communication_level >= 1) 
    {
      PRINT_WITH_FILE_LOCATION("x (from Python): " << modelX.transpose().format(fmt));
      PRINT_WITH_FILE_LOCATION("t_delay_prev (from Python): " << t_delay_prev);
    }
    
    // Set the model state to the next value of x from the optimal sequence. 
    // y = C * modelX;
    // if ((y-controller.yRef).norm() < controller.convergence_termination_tol)
    // {
    //   PRINT_WITH_FILE_LOCATION("Converged to the origin after " << i << " steps.");
    //   break;
    // }
    
  } 
  PRINT_WITH_FILE_LOCATION("Finished looping through " << max_time_steps << " time steps. Closing files...")

  u_writer.close();
  x_prediction_writer.close();
  t_predict_writer.close();
  iterations_writer.close();
  x_reader.close();
  delays_reader.close();

  // PRINT_WITH_FILE_LOCATION("Closing optimizer_info_out_file...")
  // optimizer_info_out_file.close();
  PRINT("Finished closing files. All done!.")

  return 0;
}
