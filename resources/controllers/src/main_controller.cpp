#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <regex> // Used to find and rename Dynamorio trace directories.
#include "controller.h"

#include <chrono>
#include <thread>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

template <int M = Eigen::Dynamic,
          int N = Eigen::Dynamic>
using mat = Eigen::Matrix<double, M, N>;

// Define a column vector type.
template <int N = Eigen::Dynamic>
using cvec = Eigen::Matrix<double, N, 1>;

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

int debug_program_flow_level;
int debug_interfile_communication_level;
int debug_optimizer_stats_level;
int debug_dynamics_level;
int debug_scarab_level;

// fstream provides I/O to file pipes.
#include <fstream>
 
Eigen::IOFormat fmt(3,    // precision
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
    
      // Check debug level and print the appropriate message
      if (debug_interfile_communication_level >= 2) {
          PRINT_WITH_FILE_LOCATION("Building a matrix from key='" << key << "'...");
          PRINT("Value: " << json_data.at(key))
      }

      if (json_data.at(key).size() != rows*cols)
      {
        throw std::invalid_argument(string_format(
              "loadMatrixValuesFromJson(): The number of entries (%d) in json_data[\"%s\"] does not match the expected number of entries in mat_out (%dx%d).", 
              json_data.at(key).size(),  key.c_str(), rows, cols));
      }

      int i = 0;
      for (auto& element : json_data.at(key)) {
        mat_out[i] = element;
        i++;
      }
      
      // Check debug level and print the appropriate message
      if (debug_interfile_communication_level >= 2) {
          PRINT_WITH_FILE_LOCATION("Built a matrix from key='" << key << "': " << mat_out);
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

class PipeReader {
  protected:
    std::string filename;
    std::ifstream pipe_file_stream;

  public:
    PipeReader(std::string _filename) {
      filename = _filename;
    }

    void open() {
      assertFileExists(filename);
      PRINT_WITH_FILE_LOCATION("Opening " << filename << ". Waiting for writer...");
      pipe_file_stream.open(filename, std::ofstream::in);
    }

    void close() {
      PRINT("Closing " << filename << "...")
      pipe_file_stream.close();
    }

  protected:
    void read_line(std::string label, std::string& entire_line) {
      // Read an entire line, rereading until the line ends with a '\n' character.
      
      // Check debug level and print the appropriate message
      if (debug_interfile_communication_level >= 2) {
          PRINT_WITH_FILE_LOCATION("Reading '" << label << "' line from Python...");
      }
      for (int i = 0; i < 300; i++) {
        std::string line_from_py;
        std::getline(pipe_file_stream, line_from_py);
        entire_line += line_from_py;
        if (line_from_py.length() > 0 && line_from_py.back() != '\n') {
          if (debug_interfile_communication_level >= 2)
          {
            PRINT_WITH_FILE_LOCATION("Line received from Python:")
            PRINT_WITH_FILE_LOCATION("\t" << label << ": " << line_from_py);
            PRINT_WITH_FILE_LOCATION("Entire Line received from Python:")
            PRINT_WITH_FILE_LOCATION("\t" << label << ": " << entire_line);
          }
          return;
        } else {
          // Sleep a very short time so that the process is not running at 100%.
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          if (i % 100 == 0){
            PRINT_WITH_FILE_LOCATION("WAITNG FOR " << label << " (i=" << i << "). entire_line: " << entire_line << ". line_from_py:" << line_from_py);
          }
        }
      }
      PRINT_WITH_FILE_LOCATION("Never received full line from Python:")
      PRINT_WITH_FILE_LOCATION("\t" << label << ": " << entire_line);
      throw std::runtime_error("Reading " + label + " timed out!");
    }
};

class PipeDoubleReader : public PipeReader {
  public:
    // Constructor (call parent constructor).
    PipeDoubleReader (std::string filename) : PipeReader (filename) {}

    void read(std::string label, double& value) {
      std::string value_in_line_from_py;
      read_line(label, value_in_line_from_py);

      // Create a stringstream to parse the string.
      std::stringstream value_ss(value_in_line_from_py);
      value_ss >> value;
  }
};

class PipeVectorReader : public PipeReader {
  public:
    // Constructor (call parent constructor).
    PipeVectorReader (std::string filename) : PipeReader (filename) {}
    
    template<int rows>
    void read(std::string label, cvec<rows>& x) {
        std::string x_in_line_from_py;
        read_line(label, x_in_line_from_py);

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
    std::ofstream pipe_file_stream;

  public:
    PipeWriter(std::string _filename) {
      filename = _filename;
      assertFileExists(filename);
    }

    void open() {
      assertFileExists(filename);
      PRINT_WITH_FILE_LOCATION("Opening " << filename << ". Waiting for reader...");
      pipe_file_stream.open(filename, std::ofstream::out);
    }

    void close() {
      PRINT_WITH_FILE_LOCATION("Closing " << filename << "...")
      pipe_file_stream.close();
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
      pipe_file_stream << "Loop " << i_loop << ": " << x << std::endl;
    }
};

class PipeVectorWriter : public PipeWriter {
  public:
    // Constructor (call parent constructor).
    PipeVectorWriter (std::string filename) : PipeWriter (filename) {}

    template<int rows>
    void write(std::string label, int i_loop, cvec<rows> x) 
    {
      auto out_str = x.transpose().format(fmt_high_precision);
      if (debug_interfile_communication_level >= 1) 
      {
        PRINT_WITH_FILE_LOCATION("Send \"" << label << "\" to Python: " << out_str);
      } 
      pipe_file_stream << "Loop " << i_loop << ": " << out_str << std::endl;
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


class StatusReader {
  protected:
    std::string filename;
    std::ifstream file_stream;

  public:
    StatusReader(std::string _filename) {
      filename = _filename;
    }

    void open() {
      assertFileExists(filename);
      PRINT_WITH_FILE_LOCATION("Opening status file: " << filename << ".");
      file_stream.open(filename, std::ofstream::in);
    }

    void close() {
      PRINT("Closing status file: " << filename << ".")
      file_stream.close();
    }

    bool is_simulator_running() {
      // Read an entire line, rereading until the line ends with a '\n' character.
      
      for (int i = 0; i < 100; i++){
        // Check debug level and print the appropriate message
        if (debug_interfile_communication_level >= 2) {
            PRINT_WITH_FILE_LOCATION("Reading status line from Python...");
        }
        std::string line_from_py;
        file_stream.clear(); 
        file_stream.seekg(0);
        std::getline(file_stream, line_from_py);
        if (debug_interfile_communication_level >= 2) {
          PRINT_WITH_FILE_LOCATION("Line received from Python:")
          PRINT_WITH_FILE_LOCATION("\tstatus: " << line_from_py);
        }
        if (line_from_py == "RUNNING") {
          return true;
        } else if (line_from_py == "FINISHED") {
          return false;
        } else if (line_from_py == "ERROR") {
          throw std::runtime_error("The Python script posted an error status to " + filename);
        } else {
          PRINT_WITH_FILE_LOCATION("Pausing status check becuase an unrecognized status was received: " << line_from_py)
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
      throw std::runtime_error("StatusReader.is_simulator_running() timed out.");
    }
};

int main()
// int main(int argc, char *argv[])
{
  PRINT("==== Start of " << __FILE__ << ".main() ====")
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
  PipeVectorWriter            u_writer(sim_dir + "u_c++_to_py");
  PipeVectorWriter x_prediction_writer(sim_dir + "x_predict_c++_to_py");
  PipeDoubleWriter    t_predict_writer(sim_dir + "t_predict_c++_to_py");
  PipeDoubleWriter   iterations_writer(sim_dir + "iterations_c++_to_py");
  // Readers
  StatusReader     status_reader(sim_dir + "status_py_to_c++");
  PipeVectorReader      x_reader(sim_dir + "x_py_to_c++");
  PipeDoubleReader delays_reader(sim_dir + "t_delay_py_to_c++");
  std::string optimizer_info_out_filename = sim_dir + "optimizer_info.csv";

  // Open and read JSON 
  nlohmann::json json_data = readJson(sim_dir + "config.json");

  int n_time_steps = json_data.at("n_time_steps");

  nlohmann::json debug_config = json_data.at("==== Debgugging Levels ====");
  debug_program_flow_level            = debug_config.at("debug_program_flow_level");
  debug_interfile_communication_level = debug_config.at("debug_interfile_communication_level");
  debug_optimizer_stats_level         = debug_config.at("debug_optimizer_stats_level");
  debug_dynamics_level                = debug_config.at("debug_dynamics_level");
  debug_scarab_level                  = debug_config.at("debug_scarab_level");

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
  if (debug_interfile_communication_level >= 1)
  {
    PRINT_WITH_FILE_LOCATION("Starting Pipe Readers/Writers setup.");
  }

  // Open out files.
  u_writer.open();
  x_prediction_writer.open();
  t_predict_writer.open();
  iterations_writer.open();

  // In files.
  status_reader.open();
  x_reader.open();
  delays_reader.open();
  if (debug_interfile_communication_level >= 1) {
    PRINT_WITH_FILE_LOCATION("All files open.");
  }

  // State vector state.
  cvec<Tnx> modelX, modeldX;
  cvec<Tnx> x_predict;
  
  // Predicted time of finishing computation.
  double t_predict;

  // Output vector.
  cvec<Tny> y;

  // Create a vector for storing the control input from the previous time step.
  cvec<Tnu> u;
  
  loadMatrixValuesFromJson(u, json_data, "u0");

  // Store the delay time required to compute the previous controller value.
  double t_delay_prev = 0;

  #if defined(USE_EXECUTION_DRIVEN_SCARAB)
    if (!json_data.at("Simulation Options").at("use_fake_delays")) {
      scarab_begin(); // Tell Scarab to stop "fast forwarding". This is needed for '--pintool_args -fast_forward_to_start_inst 1'
    }
  #endif

  int i = 0;
  while (true)
  {
    if (debug_program_flow_level >= 1) {
      PRINT(std::endl << "====== Starting loop #" << i << " of " << n_time_steps << " ======");
    }
    
    // Read the value of 'x' for this iteration.
    x_reader.read("x", modelX);
    if (!status_reader.is_simulator_running()){
      break;
    }

    // Begin a batch of Scarab statistics
    #if defined(USE_DYNAMORIO)
      if (debug_scarab_level >= 1) {
        PRINT_WITH_FILE_LOCATION("Starting DynamoRIO region of interest.")
      }
      dr_app_setup_and_start();
    #elif defined(USE_EXECUTION_DRIVEN_SCARAB)
        if (debug_scarab_level >= 1) {
          PRINT_WITH_FILE_LOCATION("Starting Scarab region of interest.")
        }
        scarab_roi_dump_begin(); 
    #else
      if (debug_scarab_level >= 1) {
        PRINT_WITH_FILE_LOCATION("Starting region of interest without statistics recording.")
      }
    #endif

    // if (use_state_after_delay_prediction)
    // {
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
    // } else {
      x_predict = modelX;
      t_predict = 0;
    // }
    
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
    
      if (debug_scarab_level >= 1) {
        PRINT_WITH_FILE_LOCATION("End of DynamoRIO region of interest.")
      }
      dr_app_stop_and_cleanup();

      std::filesystem::path folder(sim_dir);
      if(!std::filesystem::is_directory(folder))
      {
          throw std::runtime_error(folder.string() + " is not a folder");
      }
      std::vector<std::string> file_list;
      
      if (debug_scarab_level >= 2) {
        PRINT("DynamoRIO Trace Files in " << folder.string())
      }

      // Regex to match stings like "<path to sim dir>/drmemtrace.acc_controller_5_2_dynamorio.268046.6519.dir"
      std::regex dynamorio_trace_filename_regex("(.*)drmemtrace\\.main_controller\\w*_dynamorio\\.\\d+\\.\\d+\\.dir");
      
      bool is_first_dir_found = true;
      for (const auto& folder_entry : std::filesystem::directory_iterator(folder)) {
          const auto full_path = folder_entry.path().string();
          bool is_dynamorio_trace = std::regex_match(full_path, dynamorio_trace_filename_regex);
          if (is_dynamorio_trace) {
            PRINT("- " << folder_entry.path().string())
            std::string new_dir_name("dynamorio_trace_" +  std::to_string(i));
            std::string new_path = std::regex_replace(full_path, dynamorio_trace_filename_regex, new_dir_name);
            
            if (debug_scarab_level >= 2) {
              PRINT("Renaming")
              PRINT("\t     " << folder_entry.path().string())
              PRINT("\t to: " << new_path)
            }
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
          if (debug_scarab_level >= 1) {
            PRINT_WITH_FILE_LOCATION("End of Scarab region of interest.")
          }
        scarab_roi_dump_end();  
    #else
      if (debug_scarab_level >= 1) {
        PRINT_WITH_FILE_LOCATION("End of region of interest. Statistics recording is disabled.")
      }
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
    t_predict_writer.write(   "t prediction", i, t_predict);
    iterations_writer.write(   "iterations ", i, -1); // TODO: Fix iterations. Currently sending "-1" because we aren't reading the iterations.

    delays_reader.read("t_delay", t_delay_prev);

    PRINT("controller mapped x = " << modelX << " to " << u)
    
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
    i++;
  } 
  
  if (debug_scarab_level >= 1) {
    PRINT_WITH_FILE_LOCATION("Finished looping through " << i << " time steps. Closing files...")
  }

  u_writer.close();
  x_prediction_writer.close();
  t_predict_writer.close();
  iterations_writer.close();
  status_reader.close();
  x_reader.close();
  delays_reader.close();

  // PRINT_WITH_FILE_LOCATION("Closing optimizer_info_out_file...")
  // optimizer_info_out_file.close();
  
  if (debug_scarab_level >= 1) {
    PRINT("Finished closing files. All done!.")
  }

  return 0;
}
