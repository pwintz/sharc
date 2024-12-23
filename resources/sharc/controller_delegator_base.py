import os
from abc import ABC, abstractmethod #  AbstractBaseClass
import sharc.debug_levels as debug_levels
from sharc.utils import run_shell_cmd, assertFileExists

CMAKE_VERBOSITY_FROM_DEBUG_LEVEL = {
  0: "ERROR",
  1: "NOTICE",
  2: "VERBOSE"
}

class BaseControllerExecutableProvider(ABC):
  def __init__(self, example_dir):
    self.example_dir = os.path.abspath(example_dir)
    self.build_dir = os.path.join(self.example_dir, "build")
    os.makedirs(self.build_dir, exist_ok=True)

  @abstractmethod
  def get_controller_executable(self, build_config:dict) -> str:
    pass

class CmakeControllerExecutableProvider(BaseControllerExecutableProvider):

  def get_controller_executable(self, build_config:dict) -> str:

    simulation_options      = build_config["Simulation Options"]
    use_parallel_simulation = simulation_options["parallel_scarab_simulation"]
    use_fake_delays         = build_config["fake_delays"]["enable"]
    prediction_horizon      = build_config["system_parameters"]["mpc_options"]["prediction_horizon"]
    control_horizon         = build_config["system_parameters"]["mpc_options"]["control_horizon"]
    state_dimension         = build_config["system_parameters"]["state_dimension"]
    input_dimension         = build_config["system_parameters"]["input_dimension"]
    exogenous_input_dimension = build_config["system_parameters"]["exogenous_input_dimension"]
    output_dimension        = build_config["system_parameters"]["output_dimension"]

    # Construct the base executable name, given the system and simulation options.
    executable_name = f"main_controller_{prediction_horizon}_{control_horizon}"

    # We use DynamoRio only if we are using the parallelized scheme with real delays.
    use_dynamorio = use_parallel_simulation and not use_fake_delays
    if use_dynamorio:
      executable_name += "_dynamorio"

    executable_path = os.path.join(self.build_dir, executable_name)

    cmake_arguments_from_config = [
                f"-DPREDICTION_HORIZON={prediction_horizon}", 
                f"-DCONTROL_HORIZON={control_horizon}",
                f"-DTNX={state_dimension}",
                f"-DTNU={input_dimension}",
                f"-DTNDU={exogenous_input_dimension}",
                f"-DTNY={output_dimension}",
              ]

    if use_dynamorio:
      cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=ON"]
    else:
      cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=OFF"]

    if debug_levels.debug_build_level:
      print("== Running CMake to generate build tree ==")

    cmake_generate_tree_args = ["-S", f"{self.example_dir}", 
                                "-B", f"{self.build_dir}"] + cmake_arguments_from_config
    self.cmake(cmake_generate_tree_args)
    
    if debug_levels.debug_build_level:
      print("==== Running CMake to build controller executable ====")
    # When we build the project, we don't need to pass the arguments from "build_config" to cmake, since they have already been incorporated into the build process during the prior call of cmake. 
    self.cmake_build(executable_path)

    return executable_path

  def cmake(self, args:list):
    cmake_verbosity = CMAKE_VERBOSITY_FROM_DEBUG_LEVEL[debug_levels.debug_build_level]
    cmake_cmd = ["cmake", "--log-level=" + cmake_verbosity] + args
    run_shell_cmd(cmake_cmd, working_dir=self.build_dir)

  def cmake_build(self, executable_path, args:list=[]):
    cmake_cmd = ["cmake", "--build", self.build_dir] 
    
    if debug_levels.debug_build_level >= 2:
      cmake_cmd += ["--verbose"]
      
    cmake_cmd += args

    if debug_levels.debug_build_level == 0:
      cmake_cmd += ["--", "--quiet"]

    jobs = os.cpu_count()  # Automatically set to the number of available CPU cores
    cmake_cmd += ["-j", str(jobs)]

    run_shell_cmd(cmake_cmd)
    assertFileExists(executable_path, f'The expected executable file "{executable_path}" was not built.')
