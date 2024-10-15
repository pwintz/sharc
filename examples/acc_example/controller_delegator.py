"""
This module provides a function "get_controller_executable" for generating
the controller executable and returning the controller executable, given the 
example directory and expirement configuration values. 
"""

import subprocess
import os

CMAKE_VERBOSITY_FROM_DEBUG_LEVEL = {
  0: "ERROR",
  1: "NOTICE",
  2: "VERBOSE"
}

example_dir = os.path.abspath('.')

def get_controller_executable(build_config:dict) -> str:
  build_dir = example_dir + "/build" 
  os.makedirs(build_dir, exist_ok=True)

  debug_build_level = build_config["==== Debgugging Levels ===="]["debug_build_level"]

  simulation_options = build_config["Simulation Options"]
  use_parallel_simulation = simulation_options["parallel_scarab_simulation"]
  prediction_horizon = build_config["system_parameters"]["mpc_options"]["prediction_horizon"]
  control_horizon = build_config["system_parameters"]["mpc_options"]["control_horizon"]

  use_fake_delays = build_config["Simulation Options"]["use_fake_delays"]

  executable_name = f"acc_controller_{prediction_horizon}_{control_horizon}"

  use_dynamorio = use_parallel_simulation and not use_fake_delays

  if use_dynamorio:
    executable_name += "_dynamorio"

  executable_path = build_dir + "/" + executable_name

  def shell(cmd_list):
    if debug_build_level >= 2:
      print(f"Calling subprocess.check_call({cmd_list})")
    if debug_build_level >= 1:
      print(f">> {' '.join(cmd_list)}")
    subprocess.check_call(cmd_list)

  def cmake(args:list):
    cmake_verbosity = CMAKE_VERBOSITY_FROM_DEBUG_LEVEL[debug_build_level]
    cmake_cmd = ["cmake", "--log-level=" + cmake_verbosity] + args
    shell(cmake_cmd)

  def cmake_build(build_dir:str, args:list=[]):
    cmake_cmd = ["cmake", "--build", build_dir] 
    
    if debug_build_level >= 2:
      cmake_cmd += ["--verbose"]
      
    cmake_cmd += args

    if debug_build_level == 0:
      cmake_cmd += ["--", "--quiet"]

    shell(cmake_cmd)
    if not os.path.exists(executable_path):
      raise IOError(f'The expected executable file "{executable_path}" was not build.')

  cmake_arguments_from_config = [f"-DPREDICTION_HORIZON={prediction_horizon}", f"-DCONTROL_HORIZON={control_horizon}"]

  if use_dynamorio:
    cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=ON"]
  else:
    cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=OFF"]

  if debug_build_level:
    print("== Running CMake to generate build tree ==")

  cmake_generate_tree_args = ["-S", f"{example_dir}", 
                              "-B", f"{build_dir}"] + cmake_arguments_from_config
  cmake(cmake_generate_tree_args)
  
  if debug_build_level:
    print("== Running CMake to build controller executable ==")
  # When we build the project, we don't need to pass the arguments from "build_config" to cmake, since they have already been incorporated into the build process during the prior call of cmake. 
  cmake_build(build_dir)

  return executable_path