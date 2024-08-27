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

def get_controller_executable(example_dir:str, build_config:dict) -> str:
  build_dir = example_dir + "/build" 
  os.makedirs(build_dir, exist_ok=True)
  # os.chdir(build_dir)

  debug_build_level = build_config["==== Debgugging Levels ===="]["debug_build_level"]

  # if debug_build_level == 0:
  #   cmake_verbosity = "ERROR"
  # elif debug_build_level == 1:
  #   cmake_verbosity = "NOTICE"
  # elif debug_build_level == 2:
  #   cmake_verbosity = "VERBOSE"
  # else:
  #   raise ValueError(f"Unexpected value: debug_build_level={debug_build_level}. Only 0, 1, and 2 are supported.")

  use_dynamorio = build_config["parallel_scarab_simulation"]
  prediction_horizon = build_config["prediction_horizon"]
  control_horizon = build_config["control_horizon"]
  options_to_pass_to_cmake = ["prediction_horizon", "control_horizon"]

  executable_name = f"acc_controller_{prediction_horizon}_{control_horizon}"
  # TODO: Change executable name if using DyanmoRio
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

  cmake_arguments_from_config = [f"-D{key.upper()}={build_config[key]}" for key in options_to_pass_to_cmake]

  if use_dynamorio:
    cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=ON"]
  else:
    cmake_arguments_from_config += [f"-DUSE_DYNAMORIO=OFF"]

  # prediction_horizon = build_config["prediction_horizon"]
  # control_horizon = build_config["control_horizon"]
  # print(cmake_arguments_from_config)

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