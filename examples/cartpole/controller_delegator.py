"""
This module provides a function "get_controller_executable" for generating
the controller executable and returning the controller executable, given the 
example directory and expirement configuration values. 
"""

import subprocess
import os
from sharc.utils import run_shell_cmd
import sharc.debug_levels as debug_levels
from sharc.controller_delegator_base import CmakeControllerExecutableProvider



class ControllerExecutableProvider(CmakeControllerExecutableProvider):

  def get_controller_executable(self, build_config: dict) -> str:
    import os, shutil, subprocess

    simulation_options          = build_config["Simulation Options"]
    use_parallel_simulation     = simulation_options["parallel_scarab_simulation"]
    use_fake_delays             = build_config["fake_delays"]["enable"]
    prediction_horizon          = build_config["system_parameters"]["mpc_options"]["prediction_horizon"]
    control_horizon             = build_config["system_parameters"]["mpc_options"]["control_horizon"]
    state_dimension             = build_config["system_parameters"]["state_dimension"]
    input_dimension             = build_config["system_parameters"]["input_dimension"]
    exogenous_input_dimension   = build_config["system_parameters"]["exogenous_input_dimension"]
    output_dimension            = build_config["system_parameters"]["output_dimension"]

    executable_name = f"main_controller_{prediction_horizon}_{control_horizon}"
    use_dynamorio = use_parallel_simulation and not use_fake_delays
    if use_dynamorio:
        executable_name += "_dynamorio"

    executable_path = os.path.join(self.build_dir, executable_name)

    # Detect if we have build tools (CMake + make or Ninja)
    has_cmake = shutil.which("cmake") is not None
    has_make = shutil.which("make") or shutil.which("ninja")

    if not (has_cmake and has_make):
        # We're probably inside the Apptainer — skip build
        if os.path.exists(executable_path):
            print(f"✅ Found prebuilt controller binary: {executable_path}")
            print("⚙️  Skipping CMake rebuild (no build tools available).")
            return executable_path
        else:
            raise FileNotFoundError(
                f"❌ No build tools found and no prebuilt binary at {executable_path}.\n"
                f"Please compile it on the host first (outside Apptainer):\n"
                f"  cd examples/CarExample/build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j"
            )

    # Otherwise, run normal build flow (host environment)
    cmake_args = [
        f"-DPREDICTION_HORIZON={prediction_horizon}",
        f"-DCONTROL_HORIZON={control_horizon}",
        f"-DTNX={state_dimension}",
        f"-DTNU={input_dimension}",
        f"-DTNDU={exogenous_input_dimension}",
        f"-DTNY={output_dimension}",
        f"-DUSE_DYNAMORIO={'ON' if use_dynamorio else 'OFF'}",
    ]
    cmake_generate_tree_args = ["-S", self.example_dir, "-B", self.build_dir] + cmake_args

    print(f"🛠️  Building controller via CMake in {self.build_dir}")
    self.cmake(cmake_generate_tree_args)
    self.cmake_build(executable_path)

    return executable_path
