"""
Controller delegator for the MPC example.

Builds the CarlaMPCController executable, passing prediction horizon,
control horizon, and system dimensions to CMake.
"""

import os
from sharc.controller_delegator_base import CmakeControllerExecutableProvider
import sharc.debug_levels as debug_levels


class ControllerExecutableProvider(CmakeControllerExecutableProvider):

    def get_controller_executable(self, build_config: dict) -> str:
        simulation_options        = build_config["Simulation Options"]
        use_parallel_simulation   = simulation_options["parallel_scarab_simulation"]
        use_fake_delays           = build_config["fake_delays"]["enable"]

        prediction_horizon        = build_config["system_parameters"]["mpc_options"]["prediction_horizon"]
        control_horizon           = build_config["system_parameters"]["mpc_options"]["control_horizon"]
        state_dimension           = build_config["system_parameters"]["state_dimension"]
        input_dimension           = build_config["system_parameters"]["input_dimension"]
        exogenous_input_dimension = build_config["system_parameters"]["exogenous_input_dimension"]
        output_dimension          = build_config["system_parameters"]["output_dimension"]

        executable_name = "main_controller_MPC_v1"

        # DynamoRIO is needed only for parallel mode with real delays.
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
            cmake_arguments_from_config += ["-DUSE_DYNAMORIO=ON"]
        else:
            cmake_arguments_from_config += ["-DUSE_DYNAMORIO=OFF"]

        if debug_levels.debug_build_level:
            print("== Running CMake to generate build tree ==")

        cmake_generate_tree_args = [
            "-S", f"{self.example_dir}",
            "-B", f"{self.build_dir}",
        ] + cmake_arguments_from_config
        self.cmake(cmake_generate_tree_args)

        if debug_levels.debug_build_level:
            print("==== Running CMake to build controller executable ====")

        self.cmake_build(executable_path)

        return executable_path
