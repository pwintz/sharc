"""
This package stores global debugging message logging levels.
"""

# Define default debugging levels.
debug_program_flow_level            = 0
debug_interfile_communication_level = 0
debug_optimizer_stats_level         = 0
debug_dynamics_level                = 0
debug_configuration_level           = 0
debug_build_level                   = 0
debug_batching_level                = 0
debug_shell_calls_level             = 0

def set_from_dictionary(debug_levels_dict):
  global debug_program_flow_level
  global debug_interfile_communication_level
  global debug_optimizer_stats_level
  global debug_dynamics_level
  global debug_configuration_level
  global debug_build_level
  global debug_shell_calls_level
  global debug_batching_level
  global debug_scarab_level

  debug_program_flow_level            = debug_levels_dict["debug_program_flow_level"]
  debug_interfile_communication_level = debug_levels_dict["debug_interfile_communication_level"]
  debug_optimizer_stats_level         = debug_levels_dict["debug_optimizer_stats_level"]
  debug_dynamics_level                = debug_levels_dict["debug_dynamics_level"]
  debug_configuration_level           = debug_levels_dict["debug_configuration_level"]
  debug_build_level                   = debug_levels_dict["debug_build_level"]
  debug_shell_calls_level             = debug_levels_dict["debug_shell_calls_level"]
  debug_batching_level                = debug_levels_dict["debug_batching_level"]
  debug_scarab_level                  = debug_levels_dict["debug_scarab_level"]
  
