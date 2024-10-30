"""
This package stores global debugging message levels.
"""

# Define default debugging levels.
debug_interfile_communication_level = 0
debug_optimizer_stats_level         = 0
debug_dynamics_level                = 0
debug_configuration_level           = 0
debug_build_level                   = 0
debug_batching_level                = 0
debug_shell_calls_level             = 0

def set_from_dictionary(debug_levels_dict):
  debug_interfile_communication_level = debug_levels_dict["debug_interfile_communication_level"]
  debug_optimizer_stats_level         = debug_levels_dict["debug_optimizer_stats_level"]
  debug_dynamics_level                = debug_levels_dict["debug_dynamics_level"]
  debug_configuration_level           = debug_levels_dict["debug_configuration_level"]
  debug_build_level                   = debug_levels_dict["debug_build_level"]
  debug_shell_calls_level             = debug_levels_dict["debug_shell_calls_level"]
  debug_batching_level                = debug_levels_dict["debug_batching_level"]
  debug_scarab_level                  = debug_levels_dict["debug_scarab_level"]
  
