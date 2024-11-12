#pragma once
#include <iostream>
#include "nlohmann/json.hpp"

struct DebugLevels {
  int debug_program_flow_level            = 0;
  int debug_interfile_communication_level = 0;
  int debug_optimizer_stats_level         = 0;
  int debug_dynamics_level                = 0;
  int debug_scarab_level                  = 0;

  void from_json(const nlohmann::json& debug_config) {
    debug_program_flow_level            = debug_config.at("debug_program_flow_level");
    debug_interfile_communication_level = debug_config.at("debug_interfile_communication_level");
    debug_optimizer_stats_level         = debug_config.at("debug_optimizer_stats_level");
    debug_dynamics_level                = debug_config.at("debug_dynamics_level");
    debug_scarab_level                  = debug_config.at("debug_scarab_level");

    if (debug_program_flow_level >= 2){
      std::cout << "debug_program_flow_level: "            << debug_program_flow_level << std::endl;
      std::cout << "debug_interfile_communication_level: " << debug_interfile_communication_level << std::endl;
      std::cout << "debug_optimizer_stats_level: "         << debug_optimizer_stats_level << std::endl;
      std::cout << "debug_dynamics_level: "                << debug_dynamics_level << std::endl;
      std::cout << "debug_scarab_level: "                  << debug_scarab_level << std::endl;
    }
  }
};

// Create a debug_levels global object.
extern DebugLevels global_debug_levels;