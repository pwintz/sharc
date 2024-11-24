#! /bin/env python3

import shutil
import unittest

from sharc import BatchInit, Simulation, load_controller_delegator
from sharc.data_types import ComputationData, TimeStepSeries
from sharc.scarabizor import ParamsData
from sharc.utils import list_to_column_vec, printJson, readJson

import sharc.debug_levels as debug_levels

# Turn off debugging statements.
debug_levels.debug_program_flow_level = 0

example_dir = '../examples/acc_example'
load_controller_delegator(example_dir)

@unittest.skip
class TestACCExample(unittest.TestCase):
  def test_run_fake_serial(self):
    # ---- Clean Slate ----
    sim_dir = 'experiments/test_run_fake_serial'
    shutil.rmtree(sim_dir, ignore_errors=True)

    # ---- Setup ---- 
    experiment_config = readJson(f'{example_dir}/base_config.json')
    params = ParamsData.from_file(f'{example_dir}/chip_configs/PARAMS.base')
    experiment_config['Simulation Options']['use_fake_delays'] = True
    simulation = Simulation('test_run_fake_serial', 
                            first_time_index=1, 
                            n_time_steps=4, 
                            x0=list_to_column_vec([1, 2, 3, 4]), 
                            u0=list_to_column_vec([-2.1, -2.2]), 
                            pending_computation=None, 
                            simulation_dir=sim_dir, 
                            experiment_config=experiment_config, 
                            params=params)
    # ---- Execution ----
    simulation.setup_files()
    simulation_time_series = simulation.run()
  
    # ---- Assertions ---- 
    self.assertEqual(simulation_time_series.n_time_steps, 4)

  def test_run_fake_parallel(self):
    # ---- Clean Slate ----
    sim_dir = 'experiments/test_run_fake_parallel'
    shutil.rmtree(sim_dir, ignore_errors=True)
    
    # ---- Setup ---- 
    experiment_config = readJson(f'{example_dir}/base_config.json')
    params = ParamsData.from_file(f'{example_dir}/chip_configs/PARAMS.base')
    experiment_config['Simulation Options']['use_fake_delays'] = True
    experiment_config['Simulation Options']['in-the-loop_delay_provider'] = "onestep"
    experiment_config['Simulation Options']['parallel_scarab_simulation'] = True

    simulation = Simulation('test_run_fake_parallel', 
                            first_time_index=1, 
                            n_time_steps=4, 
                            x0=list_to_column_vec([1, 2, 3, 4]), 
                            u0=list_to_column_vec([-2, -3]), 
                            pending_computation=None, 
                            simulation_dir=sim_dir, 
                            experiment_config=experiment_config, 
                            params=params)
    
    # ---- Execution ----
    simulation.setup_files()
    simulation_time_series: TimeStepSeries = simulation.run()
  
    # ---- Assertions ---- 
    self.assertEqual(simulation_time_series.n_time_steps,   4)
    self.assertEqual(simulation_time_series.cause_of_termination, 'Finished')


if __name__ == '__main__':
  unittest.main()