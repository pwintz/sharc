#! /bin/env python3

import shutil
import unittest

from scarabintheloop import BatchInit, Simulation, load_controller_delegator
from scarabintheloop.data_types import ComputationData, TimeStepSeries
from scarabintheloop.scarabizor import ParamsData
from scarabintheloop.utils import list_to_column_vec, printJson, readJson

import scarabintheloop.debug_levels as debug_levels

# Turn off debugging statements.
debug_levels.debug_program_flow_level = 0

example_dir = '../examples/acc_example'
load_controller_delegator(example_dir)

class Test_Simulation_run(unittest.TestCase):
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
                            x0=list_to_column_vec([1, 2, 3, 4, 5]), 
                            u0=list_to_column_vec([-2]), 
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
                            x0=list_to_column_vec([1, 2, 3, 4, 5]), 
                            u0=list_to_column_vec([-2]), 
                            pending_computation=None, 
                            simulation_dir=sim_dir, 
                            experiment_config=experiment_config, 
                            params=params)
    
    # ---- Execution ----
    simulation.setup_files()
    simulation_time_series: TimeStepSeries = simulation.run()
  
    # ---- Assertions ---- 
    self.assertEqual(simulation_time_series.n_time_steps,   4)
    self.assertEqual(simulation_time_series.n_time_indices, 5)
    self.assertEqual(simulation_time_series.first_time_step,  1)
    self.assertEqual(simulation_time_series.first_time_index, 1)
    self.assertEqual(simulation_time_series.last_time_step,  4)
    self.assertEqual(simulation_time_series.last_time_index, 5)
    self.assertIsNone(simulation_time_series.pending_computation_prior, None)
    self.assertIsNotNone(simulation_time_series.pending_computation[-1])
    self.assertEqual(simulation_time_series.cause_of_termination, 'Finished')

  def test_run_fake_batched_with_late_pending_computation(self):
    # ---- Clean Slate ----
    sim_dir = 'experiments/test_run_fake_batched_with_late_computation'
    shutil.rmtree(sim_dir, ignore_errors=True)
    
    # ---- Setup ---- 
    experiment_config = readJson(f'{example_dir}/base_config.json')
    params = ParamsData.from_file(f'{example_dir}/chip_configs/PARAMS.base')
    experiment_config['experiment_dir'] = sim_dir
    experiment_config['system_parameters']['sample_time'] = 0.25
    experiment_config['Simulation Options']['use_fake_delays'] = True
    experiment_config['Simulation Options']['in-the-loop_delay_provider'] = "onestep"
    experiment_config['Simulation Options']['parallel_scarab_simulation'] = True
    pending_computation = ComputationData(t_start=0.5, delay=0.727, u=[1])
    batch_init = BatchInit._from_lists(i_batch=1,
                                            k0=3,
                                            t0=0.75,
                                            x0=[1, 2, 3, 4, 5],
                                            u0=[-1],
                            pending_computation=pending_computation
                          )
    n_time_steps = 1
    simulation = Simulation.from_experiment_config_batched(experiment_config=experiment_config,
                                                           params_base=params,
                                                           batch_init =batch_init,
                                                           n_time_steps_in_batch=n_time_steps)
    assert simulation.n_time_steps == n_time_steps

    simulation.setup_files()
    batch_sim_data = simulation.run()
    # batch_sim_data.printTimingData(f'batch_sim_data from running simulation in "run_batch()" with n_time_steps={n_time_steps}')
    self.assertEqual(batch_sim_data.n_time_steps, n_time_steps, \
      f'batch_sim_data.n_time_steps={batch_sim_data.n_time_steps} must equal n_time_steps={n_time_steps}')
    # 
    # batch = Batch(batch_init, batch_sim_data, sample_time)
    # assert batch.valid_simulation_data.n_time_steps <= n_time_steps, \
    #   f'batch.valid_simulation_data.n_time_steps={batch.valid_simulation_data.n_time_steps} ' \
    #   + f'must be less than n_time_steps={n_time_steps}.'
    # return batchdef test_run_fake_batched_with_late_computation(self):
    # ---- Clean Slate ----
    sim_dir = 'experiments/test_run_fake_batched_with_late_computation'
    shutil.rmtree(sim_dir, ignore_errors=True)
    
    # ---- Setup ---- 
    experiment_config = readJson(f'{example_dir}/base_config.json')
    params = ParamsData.from_file(f'{example_dir}/chip_configs/PARAMS.base')
    experiment_config['experiment_dir'] = sim_dir
    experiment_config['system_parameters']['sample_time'] = 0.25
    experiment_config['Simulation Options']['use_fake_delays'] = True
    experiment_config['Simulation Options']['in-the-loop_delay_provider'] = "onestep"
    experiment_config['Simulation Options']['parallel_scarab_simulation'] = True
    pending_computation = ComputationData(t_start=0.5, delay=0.27, u=[1])
    batch_init = BatchInit._from_lists(
                              i_batch=1,
                              k0=3,
                              t0=0.75,
                              x0=[1, 2, 3, 4, 5],
                              u0=[-1],
                              pending_computation=pending_computation
                            )
    n_time_steps = 1
    simulation = Simulation.from_experiment_config_batched(experiment_config=experiment_config,
                                                                 params_base=params,
                                                                  batch_init=batch_init,
                                                       n_time_steps_in_batch=n_time_steps)
    assert simulation.n_time_steps == n_time_steps
    # printJson("simulation config", simulation.sim_config)

    simulation.setup_files()
    batch_sim_data = simulation.run()
    print('batch_sim_data:', batch_sim_data)
    # batch_sim_data.printTimingData(f'batch_sim_data from running simulation in "run_batch()" with n_time_steps={n_time_steps}')
    self.assertEqual(batch_sim_data.n_time_steps, n_time_steps, \
      f'batch_sim_data.n_time_steps={batch_sim_data.n_time_steps} must equal n_time_steps={n_time_steps}')
    # 
    # batch = Batch(batch_init, batch_sim_data, sample_time)
    # assert batch.valid_simulation_data.n_time_steps <= n_time_steps, \
    #   f'batch.valid_simulation_data.n_time_steps={batch.valid_simulation_data.n_time_steps} ' \
    #   + f'must be less than n_time_steps={n_time_steps}.'
    # return batch


if __name__ == '__main__':
  unittest.main()