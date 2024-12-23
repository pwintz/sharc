#! /bin/env python3

import unittest
import sharc
from sharc import Simulation, BatchInit, Batch
import sharc.scarabizor as scarabizor
from sharc.data_types import *
import numpy as np
import copy
import os

class TestSimulation(unittest.TestCase):
  
  def test_initialization(self):
    # ---- Setup ---- 
    label = "My Label"
    first_time_index = 0
    last_time_index  = 2
    x0 = np.array([[1], [2], [3]])
    u0 = np.array([[-12]])
    pending_computation = None
    simulation_dir = '/my/path/'
    sim_config = {}
    params = scarabizor.ParamsData.from_file('PARAMS.generated')
    
    # ---- Execution ----
    simulation = Simulation(label, first_time_index, last_time_index, x0, u0, pending_computation, simulation_dir, sim_config, params) 
    
    # ---- Assertions ----
    self.assertEqual(simulation.label, label)
    self.assertEqual(simulation.x0.tolist(), x0.tolist())
    self.assertEqual(simulation.u0.tolist(), u0.tolist())
    self.assertEqual(simulation.simulation_dir, simulation_dir)
    self.assertEqual(simulation.first_time_index, first_time_index)
    self.assertEqual(simulation.last_time_index, last_time_index)
    self.assertEqual(simulation.n_time_steps,   2)
    self.assertEqual(simulation.n_time_indices, 3)
    self.assertEqual(simulation.time_steps,     [0, 1])
    self.assertEqual(simulation.time_indices, [0,  1,  2])
    self.assertEqual(simulation.pending_computation, pending_computation)
    self.assertEqual(simulation.params, params)

  def test_from_experiment_config_unbatched(self):
    # ---- Setup ---- 
    experiment_config = {
      "label": "My Label",
      "experiment_dir": '/my/path/',
      "x0": [1, 2, 3],
      "u0": [-12], 
      "PARAMS_patch_values": {
        "l1_size": 12334
      }
    }
    params_base = scarabizor.ParamsData.from_file('PARAMS.generated')
    
    # ---- Execution ----
    simulation = Simulation.from_experiment_config_unbatched(experiment_config, params_base, n_time_steps=3) 
    
    # ---- Assertions ----
    self.assertEqual(simulation.label, "My Label")
    self.assertEqual(simulation.x0.tolist(), [[1], [2], [3]])
    self.assertEqual(simulation.u0.tolist(), [[-12]])
    self.assertEqual(simulation.simulation_dir, '/my/path/')
    self.assertEqual(simulation.first_time_index, 0)
    self.assertEqual(simulation.last_time_index,  3)
    self.assertEqual(simulation.n_time_steps,     3)
    self.assertEqual(simulation.n_time_indices,   4)
    self.assertEqual(simulation.time_steps,       [0, 1, 2])# Three time steps
    self.assertEqual(simulation.time_indices,     [0, 1, 2, 3])# Four time indices
    self.assertEqual(simulation.pending_computation, None)

    # Check to make sure we did not modify the input PARAMS values.
    self.assertNotEqual(params_base.to_dict()["l1_size"], 12334)

    # Check that we *did* modify the params options for the simulation, 
    # based on the values given in PARAMS_patch_values in the experiment config.
    self.assertEqual(simulation.params.to_dict()["l1_size"], 12334)

    
  def test_from_experiment_config_batched(self):
    # ---- Setup ---- 
    pending_computation = None
    experiment_config = {
      "label": "My Label",
      "experiment_dir": '/my/path/',
      # 'n_time_steps': 3000, # Very large 
      "x0": [1, 2, 3],
      "u0": [-12], 
      "PARAMS_patch_values": {
        "chip_cycle_time": 555555
      },
      # "Simulation Options": {
      #   "max_batch_size": 1 # Force exactly one step.
      # }
    }
    params_base = scarabizor.ParamsData.from_file('PARAMS.generated')
    pending_computation = ComputationData(t_start=1.2, delay=0.22, u=[345])
    batch_init = BatchInit._from_lists(
      i_batch = 4,
      k0 = 15,
      pending_computation = pending_computation,
      x0 = [10, 20, 30],
      u0 = [-120],
      t0 = 0.4
    )

    # ---- Execution ----
    simulation = Simulation.from_experiment_config_batched(experiment_config, params_base, batch_init, n_time_steps_in_batch=1) 
    
    # ---- Assertions ----
    self.assertEqual(simulation.n_time_steps,        1)
    self.assertEqual(simulation.n_time_indices,      2)
    self.assertEqual(simulation.first_time_index,    15)
    self.assertEqual(simulation.last_time_index,     16)
    self.assertEqual(simulation.label,               "My Label/batch4_steps15-15")
    self.assertEqual(simulation.x0.tolist(),         [[10], [20], [30]])
    self.assertEqual(simulation.u0.tolist(),         [[-120]])
    self.assertEqual(simulation.simulation_dir,      '/my/path/batch4_steps15-15')
    self.assertEqual(simulation.time_steps,          [15])# One time steps
    self.assertEqual(simulation.time_indices,        [15, 16])# Two time indices
    self.assertEqual(simulation.pending_computation, pending_computation)

    # Check to make sure we did not modify the input PARAMS values.
    self.assertNotEqual(params_base.to_dict()["chip_cycle_time"], 555555)

    # Check that we *did* modify the params options for the simulation, 
    # based on the values given in PARAMS_patch_values in the experiment config.
    self.assertEqual(simulation.params.to_dict()["chip_cycle_time"], 555555)

#   @unittest.skipIf(os.cpu_count() < 4, 
#         """In order for this test to be valid, we need 
#         there to be more CPUs than the max batch size. 
#         Otherwise, the CPU count will restrict the max size.""")
#   def test_from_experiment_config_batched_with_steps_limited_by_max_batch_size(self):
# 
#     # ---- Setup ---- 
#     pending_computation = None
#     experiment_config = {
#       'n_time_steps': 3000, # Very large 
#       "Simulation Options": {
#         "max_batch_size": 2# This should be smaller than the number of CPUs
#       },
#       # The following values are unimportant for this test. 
#       "label": "My Label", "experiment_dir": '/my/path/',
#       "x0": [1, 2, 3], "u0": [-12], 
#       "PARAMS_patch_values": {}
#     }
#     params_base = scarabizor.ParamsData.from_file('PARAMS.generated')
#     pending_computation = ComputationData(t_start=1.2, delay=0.22, u=[345])
#     batch_init = BatchInit._from_lists(
#       i_batch = 4,
#       k0 = 15,
#       pending_computation = pending_computation,
#       x0 = [10, 20, 30],
#       u0 = [-120],
#       t0 = 0.4
#     )
# 
#     # ---- Execution ----
#     simulation = Simulation.from_experiment_config_batched(experiment_config, params_base, batch_init) 
#     
#     # ---- Assertions ----
#     # Check the indexing and time steps.
#     self.assertEqual(simulation.first_time_index, 15)
#     self.assertEqual(simulation.last_time_index, 17)
#     self.assertEqual(simulation.n_time_steps,   2)
#     self.assertEqual(simulation.n_time_indices, 3)
#     self.assertEqual(simulation.time_steps,    [15, 16])# Three time steps
#     self.assertEqual(simulation.time_indices,[15, 16, 17])# Four time indices

#   @unittest.skipIf(os.cpu_count() > 64, 
#         """In order for this test to be valid, we need 
#         there to be less CPUs than the max batch size and the experiment's n_time_steps. 
#         so that the CPU will restrict the max size.""")
#   def test_from_experiment_config_batched_with_steps_limited_by_CPU_count(self):
#     # ---- Setup ---- 
#     pending_computation = None
#     experiment_config = {
#       'n_time_steps': 3000, # Very large 
#       "Simulation Options": {
#         "max_batch_size": 100# This should be larger than the number of CPUs
#       },
#       # The following values are unimportant for this test. 
#       "label": "My Label", "experiment_dir": '/my/path/',
#       "x0": [1, 2, 3], "u0": [-12], 
#       "PARAMS_patch_values": {}
#     }
#     params_base = scarabizor.ParamsData.from_file('PARAMS.generated')
#     pending_computation = ComputationData(t_start=1.2, delay=0.22, u=[345])
#     batch_init = BatchInit._from_lists(
#       i_batch = 4,
#       k0 = 15,
#       pending_computation = pending_computation,
#       x0 = [10, 20, 30],
#       u0 = [-120],
#       t0 = 0.4
#     )
# 
#     # ---- Execution ----
#     simulation = Simulation.from_experiment_config_batched(experiment_config, params_base, batch_init) 
#     
#     # ---- Assertions ----
#     # Check the indexing and time steps.
#     self.assertEqual(simulation.n_time_steps,   os.cpu_count())
#     self.assertEqual(simulation.n_time_indices, os.cpu_count() + 1)

  @unittest.skipIf(os.cpu_count() < 4, 
      """In order for this test to be valid, we need 
      there to be less CPUs than the max batch size and the experiment's n_time_steps. 
      so that the CPU will restrict the max size.""")
  def test_from_experiment_config_batched_with_steps_limited_by_steps_remaining_in_experiment(self):
    # ---- Setup ---- 
    pending_computation = None
    experiment_config = {
      # 'n_time_steps': 100,
      # "Simulation Options": {
      #   "max_batch_size": 10000 # This should be larger than the number of CPUs and remaining steps.
      # },
      # The following values are unimportant for this test. 
      "label": "My Label", "experiment_dir": '/my/path/',
      "x0": [1, 2, 3], "u0": [-12], 
      "PARAMS_patch_values": {}
    }
    params_base = scarabizor.ParamsData.from_file('PARAMS.generated')
    pending_computation = ComputationData(t_start=1.2, delay=0.22, u=[345])
    batch_init = BatchInit._from_lists(
      i_batch = 4,
      k0 = 98,
      pending_computation = pending_computation,
      x0 = [10, 20, 30],
      u0 = [-120],
      t0 = 0.4
    )

    # ---- Execution ----
    simulation = Simulation.from_experiment_config_batched(experiment_config, params_base, batch_init, n_time_steps_in_batch=2) 
    
    # ---- Assertions ----
    # Check the indexing and time steps.
    self.assertEqual(simulation.n_time_steps,   2)
    self.assertEqual(simulation.n_time_indices, 3)
    
    self.assertEqual(simulation.last_time_index, 100)
    self.assertEqual(simulation.time_steps,    [98, 99])    # Two time steps
    self.assertEqual(simulation.time_indices,[98, 99 , 100])# Three time indices

class Test_BatchInit(unittest.TestCase):
  def test_first(self):
    # ---- Setup ---- 
    x0 = [1, 2, 3]
    u0 = [-1, 2]
    
    # ---- Execution ----
    batch_init = BatchInit.first(x0, u0)
  
    # ---- Assertions ---- 
    self.assertEqual(batch_init.i_batch, 0)
    self.assertEqual(batch_init.k0, 0)
    self.assertEqual(batch_init.t0, 0.0)
    # self.assertEqual(batch_init.x0, )
    # self.assertEqual(batch_init.u0, 0)
    self.assertEqual(batch_init.pending_computation, None)


# class TestProcessBatchSimulationData(unittest.TestCase):
#   def test_no_step_OK(self):
#     x0 = [1, 0]
#     u0 = [3]
#     x1 = [3, 4]
#     u1 = [-4]
#     t_delay = 0.1
#     sample_time = 1.0
#     batch_init = {
#       "i_batch": 0,
#       "first_time_index": 0,
#       "x0": x0,
#       "u0": u0,
#       # There are no pending (previouslly computed) control values because we are at the start of the experiment.
#       "pending_computation": None,
#     }
# 
#     batch_simulation_data = TimeStepSeries(k0=0, t0=0.0, x0=x0)
# 
#     # Save the input simulation data so we can check that it is not modified.
#     expected_valid_sim_data = copy.deepcopy(batch_simulation_data)
# 
#     valid_data_from_batch, next_batch_init = sharc.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)
# 
#     # Check that the valid simulation data contains all of the original data.
#     self.assertEqual(valid_data_from_batch, expected_valid_sim_data)
# 
#     # Check the bath_init
#     expected_next_batch_init = {
#       "i_batch":              1,
#       "first_time_index":     0,
#       "x0":                   x0,
#       "u0":                   u0,
#       "pending_computation":  None
#     }
#     self.assertEqual(next_batch_init, expected_next_batch_init)
# 
#   def test_one_step_no_misses(self):
#     x0 = [1, 0]
#     u0 = [3]
#     x1 = [3, 4]
#     u1 = [-4]
#     t_delay = 0.1
#     sample_time = 1.0
#     batch_init = {
#       "i_batch": 0,
#       "first_time_index": 0,
#       "x0": x0,
#       "u0": u0,
#       # There are no pending (previouslly computed) control values because we are at the start of the experiment.
#       "pending_computation": None,
#       "Last batch status": "None - first time step"
#     }
#     pending_computation = ComputationData(t_start=0.0, delay=t_delay, u=1.23333)
#     batch_simulation_data = TimeStepSeries(k0=0, t0=0.0, x0=x0)
#     batch_simulation_data.append(t_end=1.0, x_end=x1, u=u1, pending_computation=pending_computation)
# 
#     # Save the input simulation data so we can check that it is not modified.
#     expected_valid_sim_data = copy.deepcopy(batch_simulation_data)
# 
#     valid_data_from_batch, next_batch_init = sharc.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)
# 
#     # Check that the valid simulation data contains all of the original data.
#     self.assertEqual(valid_data_from_batch, expected_valid_sim_data)
# 
#     # Check the bath_init
#     expected_next_batch_init = {
#       "i_batch":              1,
#       "first_time_index":     1,
#       "x0":                   x1,
#       "u0":                   u1,
#       "pending_computation":  pending_computation,
#       "Last batch status": "No missed computations."
#     }
#     self.assertEqual(next_batch_init, expected_next_batch_init)
# 
# 
#   def test_miss_at_last_time_step(self):
#     x0 = [1, 0]
#     u0 = [3]
#     k0 = 10
#     x = [x0, [3, 4], [5, -2]]
#     u = [u0, [-2], [6]]
#     t = [0.0, 1.0, 2.0]
#     t_delay = [0.1, 1.1] # Last t_delay longer than sample time!
#     sample_time = 1.0
#     batch_init = {
#       "i_batch": 4,
#       "first_time_index": 10,
#       "x0": x0,
#       "u0": u0,
#       # There are no pending (previouslly computed) control values because we are at the start of the experiment.
#       "pending_computation": None,
#       'Last batch status': 'PREVIOUS STATUS',
#     }
#     pending_computations = [
#       ComputationData(t_start=t[0], delay=sample_time+0.1, u=1.23333),
#       ComputationData(t_start=t[1], delay=sample_time-0.1, u=1.23333)
#     ]
#     batch_simulation_data = TimeStepSeries(k0=k0, t0=0.0, x0=x0)
#     batch_simulation_data.append_multiple(t_end=t[1:], x_end=x[1:], u=u[1:], pending_computation=pending_computations)
# 
# #     # Save the input simulation data so we can check that it is not modified.
#     # expected_valid_sim_data = copy.deepcopy(batch_simulation_data)
# 
#     valid_data_from_batch, next_batch_init = sharc.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)
# 
#     expected_valid_sim_data = TimeStepSeries(k0=k0, t0=0.0, x0=x0)
#     expected_valid_sim_data.append(t_end=t[1], x_end=x[1], u=u[1], pending_computation=pending_computations[0])
# 
#     # Check that the valid simulation data contains all of the original data.
#     self.assertEqual(valid_data_from_batch.k0, expected_valid_sim_data.k0)
#     self.assertEqual(valid_data_from_batch.t0, expected_valid_sim_data.t0)
#     self.assertEqual(valid_data_from_batch.x0, expected_valid_sim_data.x0)
#     self.assertEqual(valid_data_from_batch.t, expected_valid_sim_data.t)
#     self.assertEqual(valid_data_from_batch.k, expected_valid_sim_data.k)
#     self.assertEqual(valid_data_from_batch.i, expected_valid_sim_data.i)
#     self.assertEqual(valid_data_from_batch.x, expected_valid_sim_data.x)
#     self.assertEqual(valid_data_from_batch.u, expected_valid_sim_data.u)
#     self.assertEqual(valid_data_from_batch.pending_computation, \
#                      expected_valid_sim_data.pending_computation)
#     self.assertEqual(valid_data_from_batch.pending_computation_prior, \
#                      expected_valid_sim_data.pending_computation_prior)
# 
#     # The assertions above should check equality exhaustively, but we check it explicitly here, 
#     # just in case we missed something.
#     self.assertEqual(valid_data_from_batch, expected_valid_sim_data)
# 
#     # Since there was a missed computation at the first time step, we continue using it. 
#     expcted_next_pending_computations = pending_computations[0]
# 
#     # Check the bath_init
#     expected_next_batch_init = {
#       "i_batch":                 5, # Previous batch was i_batch = 4
#       "first_time_index":       11, # Previous batch was first_time_index = 10
#       "x0":                   x[1],
#       "u0":                   u[1],
#       "pending_computation":  expcted_next_pending_computations,
#       'Last batch status': 'Missed computation at timestep[10]'
#     }
#     self.assertEqual(next_batch_init, expected_next_batch_init)

    
class Test_Simulation(unittest.TestCase):
  def test_with_no_batch_init(self):
    # Setup 
    cpu_count = os.cpu_count()
    experiment_config = {
      "n_time_steps": 200,
      "label": 'my_label',
      "experiment_dir": 'my_dir',
      "x0": [1,2,3],
      "u0": [3, 4], 
      "Simulation Options": {
        "max_batch_size": 12
      },
      "PARAMS_patch_values": {
        'chip_cycle_time': 999
      }
    }
    batch_init = None
    params_base = scarabizor.ParamsData(['--chip_cycle_time   100000000'])

    # Execution
    simulation = sharc.Simulation.from_experiment_config_unbatched(experiment_config, params_base, n_time_steps=200)
    
    # Assertions 
    self.assertEqual(simulation.sim_config["first_time_index"], 0)
    self.assertEqual(simulation.sim_config["n_time_steps"], 200)
    self.assertEqual(simulation.sim_config["last_time_index"], 200)
    self.assertEqual(simulation.sim_config["pending_computation"], None)
    self.assertEqual(simulation.sim_config["x0"].flatten().tolist(), [1,2,3])
    self.assertEqual(simulation.sim_config["u0"].flatten().tolist(), [3, 4])
    self.assertEqual(simulation.simulation_dir, 'my_dir')
    # self.assertEqual(simulation.sim_config["simulation_dir"], 'my_dir')
    self.assertEqual(simulation.sim_config["simulation_label"], 'my_label')
    self.assertEqual(simulation.params.to_dict()["chip_cycle_time"], 999)

  @unittest.skipIf(os.cpu_count() < 4, 
        """In order for this test to be valid, we need 
        there to be more CPUs than the max batch size. 
        Otherwise, the CPU count will restric the max size.""")
  def test_with_batch_init(self):
    # ----- Setup -----  
    max_batch_size = 3 # We assume CPU count > 3
    experiment_config = {
      "label": 'my_label',
      "experiment_dir": '.',
      "n_time_steps": 200,
      "x0":             [1,2,3],
      "u0":             [3, 4],  # Ignored. Replaced by u0 from batch.
      # "Simulation Options": {
      #   "max_batch_size": max_batch_size
      # },
      "PARAMS_patch_values": {}
    }

    batch_first_time_index = 6
    batch_init_pending_computation = ComputationData(t_start=0.4, delay=1.2, u=[2,1234])
    batch_init = BatchInit(
      i_batch             = 3,
      k0                  = batch_first_time_index,
      t0                  = 1.2,
      x0                  = np.array([43, 43, 43]),
      u0                  = np.array([-11, 22222]),
      pending_computation = batch_init_pending_computation
    )
    params_base = scarabizor.ParamsData([])
    
    # ----- Execution ----- 
    simulation = sharc.Simulation.from_experiment_config_batched(
                                              experiment_config, params_base, batch_init, n_time_steps_in_batch=max_batch_size)
    
    # ----- Assertions ----- 

    # The expeirment "n_time_steps" is very large, so the n_time_steps is reduced to 
    # the smaller betweeen the number of CPUs or "max_batch_size". But we assume that the 
    # number of CPUS is larger that max_batch_size, the expected batch size is max_batch_size.
    expected_batch_size = max_batch_size

    self.assertEqual(simulation.first_time_index, batch_first_time_index)
    self.assertEqual(simulation.n_time_steps, expected_batch_size)
    self.assertEqual(simulation.last_time_index, batch_first_time_index + expected_batch_size)
    self.assertEqual(simulation.pending_computation, batch_init_pending_computation)
    np.testing.assert_equal(simulation.x0, list_to_column_vec([43, 43, 43]))
    np.testing.assert_equal(simulation.u0, list_to_column_vec([-11, 22222]))


  @unittest.skipIf(os.cpu_count() < 4, 
    """In order for this test to be valid, we need 
    there to be more CPUs than the max batch size. 
    Otherwise, the CPU count will restric the max size.""")
  def test_with_batch_init_at_end_of_experiment(self):
    # ----- Setup -----  
    # The expeirment "n_time_steps" is one more than the number of time steps so far,
    # so the batch should only take one step.
    max_batch_size = 3 # We assume CPU count > 3
    experiment_max_time_steps = 8
    batch_first_time_index = 7
    expected_batch_size = 1 # 1 time step remaining

    experiment_config = {
      "label": 'my_label',
      "experiment_dir": '.',
      "n_time_steps": experiment_max_time_steps,
      "x0":             [1,2,3],
      "u0":             [3, 4],  # Ignored. Replaced by u0 from batch.
      "Simulation Options": {
        "max_batch_size": max_batch_size
      },
      "PARAMS_patch_values": {}
    }

    batch_init_pending_computation = ComputationData(t_start=0.4, delay=1.2, u=[2,1234])
    batch_init = BatchInit(
      i_batch=3,
      k0=batch_first_time_index,
      t0=0.0,
      x0=np.array([43, 43, 43]),
      u0=np.array([-11, 22222]),
      pending_computation=batch_init_pending_computation
    )
    params = scarabizor.ParamsData([])

    # ----- Execution ----- 
    simulation = sharc.Simulation.from_experiment_config_batched(
                                              experiment_config, 
                                              params, 
                                              batch_init, 
                                              n_time_steps_in_batch = expected_batch_size
                                            )
    
    # ----- Assertions ----- 
    self.assertEqual(simulation.first_time_index, batch_first_time_index)
    self.assertEqual(simulation.n_time_steps,   expected_batch_size)
    self.assertEqual(simulation.last_time_index,  experiment_max_time_steps)


if __name__ == '__main__':
  unittest.main()