#! /bin/env python3

import unittest
import scarabintheloop
from scarabintheloop.plant_runner import TimeStepSeries, ComputationData
import copy
import os

class TestProcessBatchSimulationData(unittest.TestCase):
  def test_no_step_OK(self):
    x0 = [1, 0]
    u0 = [3]
    x1 = [3, 4]
    u1 = [-4]
    t_delay = 0.1
    sample_time = 1.0
    batch_init = {
      "i_batch": 0,
      "first_time_index": 0,
      "x0": x0,
      "u0": u0,
      # There are no pending (previouslly computed) control values because we are at the start of the experiment.
      "pending_computation": None,
    }

    batch_simulation_data = TimeStepSeries(k0=0, t0=0.0, x0=x0)

    # Save the input simulation data so we can check that it is not modified.
    expected_valid_sim_data = copy.deepcopy(batch_simulation_data)

    valid_data_from_batch, next_batch_init = scarabintheloop.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)

    # Check that the valid simulation data contains all of the original data.
    self.assertEqual(valid_data_from_batch, expected_valid_sim_data)

    # Check the bath_init
    expected_next_batch_init = {
      "i_batch":              1,
      "first_time_index":     0,
      "x0":                   x0,
      "u0":                   u0,
      "pending_computation":  None
    }
    self.assertEqual(next_batch_init, expected_next_batch_init)

  def test_one_step_no_misses(self):
    x0 = [1, 0]
    u0 = [3]
    x1 = [3, 4]
    u1 = [-4]
    t_delay = 0.1
    sample_time = 1.0
    batch_init = {
      "i_batch": 0,
      "first_time_index": 0,
      "x0": x0,
      "u0": u0,
      # There are no pending (previouslly computed) control values because we are at the start of the experiment.
      "pending_computation": None,
      "Last batch status": "None - first time step"
    }
    pending_computation = ComputationData(t_start=0.0, delay=t_delay, u=1.23333)
    batch_simulation_data = TimeStepSeries(k0=0, t0=0.0, x0=x0)
    batch_simulation_data.append(t_end=1.0, x_end=x1, u=u1, pending_computation=pending_computation)

    # Save the input simulation data so we can check that it is not modified.
    expected_valid_sim_data = copy.deepcopy(batch_simulation_data)

    valid_data_from_batch, next_batch_init = scarabintheloop.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)

    # Check that the valid simulation data contains all of the original data.
    self.assertEqual(valid_data_from_batch, expected_valid_sim_data)

    # Check the bath_init
    expected_next_batch_init = {
      "i_batch":              1,
      "first_time_index":     1,
      "x0":                   x1,
      "u0":                   u1,
      "pending_computation":  pending_computation,
      "Last batch status": "No missed computations."
    }
    self.assertEqual(next_batch_init, expected_next_batch_init)


  def test_miss_at_last_time_step(self):
    x0 = [1, 0]
    u0 = [3]
    k0 = 10
    x = [x0, [3, 4], [5, -2]]
    u = [u0, [-2], [6]]
    t = [0.0, 1.0, 2.0]
    t_delay = [0.1, 1.1] # Last t_delay longer than sample time!
    sample_time = 1.0
    batch_init = {
      "i_batch": 4,
      "first_time_index": 10,
      "x0": x0,
      "u0": u0,
      # There are no pending (previouslly computed) control values because we are at the start of the experiment.
      "pending_computation": None,
      'Last batch status': 'PREVIOUS STATUS',
    }
    pending_computations = [
      ComputationData(t_start=t[0], delay=sample_time+0.1, u=1.23333),
      ComputationData(t_start=t[1], delay=sample_time-0.1, u=1.23333)
    ]
    batch_simulation_data = TimeStepSeries(k0=k0, t0=0.0, x0=x0)
    batch_simulation_data.append_multiple(t_end=t[1:], x_end=x[1:], u=u[1:], pending_computation=pending_computations)

#     # Save the input simulation data so we can check that it is not modified.
    # expected_valid_sim_data = copy.deepcopy(batch_simulation_data)

    valid_data_from_batch, next_batch_init = scarabintheloop.processBatchSimulationData(batch_init, batch_simulation_data, sample_time)

    expected_valid_sim_data = TimeStepSeries(k0=k0, t0=0.0, x0=x0)
    expected_valid_sim_data.append(t_end=t[1], x_end=x[1], u=u[1], pending_computation=pending_computations[0])

    # Check that the valid simulation data contains all of the original data.
    self.assertEqual(valid_data_from_batch.k0, expected_valid_sim_data.k0)
    self.assertEqual(valid_data_from_batch.t0, expected_valid_sim_data.t0)
    self.assertEqual(valid_data_from_batch.x0, expected_valid_sim_data.x0)
    self.assertEqual(valid_data_from_batch.t, expected_valid_sim_data.t)
    self.assertEqual(valid_data_from_batch.k, expected_valid_sim_data.k)
    self.assertEqual(valid_data_from_batch.i, expected_valid_sim_data.i)
    self.assertEqual(valid_data_from_batch.x, expected_valid_sim_data.x)
    self.assertEqual(valid_data_from_batch.u, expected_valid_sim_data.u)
    self.assertEqual(valid_data_from_batch.pending_computation, \
                     expected_valid_sim_data.pending_computation)
    self.assertEqual(valid_data_from_batch.pending_computation_prior, \
                     expected_valid_sim_data.pending_computation_prior)

    # The assertions above should check equality exhaustively, but we check it explicitly here, 
    # just in case we missed something.
    self.assertEqual(valid_data_from_batch, expected_valid_sim_data)

    # Since there was a missed computation at the first time step, we continue using it. 
    expcted_next_pending_computations = pending_computations[0]

    # Check the bath_init
    expected_next_batch_init = {
      "i_batch":                 5, # Previous batch was i_batch = 4
      "first_time_index":       11, # Previous batch was first_time_index = 10
      "x0":                   x[1],
      "u0":                   u[1],
      "pending_computation":  expcted_next_pending_computations,
      'Last batch status': 'Missed computation at timestep[10]'
    }
    self.assertEqual(next_batch_init, expected_next_batch_init)

    


class Test_create_simulation_config(unittest.TestCase):
  def test_with_no_batch_init(self):
    # Setup 
    cpu_count = os.cpu_count()
    experiment_config = {
      "max_time_steps": 200,
      "label": 'my_label',
      "experiment_dir": '.',
      "x0": [1,2,3],
      "u0": [3, 4], 
      "Simulation Options": {
        "max_batch_size": 12
      }
    }
    batch_init = None

    # Execution
    sim_config = scarabintheloop.create_simulation_config(experiment_config, batch_init)
    
    # Assertions 
    self.assertEqual(sim_config["first_time_index"], 0)
    self.assertEqual(sim_config["max_time_steps"], 200)
    self.assertEqual(sim_config["last_time_index"], 200)
    self.assertEqual(sim_config["pending_computation"], None)
    self.assertEqual(sim_config["x0"], [1,2,3])
    self.assertEqual(sim_config["u0"], [3, 4])
    self.assertEqual(sim_config["simulation_dir"], '.')
    self.assertEqual(sim_config["simulation_label"], 'my_label')

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
      "max_time_steps": 200,
      "x0":             [1,2,3],
      "u0":             [3, 4],  # Ignored. Replaced by u0 from batch.
      "Simulation Options": {
        "max_batch_size": max_batch_size
      }
    }

    batch_first_time_index = 6
    batch_init_pending_computation = ComputationData(t_start=0.4, delay=1.2, u=[2,1234])
    batch_init = {
      "i_batch":              3,
      "first_time_index":     batch_first_time_index,
      "x0":                   [43, 43, 43],
      "u0":                   [-11, 22222],
      "pending_computation":  batch_init_pending_computation
    }

    # ----- Execution ----- 
    sim_config = scarabintheloop.create_simulation_config(experiment_config, batch_init)
    
    # ----- Assertions ----- 

    # The expeirment "max_time_steps" is very large, so the max_time_steps is reduced to 
    # the smaller betweeen the number of CPUs or "max_batch_size". But we assume that the 
    # number of CPUS is larger that max_batch_size, the expected batch size is max_batch_size.
    expected_batch_size = max_batch_size

    self.assertEqual(sim_config["first_time_index"], batch_first_time_index)
    self.assertEqual(sim_config["max_time_steps"], expected_batch_size)
    self.assertEqual(sim_config["last_time_index"], batch_first_time_index + expected_batch_size)
    self.assertEqual(sim_config["pending_computation"], batch_init_pending_computation)
    self.assertEqual(sim_config["x0"], [43, 43, 43])
    self.assertEqual(sim_config["u0"], [-11, 22222])


  @unittest.skipIf(os.cpu_count() < 4, 
    """In order for this test to be valid, we need 
    there to be more CPUs than the max batch size. 
    Otherwise, the CPU count will restric the max size.""")
  def test_with_batch_init_at_end_of_experiment(self):
    # ----- Setup -----  
    # The expeirment "max_time_steps" is one more than the number of time steps so far,
    # so the batch should only take one step.
    max_batch_size = 3 # We assume CPU count > 3
    experiment_max_time_steps = 8
    batch_first_time_index = 7
    expected_batch_size = 1 # 1 time step remaining

    experiment_config = {
      "label": 'my_label',
      "experiment_dir": '.',
      "max_time_steps": experiment_max_time_steps,
      "x0":             [1,2,3],
      "u0":             [3, 4],  # Ignored. Replaced by u0 from batch.
      "Simulation Options": {
        "max_batch_size": max_batch_size
      }
    }

    batch_init_pending_computation = ComputationData(t_start=0.4, delay=1.2, u=[2,1234])
    batch_init = {
      "i_batch":              3,
      "first_time_index":     batch_first_time_index,
      "x0":                   [43, 43, 43],
      "u0":                   [-11, 22222],
      "pending_computation":  batch_init_pending_computation
    }

    # ----- Execution ----- 
    sim_config = scarabintheloop.create_simulation_config(experiment_config, batch_init)
    
    # ----- Assertions ----- 
    self.assertEqual(sim_config["first_time_index"], batch_first_time_index)
    self.assertEqual(sim_config["max_time_steps"], expected_batch_size)
    self.assertEqual(sim_config["last_time_index"], experiment_max_time_steps)

if __name__ == '__main__':
  unittest.main()