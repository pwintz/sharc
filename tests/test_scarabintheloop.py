#! /bin/env python3

import unittest
import scarabintheloop
from scarabintheloop import computeTimeStepsDelayed, findFirstExcessiveComputationDelay
import copy

class TestComputeTimeStepsDelayed(unittest.TestCase):

  def test_no_step_OK(self):
    t_delay = [None]
    sample_time = 1.0
    time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
    self.assertEqual(time_steps_delayed, [0])

  def test_one_step_no_misses(self):
    t_delay = [None, 0.1]
    sample_time = 1.0
    time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
    self.assertEqual(time_steps_delayed, [0, 1])

  def test_twosteps_no_misses(self):
    t_delay = [None, 0.1, 0.1]
    sample_time = 1.0
    time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
    self.assertEqual(time_steps_delayed, [0, 1, 1])
    
    # Check findFirstExcessiveComputationDelay while we're at it.
    first_excessive_delay_ndx, first_excessive_delay = findFirstExcessiveComputationDelay(time_steps_delayed)
    self.assertEqual(first_excessive_delay_ndx, None)
    self.assertEqual(first_excessive_delay, None)

  def test_miss_by_one(self):
    t_delay = [None, 0.2, 0.1, 1.1, 0.1]
    sample_time = 1.0
    time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
    self.assertEqual(time_steps_delayed, [0, 1, 1, 2, 1])

    # Check findFirstExcessiveComputationDelay while we're at it.
    first_excessive_delay_ndx, first_excessive_delay = findFirstExcessiveComputationDelay(time_steps_delayed)
    self.assertEqual(first_excessive_delay_ndx, 3)
    self.assertEqual(first_excessive_delay, 2)

  def test_miss_by_many(self):
    t_delay = [None, 0.1, 9.1]
    sample_time = 1.0
    
    time_steps_delayed = computeTimeStepsDelayed(sample_time, t_delay)
    self.assertEqual(time_steps_delayed, [0, 1, 10])

    # Check findFirstExcessiveComputationDelay while we're at it.
    first_excessive_delay_ndx, first_excessive_delay = findFirstExcessiveComputationDelay(time_steps_delayed)
    self.assertEqual(first_excessive_delay_ndx, 2)
    self.assertEqual(first_excessive_delay, 10)

# class TestCheckMissedComputations(unittest.TestCase):
#   def test

class TestFindFirstExcessiveComputationDelay(unittest.TestCase):
  pass

class TestProcessBatchSimulationData(unittest.TestCase):
  def test_no_step_OK(self):
    x = [[1, 0]]
    u = [1.4]
    t = [0.0]
    t_delay = [None]
    sample_time = 1.0
    n_time_steps = len(t) - 1
    batch_simulation_data = {
      "x": x,
      "u": u,
      "t": t,
      "t_delay": t_delay, 
      "sample_time": sample_time,
      "n_time_steps": n_time_steps, 
      "first_time_step": 0, 
      "last_time_step": 0
    }

    # Save the input simulation data so we can check that it is not modified.
    expected_batch_simulation_data = copy.deepcopy(batch_simulation_data)

    result = scarabintheloop.processBatchSimulationData(batch_simulation_data)

    # Check that batch_simulation_data is not modified.
    self.assertEqual(batch_simulation_data, expected_batch_simulation_data)

    ##### Construct expected data #####
    expected_all_data_from_batch = {
      "x": x,
      "u": u,
      "t": t, 
      "t_delay": t_delay,
      "first_time_step": 0,
      "last_time_step": 0
    }

    # There are no missed computations, so the valid data should equal all the data.
    expected_valid_data_from_batch = expected_all_data_from_batch

    expected_next_batch_init = {
      "x0": x[-1],
      "u0": u[-1],
      "start_time_step": 1,
      "u_pending": None,
      "next_u_update_time_step": None
    }
    
    self.assertEqual(result["all_data_from_batch"], expected_all_data_from_batch)
    self.assertEqual(result["valid_data_from_batch"], expected_valid_data_from_batch)
    self.assertEqual(result["next_batch_init"], expected_next_batch_init)
  
  def test_one_step_no_misses(self):
    x = [[1, 0], [3, 4]]
    u = [[3], [-4]]
    t = [0.0, 1.0]
    t_delay = [0.1]
    sample_time = 1.0
    n_time_steps = len(t) - 1
    batch_simulation_data = {
      "x": x,
      "u": u,
      "t": t,
      "t_delay": t_delay, 
      "sample_time": sample_time,
      "n_time_steps": n_time_steps, 
      "first_time_step": 0, 
      "last_time_step": 1
    }

    # Save the input simulation data so we can check that it is not modified.
    expected_batch_simulation_data = copy.deepcopy(batch_simulation_data)

    result = scarabintheloop.processBatchSimulationData(batch_simulation_data)

    # Check that batch_simulation_data is not modified.
    self.assertEqual(batch_simulation_data, expected_batch_simulation_data)

    ##### Construct expected data #####
    expected_all_data_from_batch = {
      "x": x,
      "u": u,
      "t": t, 
      "t_delay": t_delay,
      "first_time_step": 0,
      "last_time_step": 1
    }

    # There are no missed computations, so the valid data should equal all the data.
    expected_valid_data_from_batch = expected_all_data_from_batch

    expected_next_batch_init = {
      "x0": x[-1],
      "u0": u[-1],
      "start_time_step": 2,
      "u_pending": None,
      "next_u_update_time_step": None
    }
    
    self.assertEqual(result["all_data_from_batch"], expected_all_data_from_batch)
    self.assertEqual(result["valid_data_from_batch"], expected_valid_data_from_batch)
    self.assertEqual(result["next_batch_init"], expected_next_batch_init)


  def test_miss_at_last_time_step(self):
    x = [[1, 0], [3, 4], [5, -2]]
    u = [[3], [-2], [6]]
    t = [0.0, 1.0, 2.0]
    t_delay = [None, 0.1, 1.1] # Last t_delay longer than sample time!
    sample_time = 1.0
    n_time_steps = len(t) - 1
    batch_simulation_data = {
      "x": x,
      "u": u,
      "t": t,
      "t_delay": t_delay, 
      "sample_time": sample_time,
      "n_time_steps": n_time_steps, 
      "first_time_step": 0, 
      "last_time_step": 2
    }
    result = scarabintheloop.processBatchSimulationData(batch_config, batch_simulation_data)

    ##### Construct expected data for result #####
    expected_all_data_from_batch = {
      "x": x,
      "u": u,
      "t": t, 
      "t_delay": t_delay,
      "first_time_step": 0,
      "last_time_step": 2
    }

    # The missed computation occured at the last time step, 
    # so all of the data is valid data---effects of the missed computation 
    # on u and t only appears in the next batch
    expected_valid_data_from_batch = {
      "x": x,
      "u": u,
      "t": t, 
      "t_delay": t_delay,
      "first_time_step": 0,
      "last_time_step": 2
    }

    expected_next_batch_init = {
      "x0": x[-1],
      "u0": u[-2],
      "u_pending": u[-1],
      "start_time_step": 3,
      "next_u_update_time_step": 4
    }
    
    self.assertEqual(result["all_data_from_batch"],   expected_all_data_from_batch)
    self.assertEqual(result["valid_data_from_batch"], expected_valid_data_from_batch)
    self.assertEqual(result["next_batch_init"],       expected_next_batch_init)

    
  def test_miss_before_last_time_step(self):
    x = [[1, 0], [3, 4], [5, -2]]
    u = [[3], [-2], [6]]
    t = [0.0, 1.0, 2.0]
    t_delay = [None, 1.1, 0.2] # Last t_delay longer than sample time!
    sample_time = 1.0
    n_time_steps = len(t) - 1
    batch_config = {
      "simulation_label": "test",
      "x0": [0],
      "u0": [0],
      "system_parameters": {
        "sample_time": sample_time,
      },
      "n_time_steps": n_time_steps, 
      "first_time_step": 0, 
      "last_time_step": 2
    }
    batch_simulation_data = {
      "x": x,
      "u": u,
      "t": t,
      "t_delay": t_delay, 
    }
    result = scarabintheloop.processBatchSimulationData(batch_config, batch_simulation_data)

    ##### Construct expected data for result #####
    expected_all_data_from_batch = {
      "x": x,
      "u": u,
      "t": t, 
      "t_delay": t_delay,
      "first_time_step": 0,
      "last_time_step": 2
    }
    # The missed computation occured at the last time step, 
    # so the first and second teim steps are valid.
    expected_valid_data_from_batch = {
      "x": [[1, 0], [3, 4]],
      "u": [[3], [-2]],
      "t": [0.0, 1.0], 
      "t_delay": [None, 1.1],
      "first_time_step": 0,
      "last_time_step": 1
    }

    expected_next_batch_init = {
      "x0": x[1],
      "u0": u[0],
      "u_pending": u[1],
      "start_time_step": 2,
      "next_u_update_time_step": 3
    }
    
    self.assertEqual(result["all_data_from_batch"],   expected_all_data_from_batch)
    self.assertEqual(result["valid_data_from_batch"], expected_valid_data_from_batch)
    self.assertEqual(result["next_batch_init"],       expected_next_batch_init)


  # def test_error_if_lengths_dont_match(self):
  #   pass
    
if __name__ == '__main__':
  unittest.main()