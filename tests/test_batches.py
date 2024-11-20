#! /bin/env python3

import unittest

from sharc import Batcher, BatchInit, Batch
from sharc.data_types import TimeStepSeries
import dataclasses 
from sharc.utils import list_to_column_vec
import numpy as np

SAMPLE_TIME = 1.0

def create_batch(i_batch:int, k0: int, n_time_steps: int, k_of_missed_time_step=None):
  assert n_time_steps > 0
  assert isinstance(k0, int)
  assert isinstance(n_time_steps, int)

  k_end = k0 + n_time_steps
  t_list = [SAMPLE_TIME*k for k in range(k0, k_end + 1)]
  x_list = [[ 10.0*k, 10.0*k+0.1] for k in range(k0, k_end + 1)]
  u_list = [[-10.0*k,-10.0*k-0.1] for k in range(k0, k_end)]
  w_list = [[-11.0*k,-11.0*k-0.1] for k in range(k0, k_end)]
  delay_list = [0.1*SAMPLE_TIME if k != k_of_missed_time_step else 1.1* SAMPLE_TIME for k in range(k0, k_end)]
  assert len(t_list) == n_time_steps + 1
  assert len(x_list) == n_time_steps + 1
  assert len(u_list) == n_time_steps
  assert len(delay_list) == n_time_steps

  batch_init = BatchInit(i_batch=i_batch, 
                         k0=k0, 
                         t0=t_list[0], 
                         x0=list_to_column_vec(x_list[0]), 
                         u0=list_to_column_vec(u_list[0]), 
                         pending_computation=None)
  full_simulation_data = TimeStepSeries._from_lists(
      k0=k0, 
      t_list=t_list,
      x_list=x_list,
      u_list=u_list,
      w_list=w_list,
      delay_list=delay_list
    )
  return Batch(batch_init, full_simulation_data, SAMPLE_TIME)

def create_batcher_without_misses(n_time_steps, max_batch_size=99999):
  x0 = [ 0.0,  0.1]
  u0 = [-0.0, -0.1]
  
  def run_batch(batch_init: BatchInit, batch_n_time_steps) -> Batch:
    return create_batch(batch_init.i_batch, batch_init.k0, batch_n_time_steps, k_of_missed_time_step=None)
    
  return Batcher(x0, u0, run_batch, n_time_steps, max_batch_size=max_batch_size)

class Test_BatchInit(unittest.TestCase):
  def test_BatchInit_is_immutable(self):
    # ---- Setup ---- 
    x0 = list_to_column_vec([3, 4])
    u0 = list_to_column_vec([1, 2])
    batch = BatchInit(i_batch=12, k0=2, t0=0.5, x0=x0, u0=u0, pending_computation=None)
    
    # ---- Execution and Assertion ----
    with self.assertRaises(dataclasses.FrozenInstanceError):
      batch.k0 = None
      
  def test_first(self):
    # ---- Setup ----  
    x0 = list_to_column_vec([3, 4])
    u0 = list_to_column_vec([1, 2])
    
    # ---- Execution ----
    batch = BatchInit.first(x0=x0, u0=u0)
  
    # ---- Assertions ---- 
    self.assertEqual(batch.i_batch, 0)
    self.assertEqual(batch.k0,   0)
    self.assertEqual(batch.t0, 0.0)
    np.testing.assert_array_equal(batch.u0,  u0)
    self.assertEqual(batch.pending_computation,  None)

  def test__dict__(self):
    batch = create_batch(i_batch=3, k0=6, n_time_steps=2)
    batch_dictionary = batch.to_dict()
    self.assertEqual(batch_dictionary['batch_init'].i_batch, 3)
    self.assertEqual(batch_dictionary['batch_init'].k0, 6)
    self.assertEqual(batch_dictionary['full_simulation_data'].n_time_steps, 2)
  
class Test_create_batch(unittest.TestCase):
  """
  Check that our fake batch funciton works OK.
  """
  def test_multiple_steps_without_misses(self):
    # ---- Execution ----
    k0=10
    k_of_missed_time_step=None
    n_time_steps=2
    first_batch = create_batch(i_batch=2, k0=k0, n_time_steps=n_time_steps, k_of_missed_time_step=k_of_missed_time_step)
  
    full_simulation_data = TimeStepSeries._from_lists(
        k0=k0, 
        t_list=[10*SAMPLE_TIME, 11*SAMPLE_TIME, 12*SAMPLE_TIME],
        x_list=[[100, 100.1],   [110, 110.1],          [120, 120.1]],
        u_list=[        [-100, -100.1],   [-110, -110.1]],
        w_list=[        [-111, -111.1],   [-220, -220.1]],
        delay_list=[0.1*SAMPLE_TIME, 0.1*SAMPLE_TIME]
      )
    batch_init = BatchInit(i_batch=2, 
                           k0=k0, 
                           t0=k0*SAMPLE_TIME, 
                           x0=list_to_column_vec(full_simulation_data.x0), 
                           u0=list_to_column_vec(full_simulation_data.u[0]), 
                           pending_computation=None)
    # ---- Execution ----
    second_batch = Batch(batch_init, full_simulation_data, SAMPLE_TIME)

    # ---- Assertions ---- 

    # Check the batch created by "create_batch"
    self.assertIsInstance(first_batch.batch_init, BatchInit)
    self.assertEqual(first_batch.full_simulation_data.k0, k0)
    self.assertEqual(first_batch.batch_init.k0, k0)
    self.assertIsNone(first_batch.first_late_timestep)
    self.assertEqual(first_batch.last_valid_timestep,        k0 + n_time_steps - 1)
    self.assertEqual(first_batch.full_simulation_data.i[-1], k0 + n_time_steps)

    # Compare first_batch (created by "create_batch") with second_batch
    self.assertIsInstance(second_batch.batch_init, BatchInit)
    self.assertEqual(first_batch.batch_init, second_batch.batch_init, f'first batch_init: {first_batch.batch_init}, second batch_init: {second_batch.batch_init}')
    self.assertEqual(first_batch.full_simulation_data, second_batch.full_simulation_data)
    self.assertEqual(first_batch.valid_simulation_data, second_batch.valid_simulation_data)
    self.assertEqual(first_batch.first_late_timestep, second_batch.first_late_timestep)
    self.assertEqual(first_batch.last_valid_timestep, second_batch.last_valid_timestep)
    self.assertEqual(first_batch, second_batch)
  
  def test_multiple_steps_with_miss(self):
    # ---- Execution ----
    k0=10
    k_of_missed_time_step=11
    n_time_steps=3
    first_batch = create_batch(i_batch=2, k0=k0, n_time_steps=n_time_steps, k_of_missed_time_step=k_of_missed_time_step)
  
    full_simulation_data = TimeStepSeries._from_lists(
        k0=k0, 
        t_list=[10*SAMPLE_TIME, 11*SAMPLE_TIME, 12*SAMPLE_TIME, 13*SAMPLE_TIME],
        x_list=[[100, 100.1],   [110, 110.1],          [120, 120.1],       [130, 130.1]],
        u_list=[        [-100, -100.1],   [-110, -110.1],   [-120, -120.1]],
        delay_list=[0.1*SAMPLE_TIME, 1.1*SAMPLE_TIME, 0.1*SAMPLE_TIME]
        #                             ^--- This is how we specify the miss 
      )
    batch_init = BatchInit(i_batch=2, 
                           k0=k0, 
                           t0=k0*SAMPLE_TIME, 
                           x0=list_to_column_vec(full_simulation_data.x0), 
                           u0=list_to_column_vec(full_simulation_data.u[0]), 
                           pending_computation=None)
    # ---- Execution ----
    second_batch = Batch(batch_init, full_simulation_data, SAMPLE_TIME)

    # ---- Assertions ---- 
    # # These print statements are useful for debugging.
    # print('\nfirst_batch:', first_batch)
    # print('\nsecond_batch:', second_batch)

    # print('\nfirst_batch data:', first_batch.full_simulation_data)
    # print('\nsecond_batch data:', second_batch.full_simulation_data)

    # Check the batch created by "create_batch"
    self.assertIsInstance(first_batch.batch_init, BatchInit)
    self.assertEqual(first_batch.full_simulation_data.k0,    k0)
    self.assertEqual(first_batch.batch_init.k0,              k0)
    self.assertEqual(first_batch.first_late_timestep,        k_of_missed_time_step)
    self.assertEqual(first_batch.last_valid_timestep,        k_of_missed_time_step)
    self.assertEqual(first_batch.full_simulation_data.i[-1], k0 + n_time_steps)

    # Compare first_batch (created by "create_batch") with second_batch.
    self.assertIsInstance(second_batch.batch_init, BatchInit)
    self.assertEqual(first_batch.batch_init,            second_batch.batch_init)
    self.assertEqual(first_batch.full_simulation_data,  second_batch.full_simulation_data)
    self.assertEqual(first_batch.valid_simulation_data, second_batch.valid_simulation_data)
    self.assertEqual(first_batch.first_late_timestep,   second_batch.first_late_timestep)
    self.assertEqual(first_batch.last_valid_timestep,   second_batch.last_valid_timestep)

    # Compare equality of the entire batch objects.
    self.assertEqual(first_batch,                       second_batch)


class Test_Batch(unittest.TestCase):
  def test_one_step_with_no_missed_computations(self):
    # ---- Setup ---- 
    x0 = list_to_column_vec([3, 4])
    u0 = list_to_column_vec([1, 2])
    
    batch_init = BatchInit.first(x0=x0, u0=u0)
    sample_time = 1.1
    full_simulation_data = TimeStepSeries._from_lists(
        k0=0, 
        t_list=[0.0, sample_time],
        x_list=[[x0], [ 20]],
        u_list=[[u0]],
        delay_list=[0.1*sample_time]
      )
    # ---- Execution ----
    batch = Batch(batch_init, full_simulation_data, sample_time)
  
    # ---- Assertions ---- 
    # Since there are no missed computations, check that full and valid data are 
    # equal to the given data. 
    self.assertEqual(batch.full_simulation_data, full_simulation_data)
    self.assertEqual(batch.valid_simulation_data, full_simulation_data)
    # Check that they are not the same objects.
    self.assertIsNot(batch.full_simulation_data, full_simulation_data)
    self.assertIsNot(batch.valid_simulation_data, full_simulation_data)
    # Check batch init is saved.
    self.assertEqual(batch.batch_init, batch_init)
    # No late time steps.
    self.assertFalse(batch.has_missed_computation)
    self.assertIsNone(batch.first_late_timestep) 
    # Only one time step, and it is valid (note zero-indexing), so the last 
    # valid time step is k=0.
    self.assertEqual(batch.last_valid_timestep, 0) 

  def test_two_steps_with_no_missed_computations(self):
    # ---- Setup ---- 
    x0 = list_to_column_vec([10])
    u0 = list_to_column_vec([-10])
    
    batch_init = BatchInit.first(x0=x0, u0=u0)
    sample_time = 1.1
    full_simulation_data = TimeStepSeries._from_lists(
        k0=0, 
        t_list=[0.0, sample_time, 2*sample_time],
        x_list=[[x0], [ 20], [30]],
        u_list=[[u0], [-20]],
        delay_list=[0.1*sample_time, 0.1*sample_time]
      )
    # ---- Execution ----
    batch = Batch(batch_init, full_simulation_data, sample_time)
  
    # ---- Assertions ---- 
    # Since there are no missed computations, check that full and valid data are 
    # equal to the given data. 
    self.assertEqual(batch.full_simulation_data, full_simulation_data)
    self.assertEqual(batch.valid_simulation_data, full_simulation_data)
    # Check that they are not the same objects.
    self.assertIsNot(batch.full_simulation_data, full_simulation_data)
    self.assertIsNot(batch.valid_simulation_data, full_simulation_data)
    # Check batch init is saved.
    self.assertEqual(batch.batch_init, batch_init)
    # No late time steps.
    self.assertFalse(batch.has_missed_computation)
    self.assertIsNone(batch.first_late_timestep) 
    # Only one time step, and it is valid (note zero-indexing), so the last 
    # valid time step is k=0.
    self.assertEqual(batch.last_valid_timestep, 1) 

  def test_multiple_steps_with_miss_in_middle(self):
    # ---- Execution ----
    k0=10
    k_of_missed_time_step=11
    n_time_steps=4
    batch = create_batch(i_batch=2, 
                              k0=k0, 
                              n_time_steps=n_time_steps, 
                              k_of_missed_time_step=k_of_missed_time_step)
  
    # ---- Assertions ---- 
    self.assertEqual(batch.first_late_timestep, k_of_missed_time_step)
    self.assertEqual(batch.last_valid_timestep, k_of_missed_time_step)
    self.assertEqual(batch.full_simulation_data.i[-1], k0+n_time_steps)
    self.assertEqual(batch.valid_simulation_data.i[-1], k_of_missed_time_step+1)
    self.assertEqual(batch.valid_simulation_data.k[-1], k_of_missed_time_step)

class Test_Batcher(unittest.TestCase):
  def test_initialization(self):
    # ---- Setup ---- 
    x0 = [ 1]
    u0 = [-1]
    
    n_time_steps = 2

    def run_batch(batch_init: BatchInit, n_time_steps) -> Batch:
      return create_batch(batch_init.i_batch, batch_init.k0, n_time_steps, k_of_missed_time_step=None)
    
    # ---- Execution and ASSERTIONS----
    batcher = Batcher(x0, u0, run_batch, n_time_steps)
    self.assertEqual(batcher._next_batch_init.i_batch, 0)
    self.assertEqual(batcher._next_batch_init.t0, 0)
    self.assertEqual(batcher._next_batch_init.k0, 0)
    self.assertEqual(batcher._next_batch_init.x0, x0)
    self.assertEqual(batcher._next_batch_init.u0, u0)

    batch = next(batcher)
    self.assertEqual(batch.batch_init.i_batch, 0)
    self.assertEqual(batch.last_valid_timestep, 1) # Two time steps: k=0 and k=1
    
    # Check next batch.
    self.assertEqual(batcher._next_batch_init.i_batch, 1)
    self.assertEqual(batcher._next_batch_init.k0, 2)

    with self.assertRaises(StopIteration):
      # There are no more batches, so StopIteration should be raise.
      next(batcher)

  def test_one_step(self):
    # ---- Setup ---- 
    n_time_steps = 8
    max_batch_size = 8
    batcher = create_batcher_without_misses(n_time_steps, max_batch_size=max_batch_size)
    
    # ---- Assert values before first batch ----
    self.assertFalse(batcher._is_done())
    self.assertEqual(batcher._next_k0(), 0)
    self.assertEqual(batcher._last_valid_k(), None) # No k's yet.
    self.assertEqual(batcher._n_time_steps_remaining(), n_time_steps) # All time steps remaining
    self.assertEqual(batcher._next_batch_size(), max_batch_size) # Batch size is only constrained by given maximum.

    # ---- Execution ----
    batch = next(batcher)

    # Check that the batch is sensible.
    self.assertFalse(batch.has_missed_computation)
    self.assertEqual(batch.last_valid_timestep, n_time_steps - 1)# "-1" because of zero-indexing.

    # Check the batcher
    self.assertTrue(batcher._is_done()) # No time steps remaining
    self.assertEqual(batcher._last_valid_k(), 7)
    self.assertEqual(batcher._next_k0(), 8)
    self.assertEqual(batcher._n_time_steps_remaining(), 0) # No time steps remaining
    self.assertEqual(batcher._next_batch_size(), 0) # No time steps remaining

    with self.assertRaises(StopIteration):
      # There are no more batches, so StopIteration should be raise.
      next(batcher)

  def test_two_steps_without_miss(self):
    # ---- Setup ---- 
    n_time_steps = 16
    max_batch_size = 8
    batcher = create_batcher_without_misses(n_time_steps, max_batch_size=max_batch_size)
    
    # ---- Assert values before first batch ----
    self.assertFalse(batcher._is_done())
    self.assertEqual(batcher._next_k0(), 0)
    self.assertEqual(batcher._last_valid_k(), None) # No k's yet.
    self.assertEqual(batcher._n_time_steps_remaining(), n_time_steps) # All time steps remaining
    self.assertEqual(batcher._next_batch_size(), max_batch_size) # Batch size is only constrained by given maximum.

    # ---- Execution ----
    first_batch = next(batcher)

    # Check that the first_batch is sensible.
    self.assertFalse(first_batch.has_missed_computation)
    self.assertEqual(first_batch.last_valid_timestep, max_batch_size - 1)# "-1" because of zero-indexing.

    # Check the batcher
    self.assertFalse(batcher._is_done())
    self.assertEqual(batcher._last_valid_k(),           7)
    self.assertEqual(batcher._next_k0(),                8)
    self.assertEqual(batcher._n_time_steps_remaining(), 8)
    self.assertEqual(batcher._next_batch_size(),        8)

    second_batch = next(batcher)
    self.assertFalse(second_batch.has_missed_computation)
    self.assertEqual(second_batch.batch_init.k0,           8)
    self.assertEqual(second_batch.full_simulation_data.k0, 8)
    self.assertEqual(second_batch.last_valid_timestep, n_time_steps - 1)# "-1" because of zero-indexing.

    self.assertTrue(batcher._is_done())
    self.assertEqual(batcher._last_valid_k(),          15)
    self.assertEqual(batcher._next_k0(),               16)
    self.assertEqual(batcher._n_time_steps_remaining(), 0)
    self.assertEqual(batcher._next_batch_size(),        0)
    
    with self.assertRaises(StopIteration):
      # There are no more batches, so StopIteration should be raise.
      next(batcher)

  def test_looping_over_multiple_batches_without_misses(self):
    # ---- Setup ---- 
    x0 = [1]
    u0 = [-1]
    
    n_time_steps = 32
    max_batch_size = 8

    def run_batch(batch_init: BatchInit, n_time_steps) -> Batch:
      return create_batch(batch_init.i_batch, batch_init.k0, n_time_steps, k_of_missed_time_step=None)
    
    # ---- Execution ----
    batch_list = []
    batcher = Batcher(x0, u0, run_batch, n_time_steps, max_batch_size=max_batch_size)
    for batch in batcher:
      batch_list.append(batch)
  
    # ---- Assertions ---- 
    last_batch = batch_list[-1]
    self.assertEqual(last_batch.batch_init.i_batch, 3) # "-1" because of zero indexing 
    self.assertEqual(last_batch.last_valid_timestep, n_time_steps - 1) # "-1" because of zero indexing 

    self.assert_batch_list_consistency(batch_list)

  def test_n_time_steps_remaining_constrained_by_total_steps(self):
    # ---- Setup ---- 
    n_time_steps = 32
    max_batch_size = 8

    def run_batch(batch_init: BatchInit, n_time_steps) -> Batch:
      return create_batch(batch_init.i_batch, batch_init.k0, n_time_steps, k_of_missed_time_step=None)
    
    # ---- Execution ----
    batch_list = []
    x0 = [1]
    u0 = [-1]
    batcher = Batcher(x0, u0, run_batch, n_time_steps, max_batch_size=max_batch_size)
    for batch in batcher:
      batch_list.append(batch)
  
    # ---- Assertions ---- 
    last_batch = batch_list[-1]
    self.assertEqual(last_batch.batch_init.i_batch, 3) # "-1" because of zero indexing 
    self.assertEqual(last_batch.last_valid_timestep, n_time_steps - 1) # "-1" because of zero indexing 

    self.assert_batch_list_consistency(batch_list)

  def test_multiple_batches_without_misses_with_last_batch_shorter_than_max(self):
    # ---- Setup ---- 
    n_time_steps = 31
    max_batch_size = 8
    batcher = create_batcher_without_misses(n_time_steps, max_batch_size=max_batch_size)
    
    # ---- Execution ----
    batch_list = []
    for batch in batcher:
      batch_list.append(batch)
  
    # ---- Assertions ---- 
    last_batch = batch_list[-1]
    self.assertEqual(last_batch.last_valid_timestep, n_time_steps - 1) # "-1" because of zero indexing 

    self.assert_batch_list_consistency(batch_list)


  def assert_batch_list_consistency(self, batch_list: list):
    for i, batch in enumerate(batch_list):
      if i == 0: 
        continue
      np.testing.assert_array_equal(batch.batch_init.x0.flatten(), batch_list[i-1].valid_simulation_data.x[-1])
      self.assertEqual(batch.batch_init.t0, batch_list[i-1].valid_simulation_data.t[-1])
      # When moving to a new batch, the time step k increments by one (+1)
      self.assertEqual(batch.batch_init.k0, batch_list[i-1].valid_simulation_data.k[-1] + 1)



if __name__ == '__main__':
  unittest.main()