#! /bin/env python3

import unittest
from scarabintheloop.plant_runner import TimeStepSeries, ComputationData
import copy
import numpy as np
from io import StringIO

from unittest.mock import patch, mock_open

class Test_TimeStepSeries(unittest.TestCase):

  def test_initialization(self):
    k0 = 0
    t0 = 0
    x0 = [1.2]
    time_step_series = TimeStepSeries(k0, t0, x0)
    self.assertEqual(time_step_series.x, [])
    
  def test___repr__(self):
    k0 = 0
    t0 = 0
    x0 = [1.2]
    time_step_series = TimeStepSeries(k0, t0, x0)
    representation = repr(time_step_series)
    # print('representation: ', representation)


  def test___repr__long(self):
    k0 = 0
    t0 = 0
    x0 = np.array([[1.1], [1.2]])
    t = np.linspace(0.0, 2.0, num=21)
    t_starts = t[:-1]
    t_ends   = t[1:]
    ts_series = TimeStepSeries(k0, t0, x0)
    pending_computations = [ComputationData(t_start=t, delay=0.01, u=[11]) for t in t_starts]
    pending_computations[-1] = None # Make one value "None" to check this is OK.
    ts_series.append_multiple(
                              t_end=t_ends, 
                              x_end=[np.array([[10], [20]])]*len(t_ends), 
                                  u=[np.array([[-1], [-2]])]*len(t_ends), 
                pending_computation=pending_computations,
              )

    representation = repr(ts_series)
    # print('representation: ', representation)
#     expected_repr = """TimeStepSeries(k0=0, t0=0, x0=1.2, pending_computation_prior=None,
#      k (time steps)=            0,             0,             1,             1, [19 of 27 entries hidden],            18,            18,            19,            19
#      i (time indxs)=            0,             1,             1,             2, [19 of 27 entries hidden],            18,            19,            19,            20
#      t             =         0.00,          0.10,          0.10,          0.20, [19 of 27 entries hidden],          1.80,          1.90,          1.90,          2.00
# pending_computation=[ 0.00, 0.10], [ 0.00, 0.10], [ 0.00, 0.10], [ 0.00, 0.10], [19 of 27 entries hidden], [ 0.00, 0.10], [ 0.00, 0.10],          None,          None)"""
#     self.assertEqual(representation, expected_repr)

  def test_print(self):
    k0 = 0
    t0 = 0
    x0 = [1.2, 2.3]
    time_step_series = TimeStepSeries(k0, t0, x0)
    time_step_series.append(
      t_end=1, 
      x_end=[3.4, 0.1],
      u=[0.0001], 
      pending_computation=None
    )
    with patch('sys.stdout', new = StringIO()) as std_out:
      time_step_series.print('My label')
      time_step_series.printTimingData('My label')
    # print('representation: ', representation)


class TestCopy(unittest.TestCase):
  def test_copy(self):
    k0 = 0
    t0 = 0
    x0 = [1.2]
    ts_series = TimeStepSeries(k0, t0, x0)

    pending_computation = ComputationData(t_start=0, delay=0.3, u=[12])
    ts_series.append(
      t_end=1, 
      x_end=[3.4],
      u=[0.0001], 
      pending_computation=pending_computation
    )
    copied_ts_series = ts_series.copy()
    self.assertIsNot(copied_ts_series.x, ts_series.x)
    self.assertIsNot(copied_ts_series.t, ts_series.t)
    self.assertIsNot(copied_ts_series.u, ts_series.u)
    self.assertIsNot(copied_ts_series.k, ts_series.k)
    self.assertIsNot(copied_ts_series.x, ts_series.x)
    # and so on...

    self.assertEqual(copied_ts_series.k0, k0)
    self.assertEqual(copied_ts_series.t0, t0)
    self.assertEqual(copied_ts_series.x0, x0)
    self.assertEqual(copied_ts_series.k0, ts_series.k0)
    self.assertEqual(copied_ts_series.t0, ts_series.t0)
    self.assertEqual(copied_ts_series.x0, ts_series.x0)
    self.assertEqual(copied_ts_series.t,  ts_series.t)
    self.assertEqual(copied_ts_series.u,  ts_series.u)
    self.assertEqual(copied_ts_series.k,  ts_series.k)
    self.assertEqual(copied_ts_series.x,  ts_series.x)

class TestConcatenate(unittest.TestCase):
  def test_concatenate(self):
    
    k0_this = 0
    t0_this = 0
    x0_this = [1.2]
    u0_this = [55]
    this_ts_series = TimeStepSeries(k0_this, t0_this, x0_this)
    t1_this = 1.2
    x1_this = [-34]
    u1_this = [23] # "u_pending"
    u_delay0_this = 0.234
    this_ts_series.append(t1_this, x1_this, u0_this, 
                          ComputationData(t_start=t0_this, delay=0.3, u=[12]))

    k0_that = k0_this + 1
    t0_that = t1_this
    x0_that = x1_this
    u0_that = [45.6]
    that_ts_series = TimeStepSeries(k0_that, t0_that, x0_that)
    t1_that = 100.2
    x1_that = [-34]
    u1_that = [23] # "u_pending"
    u_delay0_that = 0.234
    that_ts_series.append(t1_that, x1_that, u0_that, 
                          ComputationData(t_start=t0_that, delay=0.3, u=[12]))

    # Concatenate those serieses, baby!
    concatenated_series = this_ts_series + that_ts_series
    
    self.assertEqual(concatenated_series.x, [x0_this, x1_this, x0_that, x1_that])
    self.assertEqual(concatenated_series.k, [k0_this, k0_this, k0_that, k0_that])
    self.assertEqual(concatenated_series.t, [t0_this, t1_this, t0_that, t1_that])
    self.assertEqual(concatenated_series.u, [u0_this, u0_this, u0_that, u0_that])

class Test_append(unittest.TestCase):
  def test_single_time_step_without_mid_update(self):
    k0 = 12
    t0 = 0.3
    x0 = [1.2]
    ts_series = TimeStepSeries(k0, t0, x0)

    t1 = 1.2
    x1 = [-34]
    u0 = [55]
    u1 = [23] # "u_pending"
    u_delay0 = 0.234
    
    pending_computation = ComputationData(t_start=t0, delay=0.1, u=[11])
    ts_series.append(t1, x1, u0, pending_computation)

    # Check time-index aligned data.
    self.assertEqual(ts_series.x, [x0, x1])
    self.assertEqual(ts_series.t, [t0, t1])
    self.assertEqual(ts_series.u, [u0, u0])
    self.assertEqual(ts_series.k, [k0, k0])

    self.assertEqual(ts_series.pending_computation, [pending_computation, pending_computation])
    # Check the time step aligned data.
    # self.assertEqual(ts_series.step_k, [k0])
    # self.assertEqual(ts_series.step_t_start, [t0])
    # self.assertEqual(ts_series.step_t_end, [t1])
    # self.assertEqual(ts_series.step_u_pending, [u1])
    # self.assertEqual(ts_series.step_u_delay, [u_delay0])
    # self.assertEqual(ts_series.step_u_pending_n_steps_delayed, ) # TODO?

  def test_single_time_step_with_mid_update(self):
    k0 = 12
    t0 = 0.7
    x0 = [1.2]
    u0 = [0.5]
    ts_series = TimeStepSeries(k0, t0, x0)

    u_delay0 = 0.234

    t_mid = 7
    x_mid = [12]

    t1 = 70
    x1 = [120]
    u1 = [5.0]
    
    pending_computation = ComputationData(t_start=t0, delay=u_delay0, u=u1)
    ts_series.append(t1, x1, u0, pending_computation, t_mid=t_mid, x_mid=x_mid)

    # Check time-index aligned data.
    self.assertEqual(ts_series.x, [x0, x_mid, x_mid, x1])
    self.assertEqual(ts_series.t, [t0, t_mid, t_mid, t1])
    self.assertEqual(ts_series.u, [u0,    u0,    u1, u1])
    self.assertEqual(ts_series.k, [k0,    k0,    k0, k0])
    self.assertEqual(ts_series.pending_computation, [pending_computation, pending_computation, None, None])

class Test_append_multiple(unittest.TestCase):
  def test_with_mid_steps(self):
    k0 = 2
    t0 = 0.1
    x0 = [1.2]
    t = [t0, 1.1, 2.2]
    ts_series = TimeStepSeries(k0, t0, x0)
    pending_computations = [
      ComputationData(t_start=t[0], delay=0.1, u=[11]),
      ComputationData(t_start=t[1], delay=0.2, u=[22])
    ]
    expected_pending_computations = [
      ComputationData(t_start=t[0], delay=0.1, u=[11]),
      ComputationData(t_start=t[1], delay=0.2, u=[22])
    ]
    ts_series.append_multiple(
                    t_end=t[1:], 
                    x_end=[[10], [20]], 
                        u=[[-1], [-2]], 
      pending_computation=pending_computations,
                    t_mid=[0.5,   1.5], 
                    x_mid=[[ 5], [15]]
              )
    self.assertEqual(ts_series.k, [  k0,   k0,   k0,   k0, k0 + 1, k0 + 1, k0 + 1, k0 + 1])
    self.assertEqual(ts_series.t, [  t0,  0.5,  0.5, t[1],   t[1],    1.5,    1.5,   t[2]])
    self.assertEqual(ts_series.x, [  x0, [ 5], [ 5], [10],   [10],   [15],   [15],   [20]])
    self.assertEqual(ts_series.u, [[-1], [-1], [11], [11],   [-2],   [-2],   [22],   [22]])

    # print(ts_series.i)
    # print(ts_series.pending_computation)
    self.assertEqual(ts_series.get_pending_computation_started_at_sample_time_index(2), expected_pending_computations[0])
    self.assertEqual(ts_series.get_pending_computation_started_at_sample_time_index(3), expected_pending_computations[1])

class Test_overwrite_computation_times(unittest.TestCase):

  def test_two_time_steps(self):
    k0 = 2
    t0 = 0.0
    x0 = [1.2]
    t = [t0, 1.0, 2.0]
    ts_series = TimeStepSeries(k0, t0, x0)
    pending_computations = [
      ComputationData(t_start=t[0], delay=0.0, u=[11]),
      ComputationData(t_start=t[1], delay=0.0, u=[22])
    ]
    ts_series.append_multiple(
                    t_end=t[1:], 
                    x_end=[[10], [20]], 
                        u=[[-1], [-2]], 
      pending_computation=pending_computations
    )
    # self.assertEqual(ts_series.pending_computation[1], ComputationData(t_start=1.1, delay=0.0, u=22))
    new_delays = [0.1, 0.2]
    ts_series.overwrite_computation_times(new_delays)
    
    expected_pending_computations = [
      ComputationData(t_start=t[0], delay=new_delays[0], u=[11]),
      ComputationData(t_start=t[1], delay=new_delays[1], u=[22])
    ]
    self.assertEqual(ts_series.get_pending_computation_started_at_sample_time_index(2), expected_pending_computations[0])
    self.assertEqual(ts_series.get_pending_computation_started_at_sample_time_index(3), expected_pending_computations[1])

class Test_truncate(unittest.TestCase):
  def test_truncate_without_mid_steps(self):
    
    k0 = 2
    t0 = 0.1
    x0 = [1.2]
    t = [t0, 1.1, 2.2, 3.3]
    ts_series = TimeStepSeries(k0, t0, x0)
    
    pending_computations = [
      ComputationData(t_start=t[0], delay=0.1, u=[11]),
      ComputationData(t_start=t[1], delay=0.2, u=[22]),
      ComputationData(t_start=t[2], delay=0.3, u=[33])
    ]

    ts_series.append_multiple(
                    t_end=t[1:], 
                    x_end=[ [10],  [20],  [30]], 
                        u=[ [-1],  [-2],  [-3]], 
      pending_computation=pending_computations
    )
    truncated_series = ts_series.truncate(last_k=2)
    # self.assertEqual(truncated_series.n_time_steps, 0) #TODO
    self.assertEqual(truncated_series.k, [k0, k0])
    self.assertEqual(truncated_series.i, [k0, k0+1])
    self.assertEqual(truncated_series.t, [t0, 1.1])
    self.assertEqual(truncated_series.x, [x0, [10]])
    self.assertEqual(truncated_series.u, [[-1], [-1]])

  def test_truncate_without_mid_steps_giving_last_k(self):
    
    k0 = 2
    t0 = 0.1
    x0 = [1.2]
    t = [t0, 1.1, 2.2, 3.3]
    ts_series = TimeStepSeries(k0, t0, x0)
    
    pending_computations = [
      ComputationData(t_start=t[0], delay=0.1, u=[11]),
      ComputationData(t_start=t[1], delay=0.2, u=[22]),
      ComputationData(t_start=t[2], delay=0.3, u=[33])
    ]
    ts_series.append_multiple(
                    t_end=t[1:], 
                    x_end=[ [10],  [20],  [30]], 
                        u=[ [-1],  [-2],  [-3]], 
        pending_computation=pending_computations
     )
    truncated_series = ts_series.truncate(last_k=4)
    self.assertEqual(truncated_series.k[-1], 4)
    self.assertEqual(truncated_series.i[-1], 5)
    # self.assertEqual(truncated_series.step_k, [2, 3, 4])

  def test_truncate_with_mid_steps(self):
    
    k0 = 2
    t0 = 0.1
    x0 = [1.2]
    ts_series = TimeStepSeries(k0, t0, x0)
    
    pending_computations = [
      ComputationData(t_start=0.1, delay=0.1, u=[11]),
      ComputationData(t_start=1.1, delay=0.2, u=[22]),
      ComputationData(t_start=2.2, delay=0.3, u=[33])
    ]
    ts_series.append_multiple(
                    t_end=[1.1,   2.2,  3.3], 
                    x_end=[[10], [20], [30]], 
                        u=[[-1], [-2], [-3]], 
      pending_computation=pending_computations,
                    t_mid=[0.5,   1.5,  2.5], 
                    x_mid=[[ 5], [15], [25]]
              )
    truncated_series = ts_series.truncate(last_k=2)
    # self.assertEqual(truncated_series.n_time_steps, 1)
    self.assertEqual(truncated_series.k, [k0,     k0,    k0,     k0])
    self.assertEqual(truncated_series.t, [t0,    0.5,   0.5,    1.1])
    self.assertEqual(truncated_series.x, [x0,    [ 5],  [ 5],  [10]])
    self.assertEqual(truncated_series.u, [[-1],  [-1],  [11],  [11]])


class Test_from_list(unittest.TestCase):
  def test_fail_on_empty_list(self):
    # ---- Setup ---- 
    k0 = 3
    t_list = []
    x_list = []
    u_list = []
    delay_list = []

    # ---- Execution and Assertion ----
    with self.assertRaises(AssertionError):
      TimeStepSeries._from_lists(k0, t_list, x_list, u_list, delay_list)

  def test_no_step(self):
    # ---- Setup ---- 
    k0 = 3
    t_list = [1.1]
    x_list = [4.5]
    u_list = []
    delay_list = []
    
    # ---- Execution and Assertion ----
    with self.assertRaises(AssertionError):
      TimeStepSeries._from_lists(k0, t_list, x_list, u_list, delay_list)
  
  def test_single_step(self):
    # ---- Setup ---- 
    k0 = 3
    t_list = [1.1, 2.2]
    x_list = [[4.5], [-1]]
    u_list = [[-1.1]]
    delay_list = [0.2]
    
    # ---- Execution ----
    ts = TimeStepSeries._from_lists(k0, t_list, x_list, u_list, delay_list)
  
    # ---- Assertions ---- 
    # Check initial state
    self.assertEqual(ts.k0, k0)
    self.assertEqual(ts.i0, k0)
    self.assertEqual(ts.t0, t_list[0])
    self.assertEqual(ts.x0, x_list[0])
    # Check arrays.
    self.assertEqual(ts.k, [k0, k0])
    self.assertEqual(ts.i, [k0, k0 + 1])
    self.assertEqual(ts.t, t_list)
    self.assertEqual(ts.x, x_list)

    expected_first_pc = ComputationData(t_list[0], delay_list[0], 0.0)
    self.assertEqual(ts.pending_computation[0], expected_first_pc)
  
  def test_two_steps(self):
    # ---- Setup ---- 
    k0     = 9
    t_list = [1.1, 2.2, 3.3]
    x_list = [[10], [20], [30]]
    u_list = [[-1.1], [-2.2]]
    delay_list = [0.2, 0.3]
    
    # ---- Execution ----
    ts = TimeStepSeries._from_lists(k0, t_list, x_list, u_list, delay_list)
  
    # ---- Assertions ---- 
    # Check initial state
    self.assertEqual(ts.k0, k0)
    self.assertEqual(ts.i0, k0)
    self.assertEqual(ts.t0, t_list[0])
    self.assertEqual(ts.x0, x_list[0])
    # Check arrays.
    self.assertEqual(ts.k, [k0, k0,     k0 + 1, k0 + 1])
    self.assertEqual(ts.i, [k0, k0 + 1, k0 + 1, k0 + 2])
    self.assertEqual(ts.t, [t_list[0], t_list[1], t_list[1], t_list[2]])
    self.assertEqual(ts.x, [x_list[0], x_list[1], x_list[1], x_list[2]])

    expected_pending_computations = [
      ComputationData(t_start=t_list[0], delay=delay_list[0], u=0.0),
      ComputationData(t_start=t_list[0], delay=delay_list[0], u=0.0),
      ComputationData(t_start=t_list[1], delay=delay_list[1], u=0.0),
      ComputationData(t_start=t_list[1], delay=delay_list[1], u=0.0)
    ]
    self.assertEqual(ts.pending_computation, expected_pending_computations)

class Test_n_time_steps_AND_n_indices(unittest.TestCase):
  def test_empty_series(self):
    # ---- Setup ---- 
    ts = TimeStepSeries(1, 0.4, [3])

    # ---- Assertions ---- 
    self.assertEqual(ts.n_time_steps, 0)
    self.assertEqual(ts.n_time_indices, 1)
    self.assertTrue(ts.is_empty)

  def test_one_step_series(self):
    # ---- Setup ---- 
    ts = TimeStepSeries(1, 0.4, [3])
    ts.append(1.3, [3], [-1], ComputationData(0.4, 0.1, [0]))

    # ---- Assertions ---- 
    self.assertFalse(ts.is_empty)
    self.assertEqual(ts.n_time_steps, 1)
    self.assertEqual(ts.n_time_indices, 2)

if __name__ == '__main__':
  unittest.main()