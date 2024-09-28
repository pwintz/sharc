#! /bin/env python3

import unittest
from scarabintheloop.plant_runner import TimeStepSeries
import copy

class TestTimeStepSeries(unittest.TestCase):

  def test_initialization(self):
    k0 = 0
    t0 = 0
    x0 = 1.2
    time_step_series = TimeStepSeries(k0, t0, x0)
    self.assertEqual(time_step_series.x, [])

class TestCopy(unittest.TestCase):
  def test_copy(self):
    k0 = 0
    t0 = 0
    x0 = 1.2
    time_step_series = TimeStepSeries(k0, t0, x0)
    copied_time_step_series = time_step_series.copy()
    self.assertIsNot(copied_time_step_series.x, time_step_series.x)
    self.assertIsNot(copied_time_step_series.t, time_step_series.t)
    self.assertIsNot(copied_time_step_series.u, time_step_series.u)
    self.assertIsNot(copied_time_step_series.k, time_step_series.k)
    self.assertIsNot(copied_time_step_series.x, time_step_series.x)
    # and so on...

    self.assertEqual(copied_time_step_series.k0, k0)
    self.assertEqual(copied_time_step_series.t0, t0)
    self.assertEqual(copied_time_step_series.x0, x0)
    self.assertEqual(copied_time_step_series.k0, time_step_series.k0)
    self.assertEqual(copied_time_step_series.t0, time_step_series.t0)
    self.assertEqual(copied_time_step_series.x0, time_step_series.x0)
    self.assertEqual(copied_time_step_series.t, time_step_series.t)
    self.assertEqual(copied_time_step_series.u, time_step_series.u)
    self.assertEqual(copied_time_step_series.k, time_step_series.k)
    self.assertEqual(copied_time_step_series.x, time_step_series.x)

class TestConcatenate(unittest.TestCase):
  def test_concatenate(self):
    
    k0_this = 0
    t0_this = 0
    x0_this = 1.2
    u0_this = 55
    this_ts_series = TimeStepSeries(k0_this, t0_this, x0_this)
    t1_this = 1.2
    x1_this = -34
    u1_this = 23 # "u_pending"
    u_delay0_this = 0.234
    this_ts_series.append(t1_this, x1_this, u0_this, u1_this, u_delay0_this)

    k0_that = k0_this + 1
    t0_that = t1_this
    x0_that = x1_this
    u0_that = 45.6
    that_ts_series = TimeStepSeries(k0_that, t0_that, x0_that)
    t1_that = 100.2
    x1_that = -34
    u1_that = 23 # "u_pending"
    u_delay0_that = 0.234
    that_ts_series.append(t1_that, x1_that, u0_that, u1_that, u_delay0_that)

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
    x0 = 1.2
    ts_series = TimeStepSeries(k0, t0, x0)

    t1 = 1.2
    x1 = -34
    u0 = 55
    u1 = 23 # "u_pending"
    u_delay0 = 0.234
    ts_series.append(t1, x1, u0, u1, u_delay0)

    # Check time-index aligned data.
    self.assertEqual(ts_series.x, [x0, x1])
    self.assertEqual(ts_series.t, [t0, t1])
    self.assertEqual(ts_series.u, [u0, u0])
    self.assertEqual(ts_series.k, [k0, k0])

    # Check the time step aligned data.
    self.assertEqual(ts_series.step_k, [k0])
    self.assertEqual(ts_series.step_t_start, [t0])
    self.assertEqual(ts_series.step_t_end, [t1])
    self.assertEqual(ts_series.step_u_pending, [u1])
    self.assertEqual(ts_series.step_u_delay, [u_delay0])
    # self.assertEqual(ts_series.step_u_pending_n_steps_delayed, ) # TODO?


  def test_single_time_step_with_mid_update(self):
    k0 = 12
    t0 = 0.7
    x0 = 1.2
    u0 = 0.5
    ts_series = TimeStepSeries(k0, t0, x0)

    t_mid = 7
    x_mid = 12

    t1 = 70
    x1 = 120
    u1 = 5.0
    u_delay0 = 0.234
    ts_series.append(t1, x1, u0, u1, u_delay0, t_mid=t_mid, x_mid=x_mid)

    # Check time-index aligned data.
    self.assertEqual(ts_series.x, [x0, x_mid, x_mid, x1])
    self.assertEqual(ts_series.t, [t0, t_mid, t_mid, t1])
    self.assertEqual(ts_series.u, [u0,    u0,    u1, u1])
    self.assertEqual(ts_series.k, [k0,    k0,    k0, k0])

    self.assertEqual(ts_series.step_k, [k0])
    self.assertEqual(ts_series.step_t_start, [t0])
    self.assertEqual(ts_series.step_t_end, [t1])
    self.assertEqual(ts_series.step_u_pending, [None])
    self.assertEqual(ts_series.step_u_delay, [u_delay0])

class Test_append_multiple(unittest.TestCase):
  def test_with_mid_steps(self):
    k0 = 2
    t0 = 0.1
    x0 = 1.2
    ts_series = TimeStepSeries(k0, t0, x0)
    ts_series.append_multiple(
                    t_end=[1.1, 2.2], 
                    x_end=[ 10,  20], 
                        u=[ -1,  -2], 
                u_pending=[ 11,  22], 
                  u_delay=[0.1, 0.2], 
                    t_mid=[0.5, 1.5], 
                    x_mid=[  5,  15]
              )
    self.assertEqual(ts_series.n_time_steps(), 2)
    self.assertEqual(ts_series.k, [k0,  k0,  k0,  k0, k0 + 1, k0 + 1, k0 + 1, k0 + 1])
    self.assertEqual(ts_series.t, [t0, 0.5, 0.5, 1.1,    1.1,    1.5,    1.5,    2.2])
    self.assertEqual(ts_series.x, [x0,   5,   5,  10,     10,     15,     15,     20])
    self.assertEqual(ts_series.u, [-1,  -1,  11,  11,     -2,     -2,     22,     22])

class Test_truncate(unittest.TestCase):
  def test_truncate_without_mid_steps(self):
    
    k0 = 2
    t0 = 0.1
    x0 = 1.2
    ts_series = TimeStepSeries(k0, t0, x0)
    ts_series.append_multiple(
                    t_end=[1.1, 2.2, 3.3], 
                    x_end=[ 10,  20,  30], 
                        u=[ -1,  -2,  -3], 
                u_pending=[ 11,  22,  33], 
                  u_delay=[0.1, 0.2, 0.3]
              )
    truncated_series = ts_series.truncate(n_timesteps=1)
    self.assertEqual(truncated_series.n_time_steps(), 1)
    self.assertEqual(truncated_series.k, [k0, k0])
    self.assertEqual(truncated_series.t, [t0, 1.1])
    self.assertEqual(truncated_series.x, [x0, 10])
    self.assertEqual(truncated_series.u, [-1, -1])

    self.assertEqual(truncated_series.step_k, [k0])
    self.assertEqual(truncated_series.step_t_start, [t0])
    self.assertEqual(truncated_series.step_t_end, [1.1])
    self.assertEqual(truncated_series.step_u_pending, [11])
    self.assertEqual(truncated_series.step_u_delay, [0.1])    

  def test_truncate_without_mid_steps_giving_last_k(self):
    
    k0 = 2
    t0 = 0.1
    x0 = 1.2
    ts_series = TimeStepSeries(k0, t0, x0)
    ts_series.append_multiple(
                    t_end=[1.1, 2.2, 3.3], 
                    x_end=[ 10,  20,  30], 
                        u=[ -1,  -2,  -3], 
                u_pending=[ 11,  22,  33], 
                  u_delay=[0.1, 0.2, 0.3]
              )
    truncated_series = ts_series.truncate(last_k=4)
    self.assertEqual(truncated_series.k[-1], 4)
    self.assertEqual(truncated_series.step_k, [2, 3, 4])

  def test_truncate_with_mid_steps(self):
    
    k0 = 2
    t0 = 0.1
    x0 = 1.2
    ts_series = TimeStepSeries(k0, t0, x0)
    ts_series.append_multiple(
                    t_end=[1.1, 2.2, 3.3], 
                    x_end=[ 10,  20,  30], 
                        u=[ -1,  -2,  -3], 
                u_pending=[ 11,  22,  33], 
                  u_delay=[0.1, 0.2, 0.3], 
                    t_mid=[0.5, 1.5, 2.5], 
                    x_mid=[  5,  15,  25]
              )
    truncated_series = ts_series.truncate(n_timesteps=1)
    self.assertEqual(truncated_series.n_time_steps(), 1)
    self.assertEqual(truncated_series.k, [k0,  k0,  k0,  k0])
    self.assertEqual(truncated_series.t, [t0, 0.5, 0.5, 1.1])
    self.assertEqual(truncated_series.x, [x0,   5,   5,  10])
    self.assertEqual(truncated_series.u, [-1,  -1,  11,  11])

    self.assertEqual(truncated_series.step_k,       [k0])
    self.assertEqual(truncated_series.step_t_start, [t0])
    self.assertEqual(truncated_series.step_t_end,   [1.1])
    self.assertEqual(truncated_series.step_u_pending, [None])
    self.assertEqual(truncated_series.step_u_delay, [0.1])  



if __name__ == '__main__':
  unittest.main()