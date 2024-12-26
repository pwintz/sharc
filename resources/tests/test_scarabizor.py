#!/bin/env python3

import unittest
import os
import copy
from sharc.scarabizor import ParamsData, ScarabPARAMSReader, MockExecutionDrivenScarabRunner, ScarabStatsReader, SECONDS_PER_FEMTOSECOND
from unittest.mock import mock_open, patch

PARAMS_data_in = """# Here is a comment
--mode                          full

## Core Parameters
# Length of each chip cycle in femtosecond (10^-15 sec).
--chip_cycle_time               100000000
"""

RESOURCES_DIR = os.getenv("RESOURCES_DIR")
TESTS_FOLDER = os.path.join(RESOURCES_DIR, 'tests')

class Test_ParamsData(unittest.TestCase):

  @patch('os.path.exists')
  def test_from_file(self, patched_exists):
    # ---- Setup ---- 
    patched_exists.return_value = True
    with patch("builtins.open", mock_open(read_data=PARAMS_data_in)) as mock_file:

      # ---- Execution ----
      data = ParamsData.from_file("path/to/PARAMS.in")

    # ---- Assertions ---- 
    mock_file.assert_called_with("path/to/PARAMS.in")
    self.assertEqual(data.get_lines()[0], '# Here is a comment\n')

  def test_params_lines_to_dict(self):
    params_lines = [
      "## Simulation Parameters",
      "--mode                          full",
      "--model                         cmp",
      "# Length of each chip cycle in femtosecond (10^-15 sec).",
      "--chip_cycle_time               100000000"
    ]
    params_dict = ParamsData(params_lines).to_dict()
    self.assertEqual(len(params_dict), 1, "Only one item from key=chip_cycle_time should be saved in the dictionary.")
    self.assertEqual(params_dict["chip_cycle_time"], 100000000)

class TestScarabPARAMSReader(unittest.TestCase):
  def test_load(self):
    scarab_params_reader = ScarabPARAMSReader(TESTS_FOLDER)
    params_in = scarab_params_reader.params_in_to_dictionary()
    self.assertTrue(params_in is not None)
    
  def test_read_params_file(self):
    scarab_params_reader = ScarabPARAMSReader()
    params_lines = scarab_params_reader.read_params_file(TESTS_FOLDER + '/PARAMS.in')
    self.assertTrue(len(params_lines) > 100)


class TestMockExecutionDrivenScarabRunner(unittest.TestCase):

  @patch('shutil.copyfile')
  @patch('builtins.open', new_callable=mock_open)
  def test_run_one_step(self, mock_file, mock_copyfile):
    queued_delays={0: 0.1}
    runner = MockExecutionDrivenScarabRunner(queued_delays=queued_delays)
    runner.run('echo') # As close as we can get to a no-op

    # Check if open was called the correct number of times
    self.assertEqual(mock_file.call_count, runner.number_of_steps)

    # Check if open was called with the correct filename and mode
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.0'), 'w', buffering=1)

    # Get the contents of the Mock files.
    csv_stat_file_contents = mock_file().write.call_args_list[0][0][0]

    stats_reader = ScarabStatsReader(TESTS_FOLDER, is_using_roi=True)
    
    time_in_femtosecs = stats_reader.find_stat_in_lines('EXECUTION_TIME', csv_stat_file_contents.split('\n'))
    time_in_secs = float(time_in_femtosecs) * SECONDS_PER_FEMTOSECOND

    # Check that the other values are in the 'file'
    stats_reader.find_stat_in_lines('NODE_CYCLE', csv_stat_file_contents.split('\n'))
    stats_reader.find_stat_in_lines('NODE_INST_COUNT', csv_stat_file_contents.split('\n'))
    
    self.assertEqual(time_in_secs, queued_delays[0])

    # Assert that shutil.copyfile was called exactly twice
    self.assertEqual(mock_copyfile.call_count, 2)
    mock_copyfile.assert_any_call(os.path.abspath('PARAMS.generated'), os.path.abspath('PARAMS.in'))
    mock_copyfile.assert_any_call(os.path.abspath('PARAMS.generated'), os.path.abspath('PARAMS.out'))

 
  @patch('shutil.copyfile')
  @patch('builtins.open', new_callable=mock_open)
  def test_run_five_steps(self, mock_file, mock_copyfile):
    queued_delays = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
    runner = MockExecutionDrivenScarabRunner(queued_delays=queued_delays)
    runner.run('echo') # As close as we can get to a no-op

    # Check if open was called the correct number of times
    self.assertEqual(mock_file.call_count, len(queued_delays))

    # Check if open was called with the correct filename and mode
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.0'), 'w', buffering=1)
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.1'), 'w', buffering=1)
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.2'), 'w', buffering=1)
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.3'), 'w', buffering=1)
    mock_file.assert_any_call(os.path.abspath('core.stat.0.csv.roi.4'), 'w', buffering=1)

    
    # Assert that shutil.copyfile was called exactly twice
    self.assertEqual(mock_copyfile.call_count, 2)
    mock_copyfile.assert_any_call(os.path.abspath('PARAMS.generated'), os.path.abspath('PARAMS.in'))
    mock_copyfile.assert_any_call(os.path.abspath('PARAMS.generated'), os.path.abspath('PARAMS.out'))

    for i in range(len(queued_delays)):
      csv_stat_file_contents = mock_file().write.call_args_list[i][0][0]

      # Use the stats reader to make check the execution time in the recorded lines.
      stats_reader = ScarabStatsReader('', is_using_roi=True)
      time_in_femtosecs = stats_reader.find_stat_in_lines('EXECUTION_TIME', csv_stat_file_contents.split('\n'))
      time_in_secs = float(time_in_femtosecs) * SECONDS_PER_FEMTOSECOND
      
      self.assertEqual(time_in_secs, queued_delays[i])
    


if __name__ == '__main__':
    unittest.main()