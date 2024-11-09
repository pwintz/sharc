import os
import subprocess
import re
import shutil
import time
import sys
import queue
import warnings
from typing import List, Set, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from abc import ABC, abstractmethod #  AbstractBaseClass

from scarabintheloop.utils import run_shell_cmd, assertFileExists, openLog

from scarab_globals import *
from scarab_globals import scarab_paths

debug_scarab_level = 0

# params_in_dir = 'docker_user_home'
log_dir_regex = re.compile(r"\nLog directory is (.*?)\n")
trace_file_regex = re.compile(r"\nCreated thread trace file (.*?)\n")

# Regex pattern for extracting the time from the Scarab output. 
time_regex = re.compile(r"time:(\d+)\s*--")

### COSTANTS ###
SECONDS_PER_FEMTOSECOND = 10**(-15)
SECONDS_PER_MICROSECOND = 10**(6)
MICROSECONDS_PER_FEMTOSECOND = 10**(6-15)
MICROSECONDS_PER_SECOND = 10**(-15)
FEMTOSECOND_PER_SECONDS = 10**15
    
def run(cmd, args=[], cwd='.'):
    verbose = False
    # If a single string is given for "args", convert it to a list.
    if isinstance(args, str):
        args = [args]
    
    cmd_and_args = cmd + args

    if verbose:
        # Print the command
        print(">> {0}".format(" ".join(cmd_and_args)))

    # Execute
    result = subprocess.run(cmd_and_args, capture_output=True, cwd=cwd)
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')
    # stdout_pretty = str(result.stdout).replace('\\n', '\n').replace('\\t', '\t')
    # stderr_pretty = str(result.stderr).replace('\\n', '\n').replace('\\t', '\t')
    return result, stdout, stderr

class ScarabData:

    def __init__(self, simulated_time_femtoseconds, cmd_stdout, cmd_stderr):
        self.simulated_time_femtoseconds = simulated_time_femtoseconds
        self.simulated_time_microseconds = MICROSECONDS_PER_FEMTOSECOND*float(simulated_time_femtoseconds)
        self.simulated_time_seconds = SECONDS_PER_FEMTOSECOND*float(simulated_time_femtoseconds)
        self.cmd_stdout = cmd_stdout
        self.cmd_stderr = cmd_stderr


class ParamsData:

  # --- STATIC VARIABLES --- #
  PARAMS_FILE_KEYS = ["chip_cycle_time",
                      "l1_size",
                      "dcache_size",
                      "icache_size",
                      "decode_cycles"]
  # Regex to find text in the form of "--name value"
  PARAM_REGEX_PATTERN = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")
  PARAM_STR_FMT = '--{}\t{}'
  
  @staticmethod
  def from_file(filename):
    assertFileExists(filename)
    with open(filename) as params_file:
      PARAM_lines = params_file.readlines()
    return ParamsData(PARAM_lines)

  def __init__(self, PARAMS_lines):
    assert isinstance(PARAMS_lines, list), f'PARAMS_lines must be a list. type(PARAMS_lines)={type(PARAMS_lines)}'
    self._PARAMS_lines = PARAMS_lines.copy()

  def patch(self, patch: dict):
    """ 
    Create a new PARAMS file with the values in PARAMS_patch inserted.
    """
    # The ParamsData constructor copies self_PARAMS_lines, 
    # so the copy will have a different list instance.
    patched_params = self.copy()
    for (key, value) in patch.items():
      assert key in ParamsData.PARAMS_FILE_KEYS, \
            f'key={key} must be in {ParamsData.PARAMS_FILE_KEYS}.'
      if value is not None:
        patched_params._set_value(key, value)
    return patched_params

  def copy(self):
    return ParamsData(self._PARAMS_lines)
  
  def to_file(self, PARAMS_out_filename):
    with open(PARAMS_out_filename, 'w') as params_out_file:
      params_out_file.writelines(self._PARAMS_lines)

  # TODO: Write tests for this.
  def _set_value(self, key, value: int):
    """
    Search through PARAM_lines and modify it in-place by updating the value for the given key. 
    If the key is not found, then an error is raised.
    """
    assert key in self.PARAMS_FILE_KEYS, f'key={key} must be in {self.PARAMS_FILE_KEYS}.'
    assert isinstance(value, int), f'Only int values are supported. type(value)={type(value)}'
    for line_num, line in enumerate(self._PARAMS_lines):
      regex_match = self.PARAM_REGEX_PATTERN.match(line)
      if not regex_match:
        continue
      if key == regex_match.groupdict()['param_name']:
        # If the regex matches, then we replace the line.
        new_line_text = self.PARAM_STR_FMT.format(key, value)
        self._PARAMS_lines[line_num] = new_line_text
        # print(f"Replaced line number {line_num} with {new_line_text}")
        return
    raise ValueError(f'The key {key} was not found in the PARAM_lines {PARAM_lines}.')

  def get_lines(self):
    return self._PARAMS_lines.copy()
    
  # TODO: Test. 
  def to_dict(self):
    param_dict = {}
    for line in self._PARAMS_lines: 
      regex_match = ParamsData.PARAM_REGEX_PATTERN.match(line)
      if regex_match:
        param_name = regex_match.groupdict()['param_name']
        param_value = regex_match.groupdict()['param_value']
        if param_name in ParamsData.PARAMS_FILE_KEYS:
          param_dict[param_name] = int(param_value)
    assert param_dict, f'param_dict is expected to have values. param_dict={param_dict}.'
    return param_dict

  # def __getitem__(self, key): 
  #   return self.to_dict()[key]


class ScarabPARAMSReader:
  
  def __init__(self, sim_dir=None):
    if sim_dir:
      if not sim_dir.endswith("/"):
        sim_dir += "/"
        
      self.params_in_file_path = sim_dir + 'PARAMS.in'
      self.params_out_file_path = sim_dir + 'PARAMS.out'


  def params_in_to_dictionary(self):
    return ParamsData.from_file(self.params_in_file_path).to_dict()

  def params_out_to_dictionary(self):
    return ParamsData.from_file(self.params_out_file_path).to_dict()

  def params_file_to_dictionary(self, filename):
    return ParamsData.from_file(filename).to_dict()

  def read_params_file(self, filename):
    return ParamsData.from_file(filename).get_lines()

  def create_patched_PARAMS_file(sim_config: dict, PARAMS_src_filename: str, PARAMS_out_filename: str):
    """
    Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
    Then, modify the values for keys listed in PARAMS_FILE_KEYS to values taken from sim_config. 
    Write the resulting PARAMS data to a file at PARAMS_out_filename in the simulation directory (sim_dir).
    Returns the absolute path to the PARAMS file.
    """
    print(f'Creating chip parameter file.')
    print(f'\tSource: {PARAMS_src_filename}.')
    print(f'\tOutput: {PARAMS_out_filename}.')
    PARAMS_src = ParamsData.from_file(PARAMS_src_filename)
    # with open(PARAMS_src_filename) as params_out_file:
    #   PARAM_file_lines = params_out_file.readlines()
    PARAMS_out = PARAMS_src.patch(sim_config['PARAMS_patch_values'])
    PARAMS_out.to_file(PARAMS_out_filename)

class ScarabStatsReader:
  
  def __init__(self, stats_dir_path, is_using_roi=True):
      self.stats_dir_path = stats_dir_path
      # Create a RegEx to match a line in the form
      #   EXECUTION_TIME      294870700000000  <anything here is ignored>    
      # or the CSV notation:
      #   EXECUTION_TIME_count,      294870700000000  <anything here is ignored>    
      # and create two groups, key="EXECUTION_TIME" and value="294870700000000".
      self.stat_regex = re.compile(r"\s*(?P<key>[\w\_]+)(?:_count,)?\s*(?P<value>[\d.]+).*")
      
      # Create a RegEx to match a line in the form
      #   EXECUTION_TIME_count,      294870700000000  <anything here is ignored>    
      # and create two groups, key="EXECUTION_TIME" and value="294870700000000".
      self.csv_stat_regex = re.compile(r"\s*(?P<key>[\w\_]+)_count,\s*(?P<value>\d(?:.\d+)?).*")

      
      # Create a RegEx to match a line in the form
      #   EXECUTION_TIME,             294870700000000  <anything here is ignored>             
      # and create two groups, key="EXECUTION_TIME" and value="294870700000000".
      # self.stat_regex = re.compile(r"\s*(?P<key>[\w|\_| ]+),\s*(?P<value>\d+\.?\d*).*")
      self.is_using_roi = is_using_roi

  def getStatsFilePath(self, k) -> os.PathLike:
    if self.is_using_roi:
      file_name = f"core.stat.0.csv.roi.{k}"
    else:
      file_name = f"core.stat.{k}.csv"
    file_path = os.path.join(self.stats_dir_path, file_name)
    return file_path

  def waitForStatsFile(self,k):
    file_path = self.getStatsFilePath(k)
    while not os.path.exists(file_path):
      time.sleep(0.01)
    time.sleep(0.1)

  def readStatistic(self, k: int, stat_key: str): 
    file_path = self.getStatsFilePath(k)
    with open(file_path, 'r') as stats_file:
      return self.find_stat_in_lines(stat_key, stats_file.readlines())
              
  def find_stat_in_lines(self, stat_key: str, lines: List[str]):
    for line in lines:
      regex_match = self.csv_stat_regex.match(line)
      if regex_match and stat_key == regex_match.groupdict()['key']:
        value = regex_match.groupdict()['value']
        # Cast value to either int or float.
        try: 
          return int(value)
        except ValueError:
          pass
        return float(value)
    # After looping through all of the lines, we have not found the key.
    raise ValueError(f'The key {stat_key} was not found in \n{lines}')
      

  def readCyclesCount(self, k: int):  
    # return self.readStatistic(k, "NODE_CYCLE_count")
    return self.readStatistic(k, "NODE_CYCLE")

  def readInstructionCount(self, k: int):  
    return self.readStatistic(k, "NODE_INST_COUNT")

  def readTime(self, k: int):  
    # time_in_femtosecs = self.readStatistic(k, "EXECUTION_TIME_count")
    time_in_femtosecs = self.readStatistic(k, "EXECUTION_TIME")
    time_in_secs = float(time_in_femtosecs) * SECONDS_PER_FEMTOSECOND
    return time_in_secs


class ExecutionDrivenScarabRunner:

  def __init__(self, sim_dir='.'):
    self.instruction_limit = int(1e9)
    self.heartbeat_interval = int(1e6) # How often to print progress.
    self.sim_dir         = os.path.abspath(sim_dir)
    self.params_src_file = os.path.join(self.sim_dir, 'PARAMS.generated')
    self.params_in_file  = os.path.join(self.sim_dir, 'PARAMS.in')
    self.params_out_file = os.path.join(self.sim_dir, 'PARAMS.out')
    self.controller_log  = None

    # Check that the simulation directory exists.
    assertFileExists(self.sim_dir)
  
    # Check that the necessary PARAMs.generated file exists. 
    # This constructor is run before we start multiprocessing, so it is easier to 
    # debug errors that are raised here.
    assertFileExists(self.params_src_file, f'PARAMS.generated should exist in the sim_dir={self.sim_dir}')

  def set_log(self, controller_log):
    self.controller_log=controller_log

  def run(self, cmd):
    print(f"ExecutionDrivenScarabRunner.run({cmd}) in {self.sim_dir}")

    # If 'PARAMS.generated' does not exist, then Scarab fails with unclear 
    # error messages, so we check it here to ensure it is there.
    assertFileExists(self.params_src_file)
    scarab_cmd_argv = [
        sys.executable, # The Python executable
        scarab_paths.bin_dir + '/scarab_launch.py',
        f'--program', cmd,
        f'--param', self.params_src_file,
        f'--pintool_args',
        # Skip over anything before the start instruction.
        f'-fast_forward_to_start_inst 1',
        f'--scarab_args',
        f'--inst_limit {self.instruction_limit}' # Instruction limit
        f'--heartbeat_interval {self.heartbeat_interval}', 
        # '--num_heartbeats 1'
        # '--power_intf_on 1']
      ]
    run_shell_cmd(scarab_cmd_argv, working_dir=self.sim_dir, log=self.controller_log)
    # There is a bug in Scarab that causes it to sometimes crash when run in the terminal and the terminal is resized. To avoid this bug, we run it in a different thread, which appears to fix the problem.
    
    # There is a bug in Scarab that causes it to sometimes crash when run in the terminal and the terminal is resized. To avoid this bug, we run it in a different thread, which appears to fix the problem.
    # def task():
    #  # raise ValueError(f'FORCE ERROR')
    #  run_shell_cmd(scarab_cmd_argv, working_dir=self.sim_dir, log=self.controller_log)
    # with ThreadPoolExecutor() as executor:
    #   future = executor.submit(task)
    #   if future.exception(): # Waits until finished or failed.
    #     raise Exception(f'Failed to execute {scarab_cmd_argv} in {sim_dir}. \nLogs: {controller_log.name}.') from future.exception()
    #      
    #   # Wait for the background task to finish before exiting
    #   future.result()
    #     
    

class MockExecutionDrivenScarabRunner(ExecutionDrivenScarabRunner):
    def __init__(self, *args, **kwargs):
      # self.number_of_steps = int(kwargs.pop('number_of_steps', 1))
      queued_delays = kwargs.pop('queued_delays')
      self.delay_queue = queue.Queue()
      # Put all of the queued delays into the delay queue
      for delay in queued_delays:
        self.delay_queue.put(delay)
      self.number_of_steps = self.delay_queue.qsize()
      # print("q_size:", self.number_of_steps)
      super().__init__(*args, **kwargs)

    def run(self, cmd):
      assertFileExists(self.sim_dir)
      
      if self.delay_queue.qsize() == 0:
        raise ValueError(f'Delay queue is empty!')
      delay = self.delay_queue.get()
      delay_in_femtoseconds = int(FEMTOSECOND_PER_SECONDS*delay)

      for roi_ndx in range(self.number_of_steps):
        # stat_out_roi_file_name = f'core.stat.0.out.roi.{roi_ndx}'
        stat_csv_roi_file_name = f'core.stat.0.csv.roi.{roi_ndx}'
        # stat_out_roi_file_contents = "\n".join([
        #   "Cumulative:        Cycles: 1               Instructions: 1               IPC: 1",
        #   "Periodic:          Cycles: 1               Instructions: 1               IPC: 1",
        #   f"EXECUTION_TIME                           {delay_in_femtoseconds}                  {delay_in_femtoseconds}",
        #   "NODE_CYCLE                                     1                        1",
        #   "NODE_INST_COUNT                                1                        1"
        # ])
        mock_cycle_count       = 1000
        mock_instruction_count = 1000
        stat_csv_roi_file_contents = "\n".join([
          # "Cumulative Cycles, 1",
          # "Cumulative Instructions, 1",
          # "Cumulative IPC, 1.1",
          # "Periodic Cycles, 1",
          # "Periodic Instructions, 1",
          # "Periodic IPC, 1.1",
          f"EXECUTION_TIME_count,  {delay_in_femtoseconds}",
          f"NODE_CYCLE_count,      {mock_cycle_count}",
          # f"NODE_CYCLE_total_count,       1",
          f"NODE_INST_COUNT_count, {mock_instruction_count}",
          # f"NODE_INST_COUNT_total_count,       1"
        ])

        # print(f"About to save a mock out file: {stat_out_roi_file_name}")
        # with open(os.path.join(sim_dir, stat_out_roi_file_name), 'w', buffering=1) as stat_out_roi_file:
          # stat_out_roi_file.write(stat_out_roi_file_contents + "\n")
          # print(f"Saved a mock out file: {stat_out_roi_file.name}")
        # print(f"About to save a mock out file: {stat_csv_roi_file_name}")
        with open(os.path.join(self.sim_dir, stat_csv_roi_file_name), 'w', buffering=1) as stat_csv_roi_file:
          stat_csv_roi_file.write(stat_csv_roi_file_contents + "\n")

      # Copy file PARAMS file, imitating the behavior of Scarb.
      assertFileExists(self.params_src_file)
      shutil.copyfile(self.params_src_file, self.params_in_file)
      shutil.copyfile(self.params_src_file, self.params_out_file)

      run_shell_cmd(cmd, working_dir=self.sim_dir, log=self.controller_log)

class TracesToComputationTimesProcessor(ABC):

  def __init__(self, sim_dir):
    self.sim_dir = os.path.abspath(sim_dir)
    assertFileExists(self.sim_dir)

  @abstractmethod
  def get_computation_time_from_trace(self):
    pass

  def get_all_computation_times(self):
    sorted_indices = self._get_trace_directories_indices()
    # print('sorted_trace_dirs', sorted_trace_dirs)
    # print('self.sim_dir:', self.sim_dir)

    if len(sorted_indices) > os.cpu_count():
      warnings.warn("There were more traces generated in a batch than the number of CPUs.", Warning)

    with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
      computation_times = scarab_executor.map(self.get_computation_time_from_trace, sorted_indices)
    
    # The executor returns a iterator, but we want a list.
    return list(computation_times)

  def _get_trace_directory_from_index(self, index):
    trace_dir = os.path.join(self.sim_dir, f'dynamorio_trace_{index}')
    assertFileExists(trace_dir)
    return trace_dir
  
  def _get_trace_directories_indices(self):
    trace_dir_regex = re.compile(r".*dynamorio_trace_(?P<trace_index>\d+)")

    # Note: glob does not support 'root_dir' until Python 3.10. :(
    # dynamorio_trace_dirs = glob.glob('dynamorio_trace_*', root_dir=sim_dir)

    dynamorio_trace_dirs = []
    trace_dir_dict = {}

    # Loop through the contents of sim_dir to find the folders that match "dynamorio_trace_XX"
    for trace_dir in os.listdir(self.sim_dir):
      result = trace_dir_regex.match(trace_dir)
      if not result:
        # Not a trace directory. Skip.
        continue 
      if not trace_dir.startswith("dynamorio_trace_"):
        raise ValueError(f'Not a trace directory!')

      trace_dir = os.path.join(self.sim_dir, trace_dir)
      trace_index = int(result.group('trace_index'))
      dynamorio_trace_dirs.append(trace_dir)
      trace_dir_dict[trace_index] = trace_dir

    # if len(trace_dir_dict) == 0:
    #   raise ValueError(f"No DynamoRIO trace directories were found in {self.sim_dir}")

    sorted_indices = sorted(trace_dir_dict)
    return sorted_indices

  def _get_trace_directories(self):
    indices = self._get_trace_directories_indices()
    path_format = lambda ndx: os.path.join(self.sim_dir, f'dynamorio_trace_{ndx}')
    trace_dirs = map(path_format, indices)
    return trace_dirs

class ScarabTracesToComputationTimesProcessor(TracesToComputationTimesProcessor):

#   def __init__(self, sim_dir):
#     self.sim_dir = os.path.abspath(sim_dir)
#     pass
#     # self.log = log
#     # self.instruction_limit = int(1e9)
#     # self.heartbeat_interval = int(1e6) # How often to print progress.
#     # if sim_dir is None:
#     #   sim_dir = '.'
#     # self.sim_dir = os.path.abspath(sim_dir)
#     # self.params_in_file = os.path.join(self.sim_dir, 'PARAMS.in')
# 
#   def get_all_computation_times(self):
#     (dynamrio_trace_dir_dictionaries, sorted_trace_dirs, sorted_indices) = self._get_trace_directories_indices()
#     print('sorted_trace_dirs', sorted_trace_dirs)
#     print('self.sim_dir:', self.sim_dir)
# 
#     if len(sorted_indices) > os.cpu_count():
#       warnings.warn("There were more traces generated in a batch than the number of CPUs.", Warning)
# 
#     with ProcessPoolExecutor(max_workers = os.cpu_count()) as scarab_executor:
#       # scarab_data = scarab_executor.map(ParallelSimulationExecutor.get_computation_time_from_trace_in_scarab, sorted_indices)
#       scarab_data = scarab_executor.map(self.get_computation_time_from_trace, sorted_indices)
#       scarab_data = list(scarab_data)
#     
#     computation_time_list = [None] * len(scarab_data)
#     print('computation_time_list (before)', computation_time_list)
#     print('scarab_data', scarab_data)
#     for datum in scarab_data:
#       print('scarab_datum: ', datum)
#       dir_index = datum["index"]
#       trace_dir = datum["trace_dir"]
#       computation_time = datum["computation_time"]
#       computation_time_list[dir_index] = computation_time
# 
#     return computation_time_list

  @staticmethod 
  def portabalize_trace(trace_dir): 
    print(f"Starting portabilization of trace in {trace_dir}.")
    log_path = trace_dir + "/portabilize.log"
    with openLog(log_path, f"Portablize \"{trace_dir}\"") as portabilize_log:
      try:
          run_shell_cmd("run_portabilize_trace.sh", working_dir=trace_dir, log=portabilize_log)
      except Exception as e:
        raise Exception(f'failed to portablize trace in {trace_dir}. See logs: {portabilize_log}') from e
    if debug_scarab_level > 1:
      print(f"Finished portabilization of trace in {trace_dir}.")

  def get_computation_time_from_trace(self, dir_index):
    if not isinstance(dir_index, int):
      raise AssertionError("Assertion failed: isinstance(dir_index, int)")
    
    trace_dir = self._get_trace_directory_from_index(dir_index)
    
    ScarabTracesToComputationTimesProcessor.portabalize_trace(trace_dir)

    log_path   = os.path.join(trace_dir, 'scarab.log')
    
    # Copy PARAMS.in file
    shutil.copyfile(os.path.join(trace_dir, '..', 'PARAMS.generated'), os.path.join(trace_dir, 'PARAMS.in'))
    assertFileExists(os.path.join(trace_dir, 'PARAMS.in'))
    assertFileExists(os.path.join(trace_dir, 'bin'))
    assertFileExists(os.path.join(trace_dir, 'trace'))
    scarab_cmd = ["scarab", # Requires that the Scarab build folder is in PATH
                    "--fdip_enable", "0", 
                    "--frontend", "memtrace", 
                    "--fetch_off_path_ops", "0", 
                    "--cbp_trace_r0=trace", 
                    "--memtrace_modules_log=bin"] 
    with openLog(log_path, header_text=f"Scarab log for {trace_dir}") as scarab_log:
      run_shell_cmd(scarab_cmd, working_dir=trace_dir, log=scarab_log)

    # Read the generated statistics files.
    stats_reader = ScarabStatsReader(trace_dir, is_using_roi=False)
    stats_file_index = 0 # We only simulate one timestep.
    computation_time = stats_reader.readTime(stats_file_index)
    if computation_time == 0:
      raise ValueError(f'The computation time was zero (computation_time = {computation_time}). This typically indicates that Scarab did not find a PARAMS.in file.')
    data = {
            "index": int(dir_index), 
            "trace_dir": trace_dir, 
            "computation_time": computation_time
            }
    print(f"Finished Scarab simulation. Time to compute controller: {computation_time} seconds.")
    # return data
    return float(computation_time)


  def _clean_statistics(self):
    """
    Delete the statitics files that were created. This is designed to be used in test cases to make sure we reset the original state of the file system (more or less).
    """
    for directory in self._get_trace_directories():
      try:
        os.remove(directory + '/core.stat.0.csv')
      except FileNotFoundError:
        pass

class MockTracesToComputationTimesProcessor(TracesToComputationTimesProcessor):

  def __init__(self, *args, **kwargs):
    self.delay_list = kwargs.pop('delays')
    self.number_of_steps = len(self.delay_list)
    for delay in self.delay_list:
      if delay is None:
        raise ValueError(f'One of the provide delays was None! delays: {self.delay_list}')
        
    super().__init__(*args, **kwargs)

  def get_computation_time_from_trace(self, dir_index):
    """
    Override the superclass' simulate trace to not use Scarab to get the computation times.
    """
    trace_dir = self._get_trace_directory_from_index(dir_index)
      
    # We don't need PARAMS.in in the trace directory for the sake of the
    # Mock delays, but it is usful to have it here to check that copying it works
    # as expected.
    shutil.copyfile(os.path.join(trace_dir, '..', 'PARAMS.generated'), os.path.join(trace_dir, 'PARAMS.in'))
    assertFileExists(os.path.join(trace_dir, 'PARAMS.in'))

    # Read the fake delay data.
    delay = self.delay_list[dir_index]
    if delay == None:
      raise ValueError(f'The delay at index {dir_index} has already been read!')
    self.delay_list[dir_index] = None

    # data = {
    #         "index": dir_index, 
    #         "trace_dir": self._get_trace_directory_from_index(dir_index), 
    #         "computation_time": delay
    #         }
    # return data
    return delay

# class Scarab:
# 
#     def __init__(self, 
#                  traces_path=None, 
#                  dynamorio_root=None, 
#                  scarab_root=None, 
#                  scarab_out_path=None, 
#                  subprocess_run=run):
#         if traces_path:
#           self.TRACES_PATH = traces_path
#         else:
#           self.TRACES_PATH = os.environ['TRACES_PATH']
#           
#         if dynamorio_root:
#           self.DYNAMORIO_ROOT = dynamorio_root
#         else:
#           self.DYNAMORIO_ROOT = os.environ['DYNAMORIO_ROOT']
# 
#         if scarab_root:
#           self.SCARAB_ROOT = scarab_root
#         else:
#           self.SCARAB_ROOT = os.environ['SCARAB_ROOT']
# 
#         if scarab_out_path:
#           self.SCARAB_OUT_PATH = scarab_out_path
#         else:
#           self.SCARAB_OUT_PATH = os.environ['SCARAB_OUT_PATH']
#             
#         self.subprocess_run = subprocess_run
#         self.verbose = False
# 
#     def clear_trace_dir(self):
#         folder = self.TRACES_PATH
# 
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             if os.path.isdir(file_path):
#                 # Check that we are deleting the right directories.
#                 if not file_path.endswith('.dir'):
#                     raise ValueError('Unexpected path: ' + file_path)
#                 # Delete directory and subtree.
#                 shutil.rmtree(file_path)
# 
#     def trace_cmd(self, cmd, args=[]):
#         if self.verbose:
#             print('\n=== Starting DyanmRIO ===')
# 
#         self.clear_trace_dir()
# 
#         # Starting DynamaRIO
#         dyanmrio_cmd = ["drrun", \
#                         "-root", self.DYNAMORIO_ROOT, \
#                         # Not sure what this option does.
#                         "-t",  "drcachesim", \
#                         # Store trace files for offline analysis
#                         "-offline", \
#                         "-outdir", self.TRACES_PATH, \
#                         # Increase the verbosity so that the trace file 
#                         # path is printed, allowing us to parse it. 
#                         "-verbose", "4", \
#                         # Tell DynamoRIO which command to trace
#                         "--", cmd]
#         result, stdout, stderr = self.subprocess_run(dyanmrio_cmd, args)
#         if result.returncode: # is not 0
#             # print("result:" + str(result))
#             raise ValueError("Subprocess failed. stderr=\n" + stderr)
#         
#         log_dir_regex_result = re.search(log_dir_regex, stderr);
#         if not log_dir_regex_result:
#             raise ValueError("No log directory found in output. stderr=\n" + stderr)
#         log_dir = log_dir_regex_result.group(1)
#         # trace_file = re.search(trace_file_regex, stderr).group(1)
# 
#         # print("Log directory: " + log_dir)
#         # print("Trace file: " + trace_file)
# 
#         # Split the path, parts to greet,
#         trace_directory = "/".join(log_dir.split("/")[:-1])
# 
#         # print("Trace directory: " + trace_directory)
#         # print("")
# 
#         # Run "run_portabilize_trace.sh" to copy binary dependencies to a local directory.
#         # This produces two new 
#         self.subprocess_run(['bash', self.SCARAB_ROOT + "/utils/memtrace/run_portabilize_trace.sh"], cwd=self.TRACES_PATH)
# 
#         # print(f'Contents of {self.TRACES_PATH} after portablizing')
#         # subprocess.run(['ls', trace_directory], capture_output=True)
#         # print()
# 
#         # print("Return value: trace_directory="+trace_directory)
#         # print('stdout:')
#         # print(stdout)
#         # print('stderr:')
#         # print(stderr)
#         return trace_directory,stdout,stderr

#     def get_computation_time_from_trace_with_scarab(self, trace_dir):
#         if self.verbose:
#             print('\n=== Starting Scarab ===')
# 
#         raise NotImplementedError(f'This seems to be deprecated??')
#         
#         scarab_cmd = ["scarab", \
#                 "--output_dir", self.SCARAB_OUT_PATH, \
#                 "--frontend", "memtrace", \
#                 "--inst_limit", "150000000", \
#                 "--fetch_off_path_ops", "0", \
#                 f"--cbp_trace_r0={trace_dir}/trace",\
#                 f"--memtrace_modules_log={trace_dir}/bin"
#                 ]
#           
#         # run_shell_cmd(scarab_cmd, workind_dir=params_in_dir)
#         result, stdout, stderr = self.subprocess_run(scarab_cmd)
#         
#         # Restore working directory.
#         os.chdir(initial_wd)
#         
#         regex_result = re.search(time_regex, stdout)
#         if not regex_result:
#             raise ValueError(f"Time regular expresion not found in Scarab output. Standard out:\n{stdout}\nStandard error:\n{stderr}")
# 
#         # print(result)
#         femtoseconds_to_run = float(regex_result.group(1))
#         # print("femtoseconds_to_run: " + femtoseconds_to_run)
#         
#         return femtoseconds_to_run
    # 
    # # Trace and simulate a command using DyanmRIO and Scarab. 
    # # The return value is the (simulation) time in femptoseconds for the command to be executed on the simulated platform.
    # def simulate(self, cmd, args=[]):
    #     trace_dir,stdout,stderr = self.trace_cmd(cmd, args)
    #     time_in_fs = self.get_computation_time_from_trace_with_scarab(trace_dir)
    #     # print(f"time: {time_in_fs:0.6} seconds ({time_in_fs*SECONDS_PER_MICROSECOND:0.4} microseconds).")
    #     data = ScarabData(time_in_fs, stdout, stderr)
    #     return data


# if __name__ == "__main__":
#     scarab = Scarab();
#     # trace_dir = scarab.trace_cmd('touch', '~/mytestfile.txt')
#     # scarab.get_computation_time_from_trace_with_scarab("/workspaces/ros-docker/drmemtrace.python3.8.122943.1703.dir")
#     print(os.getcwd())
#     data = scarab.simulate('./3x3_proportional_controller', ['1.43'])
#     print(os.getcwd())
#     data = scarab.simulate('./3x3_proportional_controller', ['1.43'])
#     print(f"time: {data.simulated_time_seconds:0.6} seconds ({data.simulated_time_microseconds:0.4} microseconds).")
#     print('Standard output:')
#     print(data.cmd_stdout)