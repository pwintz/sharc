import os
import subprocess
import re
import shutil
import time
from typing import List, Set, Dict, Tuple

params_in_dir = 'docker_user_home'
log_dir_regex = re.compile(r"\nLog directory is (.*?)\n")
trace_file_regex = re.compile(r"\nCreated thread trace file (.*?)\n")

# Regex pattern for extracting the time from the Scarab output. 
time_regex = re.compile(r"time:(\d+)\s*--")

### COSTANTS ###
SECONDS_PER_FEMTOSECOND = 10**(-15)
SECONDS_PER_MICROSECOND = 10**(6)
MICROSECONDS_PER_FEMTOSECOND = 10**(6-15)
MICROSECONDS_PER_SECOND = 10**(-15)
    
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


class ScarabPARAMSReader:
  
  def __init__(self, sim_dir=None):
    if sim_dir:
      if not sim_dir.endswith("/"):
        sim_dir += "/"
        
      self.params_in_file_path = sim_dir + 'PARAMS.in'
      self.params_out_file_path = sim_dir + 'PARAMS.out'

    self.param_keys_to_save_in_data_out = ["chip_cycle_time", "l1_size", "dcache_size", "icache_size", "decode_cycles"]
    # Regex to find text in the form of "--name value"
    self.param_regex_pattern = re.compile(r"--(?P<param_name>[\w|\_]+)\s*(?P<param_value>\d+).*")

  def params_in_to_dictionary(self):
    return self.params_file_to_dictionary(self.params_in_file_path)

  def params_out_to_dictionary(self):
    return self.params_file_to_dictionary(self.params_out_file_path)

  def params_file_to_dictionary(self, filename):
    params_lines = self.read_params_file(filename)
    return self.params_lines_to_dict(params_lines)

  def read_params_file(self, filename):
    with open(filename) as params_file:
      params_lines = params_file.readlines()
    return params_lines

  def params_lines_to_dict(self, params_lines: List[str]):
    param_dict = {}
    for line in params_lines: 
      regex_match = self.param_regex_pattern.match(line)
      if regex_match:
        param_name = regex_match.groupdict()['param_name']
        param_value = regex_match.groupdict()['param_value']
        if param_name in self.param_keys_to_save_in_data_out:
          param_dict[param_name] = int(param_value)
    if not param_dict:
      raise ValueError(f'param_dict is expected to have values, but instead param_dict={param_dict}')
      
    return param_dict

  # TODO: Write tests for this.
  def changeParamsValue(self,PARAM_lines: List[str], key, value):
    """
    Search through PARAM_lines and modify it in-place by updating the value for the given key. 
    If the key is not found, then an error is raised.
    """
    for line_num, line in enumerate(PARAM_lines):
      regex_match = self.param_regex_pattern.match(line)
      if not regex_match:
        continue
      if key == regex_match.groupdict()['param_name']:
        # If the regex matches, then we replace the line.
        new_line_text = self.param_str_fmt.format(key, value)
        PARAM_lines[line_num] = new_line_text
        # print(f"Replaced line number {line_num} with {new_line_text}")
        return PARAM_lines
      
    # After looping through all of the lines, we didn't find the key.
    raise ValueError(f"Key \"{key}\" was not found in PARAM file lines.")

  def create_patched_PARAMS_file(sim_config: dict, PARAMS_src_filename: str, PARAMS_out_filename: str):
    """
    Read the baseline PARAMS file at the location given by the PARAMS_base_file option in the simulation configuration (sim_config).
    Then, modify the values for keys listed in PARAMS_file_keys to values taken from sim_config. 
    Write the resulting PARAMS data to a file at PARAMS_out_filename in the simulation directory (sim_dir).
    Returns the absolute path to the PARAMS file.
    """
    print(f'Creating chip parameter file.')
    print(f'\tSource: {PARAMS_src_filename}.')
    print(f'\tOutput: {PARAMS_out_filename}.')
    with open(PARAMS_src_filename) as params_out_file:
      PARAM_file_lines = params_out_file.readlines()
    
    for (key, value) in sim_config.items():
      if key in PARAMS_file_keys:
        PARAM_file_lines = changeParamsValue(PARAM_file_lines, key, value)

    # Create PARAMS file with the values from the base file modified
    # based on the values in sim_config.
    with open(PARAMS_out_filename, 'w') as params_out_file:
      params_out_file.writelines(PARAM_file_lines)

class ScarabStatsReader:
  
  def __init__(self, stats_dir_path, is_using_roi=True):
      self.stats_dir_path = stats_dir_path
      # Create a RegEx to match a line in the form
      #   EXECUTION_TIME      294870700000000  <anything here is ignored>    
      # or
      #   EXECUTION_TIME_count,      294870700000000  <anything here is ignored>    
      # and create two groups, key="EXECUTION_TIME" and value="294870700000000".
      self.stat_regex = re.compile(r"\s*(?P<key>[\w\_]+)(?:_count,)?\s*(?P<value>[\d.]+).*")
      
      # Create a RegEx to match a line in the form
      #   EXECUTION_TIME,             294870700000000  <anything here is ignored>             
      # and create two groups, key="EXECUTION_TIME" and value="294870700000000".
      # self.stat_regex = re.compile(r"\s*(?P<key>[\w|\_| ]+),\s*(?P<value>\d+\.?\d*).*")
      self.is_using_roi = is_using_roi

  def getStatsFilePath(self, k) -> os.PathLike:
    if self.is_using_roi:
      file_name = f"core.stat.0.csv.roi.{k}"
    else:
      file_name = f"core.stat.{k}.out"
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
        for line in stats_file:
          regex_match = self.stat_regex.match(line)
          if regex_match and stat_key == regex_match.groupdict()['key']:
            value = regex_match.groupdict()['value']
            # Cast value to either int or float.
            try: 
              return int(value)
            except ValueError:
              pass
            return float(value)
        raise ValueError(f"key: \"{stat_key}\" was not found in stats file {file_path}.")
              
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

class Scarab:

    def __init__(self, 
                 traces_path=None, 
                 dynamorio_root=None, 
                 scarab_root=None, 
                 scarab_out_path=None, 
                 subprocess_run=run):
        if traces_path:
          self.TRACES_PATH = traces_path
        else:
          self.TRACES_PATH = os.environ['TRACES_PATH']
          
        if dynamorio_root:
          self.DYNAMORIO_ROOT = dynamorio_root
        else:
          self.DYNAMORIO_ROOT = os.environ['DYNAMORIO_ROOT']

        if scarab_root:
          self.SCARAB_ROOT = scarab_root
        else:
          self.SCARAB_ROOT = os.environ['SCARAB_ROOT']

        if scarab_out_path:
          self.SCARAB_OUT_PATH = scarab_out_path
        else:
          self.SCARAB_OUT_PATH = os.environ['SCARAB_OUT_PATH']
            
        self.subprocess_run = subprocess_run
        self.verbose = False

    def clear_trace_dir(self):
        folder = self.TRACES_PATH

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isdir(file_path):
                # Check that we are deleting the right directories.
                if not file_path.endswith('.dir'):
                    raise ValueError('Unexpected path: ' + file_path)
                # Delete directory and subtree.
                shutil.rmtree(file_path)

    def trace_cmd(self, cmd, args=[]):
        if self.verbose:
            print('\n=== Starting DyanmRIO ===')

        self.clear_trace_dir()

        # Starting DynamaRIO
        dyanmrio_cmd = ["drrun", \
                        "-root", self.DYNAMORIO_ROOT, \
                        # Not sure what this option does.
                        "-t",  "drcachesim", \
                        # Store trace files for offline analysis
                        "-offline", \
                        "-outdir", self.TRACES_PATH, \
                        # Increase the verbosity so that the trace file 
                        # path is printed, allowing us to parse it. 
                        "-verbose", "4", \
                        # Tell DynamoRIO which command to trace
                        "--", cmd]
        result, stdout, stderr = self.subprocess_run(dyanmrio_cmd, args)
        if result.returncode: # is not 0
            # print("result:" + str(result))
            raise ValueError("Subprocess failed. stderr=\n" + stderr)
        
        log_dir_regex_result = re.search(log_dir_regex, stderr);
        if not log_dir_regex_result:
            raise ValueError("No log directory found in output. stderr=\n" + stderr)
        log_dir = log_dir_regex_result.group(1)
        # trace_file = re.search(trace_file_regex, stderr).group(1)

        # print("Log directory: " + log_dir)
        # print("Trace file: " + trace_file)

        # Split the path, parts to greet,
        trace_directory = "/".join(log_dir.split("/")[:-1])

        # print("Trace directory: " + trace_directory)
        # print("")

        # Run "run_portabilize_trace.sh" to copy binary dependencies to a local directory.
        # This produces two new 
        self.subprocess_run(['bash', self.SCARAB_ROOT + "/utils/memtrace/run_portabilize_trace.sh"], cwd=self.TRACES_PATH)

        # print(f'Contents of {self.TRACES_PATH} after portablizing')
        # subprocess.run(['ls', trace_directory], capture_output=True)
        # print()

        # print("Return value: trace_directory="+trace_directory)
        # print('stdout:')
        # print(stdout)
        # print('stderr:')
        # print(stderr)
        return trace_directory,stdout,stderr

    def simulate_trace_with_scarab(self, trace_dir):
        if self.verbose:
            print('\n=== Starting Scarab ===')

        # Change the working directory to the place where the Scarab parameters file is stored.
        initial_wd = os.getcwd()
        os.chdir(params_in_dir)
        scarab_cmd = ["scarab", \
                "--output_dir", self.SCARAB_OUT_PATH, \
                "--frontend", "memtrace", \
                "--inst_limit", "150000000", \
                "--fetch_off_path_ops", "0", \
                f"--cbp_trace_r0={trace_dir}/trace",\
                f"--memtrace_modules_log={trace_dir}/bin"
                ]
        result, stdout, stderr = self.subprocess_run(scarab_cmd)
        
        # Restore working directory.
        os.chdir(initial_wd)

        # print(str(stdout))

        regex_result = re.search(time_regex, stdout)
        if not regex_result:
            raise ValueError(f"Time regular expresion not found in Scarab output. Standard out:\n{stdout}\nStandard error:\n{stderr}")

        # print(result)
        femtoseconds_to_run = float(regex_result.group(1))
        # print("femtoseconds_to_run: " + femtoseconds_to_run)
        
        return femtoseconds_to_run
    
    # Trace and simulate a command using DyanmRIO and Scarab. 
    # The return value is the (simulation) time in femptoseconds for the command to be executed on the simulated platform.
    def simulate(self, cmd, args=[]):
        trace_dir,stdout,stderr = self.trace_cmd(cmd, args)
        time_in_fs = self.simulate_trace_with_scarab(trace_dir)
        # print(f"time: {time_in_fs:0.6} seconds ({time_in_fs*SECONDS_PER_MICROSECOND:0.4} microseconds).")
        data = ScarabData(time_in_fs, stdout, stderr)
        return data


# if __name__ == "__main__":
#     scarab = Scarab();
#     # trace_dir = scarab.trace_cmd('touch', '~/mytestfile.txt')
#     # scarab.simulate_trace_with_scarab("/workspaces/ros-docker/drmemtrace.python3.8.122943.1703.dir")
#     print(os.getcwd())
#     data = scarab.simulate('./3x3_proportional_controller', ['1.43'])
#     print(os.getcwd())
#     data = scarab.simulate('./3x3_proportional_controller', ['1.43'])
#     print(f"time: {data.simulated_time_seconds:0.6} seconds ({data.simulated_time_microseconds:0.4} microseconds).")
#     print('Standard output:')
#     print(data.cmd_stdout)