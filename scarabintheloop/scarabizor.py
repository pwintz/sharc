import os
import subprocess
import re
import shutil
import time

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

class ScarabStatsReader:
  
  def __init__(self, stats_dir_path):
      self.stats_dir_path = stats_dir_path
      self.stat_regex = re.compile(r"\s*(?P<key>[\w|\_| ]+),\s*(?P<value>\d+\.?\d*).*")

  def getStatsFilePath(self, k) -> os.PathLike:
    file_name = f"core.stat.0.csv.roi.{k}"
    file_path = os.path.join(self.stats_dir_path, file_name)
    return file_path

  def waitForStatsFile(self,k):
    file_path = self.getStatsFilePath(k)
    # if not file_path:
    #    raise
    while not os.path.exists(file_path):
      time.sleep(0.01)

  def readStatistic(self, k: int, stat_key: str): 
    with open(self.getStatsFilePath(k), 'r') as stats_file:
        for line in stats_file:
          regex_match = self.stat_regex.match(line)
          if regex_match and stat_key == regex_match.groupdict()['key']:
            value = regex_match.groupdict()['value']
            try: 
              return int(value)
            except ValueError:
              pass
            return float
        raise ValueError(f"key: \"{stat_key}\" was not found in stats file.")
              
  def readCyclesCount(self, k: int):  
    return self.readStatistic(k, "NODE_CYCLE_count")

  def readInstructionCount(self, k: int):  
    return self.readStatistic(k, "NODE_INST_COUNT_count")

  def readTime(self, k: int):  
    time_in_femtosecs = self.readStatistic(k, "EXECUTION_TIME_count")
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