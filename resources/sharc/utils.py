"""
This package provides several utility functions that are used by SHARC.
"""

import io
import math
import os
import sys
import re
import json
import subprocess
import copy

import importlib.util
import numpy as np

from typing import List, Set, Dict, Tuple
from typing import Union

# Import contextmanager to allow defining commands to be used to create "with" blocks.
from contextlib import contextmanager 

import sharc.debug_levels as debug_levels
# Level 1: Print shell commands that are called.
# Level 2: Also print the current working directory.

try:
    assert False
    raise Exception('Python assertions are not working. This tool uses Python assertions to check correctness. Possible causes are running with the "-O" flag or running a precompiled (".pyo" or ".pyc") module. We do not recommend removing this check, but the code should work without assertions.')
except AssertionError:
    pass

def assertFileExists(path:str, help_msg=None):
  if not os.path.exists(path):
    err_msg = f'Expected {path} to exist but it does not. '
    if os.path.abspath(path) != path:
      err_msg += f'The absolute path is {os.path.abspath(path)}'
    if help_msg:
      err_msg += '\n' + help_msg
    raise IOError(err_msg)

def printIndented(string_to_print:str, indent: int=1):
  indent_str = '\t' * indent
  indented_line_break = "\n" + indent_str
  string_to_print = string_to_print.replace('\n', indented_line_break)
  print(indent_str + string_to_print)

def readJson(filename: str) -> Union[Dict,List]:
  assertFileExists(filename)
  try:
    with open(filename, 'r') as json_file:
      json_data = json.load(json_file)
      return json_data
  except json.decoder.JSONDecodeError as err:
    raise ValueError(f'An error occured while parsing {filename}.') from err

# # Define a custom JSON encoder for the class
# class MyJsonEncoder(json.JSONEncoder):
#   def default(self, obj):
#     if isinstance(obj, np.ndarray):
#       return 
#     if isinstance(obj, MyClass):
#       # Convert the object to a dictionary
#       return obj.__dict__
#     # For other objects, use the default serialization
#     return super().default(obj)


# # Define a custom JSON encoder for the class
# class MyJsonEncoder(json.JSONEncoder):
#   def default(self, obj):
#     if isinstance(obj, np.ndarray):
#       return column_vec_to_list(obj)
#     elif str(type(obj)) == "TimeStepSeries":
#       # Convert the object to a dictionary
#       return obj.__dict__
#     # For other objects, use the default serialization
#     return super().default(obj)

def _create_json_string(json_data):
  # Use "default=vars" to automatically convert many objects to JSON data.
  try:
    def encode_objs(obj):
      if hasattr(obj, "to_dict"):
        return obj.to_dict()
      if hasattr(obj, "__dict__"):
        return vars(obj)
      elif isinstance(obj, np.ndarray):
        return column_vec_to_list(obj)
        # return '[' + nump_vec_to_csv_string(obj) + ']'
      else:
        return repr(obj)
    json_string = json.dumps(json_data, indent=2, default=encode_objs)
    json_string = _remove_linebreaks_in_json_dump_between_number_list_items(json_string)
    return json_string
  except Exception as err:
    raise ValueError(f"Failed to create JSON string for:\n{json_data}") from err

def writeJson(filename: str, json_data: Union[Dict,List], label:str=None):
  """
  Write a dictionary to a file in JSON format. 
  If 'label' is given, then the path is printed to the stdout with the given label.
  """
  json_string = _create_json_string(json_data)

  with open(filename, 'w') as file:
    file.write(json_string)
  if label:
    print(f'{label}:\n\t' + os.path.abspath(filename))
    

def printJson(label: str, json_data: Union[Dict,List]):
  """
  Pretty-print JSON to standard out. Loses percision, so do not parse this output to recove data!
  """
  try:
    json_string = _create_json_string(json_data)
    
    max_replacements = 0
    double_regex = r"(-?\d+(?:\.\d{1,2})?)\d*"
    truncated_double_subst = "\\1"
    json_string = re.sub(double_regex, truncated_double_subst, json_string, max_replacements, re.MULTILINE)
    print(f"{label}:\n{json_string}")
  except Exception as err:
    raise ValueError(f'ERROR: Could not print as json: label="{label}", json_data={json_data}.') from err

def _remove_linebreaks_in_json_dump_between_number_list_items(json_string:str):
  max_replacements = 0 # As many as found.
  
  # Change the starts of a list of strings to be on a single line.
  first_item_regex = r"\[\s*((?:-?\d+(?:\.\d+)?))"
  first_item_subst = "[\\1"
  json_string = re.sub(first_item_regex, first_item_subst, json_string, max_replacements, re.MULTILINE)

  # Remove the line breaks after between item in lists of numbers. 
  list_items_regex = r"(-?\d+(?:\.\d+)?)(,)\s*(?=-?\d+(?:\.\d+)?)"
  list_items_subst = "\\1\\2 "
  json_string = re.sub(list_items_regex, list_items_subst, json_string, max_replacements, re.MULTILINE)
  
  # Remove the line break after the last item in a list of numbers.
  last_item_regex = r"(-?\d+(?:\.\d+)?)\s*\]"
  last_item_subst = "\\1]"
  json_string = re.sub(last_item_regex, last_item_subst, json_string, max_replacements, re.MULTILINE)

  return json_string

def patch_dictionary(base: dict, patch: dict) -> dict:
  """
  Create a new config dictionary by copying the the base dictionary and replacing any keys with the values in the patch dictionary. The base and patch dictionaries are not modified. All keys in the patch dictionary must already exist in base. If any value is itself a dictionary, then patching is done recursively.
  The returned value should have exactly the same hierarchy of keys 
  """
  # Make a copy of the base data so that modifications the base dictionary is not modified.
  patched = copy.deepcopy(base)

  for (key, value) in patch.items():
    if not key in base: 
      raise ValueError(f'The key "{key}" was given in the patch dictionary but not present in the base dictionary.')

    # Check that the dictionary-ness of the values match.
    if isinstance(value, dict) != isinstance(base[key], dict):
      raise ValueError(f'For each key, the value in the patch and base dictionary must either both be dictionaries or both not dictionaries, but for key="{key}", the type of patch["{key}"] is {type(value)} whereas the type of base["key"] {type(base[key])}.')

    if isinstance(value, dict):
      # If the value is a dictionary, then we need to recursively patch it, so that we don't just replace the whole dictionary resulting in possibly missing values, or 
      value = patch_dictionary(base[key], patch[key]) 

    patched[key] = value
    
  return patched

# TODO: Define a function for finding any numpy arrays in a dictionary and converting them to lists.
# def sanitize_np_arrays_in_dictionary(dictionary):
#   dictionary = copy.deepcopy(dictionary)
#   

def assert_is_column_vector(array: np.ndarray, label="array") -> None:
  assert isinstance(array, np.ndarray), f'array={array} is not a np.ndarray'
  assert array.ndim == 2, \
    f'Expected {label}={array} to be two dimensional, but instead {label}.ndim={array.ndim}'
  assert array.shape[1] == 1, \
    f'array must be a column vector. Instead it was {label}.shape={array.shape}'

#################################
#####  PIPE READER CLASSES ######
#################################
class PipeReader:
  filename: str
  file: io.TextIOWrapper

  def __init__(self, filename: str):
    self.filename = filename
    self.file = None
    try:
      os.mkfifo(filename)
    except FileExistsError: 
      raise FileExistsError(f"A pipe at {filename} cannot be created because it already exists.")
    
  @property
  def is_open(self):
    return not self.is_closed
  
  @property
  def is_closed(self):
    return self.file is None or self.file.closed

  def open(self):
    assertFileExists(self.filename)
    assert self.is_closed, 'File must not already be open.'
    if debug_levels.debug_interfile_communication_level >= 1:
      print(f"About to open a Python reader for {os.path.basename(self.filename)}...")

    # Open file as read-only.
    self.file = open(self.filename, 'r', buffering=1)

    if debug_levels.debug_interfile_communication_level >= 1:
      print(f"                ...the reader for {os.path.basename(self.filename)} is now open.")

  def close(self):
    if self.is_closed:
      if debug_levels.debug_interfile_communication_level >= 1:
        print(f'{self} is already closed.')
      return
    if debug_levels.debug_interfile_communication_level >= 1:
      print(f'Closing {self}')
    self.file.close()
    self.file = None

  def read(self):
    assert self.is_open, 'File must be open.'
    self._wait_for_pipe_to_be_nonempty()
    input_line = self._waitForLineFromFile()
    # input_line = PipeReader.checkAndStripInputLoopNumber(input_line)
    return input_line

  def __repr__(self):
    return f'PipeReader(file={self.filename}. Is open? {self.is_open})'

  def _waitForLineFromFile(self):
    if debug_levels.debug_interfile_communication_level >= 1:
      print(f"Waiting for input_line from {self.file.name}.")

    input_line = ""
    while not input_line.endswith("\n"):
      #!! Caution, printing out everytime we read a line will cause massive log 
      #!! files that can (will) grind my system to a hault.
      input_line += self.file.readline()

    if debug_levels.debug_interfile_communication_level >= 1:
      print(f'Received input_line from {os.path.basename(self.filename)}: {repr(input_line)}.')
    return input_line

  def _wait_for_pipe_to_be_nonempty(self):

    # Wait for the pipe.
    stat_info = os.stat(self.filename)
  
    # Check if the file size is greater than zero (some data has been written)
    return stat_info.st_size > 0

  @staticmethod
  def checkAndStripInputLoopNumber(input_line):
    """ Check that an input line, formatted as "Loop <k>: <data>" 
    has the expected value of k (given by the argument "expected_k") """

    split_input_line = input_line.split(':')
    loop_input_str = split_input_line[0]

    # Check that the input line is in the expected format. 
    if not loop_input_str.startswith('Loop '):
      raise ValueError(f'The input_line "{input_line}" did not start with a loop label.')
    
    # Extract the loop number and convert it to an integer.
    input_loop_number = int(loop_input_str[len('Loop:'):])

    # if input_loop_number != expected_k:
    #   # If the first piece of the input line doesn't give the correct loop number.
    #   raise ValueError(f'The input_loop_number="{input_loop_number}" does not match expected_k={expected_k}.')
    #   
    # Return the part of the input line that doesn't contain the loop info.
    return split_input_line[1]

class PipeFloatReader(PipeReader):
  
  def __init__(self, filename: str):
    super().__init__(filename)

  def read(self):
    return float(super().read())

class PipeVectorReader(PipeReader):

  def __init__(self, filename: str):
    super().__init__(filename)

  def read(self):
    return PipeVectorReader.convertStringToVector(super().read())
  
  @staticmethod
  def convertStringToVector(vector_str: str):
    vector_str_list = vector_str.split(',') #.strip().split("\t")

    # Convert the list of strings to a list of floats.
    chars_to_strip = ' []\n'
    v = np.array([[np.float64(x.strip(chars_to_strip)),] for x in vector_str_list])
    
    if debug_levels.debug_interfile_communication_level >= 3:
      print('convertStringToVector():')
      printIndented('vector_str:', 1)
      printIndented(repr(vector_str), 1)
      printIndented('v:', 1)
      printIndented(repr(v), 1)

    return v

class PipeJsonReader(PipeReader):
  
  def __init__(self, filename: str):
    super().__init__(filename)

  def read(self) -> dict:
    line = super().read()
    json_dict = json.loads(line)
    return json_dict

##########################
#####  PIPE WRITER  ######
##########################
class PipeWriter:
  filename: str
  file: io.TextIOWrapper
  
  def __init__(self, filename: str):
    self.filename = filename
    if debug_levels.debug_interfile_communication_level >= 1:
      print(f"Opening writer for {filename}....")
    self.file = None
    os.mkfifo(filename)
  
  @property
  def is_open(self):
    return self.file is not None
  
  @property
  def is_closed(self):
    return self.file is None
  
  def open(self):
    assert self.file is None, "File must not be open to open it."
    self.file = open(self.filename, 'w', buffering=1)

  def close(self):
    if self.file is None:
      if debug_levels.debug_interfile_communication_level >= 1:
        print(f'The file "{self.filename}" is already closed.')
      return
    if debug_levels.debug_interfile_communication_level >= 1:
      print(f'Closing {self}')
    # assert self.file is not None, "File must be open to close it."
    try:
      self.file.write("END OF PIPE\n")
      self.file.close()
    except:
      pass
    self.file = None

  def write(self, value):
    assert self.file is not None, "File must be open to write to it."
      
    if debug_levels.debug_interfile_communication_level >= 2:
      print(f'Writing "{value}" to {self.filename}')

    self.file.write(value + "\n", )

  def __repr__(self):
    return f'PipeWriter(file={self.filename}. Is open? {self.file is not None})'

class PipeVectorWriter(PipeWriter):

  def write(self, vec: np.ndarray):
    assert isinstance(vec, np.ndarray), f'Expected vec to have type np.ndarray. Instead it was {type(vec)}.'
    vec_string = nump_vec_to_csv_string(vec)
    super().write(vec_string)

class PipeFloatWriter(PipeWriter):
  def write(self, val: float):
    assert isinstance(val, (float, int)), f'Expected value to be a float or int. Instead it was {type(val)}.'
    val_string = f"{val:.8g}"
    assert val_string, f'val_string={val_string} must not be empty.'
    super().write(val_string)

class PipeIntWriter(PipeWriter):
  def write(self, val: int):
    assert isinstance(val, int), f'Expected value to be an int. Instead it was {type(val)}.'
    val_string = f"{val:d}"
    assert val_string, f'val_string={val_string} must not be empty.'
    super().write(val_string)

#############################
#####  NUMPY FUNCTIONS ######
#############################
def column_vec_to_list(array: np.ndarray) -> List[float]:
  assert_is_column_vector(array)
  vec_as_list = array.transpose().tolist()[0]
  assert isinstance(vec_as_list, list), f'vec_as_list={vec_as_list} must be a list. The original array was "{array}". Shape: {array.shape}'
  return vec_as_list

def list_to_column_vec(list_vec: List[float]) -> np.ndarray:
  return np.array(list_vec).reshape(-1, 1)

def nump_vec_to_csv_string(array: np.ndarray) -> str:
  if not isinstance(array, np.ndarray):
    raise ValueError(f'Expected array={array} to be an np.array but instead it is a {type(array)}')
  if array.ndim == 2:
    assert array.shape[0] == 1 or array.shape[1] == 1, f'Array must only have one dimension that is not equal to 1. Its dimension is {array.ndim} and shape is {array.shape}.' 
    array = array.flatten()
  assert array.ndim == 1, f'After flattening, array must be 1-dimensional.'
  string = ', '.join(map(str, array))
  return string

# def checkBatchConfig(batch_config):
#   pass
#   simulation_label = batch_config["simulation_label"]
#   max_time_steps = batch_config["max_time_steps"]
#   x0 = batch_config["x0"]
#   u0 = batch_config["u0"]
#   first_time_index = batch_config["first_time_index"]
#   # last_time_index = batch_config["last_time_index"]
# 
#   max_time_steps_in_batch = batch_config["max_time_steps"]
#   n_time_indices_in_bath = max_time_steps_in_batch + 1
#   simulation_dir = batch_config["simulation_dir"]
# 
#   if last_time_index - first_time_index != max_time_steps_in_batch:
#     raise ValueError(f"last_time_index - first_time_index = {last_time_index - first_time_index} != max_time_steps_in_batch = {max_time_steps_in_batch}")
# 
#   if max_time_steps_in_batch < 0:
#     raise ValueError(f"A negative max_time_steps_in_batch was given: {max_time_steps_in_batch}")
# 
#   if batch_config["time_indices"][0] != first_time_index or batch_config["time_indices"][-1] != last_time_index:
#     raise ValueError(f"time_indices doen't align.")
#     

# def checkSimulationData(batch_simulation_data, max_time_steps):
#   n_time_indices = len(batch_simulation_data['time_indices']) #max_time_steps + 1
# 
#   def assert_len_equals_n_time_indices(name:str):
#     array = batch_simulation_data[name]
#     if len(array) != n_time_indices:
#       raise ValueError(f'the length of {name} ({len(array)}) is not equal to the number of time indices ({n_time_indices}). The time indices are {batch_simulation_data["time_indices"]}')
# 
#   # There is one fewer time steps than time indices because each step is a step between indices.
#   def assert_len_equals_n_time_steps(name:str):
#     array = batch_simulation_data[name]
#     if len(array) != n_time_indices - 1:
#       raise ValueError(f"the length of {name} ({len(array)}) is not equal to the number of time steps ({n_time_indices - 1}).")
# 
#   # The control values applied. The first entry is the u0 previously computed. Subsequent entries are the computed values. The control u[i] is applied from t[i] to t[i+1], with u[-1] not applied during this batch. 
#   assert_len_equals_n_time_indices("x")
#   assert_len_equals_n_time_indices("u")
#   assert_len_equals_n_time_indices("t")
#   assert_len_equals_n_time_steps("t_delay")


def printHeader1(header: str):
  """ 
  Print a nicely formatted header box (level 1) to the console.
  """
  header = "|| " + header + " ||"
  print()
  print('=' * len(header))
  print(header)
  print('=' * len(header))


def printHeader2(header: str):
  """ 
  Print a nicely formatted header box (level 2) to the console.
  """
  header = "| " + header + " |"
  print()
  print('-' * len(header))
  print(header)
  print('-' * len(header))

def printHeader(header: str, level=1):
  """ 
  Print a nicely formatted header box to the console.
  """
  if level==1:
    printHeader1(header)
  elif level==2:
    printHeader2(header)
  else:
    raise ValueError(f"Unexpected value of level = {level}")

@contextmanager
def openLog(filename, header_text=None):
  if not header_text:
    header_text=f"Log: {filename}"
  log_path = os.path.abspath(filename)
  if debug_levels.debug_program_flow_level >= 2:
    print(f"Opening log: {log_path}")
  if not os.path.isdir(os.path.dirname(log_path)):
    raise ValueError(f'The parent of the given path is not a directory: {log_path}')

  try:
    log_file = open(filename, 'w+', buffering=1)
    # Write a header for the log file.
    log_file.write(f'===={"="*len(header_text)}==== \n')
    log_file.write(f'=== {header_text} === \n')
    log_file.write(f'===={"="*len(header_text)}==== \n')
    yield log_file
  finally:
    # Clean up
    log_file.close()

# @contextmanager
# def in_working_dir(path):
#   """
#   Temporarily change the working directory within a "with" block.
#   Usage:
# 
#     with cwd('/home/bwayne'):
#       <do stuff in the /home/bwayne/ directory>
# 
#   """
#   oldpwd = os.getcwd()
#   
#   assert os.getcwd() == '/dev-workspace/integration_tests'
#   try:
#     os.chdir(path)
#   except FileNotFoundError as e:
#     raise FileNotFoundError(f"The directory {path} does not exist (absolute path: {os.path.abspath(path)}).") from e
#   try:
#       yield
#   finally:
#     # Change back to initial working directory
#     os.chdir(oldpwd)


def run_shell_cmd(cmd: Union[str, List[str]], log=None, working_dir=None):
  if isinstance(cmd, str):
    cmd_print_string = ">> " + cmd
  else:
    cmd_print_string = ">> " + " ".join(cmd)
  
  # If the working directory is provided, then prepend it to the printed command.
  if working_dir:
    if debug_levels.debug_shell_calls_level >= 2:
      cmd_print_string = f"{working_dir}/" + cmd_print_string
    elif debug_levels.debug_shell_calls_level >= 1:
      cmd_print_string = os.path.basename(working_dir) + "/" + cmd_print_string

  # Print the command and if a log is given, then write the command there too.
  if debug_levels.debug_shell_calls_level >= 1:
    # Write the command string to the log, if provided.
    if log:
      log.write(cmd_print_string + "\n")
      log.flush()
    # Print to standard out.
    print(cmd_print_string)

  # We update the working directory after printing the string so that it is 
  # more evident to the developer that no working directory was explicitly given
  # but we will still print the working directory if an error is generated.
  if not working_dir:
    working_dir = os.getcwd()
    
  try:
    subprocess.check_call(cmd, stdout=log, stderr=log, cwd=working_dir)
  except Exception as e:
    err_msg = f'ERROR when running shell command "{cmd}" in {working_dir}:\n\t{str(e)}'
    if log: 
      
      # If using an external log file, print the contents.
      log.write(err_msg)
      log.flush()
      log.seek(0)
      print(f"({os.path.basename(log.name)}) ".join([''] + log.readlines()))
      print(f"End of {log.name})")
    print(err_msg)
    raise Exception(err_msg) from e
    

def loadModuleFromWorkingDir(module_name):
  """
  Given the name of a Python package (file) in the current directory, load it as a module.
  Inputs:
    - <module_name> The base name of the Python file (without the '.py' extension) and the module.
  """
  file_path = os.path.abspath(module_name + ".py")
  return loadModuleFromPath(module_name, file_path)

def loadModuleInDir(module_directory, module_name):
  """ Load a module and return it. In order for it to be accesible from the scope of the caller, you must assign the return value to a variable.
  """
  assertFileExists(module_directory, 'The module directory does not exist.')
  file_path = os.path.join(module_directory, module_name + ".py")

  return loadModuleFromPath(module_name, file_path)

def loadModuleFromPath(module_name, file_path):
  """ Load a module and return it. In order for it to be accesible from the scope of the caller, you must assign the return value to a variable.
  """
  assertFileExists(file_path, f'The requested module does not exist at {file_path}.')

  # Create a spec from the file location
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  # Create a new module based on the spec
  module = importlib.util.module_from_spec(spec)
  # Add the module to sys.modules to make it importable
  sys.modules[module_name] = module
  # Execute the module (loads the module's code)
  spec.loader.exec_module(module)
  return module



from functools import wraps

def indented_print(func):
    """ 
    This function defines a decorator that causes all of the print statements within a function to be indented. 
    Nesting is supported.
    This function was drafted by ChatGPT. 
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Define custom print with indentation. 
        previous_print_fnc = print
        indent = "|"

        # Because custom_print is defined recursively, using the previous_print_fnc, we get 
        # increasing indentations when a indented_print function is used inside another indented_print function
        def custom_print(*args, **kwargs):
            # Prepend the indentation to the first argument.
            # I'm not sure why it is necessary, but appending '+ " "' makes the alignement work out correctly (although with an extra space.)
            args = [str(arg).replace('\n', '\n' + indent + " ") for arg in args]
            previous_print_fnc(indent, *args, **kwargs)

        # Override the built-in print function
        builtins_print = sys.modules['builtins'].print
        sys.modules['builtins'].print = custom_print
        try:
            result = func(*args, **kwargs)  # Call the original function
        finally:
            # Restore the original print function and indent level
            sys.modules['builtins'].print = previous_print_fnc

        return result
    return wrapper

def seconds_to_duration_string(seconds: float) -> str:
  # Calculate hours, minutes, and seconds
  hours = int(seconds // 3600)
  mins = int((seconds % 3600) // 60)
  sec = seconds % 60  # Keep the remainder as float for fractional seconds

  # Build the formatted time string
  time_str = ""
  if hours > 0:
      time_str += f'{hours}h '
  if hours > 0 or mins > 0:
      time_str += f'{mins}m '
  
  # Format seconds to two decimal places if it's a float, or as integer if it's whole
  time_str += f'{sec:.2f}s' if sec % 1 else f'{int(sec)}s'

  return time_str.strip()
def assertLength(array, length, label=None):
  if len(array) != length:
    if None:
      err_str = f"The length of the array was {len(array)} but {length} was expected."
    else:
      err_str = f"The length of the array {label} was {len(array)} but {length} was expected."
    raise ValueError(err_str)