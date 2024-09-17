"""
This package provides several utility functions that are used by Scarab-in-the-loop.
"""

import os
import sys
import json
import subprocess
import copy

import importlib.util

from typing import List, Set, Dict, Tuple
from typing import Union

# Import contextmanager to allow defining commands to be used to create "with" blocks.
from contextlib import contextmanager 

def assertFileExists(path:str, append_msg=""):
  if not os.path.exists(path):
      raise IOError(f'Expected {path} to exist but it does not. The absolute path is {os.path.abspath(path)}' + append_msg)

def printIndented(string_to_print:str, indent: int=1):
  indent_str = '\t' * indent
  indented_line_break = "\n" + indent_str
  string_to_print = string_to_print.replace('\n', indented_line_break)
  print(indent_str + string_to_print)

def readJson(filename: str) -> Union[Dict,List]:
  assertFileExists(filename)
  with open(filename, 'r') as json_file:
    json_data = json.load(json_file)
    return json_data

def writeJson(filename: str, json_data: Union[Dict,List], label:str=None):
  """
  Write a dictionary to a file in JSON format. 
  If 'label' is given, then the path is printed to the stdout with the given label.
  """
  with open(filename, 'w') as file:
    file.write(json.dumps(json_data, indent=2))
  if label:
    print(f'{label}: ' + os.path.abspath(filename))

    
def printJson(label: str, json_data: Union[Dict,List]):
  """
  Pretty-print JSON to standard out.
  """
  print(f"{label}:\n{json.dumps(json_data, indent=2)}")


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

def checkBatchConfig(batch_config):
  simulation_label = batch_config["simulation_label"]
  max_time_steps = batch_config["max_time_steps"]
  x0 = batch_config["x0"]
  u0 = batch_config["u0"]
  first_time_index = batch_config["first_time_index"]
  last_time_index = batch_config["last_time_index"]

  max_time_steps_in_batch = batch_config["max_time_steps"]
  n_time_indices_in_bath = max_time_steps_in_batch + 1
  simulation_dir = batch_config["simulation_dir"]
  
  if max_time_steps_in_batch > os.cpu_count():
    raise ValueError(f"max_time_steps_in_batch={max_time_steps_in_batch} > os.cpu_count()")

  if last_time_index - first_time_index != max_time_steps_in_batch:
    raise ValueError(f"last_time_index - first_time_index = {last_time_index - first_time_index} != max_time_steps_in_batch = {max_time_steps_in_batch}")

  if max_time_steps_in_batch < 0:
    raise ValueError(f"A negative max_time_steps_in_batch was given: {max_time_steps_in_batch}")

  if batch_config["time_indices"][0] != first_time_index or batch_config["time_indices"][-1] != last_time_index:
    raise ValueError(f"time_indices doen't align.")
    

def checkSimulationData(batch_simulation_data, max_time_steps):
  n_time_indices = max_time_steps + 1

  def assert_len_equals_n_time_indices(name:str):
    array = batch_simulation_data[name]
    if len(array) != n_time_indices:
      raise ValueError(f"the length of x ({len(array)}) is not equal to the number of time indices ({n_time_indices}).")

  # There is one fewer time steps than time indices because each step is a step between indices.
  def assert_len_equals_n_time_steps(name:str):
    array = batch_simulation_data[name]
    if len(array) != n_time_indices - 1:
      raise ValueError(f"the length of {name} ({len(array)}) is not equal to the number of time steps ({n_time_indices - 1}).")

  # The control values applied. The first entry is the u0 previously computed. Subsequent entries are the computed values. The control u[i] is applied from t[i] to t[i+1], with u[-1] not applied during this batch. 
  assert_len_equals_n_time_indices("x")
  assert_len_equals_n_time_indices("u")
  assert_len_equals_n_time_indices("t")
  assert_len_equals_n_time_steps("t_delay")


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
def openLog(filename, headerText=None):
  if not headerText:
    headerText=f"Log: {filename}"
  print(f"Opening log: {os.path.abspath(filename)}")
  log_file = open(filename, 'w+')

  # Write a header for the log file.
  log_file.write(f'===={"="*len(headerText)}==== \n')
  log_file.write(f'=== {headerText} === \n')
  log_file.write(f'===={"="*len(headerText)}==== \n')
  log_file.flush()

  try:
    yield log_file
  finally:
    # Clean up
    log_file.close()

@contextmanager
def in_working_dir(path):
  """
  Temporarily change the working directory within a "with" block.
  Usage:

    with cwd('/home/bwayne'):
      <do stuff in the /home/bwayne/ directory>

  """
  oldpwd = os.getcwd()
  try:
    os.chdir(path)
  except FileNotFoundError as e:
    raise FileNotFoundError(f"The directory {path} does not exist (absolute path: {os.path.abspath(path)}).") from e
  try:
      yield
  finally:
    # Change back to initial working directory
    os.chdir(oldpwd)


def run_shell_cmd(cmd: Union[str, List[str]], log=None, working_dir=None):
  if isinstance(cmd, str):
    cmd_print_string = ">> " + cmd
  else:
    cmd_print_string = ">> " + " ".join(cmd)
  
  # If the working directory is provided, then prepend it to the printed command.
  if working_dir:
    cmd_print_string = f"{working_dir}/" + cmd_print_string

  # Print the command and if a log is given, then write the command there too.
  print(cmd_print_string)
  if log:
    log.write(cmd_print_string + "\n")
    log.flush()

  # We update the working directory after printing the string so that it is 
  # more evident to the developer that no working directory was explicitly given
  # but we will still print the working directory if an error is generated.
  if not working_dir:
    working_dir = os.getcwd()
    
  try:
    subprocess.check_call(cmd, stdout=log, stderr=log, cwd=working_dir)
  except Exception as e:
    err_msg = f'ERROR when running "{cmd}" in {working_dir}:\n\t{str(e)}'
    print(err_msg)
    log.write(err_msg)
    if log: 
      # If using an external log file, print the contents.
      log.flush()
      log.seek(0)
      print("(log) ".join([''] + log.readlines()))
    raise e

def loadModuleFromWorkingDir(module_name):
  """
  Given the name of a Python package (file) in the current directory, load it as a module.
  Inputs:
    - <module_name> The base name of the Python file (without the '.py' extension) and the module.
  """
  file_path = os.path.abspath(module_name + ".py")
  return loadModuleFromPath(module_name, file_path)

def loadModuleFromPath(module_name, file_path):
  """ Load a module and return it. In order for it to be accesible from the scope of the caller, you must assign the return value to a variable.
  """
  assertFileExists(file_path)

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

# Track current indent level globally
CURRENT_INDENT_LEVEL = 0

def indented_print(func):
    """ 
    This function defines a decorator that causes all of the print statements within a function to be indented. Nesting is supported.
    This function was generated by ChatGPT. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global CURRENT_INDENT_LEVEL
        
        # Increase the indent level for this function
        original_indent_level = CURRENT_INDENT_LEVEL
        CURRENT_INDENT_LEVEL += 1
        
        # Define custom print with indentation
        original_print = print
        indent = ' ' * (CURRENT_INDENT_LEVEL * 2)

        def custom_print(*args, **kwargs):
            # Prepend the indentation to the first argument
            original_print(indent, *args, **kwargs)

        # Override the built-in print function
        builtins_print = sys.modules['builtins'].print
        sys.modules['builtins'].print = custom_print
        try:
            result = func(*args, **kwargs)  # Call the original function
        finally:
            # Restore the original print function and indent level
            sys.modules['builtins'].print = original_print
            CURRENT_INDENT_LEVEL = original_indent_level  # Restore the indent level

        return result
    return wrapper