import os
import json
import subprocess

from typing import List, Set, Dict, Tuple
from typing import Union

# Import contextmanager to allow defining commands to be used to create "with" blocks.
from contextlib import contextmanager 

def assertFileExists(path:str):
  if not os.path.exists(path):
      raise IOError(f'Expected {path} to exist but it does not. The absolute path is {os.path.abspath(path)}')

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

def writeJson(filename: str, json_data: Union[Dict,List]):
  with open(filename, 'w') as file:
    file.write(json.dumps(json_data, indent=2))

    
def printJson(label: str, json_data: Union[Dict,List]):
  """
  Pretty-print JSON to standard out.
  """
  print(f"{label}:\n{json.dumps(json_data, indent=2)}")

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