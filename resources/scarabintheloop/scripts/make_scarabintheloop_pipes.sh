#! /usr/bin/bash
# Script for creating the pipe files needed to communicate between controller and plant dynamics when running a Scarab simulation in execution-driven mode. 

# Exit immediately if any command [1] has a non-zero exit status.
set -e 
# Error if using an undefined variable.
set -u 
# Make ERR trap inherited by shell functions.
set -E 
# Make the return value of a pipeline is the status of
# the last command to exit with a non-zero status,
# or zero if no command exited with a non-zero status
set -o pipefail 

## Declare an array variable
declare -a list_of_pipes=("x_predict_c++_to_py" \
                          "t_predict_c++_to_py" \
                          "u_c++_to_py" \
                          "x_py_to_c++" \
                          "iterations_c++_to_py" \
                          "t_delay_py_to_c++")

## Loop through the above array
for pipe_file in "${list_of_pipes[@]}" 
do
  # If the pipe file exists but is not pipe file, then delete it.
  if [ -e $pipe_file -a ! -p $pipe_file ]; then 
    rm $pipe_file
    echo "Deleted $pipe_file"
  fi;

  # If the pipe file does not exist, create it.
  if [ ! -e $pipe_file ]; then
    echo "Creating pipe file named \"$pipe_file\"."
    mkfifo $pipe_file
  fi;

  # Check that the file is actually a pipe.
  test -p $pipe_file
done

# # ---------------------------------------------
# #! /usr/bin/bash
# # Script for creating the file files needed to communicate between controller and plant dynamics when running a Scarab simulation in execution-driven mode. 
# 
# 
# 
# ## Declare an array variable
# declare -a list_of_files=("x_predict_c++_to_py" \
#                           "t_predict_c++_to_py" \
#                           "u_c++_to_py" \
#                           "x_py_to_c++" \
#                           "iterations_c++_to_py" \
#                           "t_delay_py_to_c++")
# 
# ## Loop through the above array
# for file in "${list_of_files[@]}" 
# do
#   # If the file file exists but is not pipe file, then delete it.
#   if [ -e $file -a ]; then 
#     rm $file
#     echo "Deleted $file"
#   fi;
# 
#   # If the file file does not exist, create it.
#   if [ ! -e $file ]; then
#     echo "Creating regular file named \"$file\"."
#     touch $file
#   fi;
# 
#   # Check that the file exists.
#   test -e $file
# done