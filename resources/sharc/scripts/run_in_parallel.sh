#!/usr/bin/env bash

# Script to run this example.
# Example usage:
# 
# >> ./main.sh 0 1 ./chip_params/raspberry_pi_5/chip_cycle_time/PARAMS.in_0 controller_parameters/inverted_pendulum/params.json test_main_sh 0 build/inverted_pendulum_simulation

# Abort the script if any commands return a non-zero value, indicating an error.
# See https://stackoverflow.com/a/821419/6651650
set -Eeuo pipefail

# Set the Scarab folder path
# scarab_path="NOT USED" # $RESOURCES_DIR/scarab

# Get the directory of the script main.sh. 
# This should be the root directory of the example, such as "inverted-pendulum/" or "acc_example/". 
# We get the path of the script and not the user's current wd. 
main_path=$(cd "$(dirname "$0")" && pwd)
echo "Example root path: $main_path"
echo ""

### Get the script arguments ###
start_idx=$1 # Typically "0". We can set larger values to start 'halfway' starting midway from a prior experiment.
time_horizon=$2 # How many timesteps to simulate, in total.
chip_params_path=$3 # E.g., "chip_params/raspberry_pi_5"
control_sampling_time=$4 # E.g., "controller_parameters/params.json"
total_timesteps=$5 # 
exp_name=$6 # Define a name for the folders, which are generated.
PARAM_INDEX=$7 # Allows selecting from different PARAMS.in files from an array of 
simulation_executable=$8 # The path the executable (already ).
controller_params_path=$9

echo "             start_idx=$start_idx"
echo "          time_horizon=$time_horizon"
echo "      chip_params_path=$chip_params_path"
echo "controller_params_path=$controller_params_path"
echo "              exp_name=$exp_name"
echo "           PARAM_INDEX=$PARAM_INDEX (UNUSED)"
echo " simulation_executable=$simulation_executable"
echo

# Calculate the total timesteps needed to reach the time_horizon
# Read and parse the JSON file to get the control_sampling_time
# json=$(cat $controller_params_path)
# control_sampling_time=$(echo "$json" | grep -o '"control_sampling_time": *[0-9.]*' | awk -F': *' '{print $2}')
# total_timesteps=$(echo "($time_horizon / $control_sampling_time + 0.99999999999)/1" | bc)

echo "control_sampling_time: $control_sampling_time"
echo "total_timesteps: $total_timesteps"
echo ""

# Initialize variables
current_timestep=1
num_proc=$(nproc)
current_time=$(date +'%Y-%m-%d_%H-%M-%S')
# Experiment path. Something in the form 
experiment_path="$main_path/Scarab-Trace-and-Simulate-Script/results/${exp_name}_time_horizon${time_horizon}_control_sampling_time${control_sampling_time}_num_proc${num_proc}/${PARAM_INDEX}_$current_time"
echo Experiment path: $experiment_path
mkdir -p "$experiment_path"
restart_flag=false

# Main loop to run the simulation and trace scripts
while [ $current_timestep -lt $total_timesteps ]; do
    cd "$experiment_path"
    echo "Current path: $(pwd)"
    echo "Current timestep: $current_timestep"
    echo "restart_flag: $restart_flag"
    echo ""

    # Run trace.sh with the restart_flag as an argument. 
    trace.sh scarab $simulation_executable $current_timestep $num_proc $restart_flag $chip_params_path $controller_params_path $control_sampling_time

    exit
    # Determine the starting simulation position based on the restart flag and current timestep
    if [ "$restart_flag" = false ] && [ "$current_timestep" -eq 1 ]; then
        start_simulation_from=0
    else
        start_simulation_from="$current_timestep"
    fi
    
    timesteps_path="$experiment_path/Timesteps_${start_simulation_from}-$(($current_timestep + num_proc - 1))"

    # Run simulate.sh to simulate using scarab (renaned to parallelize_simulation_commands.sh)
    # eval "$main_path/Scarab-Trace-and-Simulate-Script/simulate.sh" ${timesteps_path}
    $main_path/Scarab-Trace-and-Simulate-Script/simulate.sh ${timesteps_path}

    # Get the new timestep from plot.sh. The plot.sh prints a value to the terminal, which we use to determine the next timestep. These plots as-we-go are useful for tracking the progress of the simulation.
    new_timestep=$($main_path/Scarab-Trace-and-Simulate-Script/plot.sh ${timesteps_path} | tail -n 1)
    restart_flag=false
    
    # Check the condition for restarting.
    if [ $new_timestep -lt $((current_timestep + num_proc)) ]; then
        restart_flag=true
    else
        restart_flag=false
    fi
    
    # Update the current timestep
    current_timestep=$new_timestep
    
    echo "Updated current timestep: $current_timestep"
    echo ""
done