#!/usr/bin/env bash
### Script for tracing (w/ DyanmoRIO) and simulating (with Scarab) an executable, in parallel. 
### 
### Example usage:
###   ./trace.sh <path_to_scarab> <executable> <start_index> <n_cores> <restart_flag> <PARAMS path> <controller params path> <sampling time> 

echo "============= trace.sh ============="

# Abort the script if any commands return a non-zero value, indicating an error.
# See https://stackoverflow.com/a/821419/6651650
set -Eeuo pipefail

# Get the current working directory
main_path=$(pwd)
echo "Current working directory: $main_path"

# Define the script path
# script_path="${main_path}/../../.."

# Get the scarab path from the first argument. UNUSED
# scarab_path=$1
# echo "Scarab path: $scarab_path"

# Get the simulation executable from the second argument. 
simulation_executable=$2
# Get the start index from the third argument
start_from=$3
# Get the number of iterations (number of cores) from the fourth argument and compute the last timestep
num_timesteps=$4
last_timestep=$(($start_from + $num_timesteps - 1))
# Get the restart flag from the fifth argument, whether it is restarted after a run
restart_flag=$5
# Get the chip parameters path from the sixth argument
chip_params_path=$6
# Get the controller parameters path from the seventh argument
controller_params_path=$7
# Get the control sampling time from the eighth argument
control_sampling_time=$8

echo " simulation_executable=$simulation_executable"
echo "            start_from=$start_from"
echo "         num_timesteps=$num_timesteps"
echo "         last_timestep=$last_timestep"
echo "          restart_flag=$restart_flag"
echo "      chip_params_path=$chip_params_path"
echo "controller_params_path=$controller_params_path"
echo " control_sampling_time=$control_sampling_time"

echo Start from: $start_from to $last_timestep, with $num_timesteps time steps.

# Determine the starting simulation index based on the restart flag and start position
if [ "$restart_flag" = false ] && [ "$start_from" -eq 1 ]; then
    start_simulation_from=0
else
    start_simulation_from="$start_from"
fi

### Create Timesteps Path
# Define the timesteps path for file naming
timesteps_path=$main_path"/Timesteps_${start_simulation_from}-${last_timestep}"

# Add a timestamp to the folder name to create a unique path
twist="_$(date +"%Y%m%d_%H%M%S")"
timesteps_path="${timesteps_path}${twist}"
# Check if the timesteps path already exists
if [ -e "$timesteps_path" ]; then
    echo "Path already exists: $timesteps_path"
    echo "Aborting."
    return 1
fi
echo "timesteps_path: ${timesteps_path}"

# Define the traces and simulation paths
traces_path=$timesteps_path/traces
simulation_path=$timesteps_path/simulation

# Create the necessary directories
mkdir -p "$traces_path"
echo "1. Created the traces path: $traces_path"
echo ""

mkdir -p "$simulation_path"
echo -e "2. Created the simulation path: $simulation_path"
echo ""

echo "3. Starting tracing using controller parameters file: $controller_params_path"
echo ""

cp $controller_params_path $timesteps_path
cp $controller_params_path $traces_path

# Move into parent folder
cd $traces_path

IS_RESTART=1
IS_NOT_RESTART=0

# Run the dynamics simulation and record states
# Run the simulation with or without the restart flag
if [ "$restart_flag" = "true" ]; then
    echo "Simulate (is a restart):"
    $simulation_executable $start_simulation_from $num_timesteps 0 $IS_RESTART
else
    echo "Simulate (not a restart):"
    echo Executing $simulation_executable
    $simulation_executable $start_simulation_from $num_timesteps 0 $IS_NOT_RESTART
    echo "Finished executing $simulation_executable"
fi
wait
sleep 1

# Run tracing in a loop for each timestep
for ((i = start_from; i < start_from + num_timesteps; i++)); do
    echo ""
    echo "Running timestep $i of $num_timesteps..."
    trace_path_i="${traces_path}/Timestep_${i}"
    mkdir -p "${trace_path_i}"
    echo "Created directory for timestep $i at ${trace_path_i}"
    # Move into the folder for the current time-steps
    cd "${trace_path_i}"
    
    n_timesteps=1
    is_record_trace=1

    # Run tracing with or without the restart flag. The executable uses the "dr_api.h" library that  
    if [[ "$restart_flag" == "true" ]] && [[ "$i" -eq "$start_from" ]]; then
        echo "tracing, mod 2"
        is_restart=1
        $simulation_executable $i $n_timesteps $is_record_trace $is_restart
    else
        echo "tracing, mod 1"
        echo $num_timesteps
        is_restart=0

        # Execute the controller within the time-step folder, which generates the trace file in that folder.
        $simulation_executable $i $n_timesteps $is_record_trace $is_restart
    fi
done
echo -e "-- Tracing ended"
echo ""

########################################
echo -e "4. Portabilizing the trace file started"
echo ""

# Define the portabilize script path
PORTABILIZE_SCRIPT="run_portabilize_trace.sh"
cd "$traces_path"

# Run the portabilize script in the background for each subfolder
{
    for subfolder in */; do
        if [ -d "$subfolder" ]; then
            (cd "$subfolder" && bash "${PORTABILIZE_SCRIPT}") & # <- run in separate shell.
        fi
    done
    wait
} &
wait
echo -e "--  Portabilizing the trace file ended "
echo ""
# Genera

echo -e "5. Generate a list of commands to simulate (simulation_commands.txt and plot_commands.txt)"
echo ""
# Define the base file name from the second argument
base_file_name=$(basename "$simulation_executable")

# Initialize counters and create command files and write the simulation and plotting commands to disk
counter=0
cd ${timesteps_path}
touch simulation_commands.txt
touch plot_commands.txt

# Loop through each timestep and generate simulation commands.
while [ "$counter" -lt "$num_timesteps" ]; do
    current_timestep=$((start_from + counter))
    timestep_dir="${traces_path}/Timestep_${current_timestep}"

    if [ -d "$timestep_dir" ]; then
        subfolder=$(find "$timestep_dir" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)
        echo ""
        echo "Processing subfolder $counter: $subfolder"
        cd "$subfolder" || exit 1
        trace_path="$(pwd)/trace"
        bin_path="$(pwd)/bin"
        echo -e "Trace path: ${trace_path}"
        echo -e "Bin path: ${bin_path}"    

        simulation_path_i="${simulation_path}/Timestep_${current_timestep}"
        mkdir -p "${simulation_path_i}"
        cd "${main_path}" || exit 1
        cd .. || exit 1
        echo "New path: ${simulation_path_i}/PARAMS.in"
        echo ""
        cp "$chip_params_path" "${simulation_path_i}/PARAMS.in"
        cd "${timesteps_path}" || exit 1
        echo "cd ${simulation_path_i}" >> simulation_commands.txt
        # Simulation command for Scarab
        command="scarab --fdip_enable 0 --frontend memtrace --fetch_off_path_ops 0 --cbp_trace_r0=${trace_path} --memtrace_modules_log=${bin_path}"
        echo "$command" >> simulation_commands.txt
    else
        echo "Timestep directory not found: $timestep_dir"
    fi

    ((counter++))
done

# Write plot commands to the plot_commands.txt file
# TODO: Do plotting after the fact.
# echo "cd $main_path" >> plot_commands.txt
# echo "python3 ${script_path}/../plot_cycles.py ${simulation_path} ${control_sampling_time}" >> plot_commands.txt
echo -e "5. Simulation commands are written to simulation_commands.txt file "