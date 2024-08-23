#! /usr/bin/bash
# Previously named "simulate.sh". Called from main.sh.

timesteps_path=$1
input="$timesteps_path/simulation_commands.txt"
declare -a pids  # Array to store process IDs
counter=0
while IFS= read -r line; do
  ((counter++))
  echo "$line"
  if ((counter % 2 == 0)); then
    # Evaluate command in a in bacground process.
    eval "$line" &
    pids+=($!)  # Store the PID of the background process
  else
    eval "$line"
  fi
done < "$input"

# Wait for all background processes to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done
sleep 1