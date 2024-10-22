#! /usr/bin/bash
input2="$1/plot_commands.txt"
while IFS= read -r line; do
  echo "$line"
  eval "$line"
done < "$input2"
wait  # Wait for all processes in the second loop to finish
