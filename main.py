#!/usr/bin/env python3

### Read from Command Line Argument ###
# import sys
# print (sys.argv)

### Read from User Input ###
# text = input("Name: ")
# print('Hello ' + text)

import scarabizor
scarabizor

scarab = scarabizor.Scarab();
# trace_dir = scarab.trace_cmd('touch', '~/mytestfile.txt')
# scarab.simulate_trace_with_scarab("/workspaces/ros-docker/drmemtrace.python3.8.122943.1703.dir")
time_in_fs = scarab.simulate('ls')
exit

### Run External Process ###
import subprocess

# Run the given command and read the output.
cmd = ["ls", "-lah"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
output = process.stdout.read()

# Wait for the process is finished so that we can check the return code.
process.wait()

# Replace the escaped line breaks with actual line breaks.
output = str(output).replace("\\n", "\n")

# Display the output.
print(output)

# Check that the process gave a nonzero return code, indicating sucess.
did_process_suceed = not process.returncode 
assert did_process_suceed, f"Running command \"{' '.join(cmd)}\" failed."