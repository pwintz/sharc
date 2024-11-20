#!/usr/bin/env python3
# Portabilize trace by copying binary dependencies to a local directory
# Usage: python portabilize_trace.py [trace directory, e.g. ~/drmemtrace.trace1.1234.2134.dir/]
#

from shutil import copy
from os import mkdir
from os import path
import sys
import argparse


parser = argparse.ArgumentParser(description="Portabilize a trace by copying binary dependencies to a local directory.")

parser.add_argument(
    'trace_directory',
    help="Enter relative or absolute path the the trace directory, such as \"~/drmemtrace.trace1.1234.2134.dir/\"."
)

# Parse the arguments from the command line
args = parser.parse_args()

traceDir = path.abspath(args.trace_directory)

if not path.exists(traceDir):
  raise ValueError(f"The directory {traceDir} does not exist.")

if not path.exists(traceDir + "/bin/modules.log"):
  raise ValueError(f"The directory {traceDir} does not contain 'bin/modules.log'.")

print(f'Portabilizing the DynamoRIO trace in "{traceDir}"')
data = []
with open(traceDir + '/bin/modules.log', 'r') as infile:
    print(infile.readlines())
    separator = ', '
    first = 1
    col = 99
    for line in infile:
        s = line.split(separator)
        if first:
            ss = s[0].split(' ')
            
            first = 0
            if ss[2] != 'version':
                raise ValueError('Corrupt file format'+s[2])
            else:
                #version == 5
                if ss[3] == '5':
                    col = 8
                #earlier versions
                elif ss[3] < '5':
                    col = 7
                else:
                    raise ValueError('Unrecognized file format, please add support')
                    
        # Skip over but preserve lines that don't describe libraries
        if len(s) < col+1 or s[col][0] != '/':
            data.append(line);
            continue;
            
        libPath = s[col].strip()
        copy(libPath, binPath)
        # Modify the path to the library to point to new, relative path
        libName = path.basename(libPath)
        newLibPath = path.abspath(binPath + libName)
        s[col] = newLibPath + '\n'
        
        data.append(separator.join(s))

with open(traceDir + '/bin/modules.log', 'w') as outfile:
    for wline in data:
        outfile.write(wline)