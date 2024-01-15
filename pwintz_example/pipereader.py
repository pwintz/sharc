#!/usr/bin/env python3
while True:
   in_file = "c_to_python_pipe"
  out_file = "python_to_c_pipe"
  with open(in_file, 'r') as infile:
    input = infile.read()
    print(f'Text received by Python: "{input}"')

  with open(out_file, 'w') as outfile:
    msg = "This is text from Python."
    print(f'Writing "{msg}" to python_to_c.')
    outfile.write(f'{msg}')
  