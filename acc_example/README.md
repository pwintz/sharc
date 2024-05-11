# LibMPC with Scarab
This folder contains code for simulating (with Scarab) the execution of a controller on a given processor in order to determine the time to compute. 
Using the simulated computation time, we simulate the evolution of the control system incorporating the computation delays.
The solution to the ordinary differential equation (ODE) is done within Python and the communication is by writting to pipe files.

Before running this example, you need to setup this project using the directions in the README in the root of the repository.

## How-to Build and Run
This example can be built and run using `make`. The default `make` job compiles and runs the MPC closed-loop without Scarab. 
When the program starts, it will wait until another process starts reading from the pipe file it is reading from.  
In a different terminal window, run 
```
./plant_dynamics.py
``` 
which reads and writes to the necessary pipe files to communicate with the MPC executable. 
Using the values of `x` and `u` provided by the MPC process, `plant_dynamics.py` computes the evolution of the plant over some time period, then passes the new value of `x` back to the MPC process.

To simulate the MPC controller using Scarab, run 
```
make simulate
``` 
then run 
```
./plant_dynamics.py
```
in a different terminal.